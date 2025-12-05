#!/usr/bin/env python3
"""
Blind-Accessible Tic-Tac-Toe Demo Entry Point
Simplified launcher with proper threading
"""
import os
# Fix tokenizers fork crash: disable parallelism before any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import tkinter as tk
from pathlib import Path
import queue
import threading

# Add project root and demo to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "demo"))

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

# Import demo components
from blind_accessible_game import BlindAccessibleTicTacToe
from blind_demo_controller import BlindDemoController
from screen_capture import capture_region


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    
    # Check TTS (macOS 'say' command)
    import shutil
    if not shutil.which('say'):
        missing.append("macOS 'say' command")
    
    # Check screen capture
    try:
        import mss
    except ImportError:
        try:
            from PIL import ImageGrab
        except ImportError:
            missing.append("mss or Pillow")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        pass  # Will use fallback
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r demo/requirements_demo.txt")
        return False
    
    return True


class TTSEngine:
    """Thread-safe Text-to-Speech engine using macOS 'say' command"""
    
    def __init__(self):
        import subprocess
        import platform
        
        self.subprocess = subprocess
        self.platform = platform
        self.queue = queue.Queue()
        self.current_process = None  # Track current 'say' process
        self.is_speaking = False
        self.lock = threading.Lock()  # Lock for process management
        
        # Check if we're on macOS
        if platform.system() != 'Darwin':
            print("⚠ Warning: Not on macOS, 'say' command may not work")
        
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
    
    
    def _process_queue(self):
        """Process TTS queue in background thread using 'say' command"""
        while True:
            try:
                text = self.queue.get(timeout=1)
                if text is None:  # Sentinel to stop
                    break
                
                with self.lock:
                    self.is_speaking = True
                
                try:
                    # Preprocess text to fix contractions and common pronunciation issues
                    text = self._fix_pronunciation(text)
                    
                    # Use macOS 'say' command with clearer voice (Alex) and rate control
                    # -v Alex = clearer, more natural voice
                    # -r 200 = speaking rate (words per minute)
                    # Escape text for shell safety
                    import shlex
                    safe_text = shlex.quote(text)
                    cmd = ['say', '-v', 'Alex', '-r', '200', safe_text]
                    
                    # Start process (start_new_session prevents fork crash with tokenizers)
                    with self.lock:
                        self.current_process = self.subprocess.Popen(
                            cmd,
                            stdout=self.subprocess.PIPE,
                            stderr=self.subprocess.PIPE,
                            start_new_session=True
                        )
                    
                    # Wait for process to complete
                    self.current_process.wait()
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                finally:
                    with self.lock:
                        self.is_speaking = False
                        self.current_process = None
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception:
                pass
    
    def _fix_pronunciation(self, text):
        """Fix common pronunciation issues in text"""
        # Fix contractions
        replacements = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "doesn't": "does not",
            "didn't": "did not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mustn't": "must not",
            "mightn't": "might not",
            "I'm": "I am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "I've": "I have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "I'll": "I will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "I'd": "I would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "we'd": "we would",
            "they'd": "they would",
        }
        
        # Replace contractions (case-insensitive)
        import re
        for contraction, expansion in replacements.items():
            # Match whole word only
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def speak(self, text):
        """Add text to speech queue"""
        if not text or not text.strip():
            return
        
        # Check if TTS thread is alive
        if not self.thread.is_alive():
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()
        
        self.queue.put(text)
    
    def interrupt(self):
        """Interrupt current speech and clear queue immediately"""
        # Kill current 'say' process if speaking
        with self.lock:
            if self.is_speaking and self.current_process:
                try:
                    self.current_process.terminate()  # Send SIGTERM
                    try:
                        self.current_process.wait(timeout=0.5)  # Wait briefly
                    except self.subprocess.TimeoutExpired:
                        self.current_process.kill()  # Force kill if needed
                    self.current_process = None
                except Exception:
                    pass
        
        # Clear all queued items
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Shut down the TTS engine"""
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def main():
    """Main entry point"""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize TTS
    tts_engine = TTSEngine()
    
    # Create Tkinter root
    root = tk.Tk()
    
    # Create game with TTS callbacks (AI callbacks set later)
    game = BlindAccessibleTicTacToe(
        root,
        tts_callback=tts_engine.speak,
        tts_stop_callback=tts_engine.interrupt,
        ai_callback=None,  # Set after controller is created
        ai_cancel_callback=None,  # Set after controller is created
        voice_command_callback=None  # Set after controller is created
    )
    
    # Create controller
    controller = BlindDemoController(
        game=game,
        tts_callback=tts_engine.speak,
        tts_stop_callback=tts_engine.interrupt,
        screen_capture_callback=capture_region
    )
    
    # Set AI callbacks in game
    game.ai_callback = controller.trigger_ai_analysis
    game.ai_cancel = controller.cancel_ai
    game.voice_command_callback = controller.trigger_voice_command
    
    # Run game
    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        tts_engine.shutdown()


if __name__ == "__main__":
    main()


