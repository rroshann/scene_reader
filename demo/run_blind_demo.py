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
    print("Checking dependencies...")
    
    missing = []
    
    # Check TTS (macOS 'say' command)
    import shutil
    if shutil.which('say'):
        print("✓ macOS 'say' command (text-to-speech)")
    else:
        print("✗ macOS 'say' command not found")
        missing.append("macOS 'say' command")
    
    # Check screen capture
    try:
        import mss
        print("✓ mss (screen capture)")
    except ImportError:
        try:
            from PIL import ImageGrab
            print("✓ Pillow (screen capture)")
        except ImportError:
            print("✗ mss or Pillow not found")
            missing.append("mss or Pillow")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set - will use fallback board description")
    else:
        print("✓ OPENAI_API_KEY configured")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
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
        print("✓ TTS engine initialized (using macOS 'say' command)")
    
    
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
                    print(f"🔊 TTS speaking: {text[:60]}...")
                    
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
                    
                    print("✓ TTS finished speaking")
                    
                except Exception as e:
                    print(f"❌ TTS error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    with self.lock:
                        self.is_speaking = False
                        self.current_process = None
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS queue error: {e}")
                import traceback
                traceback.print_exc()
    
    def speak(self, text):
        """Add text to speech queue"""
        if not text or not text.strip():
            print("⚠ TTS: Empty text, skipping")
            return
        
        # Check if TTS thread is alive
        if not self.thread.is_alive():
            print("⚠ TTS thread died! Restarting...")
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()
            print("✓ TTS thread restarted")
        
        print(f"📝 TTS queueing: {text[:60]}... (queue size: {self.queue.qsize()})")
        self.queue.put(text)
        print(f"✓ Queued (new size: {self.queue.qsize()})")
    
    def interrupt(self):
        """Interrupt current speech and clear queue immediately"""
        # Kill current 'say' process if speaking
        with self.lock:
            if self.is_speaking and self.current_process:
                try:
                    print("✓ TTS interrupted - killing 'say' process")
                    self.current_process.terminate()  # Send SIGTERM
                    try:
                        self.current_process.wait(timeout=0.5)  # Wait briefly
                    except self.subprocess.TimeoutExpired:
                        self.current_process.kill()  # Force kill if needed
                    self.current_process = None
                except Exception as e:
                    print(f"⚠ Error stopping TTS: {e}")
        
        # Clear all queued items
        cleared = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        
        if cleared > 0:
            print(f"✓ Cleared {cleared} queued item(s)")
    
    def shutdown(self):
        """Shut down the TTS engine"""
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=1)


def main():
    """Main entry point"""
    print("="*70)
    print("Blind-Accessible Tic-Tac-Toe with AI Assistance")
    print("="*70)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    print()
    print("="*70)
    print("Starting demo...")
    print("="*70)
    print()
    
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
    
    print()
    print("="*70)
    print("CONTROLS:")
    print("  1-9  : Place your piece on squares 1-9")
    print("  D    : Get AI description of the board")
    print("  V    : Voice command mode (ask specific questions)")
    print("  R    : Reset the game")
    print("  H    : Help")
    print("="*70)
    print()
    print("Game window should now be visible.")
    print("Listen for voice instructions!")
    print()
    
    # Run game
    try:
        game.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tts_engine.shutdown()


if __name__ == "__main__":
    main()


