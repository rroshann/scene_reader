#!/usr/bin/env python3
"""
Approach 3.5 Demo: Optimized Specialized Multi-Model System
~1.5s latency - OCR/Depth + YOLO + GPT-3.5-turbo
Press D key anywhere to analyze screen
"""
import os
# Fix tokenizers fork crash: disable parallelism before any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import argparse
from pathlib import Path
import queue
import threading
import subprocess
import platform
import shlex
import time
import base64
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "demo"))

load_dotenv(dotenv_path=project_root / ".env")

# Import screen capture
from screen_capture import capture_region

# Import voice command recognition
try:
    from voice_command import VoiceCommandRecognizer
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"Warning: Voice commands not available: {e}")

# Import Approach 3.5
try:
    sys.path.insert(0, str(project_root / "code" / "approach_3_5_optimized"))
    from specialized_pipeline_optimized import run_specialized_pipeline_optimized
    APPROACH_3_5_AVAILABLE = True
except ImportError as e:
    APPROACH_3_5_AVAILABLE = False
    print(f"Warning: Approach 3.5 not available: {e}")


class TTSEngine:
    """TTS using macOS 'say' command"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.current_process = None
        self.is_speaking = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()
        print("✓ TTS engine initialized")
    
    def _process_queue(self):
        """Process TTS queue"""
        while True:
            try:
                text = self.queue.get(timeout=1)
                if text is None:
                    break
                
                with self.lock:
                    self.is_speaking = True
                
                try:
                    print(f"🔊 Speaking: {text[:60]}...")
                    # Preprocess text to fix contractions and common pronunciation issues
                    text = self._fix_pronunciation(text)
                    safe_text = shlex.quote(text)
                    cmd = ['say', '-v', 'Alex', '-r', '200', safe_text]
                    
                    with self.lock:
                        self.current_process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            start_new_session=True
                        )
                    
                    self.current_process.wait()
                    print("✓ Finished speaking")
                    
                except Exception as e:
                    print(f"❌ TTS error: {e}")
                finally:
                    with self.lock:
                        self.is_speaking = False
                        self.current_process = None
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
    
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
        """Queue text to speak"""
        if not text or not text.strip():
            return
        self.queue.put(text)
    
    def interrupt(self):
        """Interrupt current speech"""
        with self.lock:
            if self.is_speaking and self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=0.5)
                except:
                    self.current_process.kill()
                self.current_process = None
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break


class Approach35ScreenReader:
    """Screen reader using Approach 3.5: Specialized Multi-Model"""
    
    def __init__(self, tts_callback, tts_stop_callback, prompt_mode='real_world'):
        self.tts = tts_callback
        self.tts_stop = tts_stop_callback
        self.prompt_mode = prompt_mode
        self.lock = threading.Lock()
        self.thinking_sound_active = False  # Track thinking sound state
        self.user_action_start_time = None  # Track when user pressed D or finished speaking
        
        # Initialize voice recognizer
        if VOICE_AVAILABLE:
            try:
                self.voice_recognizer = VoiceCommandRecognizer(tts_callback=tts_callback)
                print("✓ Voice command recognizer initialized")
            except Exception as e:
                print(f"⚠ Voice recognizer initialization failed: {e}")
                self.voice_recognizer = None
        else:
            self.voice_recognizer = None
        
        # Use pynput for global hotkey
        try:
            from pynput import keyboard
            self.keyboard = keyboard
            
            def on_press(key):
                try:
                    if hasattr(key, 'char'):
                        if key.char == 'd':
                            self._trigger_analysis()
                        elif key.char == 'v':
                            self._handle_voice_command()
                except AttributeError:
                    pass
            
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            print("✓ Global hotkey listener started (D key for description, V key for voice command)")
            self.use_pynput = True
            
        except ImportError:
            print("⚠ pynput not available, using Tkinter (requires window focus)")
            import tkinter as tk
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.bind('<KeyPress-d>', lambda e: self._trigger_analysis())
            self.root.bind('<KeyPress-D>', lambda e: self._trigger_analysis())
            self.root.bind('<KeyPress-v>', lambda e: self._handle_voice_command())
            self.root.bind('<KeyPress-V>', lambda e: self._handle_voice_command())
            self.root.focus_set()
            self.listener = None
            self.use_pynput = False
        
        print("✓ Approach 3.5 screen reader initialized")
        print("✓ Press 'D' key anywhere to analyze screen")
        if self.voice_recognizer and self.voice_recognizer.is_available():
            print("✓ Press 'V' key for voice command mode")
    
    def _trigger_analysis(self, user_question=None):
        """Trigger screen analysis"""
        if not self.lock.acquire(blocking=False):
            self.tts("Already processing. Please wait.")
            return
        
        # Record start time for 'D' key press (user action)
        if not user_question:
            self.user_action_start_time = time.time()
        
        self.tts_stop()
        # If no user_question (pressing D), play normal "Analyzing" message
        if not user_question:
            self.tts("Analyzing screen with Approach 3.5...")
        # If user_question exists (voice command), thinking sound is already playing
        
        thread = threading.Thread(target=self._analyze_screen, daemon=True, args=(user_question,))
        thread.start()
    
    def _handle_voice_command(self):
        """Handle voice command mode"""
        if not self.voice_recognizer or not self.voice_recognizer.is_available():
            self.tts("Voice commands not available. Check microphone permissions.")
            return
        
        if not self.lock.acquire(blocking=False):
            self.tts("Already processing. Please wait.")
            return
        
        self.tts_stop()
        
        # Play beep sound to indicate ready to listen
        try:
            import subprocess
            import platform
            if platform.system() == 'Darwin':  # macOS
                # Use afplay to play a system beep
                subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Fallback: use system beep
                import sys
                sys.stdout.write('\a')
                sys.stdout.flush()
        except:
            pass  # Beep is optional
        
        # Listen for voice command in background thread (no delay - start immediately)
        def listen_and_analyze():
            try:
                question = self.voice_recognizer.listen_for_command(timeout=5, phrase_time_limit=5)
                if question:
                    print(f"Input: {question}")
                    # Record start time when user finishes speaking (voice command)
                    self.user_action_start_time = time.time()
                    # Play thinking sound instead of TTS
                    self._play_thinking_sound()
                    # Release lock before triggering analysis (it will acquire its own lock)
                    try:
                        self.lock.release()
                    except:
                        pass
                    # Now trigger analysis with the question
                    self._trigger_analysis(user_question=question)
                else:
                    # Release lock if no question received
                    try:
                        self.lock.release()
                    except:
                        pass
            except Exception as e:
                print(f"❌ Voice command error: {e}")
                try:
                    self.lock.release()
                except:
                    pass
        
        thread = threading.Thread(target=listen_and_analyze, daemon=True)
        thread.start()
    
    def _analyze_screen(self, user_question=None):
        """Analyze the current screen using Approach 3.5"""
        try:
            # Capture entire screen
            screenshot_path = "temp_screenshot_approach_3_5.png"
            try:
                import mss
                with mss.mss() as sct:
                    monitor = sct.monitors[0]
                    screen_width = monitor['width']
                    screen_height = monitor['height']
                    
                    screenshot = sct.grab(monitor)
                    from PIL import Image
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    img.save(screenshot_path)
                    
            except ImportError:
                screen_width = 1920
                screen_height = 1080
                capture_region(0, 0, screen_width, screen_height, screenshot_path)
                    
            except Exception as e:
                print(f"❌ Screenshot failed: {e}")
                self.tts("Could not capture screen.")
                return
            
            if not APPROACH_3_5_AVAILABLE:
                self.tts("Approach 3.5 not available. Please check dependencies.")
                return
            
            # Analyze with Approach 3.5
            start_time = time.time()
            
            # For gaming mode, use GPT-4-Vision directly (better for game boards, avoids OCR reading browser text)
            # For real-world mode, use OCR to catch signs
            if self.prompt_mode == 'gaming':
                # Use GPT-4-Vision directly for gaming (avoids OCR hallucination from browser text)
                try:
                    from openai import OpenAI
                    api_key = os.getenv('OPENAI_API_KEY')
                    if api_key:
                        client = OpenAI(api_key=api_key)
                        
                        with open(screenshot_path, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode()
                        
                        # Get gaming system prompt
                        sys.path.insert(0, str(project_root / "code" / "approach_3_5_optimized"))
                        from prompts_optimized import get_base_system_prompt
                        system_prompt = get_base_system_prompt('gaming')
                        user_prompt = "Your friend is playing a game and needs to know what's happening. Look at the GAME BOARD ONLY, ignore browser UI. Check for win or loss first (most important), then tell them the game state. Use simple, friendly words. Keep it very short - under 20 words."
                        
                        # Append user question if provided
                        if user_question:
                            system_prompt = f"{system_prompt}\n\nIMPORTANT: The user asked: '{user_question}'. Answer this question directly and specifically. Focus on what the user wants to know, not just general scene description."
                            user_prompt = f"{user_prompt}\n\nUser's question: {user_question}. Please answer this question directly based on what you see in the image."
                        
                        vision_start = time.time()
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                                        {"type": "text", "text": user_prompt}
                                    ]
                                }
                            ],
                            max_tokens=100,
                            temperature=0.9
                        )
                        vision_latency = time.time() - vision_start
                        description = response.choices[0].message.content
                        
                        self._stop_thinking_sound()
                        
                        # Calculate latency from user action to when answer starts speaking
                        if self.user_action_start_time:
                            user_action_latency = time.time() - self.user_action_start_time
                            self.user_action_start_time = None  # Reset
                        else:
                            user_action_latency = vision_latency
                        
                        # Print simplified output
                        if not user_question:
                            print(f"Input: Looking at the entire screen")
                        print(f"Latency: {user_action_latency:.2f}s")
                        print(f"Output: {description}")
                        
                        self.tts(description)
                        
                        # Clean up
                        if os.path.exists(screenshot_path):
                            os.remove(screenshot_path)
                        return
                    else:
                        print("⚠️  OpenAI API key not found, falling back to OCR mode")
                except Exception as e:
                    print(f"❌ GPT-4-Vision failed: {e}, falling back to OCR mode")
            
            # For real-world mode, prefer OCR to catch signs and text
            if self.prompt_mode == 'real_world':
                mode = 'ocr'  # Force OCR mode for real-world to catch signs
            else:
                mode = 'auto'  # Auto-select for gaming (fallback)
            
            # Get system prompt and append user question if provided
            sys.path.insert(0, str(project_root / "code" / "approach_3_5_optimized"))
            from prompts_optimized import get_ocr_system_prompt, get_depth_system_prompt, get_base_system_prompt
            
            # Select appropriate system prompt based on mode
            if mode == 'ocr':
                system_prompt = get_ocr_system_prompt(self.prompt_mode)
            elif mode == 'depth':
                system_prompt = get_depth_system_prompt(self.prompt_mode)
            else:
                system_prompt = get_base_system_prompt(self.prompt_mode)
            
            # Append user question if provided
            if user_question:
                system_prompt = f"{system_prompt}\n\nIMPORTANT: The user asked: '{user_question}'. Answer this question directly and specifically. Focus on what the user wants to know, not just general scene description."
            
            result = run_specialized_pipeline_optimized(
                image_path=Path(screenshot_path),
                mode=mode,
                category='text' if self.prompt_mode == 'real_world' else None,  # Force text category for real-world
                yolo_size='n',  # Nano for speed
                llm_model='gpt-3.5-turbo',
                confidence_threshold=0.25,
                system_prompt=system_prompt,
                use_cache=True,
                use_adaptive=True,
                quality_mode='balanced',
                prompt_mode=self.prompt_mode,
                user_question=user_question  # Pass user question to pipeline
            )
            
            latency = time.time() - start_time
            
            if result.get('success'):
                description = result.get('description', "Could not analyze the screen.")
                total_latency = result.get('total_latency', latency)
                mode_used = result.get('mode', 'unknown')
                ocr_latency = result.get('ocr_latency', 0)
                depth_latency = result.get('depth_latency', 0)
                detection_latency = result.get('detection_latency', 0)
                generation_latency = result.get('generation_latency', 0)
                
                # Check if YOLO detected 0 objects and we're in gaming mode - use GPT-4-Vision fallback
                generic_messages = ["no game information", "no objects detected", "could not analyze", "please share"]
                is_generic = any(msg.lower() in description.lower() for msg in generic_messages)
                
                if self.prompt_mode == 'gaming' and is_generic:
                    try:
                        from openai import OpenAI
                        api_key = os.getenv('OPENAI_API_KEY')
                        if api_key:
                            client = OpenAI(api_key=api_key)
                            
                            with open(screenshot_path, "rb") as f:
                                image_data = base64.b64encode(f.read()).decode()
                            
                            # Get gaming system prompt
                            sys.path.insert(0, str(project_root / "code" / "approach_3_5_optimized"))
                            from prompts_optimized import get_base_system_prompt
                            system_prompt = get_base_system_prompt('gaming')
                            
                            vision_start = time.time()
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                                            {"type": "text", "text": "Your friend is playing a game and needs to know what's happening. Check for win or loss first (most important), then tell them the game state. Use simple, friendly words. Keep it very short - under 20 words."}
                                        ]
                                    }
                                ],
                                max_tokens=100,
                                temperature=0.9
                            )
                            vision_latency = time.time() - vision_start
                            description = response.choices[0].message.content
                            
                            # Calculate latency from user action to when answer starts speaking
                            if self.user_action_start_time:
                                user_action_latency = time.time() - self.user_action_start_time
                                self.user_action_start_time = None  # Reset
                            else:
                                user_action_latency = vision_latency
                            
                            # Print simplified output
                            if not user_question:
                                print(f"Input: Looking at the entire screen")
                            print(f"Latency: {user_action_latency:.2f}s")
                            print(f"Output: {description}")
                            
                            self._stop_thinking_sound()
                            self.tts(description)
                        else:
                            self.tts("OpenAI API key not found.")
                    except Exception as e:
                        print(f"❌ GPT-4-Vision fallback failed: {e}")
                        # Fall back to original description
                        self._stop_thinking_sound()
                        
                        # Calculate latency from user action to when answer starts speaking
                        if self.user_action_start_time:
                            user_action_latency = time.time() - self.user_action_start_time
                            self.user_action_start_time = None  # Reset
                        else:
                            user_action_latency = total_latency
                        
                        # Print simplified output
                        if not user_question:
                            print(f"Input: Looking at the entire screen")
                        print(f"Latency: {user_action_latency:.2f}s")
                        print(f"Output: {description}")
                        
                        self.tts(description)
                else:
                    # Stop thinking sound before speaking result
                    self._stop_thinking_sound()
                    
                    # Calculate latency from user action to when answer starts speaking
                    if self.user_action_start_time:
                        user_action_latency = time.time() - self.user_action_start_time
                        self.user_action_start_time = None  # Reset
                    else:
                        user_action_latency = total_latency
                    
                    # Print simplified output
                    if not user_question:
                        print(f"Input: Looking at the entire screen")
                    print(f"Latency: {user_action_latency:.2f}s")
                    print(f"Output: {description}")
                    
                    # Speak result
                    self.tts(description)
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ Approach 3.5 failed: {error}")
                self.tts(f"Analysis failed: {str(error)[:50]}")
            
            # Clean up
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)
        
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
            self.tts(f"Error analyzing screen: {str(e)[:50]}")
        
        finally:
            self._stop_thinking_sound()
            self.lock.release()
    
    def _play_thinking_sound(self):
        """Play a subtle thinking sound continuously"""
        try:
            import subprocess
            import platform
            import threading
            self._stop_thinking_sound()  # Stop any existing sound
            
            if platform.system() == 'Darwin':  # macOS
                # Use a simple beep loop with a subtle sound
                self.thinking_sound_active = True
                
                def play_loop():
                    while getattr(self, 'thinking_sound_active', False):
                        try:
                            subprocess.Popen(
                                ['afplay', '/System/Library/Sounds/Submarine.aiff'],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True
                            )
                            import time
                            time.sleep(0.8)  # Play every 0.8 seconds
                        except:
                            break
                
                # Start a background thread to play the sound in a loop
                self.thinking_sound_thread = threading.Thread(target=play_loop, daemon=True)
                self.thinking_sound_thread.start()
            else:
                # Fallback: use system beep
                import sys
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception as e:
            print(f"⚠ Could not play thinking sound: {e}")
    
    def _stop_thinking_sound(self):
        """Stop the thinking sound"""
        try:
            self.thinking_sound_active = False
        except:
            pass
    
    def run(self):
        """Run the screen reader"""
        mode_label = "GAMING" if self.prompt_mode == 'gaming' else "TEXT/SPATIAL"
        print(f"APPROACH 3.5: OPTIMIZED SPECIALIZED MULTI-MODEL SYSTEM ({mode_label} MODE)")
        print("\nPress 'D' key anywhere to analyze what's on screen")
        print("Press Ctrl+C to exit\n")
        
        if self.use_pynput:
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            self.root.mainloop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Approach 3.5 Demo: Optimized Specialized Multi-Model')
    parser.add_argument('--mode', choices=['gaming', 'real_world'], default='real_world',
                        help='Prompt mode: gaming (for games) or real_world (for text/spatial)')
    args = parser.parse_args()
    
    if platform.system() != 'Darwin':
        print("⚠ This demo uses macOS 'say' command. May not work on other systems.")
    
    # Initialize TTS
    tts_engine = TTSEngine()
    
    # Create screen reader
    reader = Approach35ScreenReader(
        tts_callback=tts_engine.speak,
        tts_stop_callback=tts_engine.interrupt,
        prompt_mode=args.mode
    )
    
    try:
        reader.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tts_engine.queue.put(None)
        if tts_engine.thread.is_alive():
            tts_engine.thread.join(timeout=1)


if __name__ == "__main__":
    main()

