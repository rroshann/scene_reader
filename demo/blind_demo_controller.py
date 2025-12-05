"""
Blind-Accessible Demo Controller
Simplified controller with proper threading for blind accessibility
"""
import sys
import os
import time
import threading
import queue
import base64
from pathlib import Path

# Add project root to path for Approach 2.5
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "code" / "approach_2_5_optimized"))

# Import Approach 2.5
try:
    from hybrid_pipeline_optimized import run_hybrid_pipeline_optimized
    APPROACH_2_5_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Approach 2.5 not available: {e}")
    APPROACH_2_5_AVAILABLE = False

# Import GPT-4-vision for game UI analysis
try:
    from openai import OpenAI
    GPT4V_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenAI not available: {e}")
    GPT4V_AVAILABLE = False

# Import voice command recognition
try:
    from voice_command import VoiceCommandRecognizer
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"Warning: Voice commands not available: {e}")


# Game-specific prompt for concise Tic-Tac-Toe descriptions using GPT-4-vision
GAME_SYSTEM_PROMPT = """You are analyzing a Tic-Tac-Toe game window for a blind player. Look at the ENTIRE window and describe the game state EXTREMELY concisely in under 25 words.

Priority (check in order):
1. FIRST check for game outcome - Look for status text like "Player X wins!", "Player O wins!", or "It's a tie!"
2. If game is over, state the outcome clearly
3. If game is ongoing, describe whose turn it is
4. Then briefly describe the board state

Squares are numbered 1-9 (top-left to bottom-right, row by row):
1 2 3
4 5 6  
7 8 9

Examples:
- "Player X wins! X has 1, 5, 9 forming a diagonal."
- "It's a tie! Board is full."
- "Player X's turn. X on 1, 5. O on 2, 3. Empty: 4, 6, 7, 8, 9."
"""

GAME_USER_PROMPT = "Analyze this Tic-Tac-Toe game window. Check for win/tie status messages first, then describe the board. Be extremely concise."


class BlindDemoController:
    """Simplified controller for blind-accessible demo"""
    
    def __init__(self, game, tts_callback, tts_stop_callback, screen_capture_callback):
        """
        Initialize controller
        
        Args:
            game: BlindAccessibleTicTacToe instance
            tts_callback: Function to speak text
            tts_stop_callback: Function to stop/interrupt current speech
            screen_capture_callback: Function to capture screen and return path
        """
        self.game = game
        self.tts = tts_callback
        self.tts_stop = tts_stop_callback
        self.screen_capture = screen_capture_callback
        
        # Initialize voice recognizer
        if VOICE_AVAILABLE:
            try:
                self.voice_recognizer = VoiceCommandRecognizer(tts_callback=tts_callback)
            except Exception as e:
                self.voice_recognizer = None
        else:
            self.voice_recognizer = None
        
        # Processing state
        self.processing_lock = threading.Lock()
        self.is_processing = False  # Track processing state
        self.cancel_processing = False
        
        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()
        self.thinking_sound_active = False  # Track thinking sound state
        
        # Check for messages periodically - start immediately and frequently
        self._check_messages()
    
    def _check_messages(self):
        """Check message queue and process (runs in main thread)"""
        try:
            # Process all pending messages
            processed = 0
            while True:
                try:
                    message_type, data = self.message_queue.get_nowait()
                    
                    if message_type == 'speak':
                        # Stop thinking sound before speaking
                        self._stop_thinking_sound()
                        # Call TTS directly - it handles its own queue
                        self.tts(data)
                    elif message_type == 'error':
                        self.tts(f"Error: {data}")
                    
                    self.message_queue.task_done()
                    processed += 1
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error in _check_messages: {e}")
            import traceback
            traceback.print_exc()
        
        # Schedule next check - very frequent for responsiveness
        try:
            self.game.master.after(20, self._check_messages)  # Check every 20ms
        except:
            pass  # Window might be closed
    
    def _analyze_with_gpt4_vision(self, image_path, user_question=None):
        """Use GPT-4-vision to analyze the game board directly"""
        try:
            if not GPT4V_AVAILABLE:
                return None, "GPT-4-vision not available"
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None, "API key not found"
            
            client = OpenAI(api_key=api_key)
            
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            start_time = time.time()
            
            # Use different prompts based on whether user has a specific question
            if user_question:
                # When user asks a specific question, prioritize answering it directly
                system_prompt = """You are analyzing a Tic-Tac-Toe game window. Answer ONLY the user's specific question. Do not provide any additional information.

Squares are numbered 1-9:
1 2 3
4 5 6  
7 8 9

CRITICAL: Answer with ONLY a direct yes/no or brief answer. Do NOT mention other squares, board state, or any other information unless specifically asked."""
                user_prompt = f"Question: {user_question}\n\nAnswer ONLY this question. Nothing else."
            else:
                # Standard board description
                system_prompt = GAME_SYSTEM_PROMPT
                user_prompt = GAME_USER_PROMPT
            
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
                max_tokens=30 if user_question else 60  # Much shorter for questions to force brevity
            )
            
            latency = time.time() - start_time
            description = response.choices[0].message.content
            
            return {
                'success': True,
                'description': description,
                'latency': latency
            }, None
            
        except Exception as e:
            return None, str(e)
    
    def trigger_ai_analysis(self, user_question=None):
        """Trigger AI analysis of the board (called from main thread)"""
        # Check if already processing using flag (faster than lock check)
        if self.is_processing:
            self.tts("AI is already processing. Please wait.")
            return
        
        # Try to acquire lock
        if not self.processing_lock.acquire(blocking=False):
            self.tts("AI is already processing. Please wait.")
            return
        
        # If no user_question (pressing D), play normal "Analyzing" message
        if not user_question:
            self.tts("Analyzing screen...")
        # If user_question exists (voice command), thinking sound is already playing
        
        # Set processing flag
        self.is_processing = True
        self.cancel_processing = False
        
        # Start analysis in background thread
        thread = threading.Thread(target=self._run_ai_analysis, daemon=True, args=(user_question,))
        thread.start()
    
    def trigger_voice_command(self):
        """Trigger voice command mode (called from main thread)"""
        if not self.voice_recognizer or not self.voice_recognizer.is_available():
            self.tts("Voice commands not available. Check microphone permissions.")
            return
        
        if self.is_processing:
            self.tts("AI is already processing. Please wait.")
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
                    # Play thinking sound instead of TTS
                    self._play_thinking_sound()
                    # Release lock before triggering analysis (it will acquire its own lock)
                    try:
                        if self.processing_lock.locked():
                            self.processing_lock.release()
                        self.is_processing = False
                    except:
                        pass
                    # Now trigger AI analysis with the question
                    self.trigger_ai_analysis(user_question=question)
                else:
                    # Release lock if no question received
                    try:
                        if self.processing_lock.locked():
                            self.processing_lock.release()
                        self.is_processing = False
                    except:
                        pass
            except Exception as e:
                try:
                    if self.processing_lock.locked():
                        self.processing_lock.release()
                    self.is_processing = False
                except:
                    pass
        
        thread = threading.Thread(target=listen_and_analyze, daemon=True)
        thread.start()
    
    def _run_ai_analysis(self, user_question=None):
        """Run AI analysis in background thread"""
        screenshot_path = None
        try:
            # Check if cancelled before starting
            if self.cancel_processing:
                return
            
            # Get window bounds (must be done in main thread context)
            bounds = self.game.get_window_bounds()
            
            # Capture screenshot
            screenshot_path = "temp_screenshot.png"
            try:
                self.screen_capture(
                    bounds['x'],
                    bounds['y'],
                    bounds['width'],
                    bounds['height'],
                    screenshot_path
                )
            except Exception as e:
                # Fallback to internal state
                description = self.game.get_board_state_description()
                self.message_queue.put(('speak', description))
                return
            
            # Check cancellation after capture
            if self.cancel_processing:
                return
            
            # Try Approach 2.5 first (fastest for gaming: 0.56s mean latency)
            result = None
            error = None
            
            if APPROACH_2_5_AVAILABLE:
                try:
                    from pathlib import Path
                    approach_2_5_result = run_hybrid_pipeline_optimized(
                        image_path=Path(screenshot_path),
                        prompt_mode='gaming',
                        use_cache=True,
                        user_question=user_question
                    )
                    
                    # Check if Approach 2.5 succeeded and detected objects
                    if approach_2_5_result.get('success') and approach_2_5_result.get('num_objects', 0) > 0:
                        result = {
                            'success': True,
                            'description': approach_2_5_result.get('description', "Could not analyze the board."),
                            'latency': approach_2_5_result.get('total_latency', 0)
                        }
                    # If no objects detected, fallback to GPT-4V (YOLO can't detect Tic-Tac-Toe elements)
                    elif approach_2_5_result.get('num_objects', 0) == 0:
                        # Fallback to GPT-4V for better Tic-Tac-Toe detection
                        result, error = self._analyze_with_gpt4_vision(screenshot_path, user_question=user_question)
                except Exception as e:
                    # If Approach 2.5 fails, fallback to GPT-4V
                    result, error = self._analyze_with_gpt4_vision(screenshot_path, user_question=user_question)
            else:
                # Approach 2.5 not available, use GPT-4V
                result, error = self._analyze_with_gpt4_vision(screenshot_path, user_question=user_question)
            
            # Check cancellation before speaking
            if self.cancel_processing:
                return
            
            # Get description
            if result and result.get('success'):
                description = result.get('description', "Could not analyze the board.")
                latency = result.get('latency', 0)
                
                # Print minimal output like approach demos
                if user_question:
                    print(f"Input: {user_question}")
                else:
                    print(f"Input: Looking at the entire screen")
                print(f"Latency: {latency:.2f}s")
                print(f"Output: {description}")
            else:
                # Fallback: use game's internal board state
                description = self.game.get_board_state_description()
            
            # Send to main thread via queue
            if not self.cancel_processing:
                self.message_queue.put(('speak', description))
        
        except Exception as e:
            # Fallback to internal board state
            if not self.cancel_processing:
                try:
                    description = self.game.get_board_state_description()
                    self.message_queue.put(('speak', description))
                except Exception as e2:
                    self.message_queue.put(('error', f"Could not analyze board: {e2}"))
        
        finally:
            # ALWAYS clean up and release lock
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                except:
                    pass
            
            # ALWAYS release lock and reset flag
            self.is_processing = False
            self.cancel_processing = False
            try:
                self.processing_lock.release()
            except:
                pass  # Lock might already be released
    
    def cancel_ai(self):
        """Cancel any ongoing AI processing"""
        if self.is_processing:
            self.cancel_processing = True
            self.tts_stop()  # Also interrupt any ongoing speech
            self._stop_thinking_sound()  # Stop thinking sound too
    
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
        except Exception:
            pass
    
    def _stop_thinking_sound(self):
        """Stop the thinking sound"""
        try:
            self.thinking_sound_active = False
        except:
            pass


