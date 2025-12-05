"""
Voice Command Recognition Module
Handles speech-to-text for voice commands in demos
"""
import speech_recognition as sr
import threading
import time


class VoiceCommandRecognizer:
    """Voice recognition for user commands"""
    
    def __init__(self, tts_callback=None):
        """
        Initialize voice recognizer
        
        Args:
            tts_callback: Optional callback function to speak feedback (e.g., "Listening...")
        """
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.tts = tts_callback
        
        # Try to initialize microphone
        try:
            self.microphone = sr.Microphone()
            print("✓ Microphone initialized")
        except Exception as e:
            print(f"⚠ Microphone initialization failed: {e}")
            print("  Voice commands will not be available")
    
    def listen_for_command(self, timeout=5, phrase_time_limit=5):
        """
        Listen for voice command and return transcribed text
        
        Args:
            timeout: Maximum seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for speech phrase
            
        Returns:
            str: Transcribed text, or None if failed/no speech detected
        """
        if self.microphone is None:
            if self.tts:
                self.tts("Microphone not available.")
            return None
        
        try:
            # Adjust for ambient noise
            print("🎤 Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for command
            print("🎤 Listening for voice command...")
            # Note: Don't use TTS here - it gets picked up by microphone
            # User will see "Listening..." in terminal/UI instead
            # Small delay to ensure any previous TTS has stopped
            import time
            time.sleep(0.3)
            
            with self.microphone as source:
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    print("⚠ No speech detected within timeout")
                    if self.tts:
                        self.tts("No speech detected. Try again.")
                    return None
            
            # Recognize speech using Google Speech Recognition
            print("🔄 Transcribing speech...")
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✓ Transcribed: {text}")
                return text
            except sr.UnknownValueError:
                print("⚠ Could not understand audio")
                if self.tts:
                    self.tts("Could not understand. Please try again.")
                return None
            except sr.RequestError as e:
                print(f"⚠ Speech recognition service error: {e}")
                if self.tts:
                    self.tts("Speech recognition unavailable. Check internet connection.")
                return None
                
        except Exception as e:
            print(f"❌ Voice recognition error: {e}")
            if self.tts:
                self.tts("Voice recognition failed.")
            return None
    
    def is_available(self):
        """Check if microphone is available"""
        return self.microphone is not None

