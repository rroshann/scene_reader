# 🎮 Blind-Accessible Tic-Tac-Toe with AI Assistance

A fully accessible Tic-Tac-Toe game designed for blind and low-vision users, featuring real-time AI scene description powered by Approach 2.5 (Optimized YOLO+LLM).

## ✨ Features

### Full Keyboard Control
- **Number keys (1-9)**: Place your piece on any square
  - Square 1 = Top-left
  - Square 3 = Top-right  
  - Square 5 = Center
  - Square 9 = Bottom-right
- **D key**: Get AI description of the current board state
- **R key**: Reset the game
- **H key**: Hear help instructions

### Voice Feedback for Everything
- Announces whose turn it is
- Confirms each move placement
- Announces game outcomes (win/tie)
- Reads AI board descriptions
- Provides error feedback

### Error Handling
- **Buzzer sound** when trying to place on occupied square
- Clear voice messages for all errors
- Helpful guidance to retry

### AI-Powered Assistance
- Press **D** anytime for AI analysis
- Fast response (~1.1 seconds with Approach 2.5)
- Describes board state and available moves
- Helps you make strategic decisions

---

## 🚀 Quick Start

### 1. Install Dependencies

All dependencies should already be installed. If not:

```bash
cd /Users/roshansiddartha/Documents/classes/3rd\ sem/genAI/github/scene_reader
source venv/bin/activate
pip install mss pyttsx3 pynput python-dotenv Pillow
```

### 2. Ensure API Key is Set

Make sure your `.env` file contains:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Demo

**Easy way:**
```bash
./run_blind_demo.sh
```

**Manual way:**
```bash
source venv/bin/activate
python demo/run_blind_demo.py
```

---

## 🎯 How to Play

### Game Start
1. Launch the demo
2. Listen for the welcome message explaining controls
3. You'll hear: *"Welcome to Tic Tac Toe. You are playing X..."*

### Making Moves
1. Listen for whose turn it is
2. Press a number key (1-9) to place your piece
3. Hear confirmation: *"Placed X on square 5"*

### Getting AI Help
1. Press **D** at any time
2. Wait ~1 second for AI analysis
3. Hear description of board state and available moves
4. Make your decision

### Handling Errors
- If you try to place on an occupied square:
  - You'll hear a **buzzer sound**
  - Voice will say: *"Square 5 is already occupied by O. Choose another square."*
  - Try a different number

### Winning the Game
- Game announces when someone wins or it's a tie
- Press **R** to play again

---

## 🎹 Complete Controls Reference

| Key | Action |
|-----|--------|
| `1` | Top-left square |
| `2` | Top-center square |
| `3` | Top-right square |
| `4` | Middle-left square |
| `5` | Center square |
| `6` | Middle-right square |
| `7` | Bottom-left square |
| `8` | Bottom-center square |
| `9` | Bottom-right square |
| `D` | AI board description |
| `R` | Reset game |
| `H` | Help (repeat instructions) |

---

## 🏗️ Architecture

### Components

1. **BlindAccessibleTicTacToe** (`blind_accessible_game.py`)
   - Fully keyboard-driven game logic
   - Voice announcements for all state changes
   - Error handling with audio feedback
   - High-contrast visual display (for sighted assistants)

2. **BlindDemoController** (`blind_demo_controller.py`)
   - Thread-safe AI processing
   - Integrates Approach 2.5 pipeline
   - Message queue for cross-thread communication
   - Fallback to text-based description if AI unavailable

3. **TTSEngine** (`run_blind_demo.py`)
   - Thread-safe text-to-speech
   - Queue-based speech management
   - Uses `pyttsx3` for offline TTS

4. **Screen Capture** (`screen_capture.py`)
   - Captures game window for AI analysis
   - Fast screenshot using `mss` library

### Threading Model

```
Main Thread (Tkinter UI)
  ├─> Keyboard input handling
  ├─> TTS queue checking (every 100ms)
  └─> UI updates
  
Background Thread (AI Processing)
  ├─> Screen capture
  ├─> Approach 2.5 pipeline
  └─> Results sent to main thread via queue
  
Background Thread (TTS)
  └─> Speech synthesis from queue
```

---

## 🔧 Troubleshooting

### No voice output?
- Check that `pyttsx3` is installed: `pip install pyttsx3`
- On macOS, ensure system TTS voices are installed (System Preferences > Accessibility > Spoken Content)

### Keyboard not responding?
- Click on the game window to ensure it has focus
- On macOS, grant accessibility permissions (System Preferences > Security & Privacy > Accessibility)

### AI not working?
- Check that `OPENAI_API_KEY` is set in `.env`
- The game will fall back to simple board state description if AI is unavailable

### Buzzer sound not playing?
- On macOS, system sounds should work by default
- You'll still get voice feedback even if sound fails

---

## 📊 Performance

- **AI Latency**: ~1.1 seconds (Approach 2.5 with GPT-3.5-turbo)
- **Voice Response**: Immediate (<100ms)
- **Total Time-to-Feedback**: ~1.5 seconds from pressing D

---

## 🎓 Design Principles

This demo was designed following accessibility best practices:

1. **Keyboard-only operation**: No mouse required
2. **Immediate feedback**: Every action confirmed with voice
3. **Error prevention**: Clear numbering system
4. **Error recovery**: Helpful messages and retry options
5. **Non-visual feedback**: Audio cues for all state changes
6. **Fast AI responses**: Sub-2-second latency for playability
7. **Graceful degradation**: Works even if AI is unavailable

---

## 🚨 Known Issues

1. **macOS permissions**: You may need to grant accessibility permissions to your terminal/IDE for keyboard listening to work globally
2. **First TTS may be slow**: The first speech output may have a slight delay while the engine initializes
3. **System beep**: Error buzzer uses system sounds which may vary by OS

---

## 🎬 Recording the Demo

For recording:
1. Use QuickTime or OBS to capture screen + audio
2. Speak your actions: "I'm pressing 5 to place X in the center"
3. Let AI voice play clearly in recording
4. Show error handling by trying to place on occupied square
5. Demonstrate D key for AI assistance

---

## 📝 Credits

- **Vision AI**: Approach 2.5 (Optimized YOLO + GPT-3.5-turbo)
- **TTS**: pyttsx3 library
- **Screen Capture**: mss library
- **Game Design**: Optimized for blind accessibility

---

## 🔜 Future Improvements

- Voice-activated commands (in addition to keyboard)
- Difficulty levels with computer opponent
- Sound effects for moves (distinct sounds for X vs O)
- Haptic feedback support
- Multi-language support


