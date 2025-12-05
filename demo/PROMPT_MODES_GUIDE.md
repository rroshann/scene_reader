# Prompt Modes Guide - Top 3 Approaches

All three top approaches now support **two prompt modes**: `gaming` and `real_world`.

## Quick Start

### Approach 2.5 (YOLO+LLM)
```bash
# Real-world mode (default) - Best for navigation, outdoor scenes
./run_approach_2_5.sh

# Gaming mode - Best for game screens
./run_approach_2_5.sh --mode gaming
```

### Approach 3.5 (Specialized Multi-Model)
```bash
# Real-world mode (default) - Best for text/spatial scenes
./run_approach_3_5.sh

# Gaming mode - Best for game screens with text
./run_approach_3_5.sh --mode gaming
```

### Approach 5 (Streaming)
```bash
# Gaming mode (default) - Best for interactive games
./run_approach_5.sh

# Real-world mode - Best for navigation scenes
./run_approach_5.sh --mode real_world
```

## What Each Mode Does

### Gaming Mode
- **Focus:** Game outcomes, player status, game state, UI elements
- **Prioritizes:** Win/loss detection, turn information, game board analysis
- **Best for:** Tic-Tac-Toe, chess, puzzle games, any game with UI

### Real-World Mode
- **Focus:** Safety, navigation, obstacles, spatial layout
- **Prioritizes:** Hazards, crosswalks, doors, stairs, actionable navigation info
- **Best for:** Street scenes, indoor navigation, outdoor environments

## Examples

### Testing Gaming Mode
1. Open Tic-Tac-Toe game
2. Run: `./run_approach_2_5.sh --mode gaming`
3. Press 'D' - Should detect game state, win/loss, board layout

### Testing Real-World Mode
1. Open street view or navigation app
2. Run: `./run_approach_2_5.sh --mode real_world`
3. Press 'D' - Should detect obstacles, crosswalks, navigation cues

## Technical Details

### Approach 2.5
- **Gaming prompts:** Focus on game state, win/loss, player status
- **Real-world prompts:** Focus on obstacles, navigation, safety

### Approach 3.5
- **Gaming prompts:** OCR + Depth optimized for game boards
- **Real-world prompts:** OCR + Depth optimized for signs/spatial navigation

### Approach 5
- **Gaming prompts:** Streaming with game-focused analysis
- **Real-world prompts:** Streaming with safety-focused analysis

## Default Modes

- **Approach 2.5:** `real_world` (optimized for navigation)
- **Approach 3.5:** `real_world` (optimized for text/spatial)
- **Approach 5:** `gaming` (optimized for interactive games)

## Comparison Testing

To compare prompt effectiveness:

1. **Test same image with different modes:**
   ```bash
   # Gaming mode
   ./run_approach_2_5.sh --mode gaming
   
   # Real-world mode  
   ./run_approach_2_5.sh --mode real_world
   ```

2. **Compare descriptions:**
   - Gaming mode should prioritize game state
   - Real-world mode should prioritize safety/navigation

3. **Measure latency:** Should be similar (prompts don't affect speed much)

## Notes

- Mode only affects **prompts**, not architecture or latency
- All approaches maintain their core strengths regardless of mode
- Mode selection helps optimize **description quality** for specific use cases

