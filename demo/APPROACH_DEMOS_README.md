# Top 3 Approaches - Demo Files

Three separate demo files to test the fastest approaches independently.

## Quick Start

### Approach 2.5: Optimized YOLO+LLM (~1.1s)
```bash
./run_approach_2_5.sh
```
- **Architecture:** YOLOv8n + GPT-3.5-turbo + Caching
- **Best for:** Real-time scenes, speed-critical applications
- **Latency:** ~1.1 seconds

### Approach 3.5: Optimized Specialized (~1.5s)
```bash
./run_approach_3_5.sh
```
- **Architecture:** OCR/Depth + YOLO + GPT-3.5-turbo
- **Best for:** Text-heavy scenes, spatial understanding
- **Latency:** ~1.5 seconds

### Approach 1.5: Optimized Pure VLM (~1.73s perceived)
```bash
./run_approach_1_5.sh
```
- **Architecture:** Optimized GPT-4V (BLIP-2 optional tier1)
- **Best for:** Best perceived user experience
- **Perceived Latency:** ~1.73 seconds (first response)
- **Total Latency:** ~5.47 seconds (full description)

## How They Work

All three demos:
1. ✅ Use macOS `say` command for TTS
2. ✅ Capture full screen using `mss`
3. ✅ Global hotkey (D key) via `pynput`
4. ✅ Work with any game/application
5. ✅ Show latency in terminal output

## Differences

| Feature | Approach 2.5 | Approach 3.5 | Approach 1.5 |
|---------|--------------|--------------|--------------|
| **Speed** | Fastest (1.1s) | Fast (1.5s) | Perceived fastest (1.73s) |
| **Architecture** | YOLO+LLM | OCR/Depth+YOLO+LLM | Optimized GPT-4V |
| **Cost** | Low | Low | Medium |
| **Use Case** | General scenes | Text/spatial | User experience |
| **Output** | Single description | Single description | Two-tier (quick + detailed) |

## Usage

1. **Launch any game/application**
2. **Run one of the demos** (e.g., `./run_approach_2_5.sh`)
3. **Press 'D' key** anywhere to analyze screen
4. **Hear description** via TTS
5. **Compare** latency and quality between approaches

## Terminal Output

Each demo shows:
- Which approach is running
- Screenshot capture status
- Analysis progress
- Latency metrics
- Description preview

## Files

- `demo/run_approach_2_5.py` - Approach 2.5 implementation
- `demo/run_approach_3_5.py` - Approach 3.5 implementation
- `demo/run_approach_1_5.py` - Approach 1.5 implementation
- `run_approach_2_5.sh` - Launcher script
- `run_approach_3_5.sh` - Launcher script
- `run_approach_1_5.sh` - Launcher script

## Notes

- **Approach 1.5** uses async/await - may have slightly different behavior
- **Approach 3.5** may need model warmup on first run
- All require `OPENAI_API_KEY` in `.env` file
- All require `pynput` for global hotkeys (or use Tkinter fallback)

