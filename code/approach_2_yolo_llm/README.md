# Approach 2: YOLO + LLM Hybrid Pipeline

## Overview

Approach 2 implements a two-stage hybrid pipeline combining YOLOv8 object detection with LLM-based description generation. This approach offers faster inference and lower costs compared to pure Vision-Language Models (VLMs).

## Architecture

```
Image → YOLOv8 Detection → Format Objects → LLM Generation → Description
         (10-50ms)          (spatial info)    (500-1500ms)
```

## Components

### 1. YOLO Detection (`yolo_detector.py`)
- **YOLOv8 Variants:**
  - `n` (nano): Fastest (~10ms), smallest model
  - `m` (medium): Balanced (~30ms), better accuracy
  - `x` (xlarge): Most accurate (~50ms), best detection

- **Features:**
  - Object detection with bounding boxes
  - Spatial position calculation (left/center/right, top/center/bottom)
  - Confidence scoring
  - Object formatting for LLM prompts

### 2. LLM Generation (`llm_generator.py`)
- **Supported Models:**
  - GPT-4o-mini (OpenAI): Fast, cost-effective
  - Claude 3.5 Haiku (Anthropic): Good quality, reasonable cost

- **Features:**
  - Description generation from detected objects
  - Token usage tracking
  - Error handling and fallback models

### 3. Hybrid Pipeline (`hybrid_pipeline.py`)
- Orchestrates the two-stage process
- Tracks separate latencies for detection and generation
- Returns comprehensive results with metadata

## Usage

### Installation

```bash
# Install dependencies
pip install ultralytics opencv-python torch torchvision

# Or install all requirements
pip install -r requirements.txt
```

### Testing Single Image

```bash
# Test with default configuration (YOLOv8n + GPT-4o-mini)
python code/approach_2_yolo_llm/hybrid_pipeline.py <path_to_image>

# Or in Python
from code.approach_2_yolo_llm.hybrid_pipeline import run_hybrid_pipeline
from pathlib import Path

result = run_hybrid_pipeline(
    Path("data/images/gaming/tic_tac_toe-opp_move_1.png"),
    yolo_size='n',
    llm_model='gpt-4o-mini'
)
```

### Batch Testing

```bash
# Test all 6 configurations on all images
python code/approach_2_yolo_llm/batch_test_yolo_llm.py
```

**Configurations tested:**
- YOLOv8n + GPT-4o-mini
- YOLOv8n + Claude Haiku
- YOLOv8m + GPT-4o-mini
- YOLOv8m + Claude Haiku
- YOLOv8x + GPT-4o-mini
- YOLOv8x + Claude Haiku

**Total:** 42 images × 6 configurations = 252 API calls

## Analysis

### Quantitative Analysis

```bash
# Run comprehensive analysis
python code/evaluation/analyze_yolo_llm_results.py
```

**Outputs:**
- Latency statistics (detection, generation, total)
- Object detection statistics
- Response length analysis
- Cost analysis
- Summary saved to `results/approach_2_yolo_llm/analysis/yolo_llm_analysis_summary.txt`

### Visualizations

```bash
# Create all visualizations
python code/evaluation/create_yolo_llm_visualizations.py
```

**Generated charts:**
- Latency comparison by configuration
- Latency by YOLO variant
- Latency by LLM model
- Object detection statistics
- Cost comparison
- Latency by category

### Comparison with Approach 1

```bash
# Compare with pure VLM baseline
python code/evaluation/compare_yolo_llm_vs_vlm.py
```

## Results Structure

```
results/approach_2_yolo_llm/
├── raw/
│   └── batch_results.csv          # All test results
├── analysis/
│   ├── yolo_llm_analysis_summary.txt
│   └── yolo_llm_vs_vlm_comparison.txt
├── figures/
│   ├── latency_comparison.png
│   ├── latency_by_yolo_variant.png
│   ├── latency_by_llm_model.png
│   ├── object_detection_stats.png
│   └── cost_comparison.png
└── evaluation/
    └── qualitative_scores.csv     # Template for manual scoring
```

## Expected Performance

### Latency
- **Detection:** 10-50ms (YOLO)
- **Generation:** 500-1500ms (LLM)
- **Total:** 1-2 seconds (faster than pure VLMs)

### Cost
- **YOLO:** Free (runs locally)
- **LLM:** $0.00075-0.0015 per query
- **Total:** ~$0.001-0.002 per query (much cheaper than pure VLMs)

### Accuracy
- **Object Detection:** 80+ COCO classes
- **Spatial Relationships:** Based on bounding boxes
- **Description Quality:** Depends on LLM model

## Strengths

- Faster than pure VLMs (1-2s vs 3-5s)
- Much cheaper (LLM costs only, YOLO is free)
- Structured intermediate representation (debuggable)
- Can swap components independently
- More reliable object identification (COCO-trained)

## Weaknesses

- Two points of failure (detector OR generator can fail)
- May miss contextual relationships between objects
- Limited to 80 pre-defined COCO classes
- More complex implementation
- Bounding boxes don't capture everything (posture, actions)

## Use Cases

- Indoor/outdoor navigation (obstacle detection critical)
- When speed matters (near real-time requirement)
- Cost-sensitive applications
- When interpretability matters

## Notes

- YOLO models auto-download on first use
- GPU recommended but not required (CPU works, slower)
- LLM API keys must be configured in `.env` file
- Rate limiting: 1 second between API calls recommended

## Troubleshooting

### YOLO Model Not Found
```bash
# Models will auto-download, but you can manually download:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

### API Key Errors
- Ensure `.env` file contains `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`
- Check API key validity
- Verify API credits/quota

### Import Errors
```bash
# Install missing dependencies
pip install ultralytics opencv-python torch torchvision
```

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com/)

