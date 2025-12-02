# Approach 4: Local Models

## Overview

Approach 4 implements local vision-language models (BLIP-2) that run on-device without cloud APIs. This provides privacy, offline capability, and zero API cost. Optimized for M1 Mac with Metal Performance Shaders (MPS) support.

## Architecture

```
Image
    ↓
Local VLM (BLIP-2)
    - Runs on MPS (Metal) or CPU
    - No internet required
    - No API calls
    ↓
Description Output
```

## Models

### BLIP-2 (2.7B parameters) ✅ Implemented
- **Model:** `Salesforce/blip2-opt-2.7b`
- **Size:** ~14GB (with dependencies)
- **Best for:** Fast inference, smaller memory footprint
- **Quality:** Good for size, optimized for M1 Mac
- **Status:** Fully tested on all 42 images (100% success rate)


## Installation

### 1. Install Dependencies

```bash
pip install transformers accelerate huggingface-hub torch torchvision
# Or install all requirements
pip install -r requirements.txt
```

### 2. Download Models (Optional)

You can pre-download models before testing:

```bash
python code/approach_4_local/download_models.py
```

This will download BLIP-2 model to `data/models/` directory (~5GB).

**Note:** Models will also download automatically on first use if not pre-downloaded.

### 3. Model Storage

Models are stored in `data/models/` directory (project-local):
- **BLIP-2:** `data/models/blip2-opt-2.7b/`

This keeps everything self-contained in the project directory.

## Usage

### Single Image Test

```python
from blip2_model import BLIP2Model
from pathlib import Path

# Test BLIP-2
blip2 = BLIP2Model()
result, error = blip2.describe_image(Path('data/images/gaming/game_01.png'))
if not error:
    print(f"Description: {result['description']}")
    print(f"Latency: {result['latency']:.2f}s")
    print(f"Device: {result['device']}")
```

### Batch Testing

```bash
python code/approach_4_local/batch_test_local.py
```

This will:
- Test BLIP-2 on all 42 images
- Save results incrementally to `results/approach_4_local/raw/batch_results.csv`
- Show progress and handle errors gracefully

**Expected:**
- 42 tests total (BLIP-2 × 42 images)
- ~25-30 minutes runtime (MPS acceleration on M1 Mac)
- $0.00 cost (no API calls)
- 100% success rate

## Analysis

### Quantitative Analysis

```bash
python code/evaluation/analyze_local_results.py
```

Generates:
- Latency statistics (overall, by model, by category)
- Device usage statistics (MPS vs CPU)
- Response length analysis
- Cost analysis (always $0.00)

### Visualizations

```bash
python code/evaluation/create_local_visualizations.py
```

Creates charts:
- Latency by category
- Device usage distribution
- Response length analysis

### Statistical Tests

```bash
python code/evaluation/statistical_tests_local.py
```

Performs:
- ANOVA: Latency by category
- Response length analysis
- Statistical significance tests

### Baseline Comparison

```bash
python code/evaluation/compare_local_vs_cloud.py
```

Compares with Approach 1 (Pure VLMs):
- Latency tradeoffs
- Quality tradeoffs
- Cost comparison
- Privacy/offline benefits

## Hardware Requirements

### M1 Mac (Recommended)
- **Device:** MacBook Pro M1 2021 or later
- **Backend:** MPS (Metal Performance Shaders)
- **Memory:** 16GB+ recommended (8GB minimum)
- **Storage:** ~5GB for models

### CPU Fallback
- Works on any system with PyTorch
- Slower inference (5-10x slower than MPS)
- No GPU acceleration

## Performance Results (BLIP-2 on M1 Mac)

### Actual Performance
- **MPS (Metal):** 35.4 seconds average per image
- **Range:** 6.8s - 61.2s (varies by image complexity)
- **Memory:** ~4GB
- **Device:** MPS (Metal) acceleration active
- **Success Rate:** 100% (42/42 images)

### Latency Breakdown
- **Mean:** 35.4s
- **Median:** 36.8s
- **Std Dev:** 13.2s
- **Fastest:** 6.8s (simple images)
- **Slowest:** 61.2s (complex scenes)

### Latency by Category
- **Gaming:** 26.9s average (fastest)
- **Outdoor:** 33.6s average
- **Text:** 36.4s average
- **Indoor:** 46.4s average (slowest - complex scenes)

## Troubleshooting

### Model Download Issues
- **Problem:** Download fails or is slow
- **Solution:** Use pre-download script, check internet connection, ensure ~5GB free space

### Memory Issues
- **Problem:** Out of memory errors
- **Solution:** 
  - Close other applications
  - Use beam search optimization (num_beams=1) to reduce memory usage
  - Consider quantization (4-bit) if needed (MPS compatibility uncertain)

### MPS Not Available
- **Problem:** Falls back to CPU
- **Solution:** 
  - Ensure PyTorch 2.0+ is installed
  - Check macOS version (M1 requires macOS 12.3+)
  - CPU fallback will work but is slower

### Slow Inference
- **Problem:** Very slow on CPU
- **Solution:** This is expected. MPS provides 5-10x speedup. Consider using M1 Mac or GPU.

## File Structure

```
code/approach_4_local/
├── __init__.py
├── local_vlm.py              # Device detection, utilities
├── blip2_model.py            # BLIP-2 implementation
├── prompts.py                # Prompts
├── download_models.py        # Pre-download script
├── batch_test_local.py       # Batch testing
└── README.md                 # This file

data/models/                  # Model storage (gitignored)
└── blip2-opt-2.7b/

results/approach_4_local/
├── raw/
│   └── batch_results.csv
├── analysis/
│   ├── local_analysis_summary.txt
│   ├── statistical_tests.txt
│   └── local_vs_cloud_comparison.txt
└── figures/
    ├── latency_comparison.png
    ├── latency_by_category.png
    ├── device_usage.png
    └── response_length_comparison.png
```

## Key Features

- **Privacy:** All processing on-device, no data sent to cloud
- **Offline:** Works without internet connection
- **Zero Cost:** No API calls, completely free after setup
- **M1 Optimized:** Uses Metal Performance Shaders for acceleration
- **Project-Local Storage:** Models stored in project directory

## Comparison with Cloud VLMs

| Metric | Local Models | Cloud VLMs |
|--------|--------------|------------|
| **Latency** | 35.4s avg (MPS, beams=3) | 3-5s |
| **Cost** | $0.00 | $0.013/query |
| **Privacy** | ✅ On-device | ❌ Sent to API |
| **Offline** | ✅ Works offline | ❌ Requires internet |
| **Quality** | 75-85% | 90-95% |
| **Setup** | Medium | Easy |

## Novel Contribution

- **Privacy-focused local deployment** for accessibility
- **M1 Mac optimization** with MPS support
- **Resource usage analysis** (memory, device utilization)
- **Offline capability demonstration**
- **Cost-free alternative** to cloud APIs

