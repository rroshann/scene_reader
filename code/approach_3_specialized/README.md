# Approach 3: Specialized Multi-Model System

**Status:** ✅ Depth Mode Working, ⚠️ OCR Mode (SSL Issue)  
**Date:** November 24, 2025

## Overview

Approach 3 combines specialized models (OCR, depth estimation) with object detection and LLM generation for task-specific accuracy.

**Two Sub-Approaches:**
- **3A: OCR-Enhanced System** - Text reading specialist (text images)
- **3B: Depth-Enhanced System** - Spatial specialist (navigation images)

**Architecture:** Local processing (OCR/depth/YOLO) + Cloud LLM (GPT-4o-mini)

## Quick Start

### Depth Mode (3B) - Working

```python
from pathlib import Path
from specialized_pipeline import run_specialized_pipeline

# Test depth mode on navigation image
result = run_specialized_pipeline(
    image_path=Path("data/images/indoor/example.jpg"),
    category='indoor',
    mode='depth'
)

if result['success']:
    print(f"Description: {result['description']}")
    print(f"Latency: {result['total_latency']:.2f}s")
```

### OCR Mode (3A) - SSL Certificate Issue

**Known Issue:** EasyOCR model download fails on Mac due to SSL certificate issues.

**Solutions:**
1. Fix SSL certificates:
   ```bash
   /Applications/Python\ 3.*/Install\ Certificates.command
   ```

2. Use alternative OCR (PaddleOCR - more accurate for English):
   ```bash
   pip install paddleocr
   ```
   Then modify `ocr_processor.py` to use PaddleOCR instead.

3. Manually download EasyOCR models and place in `~/.EasyOCR/model/`

## Architecture

```
Image
    ↓
Parallel Processing (LOCAL):
    ├─→ YOLOv8: Object detection (~0.07s)
    ├─→ EasyOCR: Text extraction (~1-3s) [3A]
    └─→ Depth-Anything: Depth estimation (~0.2-2s) [3B]
    ↓
Fusion Layer: Combine specialized data + objects
    ↓
GPT-4o-mini API: Generate description (~3-4s)
    ↓
Output: Enhanced description with OCR/depth info
```

## Components

### Core Modules

- **`specialized_pipeline.py`** - Main orchestrator with parallel processing
- **`ocr_processor.py`** - OCR module (EasyOCR)
- **`depth_estimator.py`** - Depth estimation module (Depth-Anything)
- **`prompts.py`** - Fusion prompts for OCR/depth integration
- **`device_utils.py`** - M1 Mac device detection and optimization

### Testing

- **`test_subset.py`** - Subset validation (7-10 images)
- **`batch_test_specialized.py`** - Full batch testing (30 images)

## Performance

### Depth Mode (3B) - Validated

- **Mean Latency:** 4.63s (subset test: 6 images)
- **Range:** 2.46s - 7.48s
- **Component Breakdown:**
  - Detection (YOLO): ~0.07s (1.5% of total)
  - Depth estimation: ~0.2-2.3s (5-50% of total)
  - Generation (LLM): ~3-6s (65-85% of total)

### OCR Mode (3A) - Pending SSL Fix

- **Status:** SSL certificate issue preventing model download
- **Expected Latency:** 3-5s (similar to depth mode)

## M1 Mac Optimization

- **MPS Acceleration:** Depth estimation uses MPS (Metal Performance Shaders)
- **Device Detection:** Automatic (MPS > CUDA > CPU)
- **Model Storage:** Project-local (`data/models/`)

## Code Reuse

- **YOLO Detection:** Reused from Approach 2 (no duplication)
- **LLM Generation:** Reused from Approach 2 (no duplication)
- **New Components:** OCR and depth estimation (specialized)

## Usage Examples

### Depth Mode

```python
from specialized_pipeline import run_specialized_pipeline
from depth_estimator import DepthEstimator

# Initialize depth estimator once (reuse)
depth_estimator = DepthEstimator()

# Process navigation images
result = run_specialized_pipeline(
    image_path=Path("data/images/outdoor/example.jpg"),
    category='outdoor',
    mode='depth',
    depth_estimator=depth_estimator  # Reuse initialized estimator
)
```

### OCR Mode (After SSL Fix)

```python
from specialized_pipeline import run_specialized_pipeline
from ocr_processor import OCRProcessor

# Initialize OCR processor once (reuse)
ocr_processor = OCRProcessor(languages=['en'], gpu=True)

# Process text images
result = run_specialized_pipeline(
    image_path=Path("data/images/text/example.jpg"),
    category='text',
    mode='ocr',
    ocr_processor=ocr_processor  # Reuse initialized processor
)
```

## Dependencies

- `easyocr>=1.7.0` - OCR processing (SSL certificate fix needed)
- `transformers>=4.35.0` - Depth estimation (already installed)
- `torch>=2.0.0` - MPS support (already installed)
- `ultralytics>=8.0.0` - YOLO detection (already installed)

## Known Issues

1. **OCR SSL Certificate:** EasyOCR model download fails on Mac
   - **Workaround:** Fix SSL certificates or use PaddleOCR
   - **Status:** Documented, workaround available

2. **Numpy Version Conflict:** EasyOCR installs numpy 2.x, matplotlib needs <2
   - **Workaround:** `pip install "numpy<2"` after installing easyocr
   - **Status:** Fixed

## Results

### Subset Test (10 images)

- **Depth Mode:** 6/6 successful (100%)
- **OCR Mode:** 0/4 successful (SSL issue)
- **Overall:** 6/10 successful (60%)

### Latency (Depth Mode)

- **Mean:** 4.63s
- **Median:** ~4.5s
- **Std Dev:** ~1.5s
- **Range:** 2.46s - 7.48s

## Next Steps

1. ✅ Depth mode validated - ready for full batch test
2. ⚠️ Fix OCR SSL issue or use alternative
3. Run full batch test (30 images: 10 text + 20 navigation)
4. Comprehensive analysis and comparison

---

**Last Updated:** November 24, 2025  
**Status:** Depth Mode ✅ Working, OCR Mode ⚠️ SSL Issue

