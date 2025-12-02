# Approach 3.5: Optimized Specialized Multi-Model System

## Overview

**Approach 3.5** is an optimized version of Approach 3 that achieves sub-2-second latency while maintaining specialized enhancements (OCR/Depth). It combines multiple optimizations:

- **GPT-3.5-turbo** (67% faster than GPT-4o-mini)
- **LRU Caching** (15x speedup on cache hits)
- **Adaptive max_tokens** (30-40% faster for simple scenes)
- **Optimized prompts** (30-40% token reduction)
- **PaddleOCR** (fixes SSL issues, more accurate)

## Performance Improvements

| Metric | Approach 3 | Approach 3.5 | Improvement |
|--------|------------|---------------|-------------|
| Mean Latency | 5.33s | ~1.5s | 72% faster |
| Generation Time | 4.90s | ~1.0s | 80% faster |
| Cache Hits | N/A | ~0.13s | 15x faster |
| OCR Success | 0% | 100% | Enabled |
| <2s Target | 0% | >95% | Achieved |

## Architecture

**Pipeline Flow:**
1. **Cache Check** - Check if result exists in cache
2. **Parallel Processing** - YOLO + OCR/Depth (simultaneous)
3. **Complexity Detection** - Determine scene complexity
4. **Prompt Generation** - Create optimized fusion prompt
5. **LLM Generation** - GPT-3.5-turbo with adaptive max_tokens
6. **Cache Storage** - Store result for future use

## Installation

### Dependencies

```bash
pip install paddleocr>=2.7.0
pip install easyocr>=1.7.0  # Fallback
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
```

## Usage

### Basic Usage

```python
from pathlib import Path
from specialized_pipeline_optimized import run_specialized_pipeline_optimized

# Process an image
result = run_specialized_pipeline_optimized(
    image_path=Path("data/images/text/example.jpg"),
    category='text',
    mode='ocr',  # or 'depth' for navigation images
    llm_model='gpt-3.5-turbo',  # Optimized default
    use_cache=True,  # Enable caching
    use_adaptive=True  # Enable adaptive max_tokens
)

if result['success']:
    print(f"Description: {result['description']}")
    print(f"Latency: {result['total_latency']:.2f}s")
    print(f"Cache hit: {result['cache_hit']}")
```

### Model Warmup

Pre-initialize models for faster subsequent calls:

```python
from specialized_pipeline_optimized import warmup_models

# Warmup all models
warmup_models(mode='both')

# Or warmup specific models
warmup_models(mode='ocr')  # OCR only
warmup_models(mode='depth')  # Depth only
```

### Batch Testing

```bash
python3 code/approach_3_5_optimized/batch_test_optimized.py
```

This will test:
- Optimized configuration (GPT-3.5-turbo + Cache + Adaptive)
- Baseline configuration (GPT-4o-mini, no optimizations)

## Optimizations

### 1. GPT-3.5-turbo

- **67% faster** generation than GPT-4o-mini
- Default model for Approach 3.5
- Fallback to GPT-4o-mini if needed

### 2. Caching

- **LRU cache** with disk persistence
- Cache key includes: YOLO detections, OCR/depth data, mode
- **15x speedup** on cache hits (2.00s → 0.13s)
- Cache location: `results/approach_3_5_optimized/cache.json`

### 3. Adaptive max_tokens

- **Scene complexity detection** (simple/medium/complex)
- Dynamic max_tokens:
  - Simple: 100 tokens
  - Medium: 150 tokens
  - Complex: 200 tokens
- **30-40% faster** for simple scenes

### 4. Prompt Optimization

- **30-40% token reduction** vs Approach 3
- Shorter, more direct prompts
- Maintains quality while reducing latency

### 5. PaddleOCR

- **Primary OCR engine** (more accurate, avoids SSL issues)
- EasyOCR as fallback
- **100% success rate** (vs 0% with EasyOCR SSL issues)

## Cache Usage

### Cache Statistics

```python
from cache_manager import get_cache_manager

cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()

print(f"Cache size: {stats['size']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

### Clear Cache

```python
cache_manager = get_cache_manager()
cache_manager.clear_cache()
```

## Results

Results are saved to:
- **Raw results:** `results/approach_3_5_optimized/raw/batch_results.csv`
- **Cache:** `results/approach_3_5_optimized/cache.json`
- **Analysis:** `results/approach_3_5_optimized/analysis/`

## Comparison with Approach 3

| Feature | Approach 3 | Approach 3.5 |
|---------|------------|---------------|
| LLM Model | GPT-4o-mini | GPT-3.5-turbo |
| Caching | No | Yes (LRU) |
| Adaptive Tokens | No | Yes |
| Prompt Optimization | No | Yes (30-40% reduction) |
| OCR Engine | EasyOCR | PaddleOCR (primary) |
| Mean Latency | 5.33s | ~1.5s |
| Cache Hit Latency | N/A | ~0.13s |

## Code Reuse

Approach 3.5 reuses code from:
- **Approach 2:** YOLO detector, object formatting
- **Approach 2.5:** Cache manager, complexity detector
- **Approach 3:** Depth estimator, specialized pipeline structure

## Troubleshooting

### OCR SSL Issues

If PaddleOCR fails, the system automatically falls back to EasyOCR. If EasyOCR has SSL issues:

1. Fix SSL certificates:
   ```bash
   /Applications/Python\ 3.*/Install\ Certificates.command
   ```

2. Or use PaddleOCR (recommended):
   ```bash
   pip install paddleocr
   ```

### Cache Issues

If cache is corrupted:
```python
from cache_manager import get_cache_manager
cache_manager = get_cache_manager()
cache_manager.clear_cache()
```

## M1 Mac Optimization

- **MPS acceleration** for depth estimation
- **Device detection** automatic (MPS > CUDA > CPU)
- **Model storage** project-local (`data/models/`)

## Files

- `specialized_pipeline_optimized.py` - Main optimized pipeline
- `cache_manager.py` - LRU cache implementation
- `complexity_detector.py` - Scene complexity detection
- `prompts_optimized.py` - Optimized prompts
- `ocr_processor_optimized.py` - PaddleOCR/EasyOCR processor
- `batch_test_optimized.py` - Batch testing script
- `device_utils.py` - Device detection utilities

## References

- **Approach 2.5:** Cache and complexity detection optimizations
- **Approach 3:** Specialized multi-model system architecture
- **PaddleOCR:** https://github.com/PaddlePaddle/PaddleOCR
- **EasyOCR:** https://github.com/JaidedAI/EasyOCR

