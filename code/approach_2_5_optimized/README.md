# Approach 2.5: Optimized YOLO+LLM Hybrid Pipeline

**Status:** ✅ Complete - Production Ready  
**Mean Latency:** 1.10s (67.4% faster than Approach 2)  
**<2s Target:** ✅ Achieved (95.2% of tests)

---

## Overview

Approach 2.5 is an optimized variant of Approach 2 (YOLO+LLM Hybrid Pipeline) that achieves **<2 second latency** through:

1. **Faster LLM Model:** GPT-3.5-turbo (67.4% faster than GPT-4o-mini)
2. **Caching:** LRU cache with disk persistence (15x speedup on cache hits)
3. **Adaptive Parameters:** Scene complexity-based max_tokens (optional)

## Relationship to Approach 2

- **Approach 2:** Baseline implementation (GPT-4o-mini, Claude Haiku) - 3.39s mean latency
- **Approach 2.5:** Optimized variant (GPT-3.5-turbo + optimizations) - 1.10s mean latency

**Code Reuse Strategy:**
- Imports and extends Approach 2 components (DRY principle)
- No code duplication - reuses YOLO detector and LLM generator
- Maintains backward compatibility

## Performance Metrics

### Latency Performance
- **Mean:** 1.10s (vs 3.39s baseline - 67.4% faster)
- **Median:** 0.97s (vs 3.18s baseline)
- **Std Dev:** 0.44s (vs 1.16s baseline - more consistent)
- **Range:** 0.44s - 2.67s
- **<2s Target:** ✅ 95.2% of tests (vs 2.4% baseline)

### Cache Performance
- **Cache Hit Speedup:** 15x (2.00s → 0.13s)
- **Cache Implementation:** LRU with disk persistence
- **Default Size:** 1000 entries

### Statistical Significance
- **Paired t-test:** p < 0.000001 (highly significant)
- **Effect Size:** Cohen's d = 2.61 (large effect)
- **Sample Size:** 42 paired images

## Quick Start

### Basic Usage

```python
from pathlib import Path
from hybrid_pipeline_optimized import run_hybrid_pipeline_optimized

# Process an image
result = run_hybrid_pipeline_optimized(
    image_path=Path("data/images/gaming/example.png"),
    yolo_size='n',  # Nano - fastest
    llm_model='gpt-3.5-turbo',  # Optimized model
    use_cache=True  # Enable caching
)

if result['success']:
    print(f"Description: {result['description']}")
    print(f"Latency: {result['total_latency']:.2f}s")
    print(f"Cache hit: {result.get('cache_hit', False)}")
```

### With Adaptive Parameters

```python
result = run_hybrid_pipeline_optimized(
    image_path=Path("data/images/gaming/example.png"),
    use_cache=True,
    use_adaptive=True  # Enable adaptive max_tokens
)
```

### Disable Caching

```python
result = run_hybrid_pipeline_optimized(
    image_path=Path("data/images/gaming/example.png"),
    use_cache=False  # Disable caching
)
```

## Architecture

```
Image Input
    ↓
YOLOv8N Detection (~0.08s)
    ↓
Cache Check (if enabled)
    ├─ Cache HIT → Return cached (~0.13s total)
    └─ Cache MISS → Continue
        ↓
GPT-3.5-turbo Generation (~1.0s)
    ├─ Adaptive max_tokens (if enabled)
    └─ Fixed max_tokens: 200 (default)
        ↓
Store in Cache (if enabled)
    ↓
Description Output
```

## Components

### Core Modules

- **`hybrid_pipeline_optimized.py`** - Main pipeline orchestrator
- **`cache_manager.py`** - LRU cache with persistence
- **`complexity_detector.py`** - Scene complexity detection
- **`llm_generator_optimized.py`** - Adaptive LLM generator

### Testing & Analysis

- **`batch_test_optimized.py`** - Full batch testing (42 images)
- **`test_regression.py`** - Regression tests for Approach 2 compatibility
- **`test_adaptive_params.py`** - Adaptive parameters testing

## Optimization Strategies

### 1. Faster LLM Model
- **Model:** GPT-3.5-turbo (vs GPT-4o-mini baseline)
- **Impact:** 67.4% speedup
- **Tradeoff:** Slightly shorter descriptions (58.5 vs 123.6 words)

### 2. Caching
- **Strategy:** LRU cache with disk persistence
- **Impact:** 15x speedup on cache hits
- **Use Case:** Repeated scenes (e.g., same game levels)

### 3. Adaptive Parameters
- **Strategy:** Scene complexity-based max_tokens
- **Impact:** 5-10% additional speedup for simple scenes
- **Status:** Optional (disabled by default)

## Results

### Batch Test Results (42 images)
- **Success Rate:** 100% (42/42)
- **Mean Latency:** 1.10s
- **Median Latency:** 0.97s
- **<2s Target:** ✅ 95.2% of tests

### Comparison vs Approach 2
- **Speedup:** 67.4% faster (3.39s → 1.10s)
- **Consistency:** More consistent (0.44s vs 1.16s std dev)
- **Target Achievement:** 95.2% vs 2.4% under 2s

## Cache Management

### Get Cache Statistics

```python
from cache_manager import get_cache_manager

cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()

print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

### Clear Cache

```python
cache_manager = get_cache_manager()
cache_manager.clear_cache()
```

## File Structure

```
code/approach_2_5_optimized/
├── __init__.py                    # Package initialization
├── hybrid_pipeline_optimized.py  # Main pipeline
├── cache_manager.py              # Cache implementation
├── complexity_detector.py        # Scene complexity
├── llm_generator_optimized.py   # Adaptive generator
├── prompts_optimized.py         # Optimized prompts
├── batch_test_optimized.py      # Batch testing
├── test_regression.py           # Regression tests
├── test_adaptive_params.py      # Adaptive testing
└── README.md                    # This file

results/approach_2_5_optimized/
├── raw/
│   └── batch_results.csv        # Test results
├── analysis/
│   ├── optimization_comparison.txt
│   ├── statistical_tests.txt
│   └── implementation_review.md
└── cache.json                   # Persistent cache
```

## Dependencies

- **Approach 2 components** (imported, not duplicated)
- **ultralytics** - YOLO detection
- **openai** - GPT-3.5-turbo API
- **scipy** - Statistical tests (optional)

## Notes

- Approach 2.5 extends Approach 2 without modifying it
- All optimizations are optional/configurable
- Cache persists across sessions (disk storage)
- Regression tests ensure Approach 2 compatibility

## References

- **Baseline:** See `code/approach_2_yolo_llm/README.md`
- **Plan:** See `APPROACH_2_5_PLAN.md`
- **Results:** See `results/approach_2_5_optimized/analysis/`

---

**Last Updated:** November 24, 2025  
**Status:** ✅ Production Ready

