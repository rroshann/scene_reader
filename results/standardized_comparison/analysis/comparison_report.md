# Standardized Comparison Results

## Overview

This report compares Approaches 1.5, 2.5, and 3.5 using **identical parameters** to isolate architectural differences.

### Standardized Parameters

- **max_tokens**: 100
- **temperature**: 0.7
- **top_p**: 1.0
- **cache**: DISABLED
- **adaptive parameters**: DISABLED
- **image preprocessing**: DISABLED
- **prompts**: Neutral, standardized (same style/length)

---

## Overall Statistics

### Approach 1.5: Optimized Pure VLM (GPT-4V only)

- **Success Rate**: 100.0% (42/42)
- **Mean Latency**: 3.63s
- **Median Latency**: 3.52s
- **Min Latency**: 1.83s
- **Max Latency**: 6.35s
- **Std Deviation**: 0.85s

### Approach 2.5: Optimized YOLO + LLM

- **Success Rate**: 100.0% (42/42)
- **Mean Latency**: 1.36s
- **Median Latency**: 1.34s
- **Min Latency**: 0.73s
- **Max Latency**: 1.84s
- **Std Deviation**: 0.25s

### Approach 3.5: Optimized Specialized Multi-Model

- **Success Rate**: 100.0% (42/42)
- **Mean Latency**: 1.21s
- **Median Latency**: 1.12s
- **Min Latency**: 0.62s
- **Max Latency**: 2.41s
- **Std Deviation**: 0.45s

---

## Comparison Table

| Metric | Approach 1.5 | Approach 2.5 | Approach 3.5 | Winner |
|--------|--------------|--------------|--------------|--------|
| **Mean Latency** | 3.63s | 1.36s | 1.21s | Approach 3.5 |
| **Median Latency** | 3.52s | 1.34s | 1.12s | Approach 3.5 |
| **Success Rate** | 100.0% | 100.0% | 100.0% | Approach 3.5 |
| **Consistency (Std Dev)** | 0.85s | 0.25s | 0.45s | Approach 2.5 |

---

## Key Findings

### Architectural Differences

1. **Approach 1.5 (Pure VLM)**:
   - Direct GPT-4V analysis
   - No preprocessing steps
   - Single API call

2. **Approach 2.5 (YOLO + LLM)**:
   - YOLO object detection first (~0.1s)
   - Then GPT-3.5-turbo generation
   - Two-stage pipeline

3. **Approach 3.5 (Specialized)**:
   - YOLO detection + OCR/Depth estimation
   - Then GPT-3.5-turbo generation
   - Three-stage pipeline (most complex)

### Speed Analysis

With identical parameters:
- **Fastest**: Approach 3.5 (1.21s)
- **Most Consistent**: Approach 2.5 (std dev: 0.25s)

### Comparison with Optimized Results

**Note**: These standardized results show **architectural differences** only. In practice, each approach would be optimized with different parameters, which may change the rankings.

---

## Methodology

- **Test Images**: All images from `data/images/` (gaming, indoor, outdoor, text)
- **Parameters**: Identical across all approaches (see above)
- **Measurements**: Latency from start to description received
- **Repetitions**: Single run per image

---

*Generated from standardized comparison test results*
