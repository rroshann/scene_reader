# Mode-Specific Comparison Results: REAL_WORLD Mode

## Overview

This report compares Approaches 1.5, 2.5, and 3.5 using their **actual prompt_mode parameters** (real_world mode).

### Test Configuration

- **Mode**: real_world
- **Cache**: ENABLED (real-world usage)
- **Adaptive Parameters**: ENABLED for Approach 3.5
- **Prompts**: Mode-specific (not standardized)
- **Test Images**: Indoor + Outdoor + Text images (30)

---

## Overall Statistics

### Approach 1.5: Optimized Pure VLM (GPT-4V only)

- **Success Rate**: 100.0% (30/30)
- **Mean Latency**: 1.62s
- **Median Latency**: 1.55s
- **Min Latency**: 1.03s
- **Max Latency**: 3.34s
- **P95 Latency**: 2.46s
- **Std Deviation**: 0.48s
- **Mean Cost**: $0.0101/query
- **Mean Tokens**: 961

### Approach 2.5: Optimized YOLO + LLM

- **Success Rate**: 100.0% (30/30)
- **Mean Latency**: 0.54s
- **Median Latency**: 0.64s
- **Min Latency**: 0.08s
- **Max Latency**: 1.37s
- **P95 Latency**: 0.94s
- **Std Deviation**: 0.35s
- **Mean Cost**: $0.0000/query
- **Mean Tokens**: 344

### Approach 3.5: Optimized Specialized Multi-Model

- **Success Rate**: 100.0% (30/30)
- **Mean Latency**: 0.93s
- **Median Latency**: 1.09s
- **Min Latency**: 0.10s
- **Max Latency**: 2.35s
- **P95 Latency**: 1.45s
- **Std Deviation**: 0.52s
- **Mean Cost**: $0.0000/query
- **Mean Tokens**: 381

---

## Comparison Table

| Metric | Approach 1.5 | Approach 2.5 | Approach 3.5 | Winner |
|--------|--------------|--------------|--------------|--------|
| **Mean Latency** | 1.62s | 0.54s | 0.93s | Approach 2.5 |
| **Median Latency** | 1.55s | 0.64s | 1.09s | Approach 2.5 |
| **P95 Latency** | 2.46s | 0.94s | 1.45s | Approach 2.5 |
| **Success Rate** | 100.0% | 100.0% | 100.0% | All tied |
| **Consistency (Std Dev)** | 0.48s | 0.35s | 0.52s | Approach 2.5 |
| **Cost per Query** | $0.0101 | $0.0000 | $0.0000 | Approach 2.5 |

---

## Category-Specific Statistics

No category breakdown available

---

## Statistical Significance Tests


- **1 5 Vs 2 5**: t=10.163, p=0.0000, mean_diff=1.080s (✅ Significant)
- **1 5 Vs 3 5**: t=6.283, p=0.0000, mean_diff=0.689s (✅ Significant)
- **2 5 Vs 3 5**: t=-4.191, p=0.0002, mean_diff=-0.391s (✅ Significant)

---

## Key Findings

### Speed Analysis

- **Fastest**: Approach 2.5 (0.54s mean latency)
- **Most Consistent**: Approach 2.5 (std dev: 0.35s)
- **Cheapest**: Approach 2.5 ($0.0000/query)

### Performance Characteristics

1. **Approach 1.5 (Pure VLM)**:
   - Direct GPT-4V analysis
   - Highest quality but slower
   - Most expensive

2. **Approach 2.5 (YOLO + LLM)**:
   - Fast two-stage pipeline
   - Good balance of speed and cost
   - YOLO detection + GPT-3.5-turbo

3. **Approach 3.5 (Specialized)**:
   - Most versatile (OCR/Depth + YOLO)
   - Good for text-heavy or spatial scenarios
   - Moderate speed and cost

---

## Methodology

- **Test Images**: 30 real-world images (10 indoor + 10 outdoor + 10 text)
- **Mode**: real_world
- **Parameters**: Mode-specific prompts, cache enabled, adaptive enabled for Approach 3.5
- **Measurements**: Latency from start to description received
- **Repetitions**: Single run per image

---

*Generated from mode-specific comparison test results*
