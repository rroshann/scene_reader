# Mode-Specific Comparison Results: GAMING Mode

## Overview

This report compares Approaches 1.5, 2.5, and 3.5 using their **actual prompt_mode parameters** (gaming mode).

### Test Configuration

- **Mode**: gaming
- **Cache**: ENABLED (real-world usage)
- **Adaptive Parameters**: ENABLED for Approach 3.5
- **Prompts**: Mode-specific (not standardized)
- **Test Images**: Gaming images (12)

---

## Overall Statistics

### Approach 1.5: Optimized Pure VLM (GPT-4V only)

- **Success Rate**: 100.0% (12/12)
- **Mean Latency**: 1.46s
- **Median Latency**: 1.46s
- **Min Latency**: 0.99s
- **Max Latency**: 1.90s
- **P95 Latency**: 1.90s
- **Std Deviation**: 0.28s
- **Mean Cost**: $0.0098/query
- **Mean Tokens**: 937

### Approach 2.5: Optimized YOLO + LLM

- **Success Rate**: 100.0% (12/12)
- **Mean Latency**: 0.56s
- **Median Latency**: 0.69s
- **Min Latency**: 0.08s
- **Max Latency**: 0.92s
- **P95 Latency**: 0.92s
- **Std Deviation**: 0.34s
- **Mean Cost**: $0.0000/query
- **Mean Tokens**: 320

### Approach 3.5: Optimized Specialized Multi-Model

- **Success Rate**: 100.0% (12/12)
- **Mean Latency**: 1.05s
- **Median Latency**: 0.93s
- **Min Latency**: 0.73s
- **Max Latency**: 1.55s
- **P95 Latency**: 1.55s
- **Std Deviation**: 0.26s
- **Mean Cost**: $0.0000/query
- **Mean Tokens**: 232

---

## Comparison Table

| Metric | Approach 1.5 | Approach 2.5 | Approach 3.5 | Winner |
|--------|--------------|--------------|--------------|--------|
| **Mean Latency** | 1.46s | 0.56s | 1.05s | Approach 2.5 |
| **Median Latency** | 1.46s | 0.69s | 0.93s | Approach 2.5 |
| **P95 Latency** | 1.90s | 0.92s | 1.55s | Approach 2.5 |
| **Success Rate** | 100.0% | 100.0% | 100.0% | All tied |
| **Consistency (Std Dev)** | 0.28s | 0.34s | 0.26s | Approach 3.5 |
| **Cost per Query** | $0.0098 | $0.0000 | $0.0000 | Approach 2.5 |

---

## Category-Specific Statistics

No category breakdown available

---

## Statistical Significance Tests


- **1 5 Vs 2 5**: t=8.156, p=0.0000, mean_diff=0.908s (✅ Significant)
- **1 5 Vs 3 5**: t=4.895, p=0.0005, mean_diff=0.417s (✅ Significant)
- **2 5 Vs 3 5**: t=-4.111, p=0.0017, mean_diff=-0.491s (✅ Significant)

---

## Key Findings

### Speed Analysis

- **Fastest**: Approach 2.5 (0.56s mean latency)
- **Most Consistent**: Approach 3.5 (std dev: 0.26s)
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

- **Test Images**: 12 gaming images
- **Mode**: gaming
- **Parameters**: Mode-specific prompts, cache enabled, adaptive enabled for Approach 3.5
- **Measurements**: Latency from start to description received
- **Repetitions**: Single run per image

---

*Generated from mode-specific comparison test results*
