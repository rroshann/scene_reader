# Standardized Comparison Testing

## Overview

This standardized comparison test runs Approaches 1.5, 2.5, and 3.5 with **identical parameters** to isolate architectural differences and provide a fair, scientific comparison.

## Standardized Parameters

All approaches use the same parameters:
- **max_tokens**: 100
- **temperature**: 0.7
- **top_p**: 1.0
- **cache**: DISABLED
- **adaptive parameters**: DISABLED
- **image preprocessing**: DISABLED (no resizing/compression)
- **prompts**: Neutral, standardized (same style/length)

## Quick Start

### 1. Verify Setup

```bash
python code/evaluation/verify_standardized_setup.py
```

This verifies all imports work and checks that test images are available.

### 2. Run Standardized Comparison Test

```bash
python code/evaluation/standardized_comparison_test.py
```

This will:
- Test all 3 approaches on all 42 test images
- Use identical parameters for each approach
- Save results to `results/standardized_comparison/raw/batch_results.csv`
- Print a summary of results

**Note**: This makes API calls and may take 30-60 minutes depending on API response times.

### 3. Generate Analysis and Report

After the test completes:

```bash
python code/evaluation/analyze_standardized_comparison.py
```

This will:
- Process the CSV results
- Calculate statistics (mean, median, min, max latency, success rates)
- Generate visualizations:
  - `results/standardized_comparison/analysis/latency_comparison.png`
  - `results/standardized_comparison/analysis/latency_distribution.png`
- Create markdown report:
  - `results/standardized_comparison/analysis/comparison_report.md`

## Test Images

The test uses all images from `data/images/`:
- **gaming**: 12 images
- **indoor**: 10 images
- **outdoor**: 10 images
- **text**: 10 images
- **Total**: 42 images

## Output Files

### Raw Results
- `results/standardized_comparison/raw/batch_results.csv` - Detailed results for each image

### Analysis
- `results/standardized_comparison/analysis/comparison_report.md` - Summary report
- `results/standardized_comparison/analysis/latency_comparison.png` - Bar chart comparing mean latencies
- `results/standardized_comparison/analysis/latency_distribution.png` - Box plots showing latency distributions

## What This Tests

This standardized comparison isolates **architectural differences** by:
1. Using identical LLM parameters (max_tokens, temperature)
2. Using identical prompts (neutral, standardized)
3. Disabling optimizations (cache, adaptive parameters, image preprocessing)

This allows you to see which **architecture** is inherently faster, regardless of optimizations.

## Comparison with Optimized Results

**Important**: These standardized results show architectural differences only. In practice:
- Each approach would be optimized with different parameters
- Optimizations (cache, adaptive, preprocessing) would be enabled
- Prompts would be tailored to the use case

Both standardized and optimized results are valuable:
- **Standardized**: Shows architectural tradeoffs
- **Optimized**: Shows practical performance

## Troubleshooting

### Import Errors
If you see import errors, make sure:
1. You're running from the project root directory
2. Virtual environment is activated (if using one)
3. All dependencies are installed

### API Errors
If you see API errors:
1. Check that `OPENAI_API_KEY` is set in your environment
2. Check your API quota/limits
3. Verify network connectivity

### Slow Execution
The test may take 30-60 minutes because:
- It tests 42 images × 3 approaches = 126 API calls
- Each approach makes different API calls (GPT-4V, GPT-3.5-turbo, OCR, etc.)
- API response times vary

## Next Steps

After running the tests:
1. Review `comparison_report.md` for key findings
2. Compare standardized vs optimized results
3. Update documentation (README.md, TOP_3_COMPARISON.md, FINDINGS.md) with standardized results

