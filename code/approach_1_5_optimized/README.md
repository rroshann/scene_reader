# Approach 5: Streaming/Progressive Models

## Overview

Approach 5 implements a **two-tier streaming architecture** that optimizes perceived latency for accessibility applications. Instead of waiting for a single comprehensive description, users receive:

1. **Tier 1 (Fast)**: Quick overview from BLIP-2 local model (~0.5-1s)
2. **Tier 2 (Detailed)**: Comprehensive description from GPT-4V cloud model (~3-4s)

Both models run **in parallel**, so the total latency is the maximum of the two (not the sum), while users perceive the faster Tier1 response.

## Architecture

```
Image Received
    ↓
Parallel Execution:
    ├─→ Tier 1: BLIP-2 (local) → Quick overview (0.5-1s)
    └─→ Tier 2: GPT-4V (cloud) → Detailed description (3-4s)
    ↓
Progressive Disclosure:
    - User hears Tier1 immediately (perceived latency)
    - User receives Tier2 when ready (comprehensive info)
```

## Key Features

- **Async Parallel Execution**: Both models run simultaneously using Python `asyncio`
- **Perceived Latency Optimization**: Users get feedback in <1s instead of 3-4s
- **Progressive Disclosure**: Quick overview first, detailed description follows
- **Fallback Handling**: If one tier fails, the other still provides value
- **Cost Efficient**: Only Tier2 (GPT-4V) incurs API costs

## Components

### Core Pipeline
- `streaming_pipeline.py`: Main `StreamingPipeline` class orchestrating both tiers
- `model_wrappers.py`: Async wrappers for BLIP-2 (thread pool) and GPT-4V (async HTTP)
- `prompts.py`: Optimized prompts for each tier

### Testing & Analysis
- `batch_test_streaming.py`: Batch testing on all 42 images
- `analyze_streaming_results.py`: Quantitative analysis
- `create_streaming_visualizations.py`: Charts and plots
- `compare_streaming_vs_baseline.py`: Comparison with Approach 1

## Usage

### Basic Usage

```python
import asyncio
from pathlib import Path
from streaming_pipeline import StreamingPipeline

async def main():
    pipeline = StreamingPipeline()
    result = await pipeline.describe_image(Path("image.png"))
    
    print(f"Tier1 (quick): {result['tier1']['description']}")
    print(f"Tier2 (detailed): {result['tier2']['description']}")
    print(f"Time to first output: {result['time_to_first_output']:.2f}s")

asyncio.run(main())
```

### Batch Testing

```bash
cd code/approach_5_streaming
python batch_test_streaming.py
```

Results are saved to `results/approach_5_streaming/raw/batch_results.csv`

### Analysis

```bash
# Generate analysis report
python code/evaluation/analyze_streaming_results.py

# Create visualizations
python code/evaluation/create_streaming_visualizations.py

# Compare with baseline
python code/evaluation/compare_streaming_vs_baseline.py
```

## Results Structure

### CSV Columns

- `filename`, `category`: Image identification
- `tier1_description`, `tier1_latency`, `tier1_success`: Tier1 results
- `tier2_description`, `tier2_latency`, `tier2_tokens`, `tier2_cost`, `tier2_success`: Tier2 results
- `total_latency`: Maximum of tier1 and tier2 (parallel execution)
- `time_to_first_output`: Perceived latency (tier1 latency)
- `perceived_latency_improvement`: Percentage improvement vs tier2-only

## Metrics

### Latency Metrics
- **Time to First Output**: Latency until Tier1 completes (perceived latency)
- **Tier2 Latency**: Latency until Tier2 completes
- **Total Latency**: Maximum of Tier1 and Tier2 (since they run in parallel)
- **Perceived Improvement**: Reduction in perceived latency vs single Tier2

### Success Metrics
- **Tier1 Success Rate**: Percentage of successful BLIP-2 descriptions
- **Tier2 Success Rate**: Percentage of successful GPT-4V descriptions
- **Both Success Rate**: Percentage where both tiers succeeded
- **Either Success Rate**: Percentage where at least one tier succeeded

### Cost Metrics
- **Tier1 Cost**: $0.00 (local model)
- **Tier2 Cost**: GPT-4V API cost (same as Approach 1 baseline)
- **Total Cost**: Same as single GPT-4V call (no additional cost)

## Expected Performance

Based on the architecture:

- **Tier1 Latency**: 0.5-1.5s (BLIP-2 local inference)
- **Tier2 Latency**: 3-5s (GPT-4V API call, similar to Approach 1)
- **Time to First Output**: 0.5-1.5s (perceived latency)
- **Total Latency**: 3-5s (max of tier1 and tier2)
- **Perceived Improvement**: 60-80% faster perceived response

## Dependencies

- `approach_4_local`: For BLIP-2 model implementation
- `vlm_testing`: For GPT-4V API integration
- `asyncio`: For parallel execution
- `openai` (async): For async GPT-4V API calls
- `transformers`: For BLIP-2 model

## Implementation Details

### Async Execution

BLIP-2 is synchronous, so it's wrapped in a `ThreadPoolExecutor`:

```python
executor = ThreadPoolExecutor(max_workers=1)
result = await loop.run_in_executor(executor, blip2_sync_function)
```

GPT-4V uses async HTTP client (`AsyncOpenAI`):

```python
client = AsyncOpenAI(api_key=api_key)
response = await client.chat.completions.create(...)
```

### Model Initialization

BLIP-2 model is loaded once and reused across all images (singleton pattern) to avoid repeated loading overhead.

### Error Handling

- If Tier1 fails: Log error, continue with Tier2 only
- If Tier2 fails: Log error, return Tier1 only
- Track partial success cases for analysis

## Comparison with Other Approaches

### vs Approach 1 (Pure VLMs)
- **Perceived Latency**: Much faster (0.5-1.5s vs 3-5s)
- **Total Latency**: Similar (3-5s, max of both tiers)
- **Cost**: Same (only Tier2 uses GPT-4V)
- **Quality**: Tier2 quality same as Approach 1, Tier1 provides quick overview

### vs Approach 2 (YOLO+LLM)
- **Perceived Latency**: Similar or faster (depends on BLIP-2 vs YOLO speed)
- **Total Latency**: Similar (both run components in parallel)
- **Cost**: Similar (both use one LLM API call)
- **Quality**: Tier2 quality better (GPT-4V vs GPT-4o-mini)

## Use Cases

- **Real-time Assistance**: Gaming, navigation where immediate feedback matters
- **Impatient Users**: When partial info is better than waiting
- **UX Research**: Studying perceived latency vs actual latency
- **Progressive Disclosure**: When quick overview + detailed follow-up is valuable

## Limitations

- **Complexity**: Requires async programming and error handling
- **Two Descriptions**: Users must process both quick and detailed descriptions
- **Potential Contradictions**: Tier1 and Tier2 might disagree (rare)
- **BLIP-2 Dependency**: Requires local model setup (GPU/CPU resources)

## Future Improvements

- **Streaming API**: Use GPT-4V streaming API for true progressive output
- **Adaptive Tier1**: Choose faster model based on scene complexity
- **Tier1 Quality Tuning**: Optimize BLIP-2 prompts for better quick descriptions
- **Contradiction Detection**: Automatically flag when tiers disagree
- **TTS Integration**: Stream audio output as descriptions arrive

## Results Location

- **Raw Results**: `results/approach_5_streaming/raw/batch_results.csv`
- **Analysis**: `results/approach_5_streaming/analysis/streaming_analysis.txt`
- **Visualizations**: `results/approach_5_streaming/figures/*.png`
- **Comparison**: `results/approach_5_streaming/analysis/streaming_vs_baseline_comparison.txt`

## References

- Approach 1 (Baseline): `code/vlm_testing/`
- Approach 4 (BLIP-2): `code/approach_4_local/`
- Project Documentation: `PROJECT.md`

