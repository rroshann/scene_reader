# Approach 2.5: Advanced Speed Optimization Plan

## Overview

**Approach 2.5** is an optimized variant of Approach 2 (YOLO+LLM Hybrid Pipeline) that builds upon the successful GPT-3.5-turbo optimization (1.45s mean latency) by implementing advanced optimizations: prompt optimization, caching, adaptive parameters, and comprehensive validation. Goal: achieve consistent sub-2 second latency while maintaining quality.

**Relationship to Approach 2:**
- **Approach 2:** Baseline YOLO+LLM pipeline (GPT-4o-mini, Claude Haiku) - 3.73s mean latency
- **Approach 2.5:** Optimized variant (GPT-3.5-turbo + optimizations) - targeting <1.5s mean latency

## Phase 0: Setup and Code Reuse Strategy

### 0.1 Establish Import Structure
- **File:** `code/approach_2_5_optimized/__init__.py` (new)
- **Strategy:** Import Approach 2 components (best practice: DRY, maintainability)
  - Import `yolo_detector` from `approach_2_yolo_llm` (reuse unchanged)
  - Import `llm_generator` functions (extend with optimizations)
  - Import `prompts` (extend with optimized variants)
- **Rationale:** Reduces code duplication, ensures consistency, easier maintenance

### 0.2 Regression Testing Setup
- **File:** `code/approach_2_5_optimized/test_regression.py` (new)
- **Purpose:** Ensure Approach 2 remains functional after Approach 2.5 implementation
- **Tests:**
  - Verify Approach 2 imports work correctly
  - Test Approach 2 pipeline on sample image (smoke test)
  - Verify Approach 2 results directory unchanged
- **Run:** Before starting Phase 1, and after each major phase

## Phase 1: Full Batch Validation

### 1.1 Update Batch Test Script
- **File:** `code/approach_2_5_optimized/batch_test_optimized.py` (new)
- **Based on:** `code/approach_2_yolo_llm/batch_test_yolo_llm.py`
- **Strategy:** Import YOLO detector from Approach 2, use optimized pipeline
- **Changes:**
  - Support for `gpt-3.5-turbo` model (primary)
  - Import `yolo_detector` from `approach_2_yolo_llm` (reuse)
  - Use `hybrid_pipeline_optimized` (new, wraps Approach 2)
  - Update CSV output to include model identifier and approach version ("2.5")
  - Add incremental saving after each test
  - Save to `results/approach_2_5_optimized/raw/batch_results.csv`

### 1.2 Run Full Batch Test
- **Configuration:** YOLOv8N + GPT-3.5-turbo
- **Scope:** All 42 images
- **Expected Time:** ~2-3 minutes (42 images × ~1.5s each)
- **Output:** `results/approach_2_5_optimized/raw/batch_results.csv`
- **Metrics:** Total latency, generation latency, detection latency, word count, tokens used

### 1.3 Validate Results
- Calculate mean, median, std dev, min, max latencies
- Verify <2s target achieved across all scenarios
- Compare against subset test results (should match ~1.45s mean)
- Identify any outliers or failure cases

## Phase 2: Prompt Optimization

### 2.1 Analyze Current Prompt Token Usage
- **File:** `code/approach_2_yolo_llm/prompts.py` (reference)
- **Current:** System prompt (~80 tokens) + User prompt (~150-300 tokens depending on objects)
- **Action:** Measure actual token counts for different scene complexities
- **Tool:** Add token counting utility or use API response tokens

### 2.2 Create Optimized Prompt Variants
- **File:** `code/approach_2_5_optimized/prompts_optimized.py` (new)
- **Variants to create:**
  1. **Minimal Prompt:** Reduce system prompt verbosity, condense user prompt
  2. **Structured Prompt:** Use bullet points instead of prose
  3. **Template-based:** Pre-format common patterns
- **Target:** Reduce input tokens by 20-30% while maintaining quality

### 2.3 Test Prompt Variants
- **Script:** `code/approach_2_5_optimized/test_prompt_optimization.py` (new)
- **Method:** Test each variant on 8-image subset
- **Metrics:** Input tokens, output tokens, latency, quality (word count, completeness)
- **Comparison:** Baseline prompt vs optimized variants

### 2.4 Select Optimal Prompt
- **Criteria:** Balance between token reduction and quality preservation
- **Decision:** Choose variant with best latency/quality tradeoff
- **Update:** Use selected prompt as default for Approach 2.5

## Phase 3: Caching Implementation

### 3.1 Design Cache Strategy
- **File:** `code/approach_2_5_optimized/cache_manager.py` (new)
- **Cache Key:** Hash of (YOLO model, detected objects list, prompt template)
- **Cache Value:** Generated description, latency, tokens
- **Storage:** In-memory dict + optional disk persistence (JSON/Pickle)
- **Eviction:** LRU cache with configurable size limit (e.g., 1000 entries)

### 3.2 Implement Cache Manager
- **Functions:**
  - `get_cache_key(yolo_model, objects, prompt_template)` -> str
  - `get_cached_result(cache_key)` -> Optional[Dict]
  - `store_result(cache_key, result)` -> None
  - `clear_cache()` -> None
  - `get_cache_stats()` -> Dict (hit rate, size, etc.)

### 3.3 Integrate Cache into Pipeline
- **File:** `code/approach_2_5_optimized/hybrid_pipeline_optimized.py` (new)
- **Strategy:** Import and extend Approach 2 components (best practice: DRY principle)
  - Import `yolo_detector` from `approach_2_yolo_llm` (reuse, no changes needed)
  - Import `llm_generator` functions and extend with caching wrapper
  - Create new pipeline that wraps Approach 2 pipeline with optimizations
- **Changes:**
  - Import cache_manager
  - Check cache before LLM generation
  - Store results after generation
  - Add cache hit/miss logging
  - Add `use_cache` parameter (default: True)
  - Import YOLO detector from Approach 2 (no duplication)

### 3.4 Test Caching Performance
- **Script:** `code/approach_2_5_optimized/test_caching.py` (new)
- **Method:** 
  - Run pipeline on 42 images (populate cache)
  - Re-run same images (test cache hits)
  - Measure latency improvement
- **Metrics:** Cache hit rate, latency reduction, memory usage
- **Expected:** 50-90% speedup for cached items

## Phase 4: Adaptive Parameters

### 4.1 Implement Scene Complexity Detection
- **File:** `code/approach_2_5_optimized/complexity_detector.py` (new)
- **Metrics:**
  - Number of objects detected
  - Average confidence scores
  - Object diversity (unique classes)
  - Scene category (gaming vs navigation vs text)
- **Output:** Complexity score (simple/medium/complex)

### 4.2 Create Adaptive max_tokens Strategy
- **File:** `code/approach_2_5_optimized/llm_generator_optimized.py` (new)
- **Strategy:** Import and wrap Approach 2 LLM generator functions (best practice: extend, don't duplicate)
  - Import `generate_description` and model-specific functions from `approach_2_yolo_llm.llm_generator`
  - Create wrapper functions that add adaptive max_tokens logic
  - Preserve all existing functionality from Approach 2
- **Adaptive Strategy:**
  - Simple scenes (0-2 objects): max_tokens=100
  - Medium scenes (3-5 objects): max_tokens=150
  - Complex scenes (6+ objects): max_tokens=200
- **Implementation:** Wrapper functions that call Approach 2 functions with adaptive max_tokens
- **Fallback:** Use fixed max_tokens if complexity unavailable

### 4.3 Test Adaptive Parameters
- **Script:** `code/approach_2_5_optimized/test_adaptive_params.py` (new)
- **Method:** Compare fixed vs adaptive max_tokens on 8-image subset
- **Metrics:** Latency, word count, quality (completeness)
- **Expected:** 5-10% speedup for simple scenes, no quality loss

## Phase 5: Comprehensive Analysis

### 5.1 Create Optimization Comparison Script
- **File:** `code/evaluation/compare_approach2_5.py` (new)
- **Compare:**
  - Approach 2 baseline (GPT-4o-mini, original prompt, no cache)
  - Approach 2.5: GPT-3.5-turbo (faster model)
  - Approach 2.5: GPT-3.5-turbo + optimized prompt
  - Approach 2.5: GPT-3.5-turbo + optimized prompt + cache
  - Approach 2.5: GPT-3.5-turbo + optimized prompt + cache + adaptive params
- **Metrics:** Latency (mean, median, p95), quality (word count, completeness), cost, cache hit rate

### 5.2 Generate Comparison Report
- **File:** `results/approach_2_5_optimized/analysis/optimization_comparison.txt`
- **Content:**
  - Latency improvements by optimization
  - Quality tradeoffs
  - Cost analysis
  - Cache performance
  - Recommendations

### 5.3 Statistical Analysis
- **File:** `code/evaluation/statistical_tests_approach2_5.py` (new)
- **Tests:**
  - Paired t-tests: Approach 2 vs Approach 2.5
  - Paired t-tests: baseline vs each optimization level
  - ANOVA: compare all optimization levels
  - Effect sizes (Cohen's d)
- **Output:** `results/approach_2_5_optimized/analysis/statistical_tests.txt`

## Phase 6: Documentation Updates

### 6.1 Create Approach 2.5 README
- **File:** `code/approach_2_5_optimized/README.md` (new)
- **Content:**
  - Overview of Approach 2.5
  - Relationship to Approach 2
  - Optimization strategies section
  - Caching usage instructions
  - Adaptive parameters explanation
  - Performance benchmarks
  - Usage examples

### 6.2 Update Speed Optimization Plan
- **File:** `results/approach_2_yolo_llm/analysis/speed_optimization_plan.md`
- **Add:**
  - Approach 2.5 designation
  - Results from each optimization phase
  - Final performance metrics
  - Recommendations for production

### 6.3 Update PROJECT.md
- **File:** `PROJECT.md`
- **Add:** New section for Approach 2.5
- **Update:**
  - Approach 2 status: "Complete - Baseline"
  - Approach 2.5 status: "Speed Optimization Complete"
  - Final latency metrics for both approaches
  - Optimization techniques implemented

### 6.4 Update FINDINGS.md
- **File:** `FINDINGS.md`
- **Add:**
  - Approach 2.5 section
  - Comparison of Approach 2 vs Approach 2.5
  - Comparison of optimization levels
  - Final recommendations

## Success Criteria

### Performance Targets
- Mean latency: <1.5s (current: 1.45s, maintain or improve)
- P95 latency: <2.0s
- Cache hit rate: >50% (for repeated scenes)
- Quality: Word count 50-80 words, completeness >80%

### Quality Preservation
- No significant quality degradation from optimizations
- Qualitative evaluation confirms acceptable descriptions
- Safety-critical information preserved

### Code Quality
- All new code follows existing patterns
- Proper error handling
- Comprehensive logging
- Documentation complete

## Implementation Order

0. **Phase 0:** Setup and code reuse strategy (establish imports, regression tests)
1. **Phase 1:** Full batch validation (validate GPT-3.5-turbo)
2. **Phase 2:** Prompt optimization (quick win, low risk)
3. **Phase 3:** Caching (high impact, moderate complexity)
4. **Phase 4:** Adaptive parameters (low impact, simple)
5. **Phase 5:** Analysis (comprehensive comparison)
6. **Phase 6:** Documentation (finalize findings)

**Regression Testing:** Run `test_regression.py` after Phase 0, Phase 1, Phase 3, and Phase 6

## Files to Create

### New Directory Structure
- `code/approach_2_5_optimized/` - Main directory for Approach 2.5
  - `__init__.py` - Package initialization (imports Approach 2 components)
  - `cache_manager.py` - Cache implementation (new)
  - `complexity_detector.py` - Scene complexity detection (new)
  - `prompts_optimized.py` - Optimized prompt variants (new, extends Approach 2 prompts)
  - `llm_generator_optimized.py` - Optimized LLM generator (wraps Approach 2, adds optimizations)
  - `hybrid_pipeline_optimized.py` - Optimized pipeline (imports Approach 2, adds optimizations)
  - `batch_test_optimized.py` - Batch testing script (new)
  - `test_prompt_optimization.py` - Prompt testing (new)
  - `test_caching.py` - Cache testing (new)
  - `test_adaptive_params.py` - Adaptive params testing (new)
  - `test_regression.py` - Regression tests to ensure Approach 2 still works (new)
  - `README.md` - Approach 2.5 documentation

### Results Directory
- `results/approach_2_5_optimized/`
  - `raw/batch_results.csv` - Batch test results
  - `analysis/` - Analysis reports
  - `figures/` - Visualizations

### Evaluation Scripts
- `code/evaluation/compare_approach2_5.py` - Comprehensive comparison
- `code/evaluation/statistical_tests_approach2_5.py` - Statistical tests

## Files to Modify

- `PROJECT.md` - Add Approach 2.5 section
- `FINDINGS.md` - Add Approach 2.5 findings
- `results/approach_2_yolo_llm/analysis/speed_optimization_plan.md` - Update with Approach 2.5 designation

## Timeline Estimate

- Phase 0: 15-30 minutes (setup, imports, regression tests)
- Phase 1: 30-45 minutes (batch test + validation)
- Phase 2: 1-1.5 hours (prompt optimization)
- Phase 3: 2-3 hours (caching implementation)
- Phase 4: 1 hour (adaptive parameters)
- Phase 5: 1-2 hours (analysis)
- Phase 6: 1 hour (documentation)
- **Total: 7-10 hours** (includes regression testing overhead)

## Notes

### Code Reuse Strategy (Best Practices)
- **Import, don't duplicate:** Import Approach 2 components (`yolo_detector`, `llm_generator`, `prompts`) where possible
- **Extend, don't modify:** Create wrapper functions that add optimizations without changing Approach 2 code
- **Rationale:** Reduces maintenance burden, ensures consistency, follows DRY principle
- **Exception:** Only create new files when significant architectural changes needed (e.g., `cache_manager.py`, `complexity_detector.py`)

### Testing Strategy (Best Practices)
- **Regression tests:** Ensure Approach 2 remains functional after Approach 2.5 implementation
- **Independent testing:** Test each optimization phase independently before combining
- **Smoke tests:** Quick validation after each major phase
- **Rationale:** Prevents regressions, maintains project quality, ensures both approaches work

### General Principles
- Approach 2.5 is a distinct variant, not a modification of Approach 2
- All optimizations should be optional/configurable (backward compatible)
- Preserve existing Approach 2 results and functionality
- Focus on production-ready implementations (not just prototypes)
- Consider memory usage for caching (implement LRU eviction)
- Monitor quality alongside speed improvements
- Approach 2.5 results should be clearly distinguished from Approach 2 in all outputs

## Key Differences from Approach 2

1. **Model:** GPT-3.5-turbo (vs GPT-4o-mini/Claude Haiku)
2. **Optimizations:** Prompt optimization, caching, adaptive parameters
3. **Target:** <1.5s mean latency (vs 3.73s baseline)
4. **Directory:** Separate code directory (`approach_2_5_optimized/`)
5. **Results:** Separate results directory (`results/approach_2_5_optimized/`)

