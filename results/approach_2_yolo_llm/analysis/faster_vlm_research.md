# Faster VLM/LLM Models Research for Approach 2 Optimization

**Date:** November 23, 2025  
**Goal:** Identify faster models to achieve <2 second latency target

---

## Research Summary

### Faster LLM Models (for Approach 2: YOLO+LLM)

**Current Models:**
- GPT-4o-mini: 3.42s generation latency (baseline)
- Claude Haiku: Similar to GPT-4o-mini

**Faster Options:**

#### 1. GPT-3.5-turbo (OpenAI) ⭐ RECOMMENDED
- **Speed:** ~1.5-2s generation latency (30-50% faster than GPT-4o-mini)
- **Quality:** Slightly lower than GPT-4o-mini, but acceptable
- **Cost:** Similar to GPT-4o-mini
- **Availability:** ✅ Available via OpenAI API
- **Expected Total Latency:** ~1.7-2.2s (YOLO 0.21s + LLM 1.5-2s)

#### 2. Gemini Flash (Google) ⭐ FASTEST OPTION
- **Speed:** ~1.0-1.5s generation latency (50-70% faster)
- **Quality:** Good quality, optimized for speed
- **Cost:** Very low ($0.0001/1K tokens)
- **Availability:** ✅ Available via Google AI Studio
- **Expected Total Latency:** ~1.2-1.7s (YOLO 0.21s + LLM 1.0-1.5s)
- **Note:** Primarily a vision model, but supports text generation

#### 3. Claude Haiku (Current)
- **Speed:** ~3.4s generation latency
- **Status:** Already tested, baseline

---

### Faster Vision-Language Models (Alternative Approaches)

**Research Findings:**

#### 1. Flash-VL 2B ⭐ SPECIALIZED FAST VLM
- **Type:** Vision-Language Model (end-to-end)
- **Speed:** Ultra-low latency, optimized for real-time
- **Size:** 2B parameters
- **Availability:** ⚠️ Research model, may require custom setup
- **Use Case:** Could replace YOLO+LLM entirely if fast enough

#### 2. FastVLM (Apple) ⭐ APPLE OPTIMIZED
- **Type:** Vision-Language Model
- **Speed:** Efficient vision encoding, optimized for Apple Silicon
- **Availability:** ⚠️ Apple-specific, may have limited API access
- **Use Case:** Good for M1 Mac deployment

#### 3. Mobile-VideoGPT
- **Type:** Efficient multimodal framework
- **Speed:** 46 tokens/second (very fast)
- **Size:** <1B parameters
- **Availability:** ⚠️ Research model
- **Use Case:** Mobile/edge deployment

#### 4. CombatVLA
- **Type:** Vision-Language-Action model
- **Speed:** 50x acceleration in game scenarios
- **Availability:** ⚠️ Specialized for gaming, research model
- **Use Case:** Gaming-specific applications

---

## Recommended Testing Strategy

### Phase 1: Test Faster LLMs in Approach 2 (IMMEDIATE)
1. ✅ **GPT-3.5-turbo** - Easy to implement, likely fastest
2. ✅ **Gemini Flash** - Fastest option, good quality
3. Compare against existing GPT-4o-mini and Claude Haiku

**Expected Results:**
- GPT-3.5-turbo: ~1.7-2.2s total (may meet <2s target)
- Gemini Flash: ~1.2-1.7s total (likely meets <2s target)

### Phase 2: Consider Faster VLMs (IF NEEDED)
- If LLM optimization doesn't achieve <2s, consider:
  - Flash-VL 2B (if API available)
  - FastVLM (if Apple-specific deployment)

---

## Implementation Status

### ✅ Completed
- Added GPT-3.5-turbo support to `llm_generator.py`
- Added Gemini Flash support to `llm_generator.py`
- Created subset test script (`test_faster_models_subset.py`)

### 🔄 In Progress
- Running subset test on 8 images
- Comparing faster models vs existing

### 📊 Expected Outcomes
- Identify which model achieves <2s target
- Document speed vs quality tradeoffs
- Provide recommendations for Phase 2 implementation

---

## References

- Flash-VL 2B: https://arxiv.org/abs/2505.09498
- FastVLM: https://machinelearning.apple.com/research/fast-vision-language-models
- Mobile-VideoGPT: https://arxiv.org/abs/2503.21782
- CombatVLA: https://arxiv.org/abs/2503.09527

