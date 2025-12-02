# Latency Comparison - All Approaches

**Last Updated:** November 23, 2025  
**Total Approaches Tested:** 5 (with multiple configurations)

---

## Detailed Latency Table

| Approach | Configuration | Mean (s) | Median (s) | Std Dev (s) | Range (s) | Notes |
|----------|--------------|----------|------------|-------------|-----------|-------|
| **1. Pure VLMs** | GPT-4V | 5.63 | 2.83 | 17.02 | 0.5-113.1 | Fastest median, high variability |
| **1. Pure VLMs** | Gemini 2.5 Flash | 5.88 | 4.79 | 4.73 | 1.2-25.3 | Balanced performance |
| **1. Pure VLMs** | Claude 3.5 Haiku | 4.95 | 5.04 | 0.99 | 3.1-7.2 | Most consistent |
| **2. YOLO+LLM** | YOLOv8N + GPT-4o-mini | **3.73** | 3.39 | 1.36 | 1.8-8.5 | **Fastest overall mean** |
| **4. Local Models** | BLIP-2 OPT-2.7B (MPS) | 35.40 | 36.80 | 13.20 | 6.8-61.2 | Slowest, but zero cost |
| **6. RAG-Enhanced** | GPT-4V + RAG (Gaming) | 10.60 | 10.24 | 2.15 | 6.8-16.2 | Educational context |
| **7. Chain-of-Thought** | GPT-4V + CoT | 8.48 | 5.68 | 17.02 | 0.5-113.1 | Better safety detection |

---

## Summary Statistics

### Overall Mean Latency by Approach

| Approach | Mean Latency | Rank |
|----------|--------------|------|
| **Approach 2 (YOLO+LLM)** | **3.73s** | 🥇 Fastest |
| **Approach 1 (Pure VLMs)** | 5.49s | 🥈 2nd |
| **Approach 7 (Chain-of-Thought)** | 8.48s | 🥉 3rd |
| **Approach 6 (RAG-Enhanced)** | 10.60s | 4th |
| **Approach 4 (Local Models)** | 35.40s | 5th (slowest) |

### Speed Comparison (Relative to Fastest)

| Approach | Multiplier | Interpretation |
|----------|------------|----------------|
| **Approach 2 (YOLO+LLM)** | 1.00x | Baseline (fastest) |
| **Approach 1 (Pure VLMs)** | 1.47x | 47% slower |
| **Approach 7 (Chain-of-Thought)** | 2.27x | 2.27x slower |
| **Approach 6 (RAG-Enhanced)** | 2.84x | 2.84x slower |
| **Approach 4 (Local Models)** | 9.49x | 9.49x slower |

---

## Key Insights

### 🏆 Fastest Approaches
1. **Approach 2 (YOLO+LLM)** - 3.73s mean (fastest overall)
2. **Approach 1 - GPT-4V** - 2.83s median (fastest median, but high variability)
3. **Approach 1 - Claude** - 4.95s mean (most consistent)

### 📊 Consistency Rankings
1. **Claude 3.5 Haiku** - 0.99s std dev (most consistent)
2. **Gemini 2.5 Flash** - 4.73s std dev (moderate consistency)
3. **YOLO+LLM** - 1.36s std dev (good consistency)
4. **RAG-Enhanced** - 2.15s std dev (moderate consistency)
5. **GPT-4V** - 17.02s std dev (high variability)

### ⚡ Speed vs Consistency Tradeoffs

**For Speed-Critical Applications:**
- **Best:** Approach 2 (YOLO+LLM) - 3.73s mean, consistent
- **Alternative:** GPT-4V median (2.83s) but high variability

**For Consistent Performance:**
- **Best:** Claude 3.5 Haiku - 4.95s mean, 0.99s std dev
- **Alternative:** YOLO+LLM - 3.73s mean, 1.36s std dev

**For Zero Cost:**
- **Only Option:** Approach 4 (Local Models) - 35.4s mean, but $0.00 cost

---

## Latency Breakdown by Component (Where Applicable)

### Approach 2 (YOLO+LLM)
- **Detection (YOLO):** 0.21s (5.7% of total)
- **Generation (LLM):** 3.42s (91.6% of total)
- **Total:** 3.73s

### Approach 6 (RAG-Enhanced)
- **Base VLM:** 6.77s (45.9% of total)
- **Enhancement:** 3.69s (53.0% of total)
- **Retrieval:** 0.14s (0.9% of total)
- **Total:** 10.60s

### Approach 7 (Chain-of-Thought)
- **Baseline GPT-4V:** 5.63s mean
- **CoT Overhead:** +2.85s (+94.5% slower)
- **Total:** 8.48s mean

---

## Use Case Recommendations Based on Latency

| Use Case | Recommended Approach | Latency | Reasoning |
|----------|---------------------|---------|-----------|
| **Real-time Navigation** | Approach 2 (YOLO+LLM) | 3.73s | Fastest, consistent |
| **Speed-Critical Apps** | Approach 2 (YOLO+LLM) | 3.73s | Best speed/consistency balance |
| **General Purpose** | Approach 1 (Pure VLMs) | 5.49s | Good quality, acceptable speed |
| **Consistent Performance** | Approach 1 (Claude) | 4.95s | Most reliable |
| **Safety-Critical** | Approach 7 (CoT) | 8.48s | Better hazard detection |
| **Educational/Gaming** | Approach 6 (RAG) | 10.60s | Rich context, acceptable latency |
| **Privacy/Offline** | Approach 4 (Local) | 35.4s | Zero cost, but slow |

---

## Notes

- **Approach 1** includes 3 different models (GPT-4V, Gemini, Claude)
- **Approach 2** tested multiple YOLO variants; YOLOv8N + GPT-4o-mini shown (best overall)
- **Approach 4** tested on M1 Mac with MPS acceleration; CPU would be slower
- **Approach 6** tested only on gaming scenarios (10 images)
- **Approach 7** adds CoT overhead to baseline GPT-4V
- All latencies measured end-to-end (image input to description output)
- Standard deviation indicates consistency (lower = more consistent)

---

**Data Source:** FINDINGS.md, results/approach_*_local/analysis/

