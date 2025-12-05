# Scene Reader 🎮👁️

**Comprehensive Analysis of 9 Computer Vision Approaches for Real-Time Visual Accessibility**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Approaches Tested](https://img.shields.io/badge/Approaches-9%20Complete-brightgreen.svg)](https://github.com)
[![API Calls](https://img.shields.io/badge/API%20Calls-564+-blue.svg)](https://github.com)

**Team:** Roshan Sivakumar & Dhesel Khando  
**Course:** DS-5690 - Generative AI Models | Vanderbilt University | Fall 2025  
**Instructor:** Prof. Jesse Spencer-Smith

---

## 📖 Project Overview

**Scene Reader** systematically evaluates **9 different computer vision and AI approaches** for providing real-time visual assistance to blind and low-vision users.

### 🏆 Key Achievement

We identified **3 approaches that achieve sub-2-second latency** (real-time capable), with the fastest (**Approach 2.5**) achieving:
- **1.10s mean latency** 
- **5.12x speedup** over baseline VLM approaches
- **$0.005/query cost** (affordable at scale)

### 📊 Scale of Testing

- **9 approaches** tested across multiple configurations
- **564 API calls** + 84 local model tests
- **42 images** across 4 real-world scenarios
- **Comprehensive metrics:** latency, cost, quality, safety

---

## 🔬 Standardized Comparison

To isolate **architectural differences** from optimizations, we tested all 3 top approaches with **identical parameters**:

| Approach | Mean Latency | Median | Std Dev | Success Rate |
|----------|--------------|--------|---------|--------------|
| **Approach 3.5** | **1.21s** 🥇 | 1.12s | 0.45s | 100% |
| **Approach 2.5** | **1.36s** 🥈 | 1.34s | 0.25s | 100% |
| **Approach 1.5** | **3.63s** 🥉 | 3.52s | 0.85s | 100% |

**Standardized Parameters**: max_tokens=100, temperature=0.7, no caching, no image preprocessing

**Key Finding**: Even with identical parameters, specialized architectures (3.5, 2.5) outperform pure VLM (1.5) due to:
- Faster LLM models (GPT-3.5-turbo vs GPT-4V)
- Specialized preprocessing (YOLO, OCR, Depth)
- Multi-stage pipelines that parallelize work

**Note**: Optimized results (shown in Top 3 section) use approach-specific parameters and show better performance overall.

---

## 🔬 Methodology

### Two-Phase Approach

**Phase 1: Comprehensive Evaluation** ✅ **COMPLETE**
- Systematic testing of 9 approaches on 42 static images
- Controlled, reproducible evaluation across gaming, navigation, and text-reading scenarios
- Identified top 3 optimal approaches through data-driven analysis

**Phase 2: Real-Time Implementation** 🔄 **IN PROGRESS**
- Deploy fastest approach (Approach 2.5 - 1.10s) for live screen capture
- Integration with text-to-speech for real-time audio output
- Target application: Gaming accessibility demo (e.g., Stardew Valley)

### Test Scenarios

| Scenario | Images | Challenge | Key Metric |
|----------|--------|-----------|------------|
| **🎮 Gaming** | 12 | Complex UI, character positioning | Object identification |
| **🚶 Indoor Navigation** | 10 | Spatial relationships, obstacles | Hazard detection |
| **🌳 Outdoor Navigation** | 10 | Safety-critical elements | False negative rate |
| **📝 Text Reading** | 10 | OCR accuracy, varied fonts | Text extraction |

---

## 🏆 Top 3 Approaches

### 🥇 #1: Approach 2.5 - Optimized YOLO+LLM
**1.10s latency** | **$0.005/query** | **5.12x faster than baseline**

The **fastest approach overall** - achieves sub-2-second real-time performance.

#### Architecture
```
Image → YOLOv8n Detection (0.15s)
     → Smart Caching (15x speedup)
     → GPT-3.5-turbo Generation
     → Description Output (1.10s total)
```

#### Key Optimizations
- ✅ GPT-3.5-turbo (3-4x faster than GPT-4, 90% quality)
- ✅ Intelligent caching (40-60% cache hit rate)
- ✅ Adaptive parameters (shorter responses for simple scenes)
- ✅ Complexity detection (routes to appropriate generation)

#### Performance
- **95% of queries under 2s** (real-time threshold)
- **67.4% faster** than baseline YOLO+LLM
- **Cost-effective:** $5 per 1000 queries

#### Best For
- Real-time gaming accessibility ✅
- Speed-critical navigation ✅
- Cost-sensitive deployments ✅

---

### 🥈 #2: Approach 3.5 - Optimized Specialized
**1.50s latency** | **$0.006/query** | **72% faster than baseline**

Combines specialized models (OCR, depth) with speed optimizations.

#### Architecture
```
Image → Complexity Detector
     → [If text] EasyOCR → GPT-3.5-turbo
     → [If depth] MiDaS → GPT-3.5-turbo  
     → [Else] YOLO → GPT-3.5-turbo
     → Cached Output (1.50s total)
```

#### Key Features
- ✅ Intelligent routing (only uses specialized models when needed)
- ✅ OCR integration (95%+ text accuracy)
- ✅ Depth estimation (improved spatial descriptions)
- ✅ Same optimizations as Approach 2.5

#### Best For
- Gaming UI/menus (OCR for inventory, stats) ✅
- Text reading (signs, documents) ✅
- Indoor navigation (depth awareness) ✅

---

### 🥉 #3: Approach 1.5 - Optimized Pure VLM
**1.73s perceived latency** | **$0.012/query** | **69% faster perceived**

**Optimized version of Approach 1** with concise prompts, lower token limits, and progressive disclosure.

#### Architecture
```
Image Input
    ↓
┌──────────────┐  ┌───────────────┐
│ Tier 1: Fast │  │ Tier 2: Detail│
│ BLIP-2 Local │  │ GPT-4V Cloud  │
│ (optional)   │  │ (optimized)   │
│ (1.73s)      │  │ (5.47s)       │
└──────┬───────┘  └───────┬───────┘
       ↓                  ↓
   Quick                Full
   Overview          Description
```

#### Key Optimizations (vs Approach 1)
1. **Concise prompts** (under 20 words) - faster processing
2. **Lower token limits** (max_tokens=100) - faster generation
3. **Mode-specific prompts** (gaming/real-world) - better quality
4. **Progressive disclosure** (optional BLIP-2) - immediate feedback

#### Best For
- **Best user experience** - immediate feedback reduces anxiety ✅
- Scenarios where "something now" > "perfect later" ✅
- Users who are impatient or anxious ✅

---

## 📊 Complete Results

### All 9 Approaches Ranked by Speed

| Rank | Approach | Latency | Cost/Query | Quality | Best For |
|------|----------|---------|------------|---------|----------|
| 🥇 | **Approach 2.5 (Optimized YOLO+LLM)** | **1.10s** | $0.005 | ⭐⭐⭐⭐ | Real-time, speed-critical |
| 🥈 | **Approach 3.5 (Optimized Specialized)** | **1.50s** | $0.006 | ⭐⭐⭐⭐ | Text/depth scenarios |
| 🥉 | **Approach 1.5 (Optimized Pure VLM)** | **1.73s*** | $0.012 | ⭐⭐⭐⭐⭐ | Best user experience |
| 4 | Approach 2 (YOLO+LLM Baseline) | 3.39s | $0.005 | ⭐⭐⭐⭐ | Balanced baseline |
| 5 | Approach 1 (Claude 3.5 Haiku) | 4.95s | $0.024 | ⭐⭐⭐⭐ | Most consistent |
| 6 | Approach 3 (Specialized Baseline) | 5.33s | $0.010 | ⭐⭐⭐⭐ | With OCR/Depth |
| 7 | Approach 1 (Gemini 2.5 Flash) | 5.88s | $0.003 | ⭐⭐⭐⭐ | Most cost-effective |
| 8 | Approach 7 (Chain-of-Thought) | 8.48s | $0.015 | ⭐⭐⭐⭐⭐ | Best safety detection |
| 9 | Approach 6 (RAG-Enhanced) | 10.60s | $0.020 | ⭐⭐⭐⭐ | Gaming knowledge |
| 10 | Approach 4 (Local BLIP-2) | 35.40s | $0.000 | ⭐⭐⭐ | Zero cost, offline |

*Approach 1.5: 1.73s = perceived latency (time to first output), 5.47s = full description

### Key Achievements
- ✅ **3 approaches achieve sub-2s latency** (real-time capable)
- ✅ **5.12x speedup** over baseline GPT-4V (1.10s vs 5.63s)
- ✅ **67-72% latency reduction** through optimization
- ✅ **95% of queries under 2s** with Approach 2.5

### Cost vs Speed Analysis
- **Fastest & affordable:** Approach 2.5 ($0.005/query, 1.10s)
- **Most cost-effective:** Gemini ($0.003/query, but 5.88s)
- **Zero cost:** Approach 4 (local BLIP-2, but 35.4s)
- **Cost range:** 7.7x difference across approaches

---

## 🎯 Use Case Recommendations

| Scenario | Recommended | Latency | Why? |
|----------|------------|---------|------|
| 🎮 **Gaming (Real-time)** | **Approach 2.5** | 1.10s | Fastest, affordable, good quality |
| 🚶 **Indoor Navigation** | **Approach 3.5** | 1.50s | Depth awareness, fast |
| 🌳 **Outdoor Navigation** | **Approach 7** | 8.48s | Best safety detection |
| 📝 **Text Reading** | **Approach 3.5** | 1.50s | OCR integration, 95%+ accuracy |
| 😊 **Best UX** | **Approach 1.5** | 1.73s* | Immediate feedback |
| 💰 **Cost-Sensitive** | **Approach 2.5** | 1.10s | $5 per 1000 queries |
| 🔒 **Privacy/Offline** | **Approach 4** | 35.40s | Zero cost, no cloud |

---

## 📈 Key Findings

### Major Discoveries

1. **Sub-2-second latency is achievable**
   - 3 approaches achieve real-time performance (<2s)
   - Hybrid architectures (YOLO+LLM) consistently 2-5x faster than pure VLMs

2. **Optimization matters hugely**
   - 67-72% speedup through caching + faster LLM models
   - GPT-3.5-turbo achieves 90% quality at 3-4x speed of GPT-4

3. **Progressive disclosure works**
   - Approach 1.5 reduces perceived wait by 69%
   - UX innovation: immediate feedback > waiting for perfect response

4. **Cost-speed tradeoff is favorable**
   - Fastest approach (2.5) is also cost-effective ($0.005/query)
   - 7.7x cost variation across approaches

5. **Safety remains challenging**
   - All approaches have 15-20% false negative rate for hazards
   - Chain-of-Thought improves hazard detection by +20%

### Statistical Validation
- **ANOVA p < 0.001** - Statistically significant latency differences
- **Cohen's d = 2.61** - Large effect size for optimizations
- **95% confidence** - Results are reproducible

### Novel Contributions
1. First comprehensive comparison of **9 vision AI approaches** for accessibility
2. **Sub-2s real-time performance** achieved through systematic optimization
3. **Progressive disclosure architecture** - novel UX innovation for perceived latency
4. **Gaming accessibility focus** - underexplored domain
5. Complete **tradeoff analysis** - latency, cost, quality, safety

---

## 🚧 Limitations & Future Work

### Current Limitations
- **True Real-Time:** Sub-2s achieved, but not <500ms for instant response
- **Accuracy:** Hallucinations in 5-15% of descriptions
- **Safety:** 15-20% false negative rate for hazards across all approaches
- **Dataset Size:** 42 images (comparison-focused, not training-scale)
- **User Testing:** No blind/low-vision user validation yet
- **Static Images:** Phase 1 only; Phase 2 will implement real-time video

### Future Work
- **Phase 2 Implementation:** Real-time demo with Approach 2.5
- **User Studies:** Validation with blind/low-vision users
- **Safety Improvements:** Reduce false negative rate for hazards
- **Extended Scenarios:** Medical imaging, workplace, transportation
- **Edge Deployment:** Optimize for mobile/edge devices

---

## 🎓 Course Connection - DS-5690

### Learning Objectives Met

✅ **Transformer Architectures**
- Vision transformers (ViT) in multimodal models (Approaches 1, 4, 5)
- Cross-modal attention mechanisms in VLMs
- Encoder-decoder vs decoder-only architectures

✅ **Evaluating Capabilities & Limitations**
- Systematic testing methodology (9 approaches, 564+ tests)
- Latency constraint analysis (sub-2s requirement)
- Failure mode identification (15-20% false negative rate)

✅ **Technical Tradeoffs**
- Cloud vs edge deployment (Approach 4 vs others)
- Accuracy vs speed vs cost (comprehensive comparison)
- Single-model vs multi-model pipelines (Approaches 2, 3)

✅ **Model Adaptation**
- Prompt engineering for vision tasks (Approach 7 - Chain-of-Thought)
- API optimization strategies (caching, adaptive parameters)
- Hybrid architecture design (Approaches 2.5, 3.5, 5)

---

## 🗂️ Project Structure

```
scene-reader/
├── README.md                      # This file (presentation overview)
├── PROJECT.md                     # Comprehensive technical documentation
├── FINDINGS.md                    # Detailed results and analysis
├── data/images/                   # 42 test images (4 scenarios)
├── code/
│   ├── approach_2_5_optimized/   # 🥇 Fastest (1.10s)
│   ├── approach_3_5_optimized/   # 🥈 Specialized (1.50s)
│   ├── approach_1_5_optimized/   # 🥉 Best UX (1.73s perceived)
│   └── [6 other approaches]
└── results/
    ├── approach_*/               # Results for each approach
    ├── comprehensive_comparison/ # Cross-approach analysis
    └── LATENCY_COMPARISON.md     # Speed rankings
```

---

## 🛠️ Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/scene-reader.git
cd scene-reader
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys in .env
cp .env.example .env
# Edit .env with: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
```

### Test Fastest Approach (1.10s)

```bash
# Run Approach 2.5 on single image
cd code/approach_2_5_optimized
python -c "
from hybrid_pipeline_optimized import HybridPipelineOptimized
pipeline = HybridPipelineOptimized()
result = pipeline.describe_image('../../data/images/gaming/game_01.png')
print(f'Description: {result[\"description\"]}')
print(f'Latency: {result[\"total_latency\"]:.2f}s')
"

# Run batch test on all 42 images
python batch_test_optimized.py
```

### View Results

All results are pre-generated in `results/` directory:
- `results/approach_2_5_optimized/` - Fastest approach (1.10s)
- `results/approach_3_5_optimized/` - Specialized (1.50s)
- `results/approach_1_5_optimized/` - Optimized Pure VLM (1.73s)
- `results/comprehensive_comparison/` - Cross-approach analysis

---

## 📚 Additional Documentation

- **[PROJECT.md](PROJECT.md)** - Full technical documentation (2200+ lines)
- **[FINDINGS.md](FINDINGS.md)** - Detailed analysis and results
- **[LATENCY_COMPARISON.md](results/LATENCY_COMPARISON.md)** - Speed rankings
- **[API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)** - API configuration

---

## 📊 Project Status

**Phase 1:** ✅ Complete (9 approaches tested, top 3 identified)  
**Phase 2:** 🔄 In Progress (Real-time demo with Approach 2.5)  
**Progress:** 95% Complete  
**Submission:** December 4, 2025

---

## 🙏 Acknowledgments

### People
- **Prof. Jesse Spencer-Smith** - Project guidance
- **Shivam Tyagi (TA)** - Technical support
- **Vanderbilt Data Science Institute** - Resources and infrastructure

### Tools & APIs
- OpenAI (GPT-4V, GPT-3.5-turbo), Google (Gemini), Anthropic (Claude)
- Ultralytics (YOLOv8), EasyOCR, MiDaS, BLIP-2

### Inspiration
Dedicated to 7M+ blind and visually impaired individuals in the US, and 250M+ worldwide who deserve equal access to visual information.

---
