# Approach 6: RAG-Enhanced Vision

## Overview

Approach 6 implements a RAG (Retrieval-Augmented Generation) enhanced vision pipeline that combines VLM descriptions with retrieved game knowledge to provide context-aware, educational descriptions for gaming scenarios. This is a novel contribution to gaming accessibility.

## Architecture

```
Image (Gaming Screenshot)
    ↓
VLM: Generate Base Description
    - "Character on platform, enemy bug ahead, health bar shows 6/9"
    ↓
Entity Extraction & Game Identification
    - Identified: "Slay the Spire", "Sentry", "Defect character"
    ↓
Knowledge Base Search (Vector Similarity)
    - Retrieved chunks about Sentry enemy, Defect character, combat mechanics
    ↓
Enhanced LLM Generation
    - "You're playing Slay the Spire. The Defect character is facing a Sentry enemy. 
       Sentries add Dazed cards to your deck. Your health is 6/9. Consider blocking 
       before attacking to reduce damage."
    ↓
Context-Aware, Educational Description
```

## Components

### 1. Knowledge Base (`knowledge_processor.py`, `vector_store.py`)
- **Game Knowledge:** Manually curated knowledge for Slay the Spire, Stardew Valley, Tic Tac Toe
- **Vector Database:** ChromaDB with sentence-transformers embeddings
- **Retrieval:** Semantic search using cosine similarity

### 2. Entity Extractor (`entity_extractor.py`)
- **Game Identification:** Filename-based (fast, free) or LLM-based (more accurate)
- **Entity Extraction:** Identifies game entities from descriptions
- **Fallback:** Simple pattern matching if LLM unavailable

### 3. RAG Pipeline (`rag_pipeline.py`)
- **Stage 1:** Base VLM description (GPT-4V, Gemini, Claude)
- **Stage 2:** Entity extraction and game identification
- **Stage 3:** Knowledge retrieval (vector search)
- **Stage 4:** Enhanced description generation

### 4. Prompts (`prompts.py`)
- Entity extraction prompts
- Enhanced generation prompts with context fusion
- System prompts for accessibility focus

## Usage

### Installation

```bash
# Install dependencies
pip install chromadb sentence-transformers

# Or install all requirements
pip install -r requirements.txt
```

### Initialize Knowledge Base

```bash
# Process game knowledge and create vector store
python code/approach_6_rag/knowledge_processor.py
python code/approach_6_rag/vector_store.py
```

### Testing Single Image

```bash
# Test RAG pipeline on a single image
python code/approach_6_rag/rag_pipeline.py data/images/gaming/SlayTheSpire_Defect_vs_Sentry_ZapPlus.png
```

### Batch Testing

```bash
# Test all configurations on gaming images
python code/approach_6_rag/batch_test_rag.py
```

**Configurations tested:**
- Base VLM: GPT-4V, Gemini 2.5 Flash, Claude 3.5 Haiku (3 configs)
- RAG-Enhanced: Same 3 VLMs with RAG (3 configs)
- Total: 6 configurations × 12 gaming images = 72 tests

## Analysis

### Quantitative Analysis

```bash
python code/evaluation/analyze_rag_results.py
```

Generates:
- Latency statistics (overall, by stage, by VLM)
- Cost analysis
- Retrieval quality metrics
- Response length comparison

### Visualizations

```bash
python code/evaluation/create_rag_visualizations.py
```

Creates charts:
- Latency comparison (Base vs RAG)
- Latency breakdown by stage
- Latency by VLM model
- Response length comparison
- Retrieval statistics

### Statistical Tests

```bash
python code/evaluation/statistical_tests_rag.py
```

Performs:
- Paired t-test: Base vs RAG latency
- ANOVA: Latency by VLM model
- Paired t-test: Response length

### Baseline Comparison

```bash
python code/evaluation/compare_rag_vs_baseline.py
```

Compares Approach 6 with Approach 1 on gaming subset.

### Retrieval Quality

```bash
python code/evaluation/retrieval_quality_analyzer.py
```

Analyzes:
- Retrieval success rate
- Average chunks retrieved
- Game identification accuracy

## Expected Performance

### Latency
- **Base VLM:** 2-6 seconds (depending on model)
- **Entity Extraction:** <0.1s (filename-based) or 0.5-1s (LLM-based)
- **Retrieval:** 0.01-0.05s (vector search)
- **Enhancement:** 0.5-2s (LLM generation)
- **Total RAG:** ~2-8 seconds (base + enhancement)

### Cost
- **Base VLM:** Same as Approach 1
- **Enhancement:** ~$0.0001-0.0005 per query (GPT-4o-mini)
- **Total:** ~2x base VLM cost (two LLM calls)

### Quality
- **Expected:** +20-30% improvement in gaming scenario descriptions
- **Context-aware:** Provides game-specific knowledge
- **Educational:** Teaches game mechanics while describing

## Strengths

- **Context-aware:** Provides game-specific knowledge
- **Educational:** Teaches mechanics while describing
- **Helpful:** More actionable for gameplay decisions
- **Novel:** First application of RAG to gaming accessibility
- **Extendable:** Can add any knowledge domain

## Weaknesses

- **Slower:** ~2x latency (base + enhancement calls)
- **Costlier:** ~2x cost (two LLM calls)
- **Domain-specific:** Only beneficial for games with knowledge bases
- **Complex:** More components to maintain

## Use Cases

- **Gaming accessibility** (primary innovation)
- **Educational applications**
- **Domain-specific assistance**
- **When context enhances understanding**

## File Structure

```
code/approach_6_rag/
├── __init__.py
├── knowledge_processor.py
├── vector_store.py
├── entity_extractor.py
├── rag_pipeline.py
├── prompts.py
├── batch_test_rag.py
└── README.md

data/knowledge_base/
├── games/
│   ├── slay_the_spire/
│   ├── stardew_valley/
│   └── tic_tac_toe/
└── vector_db/

results/approach_6_rag/
├── raw/
│   └── batch_results.csv
├── analysis/
│   ├── rag_analysis_summary.txt
│   ├── statistical_tests.txt
│   ├── rag_vs_baseline_comparison.txt
│   └── retrieval_quality.txt
├── figures/
│   └── *.png
└── evaluation/
    └── qualitative_scores.csv
```

## Notes

- Knowledge base is manually curated (can be expanded)
- Entity extraction uses filename-based identification by default (fast, free)
- Vector store persists between runs (ChromaDB)
- Testing focuses on gaming images only (12 images)
- RAG enhancement only applied when game is identified

