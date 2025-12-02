# Evaluation Tools

This directory contains automated analysis scripts and helper tools for evaluating VLM results.

## Automated Analysis Scripts

### 1. `analyze_vlm_results.py`
**Purpose:** Comprehensive quantitative analysis of VLM results

**What it does:**
- Calculates latency statistics (mean, median, p75, p90, p95, p99, min, max, std dev)
- Analyzes latency by scenario (gaming, indoor, outdoor, text)
- Calculates response length metrics (word count, character count, tokens)
- Estimates costs per query and per 1000 queries

**Usage:**
```bash
python code/evaluation/analyze_vlm_results.py
```

**Output:**
- Console output with all statistics
- `results/approach_1_vlm/analysis/vlm_analysis_summary.txt` - Summary file

---

### 2. `statistical_tests.py`
**Purpose:** Statistical significance testing

**What it does:**
- One-way ANOVA to test if latency differs significantly across models
- Paired t-tests for pairwise model comparisons

**Usage:**
```bash
python code/evaluation/statistical_tests.py
```

**Output:**
- Console output with test results
- `results/approach_1_vlm/analysis/statistical_tests.txt` - Detailed results

---

### 3. `create_visualizations.py`
**Purpose:** Generate charts and graphs

**What it does:**
- Creates latency box plots
- Creates latency by scenario bar charts
- Creates response length comparison charts
- Creates cost comparison charts

**Usage:**
```bash
python code/evaluation/create_visualizations.py
```

**Output:**
- `results/approach_1_vlm/figures/latency_comparison.png`
- `results/approach_1_vlm/figures/latency_by_scenario.png`
- `results/approach_1_vlm/figures/response_length_comparison.png`
- `results/approach_1_vlm/figures/cost_comparison.png`

**Requirements:**
- matplotlib
- seaborn

---

## Manual Evaluation Helper Tools

### 4. `qualitative_evaluation_helper.py`
**Purpose:** Help with manual qualitative scoring

**What it does:**
- Creates CSV template for scoring descriptions (1-5 scale)
- Displays descriptions side-by-side for comparison
- Creates text file with all descriptions for review

**Usage:**

Create scoring template:
```bash
python code/evaluation/qualitative_evaluation_helper.py template
```

Create comparison file (all descriptions):
```bash
python code/evaluation/qualitative_evaluation_helper.py all
```

Interactive mode:
```bash
python code/evaluation/qualitative_evaluation_helper.py
```

**Output:**
- `results/approach_1_vlm/evaluation/qualitative_scores.csv` - Template for scoring
- `results/approach_1_vlm/evaluation/all_descriptions_comparison.txt` - All descriptions for review

**Scoring Criteria (1-5 scale):**
- **Completeness:** Does it cover all important elements?
- **Clarity:** Is it easy to understand?
- **Conciseness:** Is it brief and to the point?
- **Actionability:** Can the user act on this information?
- **Safety Focus:** Does it prioritize safety-critical info?

---

### 5. `safety_analysis_helper.py`
**Purpose:** Safety-critical error analysis for navigation images

**What it does:**
- Creates checklist for safety-critical elements
- Performs quick keyword analysis
- Identifies potential false negatives (missed hazards)

**Usage:**

Create safety analysis template:
```bash
python code/evaluation/safety_analysis_helper.py
```

Quick keyword analysis:
```bash
python code/evaluation/safety_analysis_helper.py keywords
```

**Output:**
- `results/approach_1_vlm/evaluation/safety_analysis.csv` - Template for safety analysis
- Console output with keyword statistics

**Safety Elements Checked:**
- **Indoor:** stairs, steps, obstacles, doors, walls, furniture, barriers, openings, exits, entrances
- **Outdoor:** crosswalks, traffic, vehicles, pedestrians, roads, curbs, sidewalks, obstacles, signs, lights

---

## Running All Analyses

To run all automated analyses at once:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all analyses
python code/evaluation/analyze_vlm_results.py
python code/evaluation/statistical_tests.py
python code/evaluation/create_visualizations.py

# Create helper templates
python code/evaluation/qualitative_evaluation_helper.py template
python code/evaluation/safety_analysis_helper.py
```

---

## Output Files Structure

```
results/
└── approach_1_vlm/
    ├── raw/
    │   └── batch_results.csv         # Main data file (126 results)
    ├── analysis/
    │   ├── vlm_analysis_summary.txt  # Quantitative summary
    │   └── statistical_tests.txt     # Statistical test results
    ├── evaluation/
    │   ├── qualitative_scores.csv    # Template for manual scoring
    │   ├── safety_analysis.csv       # Template for safety analysis
    │   └── all_descriptions_comparison.txt  # All descriptions for review
    └── figures/
        ├── latency_comparison.png
        ├── latency_by_scenario.png
        ├── response_length_comparison.png
        └── cost_comparison.png
```

---

## Next Steps

1. **Quantitative Analysis:** ✅ Complete (automated)
2. **Qualitative Evaluation:** ⏳ Manual work needed
   - Open `results/evaluation/qualitative_scores.csv`
   - Score each description (1-5 scale)
   - Save results
3. **Safety Analysis:** ⏳ Manual work needed
   - Open `results/evaluation/safety_analysis.csv`
   - Review descriptions for missed hazards
   - Mark false negatives and score safety
4. **Category-Specific Analysis:** ⏳ Manual work needed
   - Review descriptions by category
   - Identify best performer per category
5. **Tradeoff Analysis:** ⏳ After quality scores are available
   - Create latency vs quality scatter plots
   - Create cost vs quality analysis

