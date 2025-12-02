# Results Directory Structure

This directory contains all results organized by approach.

## Current Structure

```
results/
└── approach_1_vlm/          # Approach 1: Pure Vision-Language Models
    ├── raw/                  # Raw API results
    │   └── batch_results.csv
    ├── analysis/             # Quantitative analysis results
    │   ├── vlm_analysis_summary.txt
    │   └── statistical_tests.txt
    ├── figures/              # Visualizations
    │   ├── latency_comparison.png
    │   ├── latency_by_scenario.png
    │   ├── response_length_comparison.png
    │   └── cost_comparison.png
    └── evaluation/           # Manual evaluation templates
        ├── qualitative_scores.csv
        ├── safety_analysis.csv
        └── all_descriptions_comparison.txt
```

## Future Approaches

As new approaches are implemented, they will be added with the same structure:

```
results/
├── approach_1_vlm/          # ✅ Complete
├── approach_2_yolo_llm/     # ⏳ Planned
├── approach_3_specialized/  # ⏳ Planned
├── approach_4_local/        # ⏳ Planned
├── approach_5_streaming/    # ⏳ Planned
├── approach_6_rag/          # ⏳ Planned
└── approach_7_cot/          # ⏳ Planned
```

Each approach folder contains:
- `raw/` - Raw test results (CSV files)
- `analysis/` - Quantitative analysis outputs
- `figures/` - Charts and visualizations
- `evaluation/` - Manual evaluation templates and results

## Accessing Results

### Approach 1 (VLMs)
- **Raw data:** `results/approach_1_vlm/raw/batch_results.csv`
- **Analysis:** `results/approach_1_vlm/analysis/`
- **Figures:** `results/approach_1_vlm/figures/`
- **Evaluation:** `results/approach_1_vlm/evaluation/`

### Running Analyses

All analysis scripts have been updated to use the new paths. They will automatically save to the correct approach folder.

```bash
# Analyze Approach 1 results
python code/evaluation/analyze_vlm_results.py
python code/evaluation/statistical_tests.py
python code/evaluation/create_visualizations.py
```

## Notes

- All scripts have been updated to use the new folder structure
- Old paths (`results/raw/`, `results/analysis/`, etc.) have been removed
- Each approach is now self-contained and easy to compare

