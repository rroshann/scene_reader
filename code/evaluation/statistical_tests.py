#!/usr/bin/env python3
"""
Statistical tests for VLM results
- ANOVA for latency differences across models
- Paired t-tests for pairwise comparisons
"""
import csv
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_rel
from pathlib import Path
import numpy as np

def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '2025-11-22T20:' in row.get('timestamp', ''):
                results.append(row)
    return results

def get_latencies_by_model(results):
    """Get latency data organized by model and image"""
    by_image = {}
    
    for r in results:
        if r['success'] == 'True' and r.get('latency_seconds'):
            filename = r['filename']
            model = r['model']
            try:
                latency = float(r['latency_seconds'])
                if latency < 20:  # Filter outliers
                    if filename not in by_image:
                        by_image[filename] = {}
                    by_image[filename][model] = latency
            except (ValueError, TypeError):
                continue
    
    return by_image

def anova_test(results):
    """Perform ANOVA test on latency across models"""
    by_image = get_latencies_by_model(results)
    
    # Get latencies for each model (only for images tested by all models)
    gpt4v = []
    gemini = []
    claude = []
    
    for filename, model_lats in by_image.items():
        if 'GPT-4V' in model_lats and 'Gemini' in model_lats and 'Claude' in model_lats:
            gpt4v.append(model_lats['GPT-4V'])
            gemini.append(model_lats['Gemini'])
            claude.append(model_lats['Claude'])
    
    if len(gpt4v) < 3:
        print("❌ Not enough paired data for ANOVA")
        return None
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(gpt4v, gemini, claude)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(gpt4v)
    }

def paired_t_tests(results):
    """Perform paired t-tests between models"""
    by_image = get_latencies_by_model(results)
    
    pairs = [
        ('GPT-4V', 'Gemini'),
        ('GPT-4V', 'Claude'),
        ('Gemini', 'Claude')
    ]
    
    results_dict = {}
    
    for model1, model2 in pairs:
        lat1 = []
        lat2 = []
        
        for filename, model_lats in by_image.items():
            if model1 in model_lats and model2 in model_lats:
                lat1.append(model_lats[model1])
                lat2.append(model_lats[model2])
        
        if len(lat1) >= 3:
            t_stat, p_value = ttest_rel(lat1, lat2)
            results_dict[f"{model1} vs {model2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n': len(lat1),
                'mean_diff': np.mean(np.array(lat1) - np.array(lat2))
            }
    
    return results_dict

def main():
    """Run statistical tests"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    print("=" * 60)
    print("STATISTICAL TESTS - Approach 1 (VLMs)")
    print("=" * 60)
    print()
    
    results = load_results(csv_path)
    
    # ANOVA
    print("1️⃣  ONE-WAY ANOVA (Latency across models)")
    print("-" * 60)
    anova_result = anova_test(results)
    if anova_result:
        print(f"F-statistic: {anova_result['f_statistic']:.4f}")
        print(f"p-value: {anova_result['p_value']:.6f}")
        print(f"Significant (p < 0.05): {'✅ YES' if anova_result['significant'] else '❌ NO'}")
        print(f"Sample size (n): {anova_result['n']}")
        print()
    
    # Paired t-tests
    print("2️⃣  PAIRED T-TESTS (Pairwise comparisons)")
    print("-" * 60)
    ttest_results = paired_t_tests(results)
    for comparison, result in ttest_results.items():
        print(f"\n{comparison}:")
        print(f"  t-statistic: {result['t_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  Significant (p < 0.05): {'✅ YES' if result['significant'] else '❌ NO'}")
        print(f"  Mean difference: {result['mean_diff']:.4f}s")
        print(f"  Sample size (n): {result['n']}")
    
    # Save results
    output_path = Path('results/approach_1_vlm/analysis/statistical_tests.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("STATISTICAL TESTS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        if anova_result:
            f.write("ANOVA Test:\n")
            f.write(f"F-statistic: {anova_result['f_statistic']:.4f}\n")
            f.write(f"p-value: {anova_result['p_value']:.6f}\n")
            f.write(f"Significant: {anova_result['significant']}\n\n")
        
        f.write("Paired T-Tests:\n")
        for comparison, result in ttest_results.items():
            f.write(f"\n{comparison}:\n")
            f.write(f"  t-statistic: {result['t_statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Significant: {result['significant']}\n")
            f.write(f"  Mean difference: {result['mean_diff']:.4f}s\n")
    
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == '__main__':
    main()

