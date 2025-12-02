#!/usr/bin/env python3
"""
Statistical comparison of CoT vs baseline GPT-4V
Performs paired t-tests and effect size calculations
"""
import csv
import scipy.stats as stats
from scipy.stats import ttest_rel
from pathlib import Path
import numpy as np

def load_cot_results(csv_path):
    """Load CoT results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('prompt_type') == 'CoT' and row['success'] == 'True':
                results.append(row)
    return results

def load_baseline_results(csv_path):
    """Load baseline GPT-4V results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == 'GPT-4V' and '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True':
                    results.append(row)
    return results

def get_paired_data(cot_results, baseline_results):
    """Get paired data for CoT and baseline by filename"""
    cot_by_file = {r['filename']: r for r in cot_results}
    baseline_by_file = {r['filename']: r for r in baseline_results}
    
    # Find common files
    common_files = set(cot_by_file.keys()) & set(baseline_by_file.keys())
    
    paired_latency = []
    paired_length = []
    paired_tokens = []
    
    for filename in common_files:
        cot = cot_by_file[filename]
        baseline = baseline_by_file[filename]
        
        # Latency
        if cot.get('latency_seconds') and baseline.get('latency_seconds'):
            try:
                cot_lat = float(cot['latency_seconds'])
                base_lat = float(baseline['latency_seconds'])
                if cot_lat < 30 and base_lat < 30:  # Filter outliers
                    paired_latency.append((cot_lat, base_lat))
            except (ValueError, TypeError):
                pass
        
        # Length
        if cot.get('description') and baseline.get('description'):
            cot_words = len(cot['description'].split())
            base_words = len(baseline['description'].split())
            paired_length.append((cot_words, base_words))
        
        # Tokens
        if cot.get('tokens_used') and baseline.get('tokens_used'):
            try:
                cot_tokens = int(cot['tokens_used'])
                base_tokens = int(baseline['tokens_used'])
                paired_tokens.append((cot_tokens, base_tokens))
            except (ValueError, TypeError):
                pass
    
    return {
        'latency': paired_latency,
        'length': paired_length,
        'tokens': paired_tokens,
        'common_files': len(common_files)
    }

def cohens_d(data1, data2):
    """Calculate Cohen's d effect size"""
    n1 = len(data1)
    n2 = len(data2)
    
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    var1 = np.var(data1, ddof=1)
    var2 = np.var(data2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    d = (mean1 - mean2) / pooled_std
    return d

def interpret_effect_size(d):
    """Interpret Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def perform_paired_t_test(paired_data, metric_name):
    """Perform paired t-test on paired data"""
    if len(paired_data) < 3:
        return None
    
    cot_values = [x[0] for x in paired_data]
    baseline_values = [x[1] for x in paired_data]
    
    # Paired t-test
    t_stat, p_value = ttest_rel(cot_values, baseline_values)
    
    # Calculate effect size (Cohen's d)
    differences = np.array(cot_values) - np.array(baseline_values)
    d = cohens_d(cot_values, baseline_values)
    
    # Calculate means
    cot_mean = np.mean(cot_values)
    baseline_mean = np.mean(baseline_values)
    mean_diff = cot_mean - baseline_mean
    
    return {
        'metric': metric_name,
        'n': len(paired_data),
        'cot_mean': cot_mean,
        'baseline_mean': baseline_mean,
        'mean_difference': mean_diff,
        'percent_change': (mean_diff / baseline_mean) * 100 if baseline_mean != 0 else 0,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': d,
        'effect_size': interpret_effect_size(d)
    }

def main():
    """Run statistical tests"""
    cot_csv_path = Path('results/approach_7_cot/raw/batch_results.csv')
    baseline_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    
    if not cot_csv_path.exists():
        print(f"❌ CoT results file not found: {cot_csv_path}")
        return
    
    if not baseline_csv_path.exists():
        print(f"❌ Baseline results file not found: {baseline_csv_path}")
        return
    
    print("=" * 60)
    print("STATISTICAL COMPARISON: CoT vs Baseline GPT-4V")
    print("=" * 60)
    print()
    
    cot_results = load_cot_results(cot_csv_path)
    baseline_results = load_baseline_results(baseline_csv_path)
    
    print(f"📊 Loaded {len(cot_results)} CoT results")
    print(f"📊 Loaded {len(baseline_results)} baseline results")
    print()
    
    # Get paired data
    paired_data = get_paired_data(cot_results, baseline_results)
    print(f"📊 Found {paired_data['common_files']} common images for comparison")
    print()
    
    # Perform tests
    test_results = {}
    
    # Latency test
    if paired_data['latency']:
        print("1️⃣  LATENCY: Paired T-Test")
        print("-" * 60)
        result = perform_paired_t_test(paired_data['latency'], 'Latency (seconds)')
        if result:
            test_results['latency'] = result
            print(f"Sample size (n): {result['n']}")
            print(f"CoT mean: {result['cot_mean']:.2f}s")
            print(f"Baseline mean: {result['baseline_mean']:.2f}s")
            print(f"Mean difference: {result['mean_difference']:+.2f}s")
            print(f"Percent change: {result['percent_change']:+.1f}%")
            print(f"t-statistic: {result['t_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.6f}")
            print(f"Significant (p < 0.05): {'✅ YES' if result['significant'] else '❌ NO'}")
            print(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)")
            print()
    
    # Length test
    if paired_data['length']:
        print("2️⃣  RESPONSE LENGTH: Paired T-Test")
        print("-" * 60)
        result = perform_paired_t_test(paired_data['length'], 'Response Length (words)')
        if result:
            test_results['length'] = result
            print(f"Sample size (n): {result['n']}")
            print(f"CoT mean: {result['cot_mean']:.1f} words")
            print(f"Baseline mean: {result['baseline_mean']:.1f} words")
            print(f"Mean difference: {result['mean_difference']:+.1f} words")
            print(f"Percent change: {result['percent_change']:+.1f}%")
            print(f"t-statistic: {result['t_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.6f}")
            print(f"Significant (p < 0.05): {'✅ YES' if result['significant'] else '❌ NO'}")
            print(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)")
            print()
    
    # Token test
    if paired_data['tokens']:
        print("3️⃣  TOKEN USAGE: Paired T-Test")
        print("-" * 60)
        result = perform_paired_t_test(paired_data['tokens'], 'Token Usage')
        if result:
            test_results['tokens'] = result
            print(f"Sample size (n): {result['n']}")
            print(f"CoT mean: {result['cot_mean']:.0f} tokens")
            print(f"Baseline mean: {result['baseline_mean']:.0f} tokens")
            print(f"Mean difference: {result['mean_difference']:+.0f} tokens")
            print(f"Percent change: {result['percent_change']:+.1f}%")
            print(f"t-statistic: {result['t_statistic']:.4f}")
            print(f"p-value: {result['p_value']:.6f}")
            print(f"Significant (p < 0.05): {'✅ YES' if result['significant'] else '❌ NO'}")
            print(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)")
            print()
    
    # Save results
    output_path = Path('results/approach_7_cot/analysis/statistical_comparison.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("STATISTICAL COMPARISON: CoT vs Baseline GPT-4V\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Common Images: {paired_data['common_files']}\n\n")
        
        for metric, result in test_results.items():
            f.write(f"{result['metric']}:\n")
            f.write(f"  Sample size (n): {result['n']}\n")
            f.write(f"  CoT mean: {result['cot_mean']:.2f}\n")
            f.write(f"  Baseline mean: {result['baseline_mean']:.2f}\n")
            f.write(f"  Mean difference: {result['mean_difference']:+.2f}\n")
            f.write(f"  Percent change: {result['percent_change']:+.1f}%\n")
            f.write(f"  t-statistic: {result['t_statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Significant (p < 0.05): {result['significant']}\n")
            f.write(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)\n\n")
    
    print(f"✅ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Key Question: Does CoT significantly improve quality metrics while maintaining acceptable latency?")
    print()
    
    if test_results.get('latency'):
        lat = test_results['latency']
        if lat['significant']:
            print(f"⚠️  Latency: CoT is {'slower' if lat['mean_difference'] > 0 else 'faster'} by {abs(lat['mean_difference']):.2f}s (statistically significant)")
        else:
            print(f"✅ Latency: No significant difference ({lat['mean_difference']:+.2f}s)")
    
    if test_results.get('length'):
        length = test_results['length']
        if length['significant']:
            print(f"✅ Response Length: CoT produces {'longer' if length['mean_difference'] > 0 else 'shorter'} descriptions ({abs(length['mean_difference']):.1f} words, statistically significant)")
        else:
            print(f"Response Length: No significant difference ({length['mean_difference']:+.1f} words)")
    
    if test_results.get('tokens'):
        tokens = test_results['tokens']
        if tokens['significant']:
            print(f"✅ Token Usage: CoT uses {'more' if tokens['mean_difference'] > 0 else 'fewer'} tokens ({abs(tokens['mean_difference']):.0f} tokens, statistically significant)")
        else:
            print(f"Token Usage: No significant difference ({tokens['mean_difference']:+.0f} tokens)")

if __name__ == '__main__':
    main()

