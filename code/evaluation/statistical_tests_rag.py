#!/usr/bin/env python3
"""
Statistical significance tests for Approach 6 RAG-Enhanced Vision
"""
import csv
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def paired_t_test_base_vs_rag(results):
    """Paired t-test: Base VLM vs RAG-Enhanced latency"""
    # Group by filename and VLM model
    pairs = defaultdict(dict)
    
    for r in results:
        filename = r.get('filename')
        vlm = r.get('vlm_model')
        use_rag = r.get('use_rag') == 'True' or r.get('use_rag') is True
        
        key = (filename, vlm)
        if key not in pairs:
            pairs[key] = {}
        
        try:
            latency = float(r.get('total_latency', 0))
            if use_rag:
                pairs[key]['rag'] = latency
            else:
                pairs[key]['base'] = latency
        except (ValueError, TypeError):
            continue
    
    # Extract paired data
    base_latencies = []
    rag_latencies = []
    
    for key, values in pairs.items():
        if 'base' in values and 'rag' in values:
            base_latencies.append(values['base'])
            rag_latencies.append(values['rag'])
    
    if len(base_latencies) < 2:
        return None
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(base_latencies, rag_latencies)
    
    return {
        'test': 'Paired t-test: Base vs RAG-Enhanced Latency',
        'base_mean': np.mean(base_latencies),
        'rag_mean': np.mean(rag_latencies),
        'mean_difference': np.mean(rag_latencies) - np.mean(base_latencies),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_pairs': len(base_latencies),
        'significant': p_value < 0.05
    }


def anova_by_vlm(results):
    """One-way ANOVA: Latency differences across VLM models"""
    vlm_groups = defaultdict(list)
    
    for r in results:
        vlm = r.get('vlm_model')
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                vlm_groups[vlm].append(latency)
        except (ValueError, TypeError):
            continue
    
    if len(vlm_groups) < 2:
        return None
    
    groups = [vlm_groups[vlm] for vlm in sorted(vlm_groups.keys())]
    labels = sorted(vlm_groups.keys())
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'test': 'One-way ANOVA: Latency by VLM Model',
        'f_statistic': f_stat,
        'p_value': p_value,
        'groups': {label: {'mean': np.mean(groups[i]), 'n': len(groups[i])} 
                   for i, label in enumerate(labels)},
        'significant': p_value < 0.05
    }


def paired_t_test_response_length(results):
    """Paired t-test: Base vs Enhanced response length"""
    pairs = defaultdict(dict)
    
    for r in results:
        filename = r.get('filename')
        vlm = r.get('vlm_model')
        use_rag = r.get('use_rag') == 'True' or r.get('use_rag') is True
        
        key = (filename, vlm)
        if key not in pairs:
            pairs[key] = {}
        
        try:
            if use_rag and r.get('enhanced_description'):
                pairs[key]['enhanced'] = len(r['enhanced_description'].split())
            elif not use_rag and r.get('base_description'):
                pairs[key]['base'] = len(r['base_description'].split())
        except (ValueError, TypeError):
            continue
    
    base_lengths = []
    enhanced_lengths = []
    
    for key, values in pairs.items():
        if 'base' in values and 'enhanced' in values:
            base_lengths.append(values['base'])
            enhanced_lengths.append(values['enhanced'])
    
    if len(base_lengths) < 2:
        return None
    
    t_stat, p_value = stats.ttest_rel(base_lengths, enhanced_lengths)
    
    return {
        'test': 'Paired t-test: Base vs Enhanced Response Length',
        'base_mean': np.mean(base_lengths),
        'enhanced_mean': np.mean(enhanced_lengths),
        'mean_difference': np.mean(enhanced_lengths) - np.mean(base_lengths),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_pairs': len(base_lengths),
        'significant': p_value < 0.05
    }


def main():
    """Main statistical tests function"""
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    print(f"Loaded {len(results)} successful results")
    
    output_dir = Path('results/approach_6_rag/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'statistical_tests.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Statistical Significance Tests - Approach 6 (RAG-Enhanced Vision)\n")
        f.write("=" * 60 + "\n\n")
        
        # Test 1: Base vs RAG latency
        test1 = paired_t_test_base_vs_rag(results)
        if test1:
            f.write(f"1. {test1['test']}\n")
            f.write(f"   Base mean: {test1['base_mean']:.3f}s\n")
            f.write(f"   RAG mean: {test1['rag_mean']:.3f}s\n")
            f.write(f"   Mean difference: {test1['mean_difference']:.3f}s\n")
            f.write(f"   t-statistic: {test1['t_statistic']:.3f}\n")
            f.write(f"   p-value: {test1['p_value']:.6f}\n")
            f.write(f"   Result: {'Significant difference' if test1['significant'] else 'No significant difference'} (p < 0.05)\n")
            f.write(f"   N pairs: {test1['n_pairs']}\n\n")
        
        # Test 2: ANOVA by VLM
        test2 = anova_by_vlm(results)
        if test2:
            f.write(f"2. {test2['test']}\n")
            f.write(f"   F-statistic: {test2['f_statistic']:.3f}\n")
            f.write(f"   p-value: {test2['p_value']:.6f}\n")
            f.write(f"   Result: {'Significant difference' if test2['significant'] else 'No significant difference'} (p < 0.05)\n")
            f.write(f"   Group means:\n")
            for label, stats in test2['groups'].items():
                f.write(f"     {label}: {stats['mean']:.3f}s (n={stats['n']})\n")
            f.write("\n")
        
        # Test 3: Response length
        test3 = paired_t_test_response_length(results)
        if test3:
            f.write(f"3. {test3['test']}\n")
            f.write(f"   Base mean: {test3['base_mean']:.1f} words\n")
            f.write(f"   Enhanced mean: {test3['enhanced_mean']:.1f} words\n")
            f.write(f"   Mean difference: {test3['mean_difference']:.1f} words\n")
            f.write(f"   t-statistic: {test3['t_statistic']:.3f}\n")
            f.write(f"   p-value: {test3['p_value']:.6f}\n")
            f.write(f"   Result: {'Significant difference' if test3['significant'] else 'No significant difference'} (p < 0.05)\n")
            f.write(f"   N pairs: {test3['n_pairs']}\n\n")
    
    print(f"✅ Statistical tests complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

