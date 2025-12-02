#!/usr/bin/env python3
"""
Statistical significance testing for Local Models results (Approach 4)
"""
import csv
import statistics
from pathlib import Path
from scipy import stats
import numpy as np


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def get_latencies_by_model(results, model_name):
    """Get latencies for a specific model"""
    latencies = []
    for r in results:
        if r.get('model') == model_name:
            try:
                lat = float(r.get('total_latency', 0) or r.get('latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def get_latencies_by_category(results, category):
    """Get latencies for a specific category"""
    latencies = []
    for r in results:
        if r.get('category') == category:
            try:
                lat = float(r.get('total_latency', 0) or r.get('latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def main():
    """Run statistical tests"""
    print("=" * 60)
    print("Statistical Significance Tests - Approach 4 (Local Models)")
    print("=" * 60)
    print()
    
    # Load results
    csv_path = Path('results/approach_4_local/raw/batch_results.csv')
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    print()
    
    # Output directory
    output_dir = Path('results/approach_4_local/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'statistical_tests.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Statistical Significance Tests - Approach 4 (Local Models)\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. BLIP-2 latency analysis
        blip2_latencies = get_latencies_by_model(results, 'blip2')
        llava_latencies = get_latencies_by_model(results, 'llava')
        
        if blip2_latencies:
            # BLIP-2 only analysis
            blip2_mean = statistics.mean(blip2_latencies)
            blip2_std = statistics.stdev(blip2_latencies) if len(blip2_latencies) > 1 else 0
            
            f.write("1. BLIP-2 Latency Statistics\n")
            f.write(f"   Mean: {blip2_mean:.3f}s\n")
            f.write(f"   Std Dev: {blip2_std:.3f}s\n")
            f.write(f"   Count: {len(blip2_latencies)}\n\n")
        
        if blip2_latencies and llava_latencies:
            # Match by image (same filename)
            blip2_by_file = {}
            llava_by_file = {}
            
            for r in results:
                filename = r.get('filename')
                model = r.get('model')
                try:
                    lat = float(r.get('total_latency', 0) or r.get('latency', 0))
                    if lat > 0:
                        if model == 'blip2':
                            blip2_by_file[filename] = lat
                        elif model == 'llava':
                            llava_by_file[filename] = lat
                except (ValueError, TypeError):
                    continue
            
            # Get paired latencies (same filename)
            paired_blip2 = []
            paired_llava = []
            for filename in blip2_by_file:
                if filename in llava_by_file:
                    paired_blip2.append(blip2_by_file[filename])
                    paired_llava.append(llava_by_file[filename])
            
            if len(paired_blip2) > 1 and len(paired_llava) > 1:
                t_stat, p_value = stats.ttest_rel(paired_blip2, paired_llava)
                blip2_mean = statistics.mean(paired_blip2)
                llava_mean = statistics.mean(paired_llava)
                mean_diff = llava_mean - blip2_mean
                
                f.write("1. Paired t-test: BLIP-2 vs LLaVA Latency\n")
                f.write(f"   BLIP-2 mean: {blip2_mean:.3f}s\n")
                f.write(f"   LLaVA mean: {llava_mean:.3f}s\n")
                f.write(f"   Mean difference: {mean_diff:.3f}s\n")
                f.write(f"   t-statistic: {t_stat:.3f}\n")
                f.write(f"   p-value: {p_value:.6f}\n")
                if p_value < 0.05:
                    f.write(f"   Result: Significant difference (p < 0.05)\n")
                else:
                    f.write(f"   Result: No significant difference (p >= 0.05)\n")
                f.write(f"   N pairs: {len(paired_blip2)}\n\n")
        
        # 2. One-way ANOVA: Latency by category
        categories = ['gaming', 'indoor', 'outdoor', 'text']
        category_latencies = {}
        
        for cat in categories:
            lats = get_latencies_by_category(results, cat)
            if lats:
                category_latencies[cat] = lats
        
        if len(category_latencies) >= 2:
            # Perform ANOVA
            groups = list(category_latencies.values())
            f_stat, p_value = stats.f_oneway(*groups)
            
            f.write("2. One-way ANOVA: Latency by Category\n")
            f.write(f"   F-statistic: {f_stat:.3f}\n")
            f.write(f"   p-value: {p_value:.6f}\n")
            if p_value < 0.05:
                f.write(f"   Result: Significant difference (p < 0.05)\n")
            else:
                f.write(f"   Result: No significant difference (p >= 0.05)\n")
            f.write(f"   Group means:\n")
            for cat, lats in category_latencies.items():
                f.write(f"     {cat}: {statistics.mean(lats):.3f}s (n={len(lats)})\n")
            f.write("\n")
        
        # 3. Response length analysis
        blip2_lengths = []
        llava_lengths = []
        
        for r in results:
            model = r.get('model')
            desc = r.get('description', '')
            if desc:
                length = len(desc.split())
                if model == 'blip2':
                    blip2_lengths.append(length)
                elif model == 'llava':
                    llava_lengths.append(length)
        
        if blip2_lengths:
            # BLIP-2 only analysis
            blip2_mean_len = statistics.mean(blip2_lengths)
            blip2_std_len = statistics.stdev(blip2_lengths) if len(blip2_lengths) > 1 else 0
            
            f.write("3. BLIP-2 Response Length Statistics\n")
            f.write(f"   Mean: {blip2_mean_len:.1f} words\n")
            f.write(f"   Std Dev: {blip2_std_len:.1f} words\n")
            f.write(f"   Count: {len(blip2_lengths)}\n\n")
        
        if blip2_lengths and llava_lengths:
            # Match by filename for paired test
            blip2_lengths_by_file = {}
            llava_lengths_by_file = {}
            
            for r in results:
                filename = r.get('filename')
                model = r.get('model')
                desc = r.get('description', '')
                if desc:
                    length = len(desc.split())
                    if model == 'blip2':
                        blip2_lengths_by_file[filename] = length
                    elif model == 'llava':
                        llava_lengths_by_file[filename] = length
            
            paired_blip2_len = []
            paired_llava_len = []
            for filename in blip2_lengths_by_file:
                if filename in llava_lengths_by_file:
                    paired_blip2_len.append(blip2_lengths_by_file[filename])
                    paired_llava_len.append(llava_lengths_by_file[filename])
            
            if len(paired_blip2_len) > 1 and len(paired_llava_len) > 1:
                t_stat, p_value = stats.ttest_rel(paired_blip2_len, paired_llava_len)
                blip2_mean_len = statistics.mean(paired_blip2_len)
                llava_mean_len = statistics.mean(paired_llava_len)
                mean_diff_len = llava_mean_len - blip2_mean_len
                
                f.write("3. Paired t-test: BLIP-2 vs LLaVA Response Length\n")
                f.write(f"   BLIP-2 mean: {blip2_mean_len:.1f} words\n")
                f.write(f"   LLaVA mean: {llava_mean_len:.1f} words\n")
                f.write(f"   Mean difference: {mean_diff_len:.1f} words\n")
                f.write(f"   t-statistic: {t_stat:.3f}\n")
                f.write(f"   p-value: {p_value:.6f}\n")
                if p_value < 0.05:
                    f.write(f"   Result: Significant difference (p < 0.05)\n")
                else:
                    f.write(f"   Result: No significant difference (p >= 0.05)\n")
                f.write(f"   N pairs: {len(paired_blip2_len)}\n")
    
    print(f"✅ Statistical tests complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

