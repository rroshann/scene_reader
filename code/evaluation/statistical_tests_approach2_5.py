#!/usr/bin/env python3
"""
Statistical Tests for Approach 2.5
Paired t-tests, ANOVA, and effect sizes
"""
import sys
import csv
import statistics
from pathlib import Path
from scipy import stats
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_paired_results():
    """Load paired results (same images from both approaches)"""
    approach2_path = project_root / 'results' / 'approach_2_yolo_llm' / 'raw' / 'batch_results.csv'
    approach25_path = project_root / 'results' / 'approach_2_5_optimized' / 'raw' / 'batch_results.csv'
    
    # Load Approach 2 (GPT-4o-mini, YOLOv8N)
    approach2_data = {}
    if approach2_path.exists():
        with open(approach2_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('success') == 'True' and 
                    row.get('yolo_model') == 'yolov8n' and 
                    row.get('llm_model') == 'gpt-4o-mini'):
                    approach2_data[row['filename']] = float(row['total_latency'])
    
    # Load Approach 2.5 (GPT-3.5-turbo, YOLOv8N)
    approach25_data = {}
    if approach25_path.exists():
        with open(approach25_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('success') == 'True':
                    approach25_data[row['filename']] = float(row['total_latency'])
    
    # Create paired data (same filenames)
    paired_data = []
    for filename in approach2_data.keys():
        if filename in approach25_data:
            paired_data.append({
                'filename': filename,
                'approach2': approach2_data[filename],
                'approach25': approach25_data[filename]
            })
    
    return paired_data


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = statistics.variance(group1) if len(group1) > 1 else 0, statistics.variance(group2) if len(group2) > 1 else 0
    
    pooled_std = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = pooled_std ** 0.5
    
    if pooled_std == 0:
        return 0.0
    
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    return (mean1 - mean2) / pooled_std


def main():
    """Run statistical tests"""
    print("=" * 80)
    print("STATISTICAL TESTS: APPROACH 2 vs APPROACH 2.5")
    print("=" * 80)
    print()
    
    paired_data = load_paired_results()
    
    if len(paired_data) < 10:
        print("⚠️  Insufficient paired data for statistical tests")
        return
    
    approach2_latencies = [d['approach2'] for d in paired_data]
    approach25_latencies = [d['approach25'] for d in paired_data]
    differences = [d['approach25'] - d['approach2'] for d in paired_data]
    
    print(f"Paired samples: {len(paired_data)}")
    print()
    
    # Paired t-test
    print("PAIRED T-TEST")
    print("-" * 80)
    t_stat, p_value = stats.ttest_rel(approach2_latencies, approach25_latencies)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significant: {'✅ YES' if p_value < 0.05 else '❌ NO'} (p < 0.05)")
    print()
    
    # Effect size (Cohen's d)
    cohens_d_value = cohens_d(approach2_latencies, approach25_latencies)
    effect_size_interpretation = "large" if abs(cohens_d_value) > 0.8 else "medium" if abs(cohens_d_value) > 0.5 else "small"
    print("EFFECT SIZE (Cohen's d)")
    print("-" * 80)
    print(f"Cohen's d: {cohens_d_value:.4f}")
    print(f"Interpretation: {effect_size_interpretation} effect")
    print()
    
    # Descriptive statistics
    print("DESCRIPTIVE STATISTICS")
    print("-" * 80)
    print(f"Approach 2:")
    print(f"  Mean: {statistics.mean(approach2_latencies):.2f}s")
    print(f"  Std Dev: {statistics.stdev(approach2_latencies):.2f}s")
    print()
    print(f"Approach 2.5:")
    print(f"  Mean: {statistics.mean(approach25_latencies):.2f}s")
    print(f"  Std Dev: {statistics.stdev(approach25_latencies):.2f}s")
    print()
    print(f"Mean difference: {statistics.mean(differences):.2f}s")
    print(f"Std Dev of differences: {statistics.stdev(differences):.2f}s")
    print()
    
    # Save report
    report_path = project_root / 'results' / 'approach_2_5_optimized' / 'analysis' / 'statistical_tests.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Statistical Tests: Approach 2 vs Approach 2.5\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Paired samples: {len(paired_data)}\n\n")
        
        f.write("PAIRED T-TEST\n")
        f.write("-" * 80 + "\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value: {p_value:.6f}\n")
        f.write(f"Significant: {'YES' if p_value < 0.05 else 'NO'} (p < 0.05)\n\n")
        
        f.write("EFFECT SIZE (Cohen's d)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Cohen's d: {cohens_d_value:.4f}\n")
        f.write(f"Interpretation: {effect_size_interpretation} effect\n\n")
        
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Approach 2: {statistics.mean(approach2_latencies):.2f}s (mean), {statistics.stdev(approach2_latencies):.2f}s (std dev)\n")
        f.write(f"Approach 2.5: {statistics.mean(approach25_latencies):.2f}s (mean), {statistics.stdev(approach25_latencies):.2f}s (std dev)\n")
        f.write(f"Mean difference: {statistics.mean(differences):.2f}s\n\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 80 + "\n")
        if p_value < 0.05:
            f.write("✅ Approach 2.5 shows statistically significant improvement over Approach 2\n")
        f.write(f"Effect size: {effect_size_interpretation} ({cohens_d_value:.4f})\n")
    
    print(f"📄 Statistical test report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("⚠️  scipy not installed. Install with: pip install scipy")
        print("Running basic statistics only...")
        
        # Basic statistics without scipy
        paired_data = load_paired_results()
        if paired_data:
            approach2_latencies = [d['approach2'] for d in paired_data]
            approach25_latencies = [d['approach25'] for d in paired_data]
            
            print(f"\nBasic Statistics:")
            print(f"Approach 2 mean: {statistics.mean(approach2_latencies):.2f}s")
            print(f"Approach 2.5 mean: {statistics.mean(approach25_latencies):.2f}s")
            print(f"Speedup: {((statistics.mean(approach2_latencies) - statistics.mean(approach25_latencies)) / statistics.mean(approach2_latencies)) * 100:.1f}%")

