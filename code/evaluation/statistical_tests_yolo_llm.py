#!/usr/bin/env python3
"""
Statistical significance testing for YOLO+LLM results (Approach 2)
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


def get_latencies_by_config(results, config_name):
    """Get latencies for a specific configuration"""
    latencies = []
    for r in results:
        if r.get('configuration') == config_name:
            try:
                lat = float(r.get('total_latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def get_latencies_by_yolo(results, yolo_size):
    """Get detection latencies for a specific YOLO variant"""
    latencies = []
    for r in results:
        if f'yolov8{yolo_size}' in r.get('yolo_model', ''):
            try:
                lat = float(r.get('detection_latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def get_latencies_by_llm(results, llm_model):
    """Get generation latencies for a specific LLM model"""
    latencies = []
    for r in results:
        if llm_model.lower() in r.get('llm_model', '').lower():
            try:
                lat = float(r.get('generation_latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def one_way_anova(groups):
    """Perform one-way ANOVA"""
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return None, None
    
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        return f_stat, p_value
    except Exception as e:
        print(f"ANOVA error: {e}")
        return None, None


def paired_t_test(group1, group2):
    """Perform paired t-test"""
    if len(group1) != len(group2):
        return None, None
    
    if len(group1) < 2:
        return None, None
    
    try:
        t_stat, p_value = stats.ttest_rel(group1, group2)
        return t_stat, p_value
    except Exception as e:
        print(f"T-test error: {e}")
        return None, None


def independent_t_test(group1, group2):
    """Perform independent samples t-test"""
    if len(group1) < 2 or len(group2) < 2:
        return None, None
    
    try:
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return t_stat, p_value
    except Exception as e:
        print(f"T-test error: {e}")
        return None, None


def main():
    csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    output_path = Path('results/approach_2_yolo_llm/analysis/statistical_tests.txt')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_yolo_llm.py first.")
        return
    
    print("=" * 60)
    print("Statistical Tests - YOLO+LLM Results")
    print("=" * 60)
    print()
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    print()
    
    output_lines = []
    output_lines.append("Statistical Significance Tests - Approach 2 (YOLO+LLM)\n")
    output_lines.append("=" * 60 + "\n\n")
    
    # 1. ANOVA: Total latency across configurations
    print("1. Testing latency differences across configurations...")
    configurations = ['YOLOv8N+GPT-4o-mini', 'YOLOv8N+Claude Haiku',
                     'YOLOv8M+GPT-4o-mini', 'YOLOv8M+Claude Haiku',
                     'YOLOv8X+GPT-4o-mini', 'YOLOv8X+Claude Haiku']
    
    config_groups = []
    config_labels = []
    for config in configurations:
        latencies = get_latencies_by_config(results, config)
        if latencies:
            config_groups.append(latencies)
            config_labels.append(config)
    
    if len(config_groups) >= 2:
        f_stat, p_value = one_way_anova(config_groups)
        if f_stat is not None:
            print(f"   One-way ANOVA: F = {f_stat:.2f}, p = {p_value:.6f}")
            significance = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
            print(f"   Result: {significance} (p < 0.05)")
            
            output_lines.append("1. Total Latency Across Configurations (One-Way ANOVA)\n")
            output_lines.append(f"   F-statistic: {f_stat:.2f}\n")
            output_lines.append(f"   p-value: {p_value:.6f}\n")
            output_lines.append(f"   Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (p < 0.05)\n\n")
    
    # 2. ANOVA: Detection latency across YOLO variants
    print("\n2. Testing detection latency differences across YOLO variants...")
    yolo_groups = []
    yolo_labels = []
    for yolo_size in ['n', 'm', 'x']:
        latencies = get_latencies_by_yolo(results, yolo_size)
        if latencies:
            yolo_groups.append(latencies)
            yolo_labels.append(f'YOLOv8{yolo_size.upper()}')
    
    if len(yolo_groups) >= 2:
        f_stat, p_value = one_way_anova(yolo_groups)
        if f_stat is not None:
            print(f"   One-way ANOVA: F = {f_stat:.2f}, p = {p_value:.6f}")
            significance = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
            print(f"   Result: {significance} (p < 0.05)")
            
            output_lines.append("2. Detection Latency Across YOLO Variants (One-Way ANOVA)\n")
            output_lines.append(f"   F-statistic: {f_stat:.2f}\n")
            output_lines.append(f"   p-value: {p_value:.6f}\n")
            output_lines.append(f"   Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (p < 0.05)\n\n")
    
    # 3. T-test: Generation latency between LLM models
    print("\n3. Testing generation latency differences between LLM models...")
    gpt_latencies = get_latencies_by_llm(results, 'gpt-4o-mini')
    claude_latencies = get_latencies_by_llm(results, 'claude-haiku')
    
    if gpt_latencies and claude_latencies:
        # Match by image (paired test)
        # Group by filename to create pairs
        gpt_by_file = {}
        claude_by_file = {}
        
        for r in results:
            filename = r.get('filename', '')
            llm_model = r.get('llm_model', '').lower()
            try:
                lat = float(r.get('generation_latency', 0))
                if lat > 0:
                    if 'gpt' in llm_model:
                        if filename not in gpt_by_file:
                            gpt_by_file[filename] = []
                        gpt_by_file[filename].append(lat)
                    elif 'claude' in llm_model:
                        if filename not in claude_by_file:
                            claude_by_file[filename] = []
                        claude_by_file[filename].append(lat)
            except (ValueError, TypeError):
                continue
        
        # Create paired samples (average if multiple configs per image)
        paired_gpt = []
        paired_claude = []
        common_files = set(gpt_by_file.keys()) & set(claude_by_file.keys())
        
        for filename in common_files:
            paired_gpt.append(statistics.mean(gpt_by_file[filename]))
            paired_claude.append(statistics.mean(claude_by_file[filename]))
        
        if len(paired_gpt) >= 2 and len(paired_claude) >= 2:
            t_stat, p_value = paired_t_test(paired_gpt, paired_claude)
            if t_stat is not None:
                mean_diff = statistics.mean(paired_gpt) - statistics.mean(paired_claude)
                print(f"   Paired t-test: t = {t_stat:.2f}, p = {p_value:.6f}")
                print(f"   Mean difference: {mean_diff:.3f}s (GPT-4o-mini - Claude Haiku)")
                significance = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
                print(f"   Result: {significance} (p < 0.05)")
                
                output_lines.append("3. Generation Latency Between LLM Models (Paired T-Test)\n")
                output_lines.append(f"   GPT-4o-mini mean: {statistics.mean(paired_gpt):.3f}s\n")
                output_lines.append(f"   Claude Haiku mean: {statistics.mean(paired_claude):.3f}s\n")
                output_lines.append(f"   Mean difference: {mean_diff:.3f}s\n")
                output_lines.append(f"   t-statistic: {t_stat:.2f}\n")
                output_lines.append(f"   p-value: {p_value:.6f}\n")
                output_lines.append(f"   Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (p < 0.05)\n\n")
    
    # 4. Pairwise comparisons: YOLO variants
    print("\n4. Pairwise comparisons: YOLO variants...")
    if len(yolo_groups) >= 2:
        for i in range(len(yolo_groups)):
            for j in range(i + 1, len(yolo_groups)):
                group1 = yolo_groups[i]
                group2 = yolo_groups[j]
                label1 = yolo_labels[i]
                label2 = yolo_labels[j]
                
                # Independent samples t-test
                t_stat, p_value = independent_t_test(group1, group2)
                if t_stat is not None:
                    mean_diff = statistics.mean(group1) - statistics.mean(group2)
                    print(f"   {label1} vs {label2}: t = {t_stat:.2f}, p = {p_value:.6f}, diff = {mean_diff:.3f}s")
                    significance = "✅" if p_value < 0.05 else "❌"
                    print(f"   {significance} {'Significant' if p_value < 0.05 else 'Not significant'}")
                    
                    output_lines.append(f"4.{i+1}.{j} {label1} vs {label2} (Independent T-Test)\n")
                    output_lines.append(f"   Mean difference: {mean_diff:.3f}s\n")
                    output_lines.append(f"   t-statistic: {t_stat:.2f}\n")
                    output_lines.append(f"   p-value: {p_value:.6f}\n")
                    output_lines.append(f"   Result: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (p < 0.05)\n\n")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"\n✅ Statistical tests complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()

