#!/usr/bin/env python3
"""
Statistical significance testing for Approach 3: Specialized Multi-Model System results
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


def get_latencies_by_mode(results, mode):
    """Get latencies for a specific mode"""
    latencies = []
    for r in results:
        if r.get('mode') == mode:
            try:
                lat = float(r.get('total_latency', 0))
                if lat > 0:
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    return latencies


def get_paired_latencies(results_a2, results_a3, filenames):
    """Get paired latencies for common images"""
    a2_latencies = []
    a3_latencies = []
    
    # Create lookup dictionaries
    a2_dict = {r['filename']: r for r in results_a2}
    a3_dict = {r['filename']: r for r in results_a3}
    
    for filename in filenames:
        if filename in a2_dict and filename in a3_dict:
            try:
                a2_lat = float(a2_dict[filename].get('total_latency', 0))
                a3_lat = float(a3_dict[filename].get('total_latency', 0))
                if a2_lat > 0 and a3_lat > 0:
                    a2_latencies.append(a2_lat)
                    a3_latencies.append(a3_lat)
            except (ValueError, TypeError):
                continue
    
    return a2_latencies, a3_latencies


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


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    if len(group1) < 2 or len(group2) < 2:
        return None
    
    try:
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                              (len(group2) - 1) * np.var(group2, ddof=1)) / 
                             (len(group1) + len(group2) - 2))
        if pooled_std == 0:
            return None
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    except Exception as e:
        print(f"Cohen's d error: {e}")
        return None


def main():
    csv_path = Path('results/approach_3_specialized/raw/batch_results.csv')
    output_path = Path('results/approach_3_specialized/analysis/statistical_tests.txt')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_specialized.py first.")
        return
    
    print("=" * 80)
    print("Statistical Tests - Approach 3: Specialized Multi-Model System")
    print("=" * 80)
    print()
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    print()
    
    output_lines = []
    output_lines.append("Statistical Significance Tests - Approach 3 (Specialized Multi-Model System)\n")
    output_lines.append("=" * 80 + "\n\n")
    
    # 1. Mode comparison: OCR vs Depth
    print("1. Testing latency difference between OCR and Depth modes...")
    ocr_latencies = get_latencies_by_mode(results, 'ocr')
    depth_latencies = get_latencies_by_mode(results, 'depth')
    
    if ocr_latencies and depth_latencies:
        t_stat, p_value = independent_t_test(ocr_latencies, depth_latencies)
        if t_stat is not None:
            print(f"   Independent t-test: t = {t_stat:.4f}, p = {p_value:.6f}")
            print(f"   OCR mean: {statistics.mean(ocr_latencies):.2f}s")
            print(f"   Depth mean: {statistics.mean(depth_latencies):.2f}s")
            
            output_lines.append("1. MODE COMPARISON: OCR vs Depth\n")
            output_lines.append("-" * 80 + "\n")
            output_lines.append(f"OCR Mode: n={len(ocr_latencies)}, mean={statistics.mean(ocr_latencies):.2f}s\n")
            output_lines.append(f"Depth Mode: n={len(depth_latencies)}, mean={statistics.mean(depth_latencies):.2f}s\n")
            output_lines.append(f"Independent t-test: t={t_stat:.4f}, p={p_value:.6f}\n")
            output_lines.append(f"Significant: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}\n\n")
        else:
            output_lines.append("1. MODE COMPARISON: OCR vs Depth\n")
            output_lines.append("-" * 80 + "\n")
            output_lines.append("Insufficient data for comparison\n\n")
    else:
        output_lines.append("1. MODE COMPARISON: OCR vs Depth\n")
        output_lines.append("-" * 80 + "\n")
        if not ocr_latencies:
            output_lines.append("No OCR mode results (SSL issue)\n\n")
        else:
            output_lines.append("Insufficient data for comparison\n\n")
    
    # 2. Component latency analysis
    print("\n2. Analyzing component latency contributions...")
    detection_latencies = []
    ocr_latencies_comp = []
    depth_latencies_comp = []
    generation_latencies = []
    
    for r in results:
        try:
            detection = float(r.get('detection_latency', 0))
            generation = float(r.get('generation_latency', 0))
            ocr = float(r.get('ocr_latency', 0)) if r.get('ocr_latency') else None
            depth = float(r.get('depth_latency', 0)) if r.get('depth_latency') else None
            
            if detection > 0:
                detection_latencies.append(detection)
            if generation > 0:
                generation_latencies.append(generation)
            if ocr and ocr > 0:
                ocr_latencies_comp.append(ocr)
            if depth and depth > 0:
                depth_latencies_comp.append(depth)
        except (ValueError, TypeError):
            continue
    
    output_lines.append("2. COMPONENT LATENCY ANALYSIS\n")
    output_lines.append("-" * 80 + "\n")
    output_lines.append(f"Detection (YOLO): n={len(detection_latencies)}, mean={statistics.mean(detection_latencies):.3f}s\n")
    if ocr_latencies_comp:
        output_lines.append(f"OCR: n={len(ocr_latencies_comp)}, mean={statistics.mean(ocr_latencies_comp):.3f}s\n")
    if depth_latencies_comp:
        output_lines.append(f"Depth: n={len(depth_latencies_comp)}, mean={statistics.mean(depth_latencies_comp):.3f}s\n")
    output_lines.append(f"Generation (LLM): n={len(generation_latencies)}, mean={statistics.mean(generation_latencies):.3f}s\n\n")
    
    # 3. Comparison with Approach 2 baseline
    print("\n3. Comparing Approach 3 vs Approach 2 baseline...")
    
    # Load Approach 2 results
    a2_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    if a2_path.exists():
        a2_results = load_results(a2_path)
        a2_filtered = [r for r in a2_results 
                      if r.get('yolo_model') == 'yolov8n' 
                      and r.get('llm_model') == 'gpt-4o-mini']
        
        # 3A vs 2 (text images)
        a2_text = [r for r in a2_filtered if r.get('category') == 'text']
        a3_ocr = [r for r in results if r.get('mode') == 'ocr']
        
        if a2_text and a3_ocr:
            common_text = list(set(r['filename'] for r in a2_text).intersection(set(r['filename'] for r in a3_ocr)))
            if common_text:
                a2_latencies, a3_latencies = get_paired_latencies(a2_text, a3_ocr, common_text)
                
                if len(a2_latencies) == len(a3_latencies) and len(a2_latencies) >= 2:
                    t_stat, p_value = paired_t_test(a2_latencies, a3_latencies)
                    cohens_d_val = cohens_d(a2_latencies, a3_latencies)
                    
                    if t_stat is not None:
                        print(f"   Approach 3A vs 2 (text): t = {t_stat:.4f}, p = {p_value:.6f}")
                        output_lines.append("3. APPROACH 3A vs APPROACH 2 (TEXT IMAGES)\n")
                        output_lines.append("-" * 80 + "\n")
                        output_lines.append(f"Paired samples: {len(a2_latencies)}\n")
                        output_lines.append(f"Approach 2 mean: {statistics.mean(a2_latencies):.2f}s\n")
                        output_lines.append(f"Approach 3A mean: {statistics.mean(a3_latencies):.2f}s\n")
                        output_lines.append(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}\n")
                        if cohens_d_val:
                            output_lines.append(f"Cohen's d: {cohens_d_val:.4f}\n")
                        output_lines.append(f"Significant: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}\n\n")
        
        # 3B vs 2 (navigation images)
        a2_nav = [r for r in a2_filtered if r.get('category') in ['indoor', 'outdoor']]
        a3_depth = [r for r in results if r.get('mode') == 'depth']
        
        if a2_nav and a3_depth:
            common_nav = list(set(r['filename'] for r in a2_nav).intersection(set(r['filename'] for r in a3_depth)))
            if common_nav:
                a2_latencies, a3_latencies = get_paired_latencies(a2_nav, a3_depth, common_nav)
                
                if len(a2_latencies) == len(a3_latencies) and len(a2_latencies) >= 2:
                    t_stat, p_value = paired_t_test(a2_latencies, a3_latencies)
                    cohens_d_val = cohens_d(a2_latencies, a3_latencies)
                    
                    if t_stat is not None:
                        print(f"   Approach 3B vs 2 (navigation): t = {t_stat:.4f}, p = {p_value:.6f}")
                        output_lines.append("4. APPROACH 3B vs APPROACH 2 (NAVIGATION IMAGES)\n")
                        output_lines.append("-" * 80 + "\n")
                        output_lines.append(f"Paired samples: {len(a2_latencies)}\n")
                        output_lines.append(f"Approach 2 mean: {statistics.mean(a2_latencies):.2f}s\n")
                        output_lines.append(f"Approach 3B mean: {statistics.mean(a3_latencies):.2f}s\n")
                        output_lines.append(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}\n")
                        if cohens_d_val:
                            output_lines.append(f"Cohen's d: {cohens_d_val:.4f}\n")
                        output_lines.append(f"Significant: {'YES (p < 0.05)' if p_value < 0.05 else 'NO (p >= 0.05)'}\n\n")
    else:
        output_lines.append("3. APPROACH 3 vs APPROACH 2 COMPARISON\n")
        output_lines.append("-" * 80 + "\n")
        output_lines.append("Approach 2 results not found for comparison\n\n")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"\n✅ Statistical tests report saved to: {output_path}")


if __name__ == "__main__":
    main()

