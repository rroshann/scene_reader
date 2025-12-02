#!/usr/bin/env python3
"""
Analyze Approach 3.5 improvements: Compare before/after results
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

project_root = Path(__file__).parent.parent.parent

def load_results():
    """Load before and after results"""
    results_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    
    # Load previous batch results (before improvements)
    before_file = results_dir / 'batch_results.csv'
    after_file = results_dir / 'batch_results_with_improvements.csv'
    
    # Fallback to subset test if full batch not ready
    if not after_file.exists():
        after_file = results_dir / 'subset_test_improvements.csv'
        print("⚠️  Using subset test results (full batch may still be running)")
    
    before = pd.read_csv(before_file) if before_file.exists() else None
    after = pd.read_csv(after_file) if after_file.exists() else None
    
    return before, after


def compare_results(before, after):
    """Compare before and after results"""
    print("=" * 70)
    print("Approach 3.5: Improvements Analysis")
    print("=" * 70)
    
    if before is None:
        print("⚠️  No 'before' results found. Using subset test as baseline.")
        return
    
    if after is None:
        print("❌ No 'after' results found!")
        return
    
    # Filter successful tests only
    before_success = before[before['success'] == True]
    after_success = after[after['success'] == True]
    
    print(f"\nBefore (baseline): {len(before_success)} successful tests")
    print(f"After (with improvements): {len(after_success)} successful tests")
    
    # Overall latency comparison
    print("\n" + "=" * 70)
    print("Overall Latency Comparison")
    print("=" * 70)
    
    before_mean = before_success['total_latency'].mean()
    after_mean = after_success['total_latency'].mean()
    improvement = (before_mean - after_mean) / before_mean * 100
    
    print(f"Mean latency:")
    print(f"  Before: {before_mean:.3f}s")
    print(f"  After:  {after_mean:.3f}s")
    print(f"  Improvement: {improvement:.1f}% ({'faster' if improvement > 0 else 'slower'})")
    
    before_median = before_success['total_latency'].median()
    after_median = after_success['total_latency'].median()
    median_improvement = (before_median - after_median) / before_median * 100
    
    print(f"\nMedian latency:")
    print(f"  Before: {before_median:.3f}s")
    print(f"  After:  {after_median:.3f}s")
    print(f"  Improvement: {median_improvement:.1f}% ({'faster' if median_improvement > 0 else 'slower'})")
    
    # Component-wise comparison
    print("\n" + "=" * 70)
    print("Component Latency Comparison")
    print("=" * 70)
    
    # Detection latency
    if 'detection_latency' in before_success.columns and 'detection_latency' in after_success.columns:
        before_det = before_success['detection_latency'].mean()
        after_det = after_success['detection_latency'].mean()
        det_improvement = (before_det - after_det) / before_det * 100 if before_det > 0 else 0
        print(f"Detection:")
        print(f"  Before: {before_det:.3f}s")
        print(f"  After:  {after_det:.3f}s")
        print(f"  Improvement: {det_improvement:.1f}%")
    
    # Depth latency (for depth mode)
    before_depth = before_success[before_success['mode'] == 'depth']
    after_depth = after_success[after_success['mode'] == 'depth']
    
    if len(before_depth) > 0 and len(after_depth) > 0:
        if 'depth_latency' in before_depth.columns and 'depth_latency' in after_depth.columns:
            before_depth_lat = before_depth['depth_latency'].mean()
            after_depth_lat = after_depth['depth_latency'].mean()
            depth_improvement = (before_depth_lat - after_depth_lat) / before_depth_lat * 100 if before_depth_lat > 0 else 0
            
            print(f"\nDepth (parallel execution):")
            print(f"  Before: {before_depth_lat:.3f}s")
            print(f"  After:  {after_depth_lat:.3f}s")
            print(f"  Improvement: {depth_improvement:.1f}%")
            
            # Check parallel execution effectiveness
            before_total_depth = before_depth['total_latency'].mean()
            after_total_depth = after_depth['total_latency'].mean()
            before_seq = before_depth['detection_latency'].mean() + before_depth_lat
            after_parallel = max(after_depth['detection_latency'].mean(), after_depth_lat)
            
            print(f"\nParallel execution analysis:")
            print(f"  Before (sequential): detection + depth = {before_seq:.3f}s")
            print(f"  After (parallel): max(detection, depth) = {after_parallel:.3f}s")
            print(f"  Parallel speedup: {(before_seq - after_parallel) / before_seq * 100:.1f}%")
    
    # Generation latency
    if 'generation_latency' in before_success.columns and 'generation_latency' in after_success.columns:
        before_gen = before_success['generation_latency'].mean()
        after_gen = after_success['generation_latency'].mean()
        gen_improvement = (before_gen - after_gen) / before_gen * 100 if before_gen > 0 else 0
        print(f"\nGeneration:")
        print(f"  Before: {before_gen:.3f}s")
        print(f"  After:  {after_gen:.3f}s")
        print(f"  Improvement: {gen_improvement:.1f}%")
    
    # Statistical significance
    print("\n" + "=" * 70)
    print("Statistical Significance")
    print("=" * 70)
    
    # Paired t-test (if we have matching images)
    if len(before_success) > 0 and len(after_success) > 0:
        # Try to match by filename
        before_dict = {row['filename']: row['total_latency'] for _, row in before_success.iterrows()}
        after_dict = {row['filename']: row['total_latency'] for _, row in after_success.iterrows()}
        
        common_files = set(before_dict.keys()) & set(after_dict.keys())
        if len(common_files) > 1:
            before_vals = [before_dict[f] for f in common_files]
            after_vals = [after_dict[f] for f in common_files]
            
            t_stat, p_value = stats.ttest_rel(before_vals, after_vals)
            print(f"Paired t-test (n={len(common_files)}):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (p < 0.05)")
            
            # Effect size (Cohen's d)
            diff = np.array(after_vals) - np.array(before_vals)
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
            if abs(cohens_d) < 0.2:
                effect_size = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size = "small"
            elif abs(cohens_d) < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            print(f"  Effect size interpretation: {effect_size}")
    
    # Under 2s target
    print("\n" + "=" * 70)
    print("Under 2s Target Achievement")
    print("=" * 70)
    
    before_under_2s = (before_success['total_latency'] < 2.0).sum()
    after_under_2s = (after_success['total_latency'] < 2.0).sum()
    
    before_pct = before_under_2s / len(before_success) * 100
    after_pct = after_under_2s / len(after_success) * 100
    
    print(f"Before: {before_under_2s}/{len(before_success)} ({before_pct:.1f}%)")
    print(f"After:  {after_under_2s}/{len(after_success)} ({after_pct:.1f}%)")
    print(f"Improvement: +{after_pct - before_pct:.1f} percentage points")
    
    # Cache hit rate
    print("\n" + "=" * 70)
    print("Cache Performance")
    print("=" * 70)
    
    if 'cache_hit' in after_success.columns:
        cache_hits = after_success['cache_hit'].sum()
        cache_rate = cache_hits / len(after_success) * 100
        print(f"Cache hits: {cache_hits}/{len(after_success)} ({cache_rate:.1f}%)")
        
        if cache_hits > 0:
            cached = after_success[after_success['cache_hit'] == True]
            non_cached = after_success[after_success['cache_hit'] == False]
            
            if len(cached) > 0 and len(non_cached) > 0:
                cached_latency = cached['total_latency'].mean()
                non_cached_latency = non_cached['total_latency'].mean()
                speedup = non_cached_latency / cached_latency if cached_latency > 0 else 0
                
                print(f"  Cached latency: {cached_latency:.3f}s")
                print(f"  Non-cached latency: {non_cached_latency:.3f}s")
                print(f"  Cache speedup: {speedup:.1f}x")
    
    print("\n" + "=" * 70)


def main():
    before, after = load_results()
    compare_results(before, after)


if __name__ == "__main__":
    main()

