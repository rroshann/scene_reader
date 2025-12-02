#!/usr/bin/env python3
"""
Compare Approach 3.5 vs Approach 3
Detailed comparison of optimized vs baseline specialized system
"""
import pandas as pd
from pathlib import Path
from datetime import datetime


def compare_results():
    """Compare Approach 3.5 vs Approach 3 results"""
    project_root = Path(__file__).parent.parent.parent
    
    # Load results
    results_3_5 = project_root / 'results' / 'approach_3_5_optimized' / 'raw' / 'batch_results.csv'
    results_3 = project_root / 'results' / 'approach_3_specialized' / 'raw' / 'batch_results.csv'
    
    if not results_3_5.exists():
        print(f"Approach 3.5 results not found: {results_3_5}")
        return
    
    df_3_5 = pd.read_csv(results_3_5)
    df_3_5_success = df_3_5[df_3_5['success'] == True].copy()
    
    df_3 = None
    df_3_success = None
    if results_3.exists():
        df_3 = pd.read_csv(results_3)
        df_3_success = df_3[df_3['success'] == True].copy()
    
    # Create output directory
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comparison report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 3.5 vs APPROACH 3 COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall comparison
    report_lines.append("OVERALL COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if len(df_3_5_success) > 0:
        report_lines.append("Approach 3.5 (Optimized):")
        report_lines.append(f"  Total tests: {len(df_3_5)}")
        report_lines.append(f"  Successful: {len(df_3_5_success)} ({100 * len(df_3_5_success) / len(df_3_5):.1f}%)")
        report_lines.append(f"  Mean latency: {df_3_5_success['total_latency'].mean():.2f}s")
        report_lines.append(f"  Median latency: {df_3_5_success['total_latency'].median():.2f}s")
        report_lines.append("")
    
    if df_3_success is not None and len(df_3_success) > 0:
        report_lines.append("Approach 3 (Baseline):")
        report_lines.append(f"  Total tests: {len(df_3)}")
        report_lines.append(f"  Successful: {len(df_3_success)} ({100 * len(df_3_success) / len(df_3):.1f}%)")
        report_lines.append(f"  Mean latency: {df_3_success['total_latency'].mean():.2f}s")
        report_lines.append(f"  Median latency: {df_3_success['total_latency'].median():.2f}s")
        report_lines.append("")
        
        # Improvement calculation
        if len(df_3_5_success) > 0:
            latency_improvement = df_3_success['total_latency'].mean() - df_3_5_success['total_latency'].mean()
            latency_improvement_pct = 100 * latency_improvement / df_3_success['total_latency'].mean()
            report_lines.append("Improvement:")
            report_lines.append(f"  Latency reduction: {latency_improvement:.2f}s ({latency_improvement_pct:.1f}% faster)")
            report_lines.append("")
    
    # Mode-specific comparison
    report_lines.append("MODE-SPECIFIC COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for mode in ['ocr', 'depth']:
        mode_3_5 = df_3_5_success[df_3_5_success['mode'] == mode] if len(df_3_5_success) > 0 else pd.DataFrame()
        
        if len(mode_3_5) > 0:
            report_lines.append(f"{mode.upper()} Mode:")
            report_lines.append(f"  Approach 3.5:")
            report_lines.append(f"    Count: {len(mode_3_5)}")
            report_lines.append(f"    Mean latency: {mode_3_5['total_latency'].mean():.2f}s")
            
            if df_3_success is not None:
                mode_3 = df_3_success[df_3_success['mode'] == mode]
                if len(mode_3) > 0:
                    report_lines.append(f"  Approach 3:")
                    report_lines.append(f"    Count: {len(mode_3)}")
                    report_lines.append(f"    Mean latency: {mode_3['total_latency'].mean():.2f}s")
                    
                    improvement = mode_3['total_latency'].mean() - mode_3_5['total_latency'].mean()
                    improvement_pct = 100 * improvement / mode_3['total_latency'].mean()
                    report_lines.append(f"  Improvement: {improvement:.2f}s ({improvement_pct:.1f}% faster)")
            report_lines.append("")
    
    # Generation latency comparison
    report_lines.append("GENERATION LATENCY COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    if len(df_3_5_success) > 0:
        gen_3_5 = df_3_5_success['generation_latency'].dropna()
        if len(gen_3_5) > 0:
            report_lines.append("Approach 3.5:")
            report_lines.append(f"  Mean: {gen_3_5.mean():.2f}s")
            report_lines.append(f"  Median: {gen_3_5.median():.2f}s")
            report_lines.append("")
    
    if df_3_success is not None:
        gen_3 = df_3_success['generation_latency'].dropna()
        if len(gen_3) > 0:
            report_lines.append("Approach 3:")
            report_lines.append(f"  Mean: {gen_3.mean():.2f}s")
            report_lines.append(f"  Median: {gen_3.median():.2f}s")
            report_lines.append("")
            
            if len(gen_3_5) > 0:
                gen_improvement = gen_3.mean() - gen_3_5.mean()
                gen_improvement_pct = 100 * gen_improvement / gen_3.mean()
                report_lines.append(f"Improvement: {gen_improvement:.2f}s ({gen_improvement_pct:.1f}% faster)")
                report_lines.append("")
    
    # Cache performance
    if 'cache_hit' in df_3_5_success.columns:
        cache_hits = df_3_5_success[df_3_5_success['cache_hit'] == True]
        if len(cache_hits) > 0:
            report_lines.append("CACHE PERFORMANCE (Approach 3.5)")
            report_lines.append("-" * 80)
            report_lines.append(f"Cache hits: {len(cache_hits)} ({100 * len(cache_hits) / len(df_3_5_success):.1f}%)")
            report_lines.append(f"Cache hit mean latency: {cache_hits['total_latency'].mean():.3f}s")
            cache_misses = df_3_5_success[df_3_5_success['cache_hit'] == False]
            if len(cache_misses) > 0:
                report_lines.append(f"Cache miss mean latency: {cache_misses['total_latency'].mean():.3f}s")
                speedup = cache_misses['total_latency'].mean() / cache_hits['total_latency'].mean()
                report_lines.append(f"Cache speedup: {speedup:.1f}x")
            report_lines.append("")
    
    # OCR success rate comparison
    ocr_3_5 = df_3_5[df_3_5['mode'] == 'ocr']
    if len(ocr_3_5) > 0:
        ocr_3_5_success = ocr_3_5[ocr_3_5['success'] == True]
        report_lines.append("OCR MODE SUCCESS RATE")
        report_lines.append("-" * 80)
        report_lines.append(f"Approach 3.5: {len(ocr_3_5_success)}/{len(ocr_3_5)} ({100 * len(ocr_3_5_success) / len(ocr_3_5):.1f}%)")
        
        if df_3 is not None:
            ocr_3 = df_3[df_3['mode'] == 'ocr']
            if len(ocr_3) > 0:
                ocr_3_success = ocr_3[ocr_3['success'] == True]
                report_lines.append(f"Approach 3: {len(ocr_3_success)}/{len(ocr_3)} ({100 * len(ocr_3_success) / len(ocr_3):.1f}%)")
        report_lines.append("")
    
    # Configuration comparison
    if 'configuration' in df_3_5_success.columns:
        report_lines.append("CONFIGURATION COMPARISON")
        report_lines.append("-" * 80)
        for config in df_3_5_success['configuration'].unique():
            config_df = df_3_5_success[df_3_5_success['configuration'] == config]
            report_lines.append(f"{config}:")
            report_lines.append(f"  Mean latency: {config_df['total_latency'].mean():.2f}s")
            report_lines.append(f"  Count: {len(config_df)}")
            report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    if len(df_3_5_success) > 0:
        report_lines.append("Approach 3.5 achieves:")
        report_lines.append(f"  - Mean latency: {df_3_5_success['total_latency'].mean():.2f}s")
        under_2s = len(df_3_5_success[df_3_5_success['total_latency'] < 2.0])
        report_lines.append(f"  - Under 2s: {under_2s}/{len(df_3_5_success)} ({100 * under_2s / len(df_3_5_success):.1f}%)")
        if 'cache_hit' in df_3_5_success.columns:
            cache_hits = len(df_3_5_success[df_3_5_success['cache_hit'] == True])
            report_lines.append(f"  - Cache hits: {cache_hits} ({100 * cache_hits / len(df_3_5_success):.1f}%)")
        report_lines.append("")
    
    # Write report
    with open(output_dir / 'comparison.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("Comparison complete!")
    print(f"Report saved to: {output_dir / 'comparison.txt'}")


if __name__ == '__main__':
    compare_results()

