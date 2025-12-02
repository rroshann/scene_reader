#!/usr/bin/env python3
"""
Analyze Approach 3.5 results
Comprehensive analysis of optimized specialized multi-model system
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def analyze_results():
    """Analyze Approach 3.5 batch test results"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_5_optimized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Filter successful results
    df_success = df[df['success'] == True].copy()
    
    # Create output directory
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 3.5: OPTIMIZED SPECIALIZED MULTI-MODEL SYSTEM - ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total tests: {len(df)}")
    report_lines.append(f"Successful: {len(df_success)} ({100 * len(df_success) / len(df):.1f}%)")
    report_lines.append(f"Failed: {len(df) - len(df_success)} ({100 * (len(df) - len(df_success)) / len(df):.1f}%)")
    report_lines.append("")
    
    if len(df_success) == 0:
        report_lines.append("No successful results to analyze.")
        with open(output_dir / 'comprehensive_analysis.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        return
    
    # Latency statistics
    report_lines.append("OVERALL LATENCY STATISTICS")
    report_lines.append("-" * 80)
    latencies = df_success['total_latency'].dropna()
    report_lines.append(f"Count: {len(latencies):.0f}")
    report_lines.append(f"Mean: {latencies.mean():.2f}s")
    report_lines.append(f"Median: {latencies.median():.2f}s")
    report_lines.append(f"P75: {latencies.quantile(0.75):.2f}s")
    report_lines.append(f"P90: {latencies.quantile(0.90):.2f}s")
    report_lines.append(f"P95: {latencies.quantile(0.95):.2f}s")
    report_lines.append(f"Min: {latencies.min():.2f}s")
    report_lines.append(f"Max: {latencies.max():.2f}s")
    report_lines.append(f"Stdev: {latencies.std():.2f}s")
    report_lines.append("")
    
    # Component latency breakdown
    report_lines.append("COMPONENT LATENCY BREAKDOWN")
    report_lines.append("-" * 80)
    
    components = [
        ('Detection', 'detection_latency'),
        ('OCR', 'ocr_latency'),
        ('Depth', 'depth_latency'),
        ('Generation', 'generation_latency')
    ]
    
    for comp_name, col_name in components:
        comp_latencies = df_success[col_name].dropna()
        if len(comp_latencies) > 0:
            report_lines.append(f"{comp_name}:")
            report_lines.append(f"  Mean: {comp_latencies.mean():.3f}s")
            report_lines.append(f"  Median: {comp_latencies.median():.3f}s")
            report_lines.append(f"  Min: {comp_latencies.min():.3f}s")
            report_lines.append(f"  Max: {comp_latencies.max():.3f}s")
            report_lines.append(f"  Stdev: {comp_latencies.std():.3f}s")
            report_lines.append(f"  Count: {len(comp_latencies):.0f}")
            report_lines.append("")
    
    # Cache performance
    cache_hits = df_success[df_success['cache_hit'] == True]
    if len(cache_hits) > 0:
        report_lines.append("CACHE PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"Cache hits: {len(cache_hits)} ({100 * len(cache_hits) / len(df_success):.1f}%)")
        report_lines.append(f"Cache hit mean latency: {cache_hits['total_latency'].mean():.3f}s")
        report_lines.append(f"Cache miss mean latency: {df_success[df_success['cache_hit'] == False]['total_latency'].mean():.3f}s")
        if len(cache_hits) > 0:
            speedup = df_success[df_success['cache_hit'] == False]['total_latency'].mean() / cache_hits['total_latency'].mean()
            report_lines.append(f"Cache speedup: {speedup:.1f}x")
        report_lines.append("")
    
    # Complexity distribution
    if 'complexity' in df_success.columns:
        complexity_counts = df_success['complexity'].value_counts()
        report_lines.append("COMPLEXITY DISTRIBUTION")
        report_lines.append("-" * 80)
        for complexity, count in complexity_counts.items():
            report_lines.append(f"{complexity}: {count} ({100 * count / len(df_success):.1f}%)")
        report_lines.append("")
    
    # Mode-specific analysis
    report_lines.append("MODE-SPECIFIC ANALYSIS")
    report_lines.append("-" * 80)
    
    for mode in ['ocr', 'depth']:
        mode_df = df_success[df_success['mode'] == mode]
        if len(mode_df) > 0:
            report_lines.append(f"{mode.upper()} Mode:")
            report_lines.append(f"  Total tests: {len(df[df['mode'] == mode])}")
            report_lines.append(f"  Successful: {len(mode_df)} ({100 * len(mode_df) / len(df[df['mode'] == mode]):.1f}%)")
            report_lines.append(f"  Mean latency: {mode_df['total_latency'].mean():.2f}s")
            report_lines.append("")
    
    # Configuration comparison
    if 'configuration' in df_success.columns:
        report_lines.append("CONFIGURATION COMPARISON")
        report_lines.append("-" * 80)
        config_stats = df_success.groupby('configuration').agg({
            'total_latency': ['mean', 'median', 'std', 'count'],
            'generation_latency': 'mean',
            'cache_hit': lambda x: (x == True).sum()
        }).round(3)
        
        for config in df_success['configuration'].unique():
            config_df = df_success[df_success['configuration'] == config]
            report_lines.append(f"{config}:")
            report_lines.append(f"  Mean latency: {config_df['total_latency'].mean():.2f}s")
            report_lines.append(f"  Mean generation: {config_df['generation_latency'].mean():.2f}s")
            report_lines.append(f"  Cache hits: {len(config_df[config_df['cache_hit'] == True])}")
            report_lines.append("")
    
    # Response length statistics
    if 'word_count' in df_success.columns:
        word_counts = df_success['word_count'].dropna()
        if len(word_counts) > 0:
            report_lines.append("RESPONSE LENGTH STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Word Count:")
            report_lines.append(f"  Mean: {word_counts.mean():.1f}")
            report_lines.append(f"  Median: {word_counts.median():.1f}")
            report_lines.append(f"  Min: {word_counts.min():.0f}")
            report_lines.append(f"  Max: {word_counts.max():.0f}")
            report_lines.append(f"  Stdev: {word_counts.std():.1f}")
            report_lines.append("")
    
    # OCR engine statistics
    if 'ocr_engine' in df_success.columns:
        ocr_df = df_success[df_success['mode'] == 'ocr']
        if len(ocr_df) > 0:
            engine_counts = ocr_df['ocr_engine'].value_counts()
            report_lines.append("OCR ENGINE STATISTICS")
            report_lines.append("-" * 80)
            for engine, count in engine_counts.items():
                report_lines.append(f"{engine}: {count} ({100 * count / len(ocr_df):.1f}%)")
            report_lines.append("")
    
    # Target analysis (<2 seconds)
    under_2s = df_success[df_success['total_latency'] < 2.0]
    report_lines.append("TARGET ANALYSIS (<2 SECONDS)")
    report_lines.append("-" * 80)
    report_lines.append(f"Under 2s: {len(under_2s)}/{len(df_success)} ({100 * len(under_2s) / len(df_success):.1f}%)")
    report_lines.append("")
    
    # Write report
    with open(output_dir / 'comprehensive_analysis.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("Analysis complete!")
    print(f"Report saved to: {output_dir / 'comprehensive_analysis.txt'}")


if __name__ == '__main__':
    analyze_results()

