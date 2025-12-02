#!/usr/bin/env python3
"""
Create final visualizations for Approach 3.5 full batch test results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

project_root = Path(__file__).parent.parent.parent

def load_results():
    """Load full batch test results"""
    results_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    
    # Load results with improvements
    after_file = results_dir / 'batch_results_with_improvements.csv'
    
    if not after_file.exists():
        print(f"❌ Results file not found: {after_file}")
        return None
    
    df = pd.read_csv(after_file)
    return df


def create_latency_distribution_plot(df, output_dir):
    """Create latency distribution plot"""
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("⚠️  No successful tests for latency distribution")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Approach 3.5: Latency Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Overall latency histogram
    ax1 = axes[0, 0]
    ax1.hist(successful['total_latency'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(successful['total_latency'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {successful["total_latency"].mean():.2f}s')
    ax1.axvline(successful['total_latency'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {successful["total_latency"].median():.2f}s')
    ax1.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='2s Target')
    ax1.set_xlabel('Total Latency (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Latency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Component latency breakdown
    ax2 = axes[0, 1]
    components = []
    latencies = []
    
    if 'detection_latency' in successful.columns:
        components.append('Detection')
        latencies.append(successful['detection_latency'].mean())
    
    ocr_lat = successful[successful['mode'] == 'ocr']['ocr_latency'].mean() if 'ocr_latency' in successful.columns else None
    if not pd.isna(ocr_lat) and ocr_lat is not None:
        components.append('OCR')
        latencies.append(ocr_lat)
    
    depth_lat = successful[successful['mode'] == 'depth']['depth_latency'].mean() if 'depth_latency' in successful.columns else None
    if not pd.isna(depth_lat) and depth_lat is not None:
        components.append('Depth')
        latencies.append(depth_lat)
    
    if 'generation_latency' in successful.columns:
        components.append('Generation')
        latencies.append(successful['generation_latency'].mean())
    
    if components:
        bars = ax2.bar(components, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(components)])
        ax2.set_ylabel('Mean Latency (seconds)')
        ax2.set_title('Component Latency Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, lat in zip(bars, latencies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{lat:.3f}s',
                    ha='center', va='bottom', fontsize=9)
    
    # Latency by mode
    ax3 = axes[1, 0]
    if 'mode' in successful.columns:
        modes = successful['mode'].unique()
        mode_data = [successful[successful['mode'] == mode]['total_latency'].values for mode in modes]
        bp = ax3.boxplot(mode_data, labels=modes, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_ylabel('Total Latency (seconds)')
        ax3.set_title('Latency by Mode (OCR vs Depth)')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Under 2s target achievement
    ax4 = axes[1, 1]
    under_2s = (successful['total_latency'] < 2.0).sum()
    over_2s = len(successful) - under_2s
    pct_under = under_2s / len(successful) * 100
    
    colors = ['#2ca02c', '#d62728']
    ax4.pie([under_2s, over_2s], labels=[f'<2s ({under_2s})', f'≥2s ({over_2s})'], 
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title(f'Under 2s Target Achievement\n({pct_under:.1f}% under 2s)')
    
    plt.tight_layout()
    output_path = output_dir / 'latency_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_comparison_plot(before_df, after_df, output_dir):
    """Create before/after comparison plot"""
    if before_df is None or after_df is None:
        print("⚠️  Skipping comparison plot (missing data)")
        return
    
    before_success = before_df[before_df['success'] == True]
    after_success = after_df[after_df['success'] == True]
    
    if len(before_success) == 0 or len(after_success) == 0:
        print("⚠️  Not enough data for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Approach 3.5: Before vs After Improvements', fontsize=16, fontweight='bold')
    
    # Overall latency comparison
    ax1 = axes[0, 0]
    data = [before_success['total_latency'].values, after_success['total_latency'].values]
    bp = ax1.boxplot(data, labels=['Before', 'After'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Total Latency (seconds)')
    ax1.set_title('Overall Latency Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    ax1.axhline(before_success['total_latency'].mean(), color='red', linestyle='--', alpha=0.5)
    ax1.axhline(after_success['total_latency'].mean(), color='green', linestyle='--', alpha=0.5)
    
    # Component comparison
    ax2 = axes[0, 1]
    components = ['Detection', 'Generation']
    before_vals = [
        before_success['detection_latency'].mean() if 'detection_latency' in before_success.columns else 0,
        before_success['generation_latency'].mean() if 'generation_latency' in before_success.columns else 0
    ]
    after_vals = [
        after_success['detection_latency'].mean() if 'detection_latency' in after_success.columns else 0,
        after_success['generation_latency'].mean() if 'generation_latency' in after_success.columns else 0
    ]
    
    x = np.arange(len(components))
    width = 0.35
    ax2.bar(x - width/2, before_vals, width, label='Before', color='lightcoral')
    ax2.bar(x + width/2, after_vals, width, label='After', color='lightgreen')
    ax2.set_ylabel('Mean Latency (seconds)')
    ax2.set_title('Component Latency Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Under 2s comparison
    ax3 = axes[1, 0]
    before_under = (before_success['total_latency'] < 2.0).sum() / len(before_success) * 100
    after_under = (after_success['total_latency'] < 2.0).sum() / len(after_success) * 100
    
    categories = ['Before', 'After']
    percentages = [before_under, after_under]
    bars = ax3.bar(categories, percentages, color=['lightcoral', 'lightgreen'])
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Under 2s Target Achievement')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Improvement summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    before_mean = before_success['total_latency'].mean()
    after_mean = after_success['total_latency'].mean()
    improvement = (before_mean - after_mean) / before_mean * 100
    
    summary_text = f"""
    Performance Improvements Summary
    
    Mean Latency:
      Before: {before_mean:.3f}s
      After:  {after_mean:.3f}s
      Improvement: {improvement:.1f}%
    
    Under 2s Target:
      Before: {before_under:.1f}%
      After:  {after_under:.1f}%
      Improvement: +{after_under - before_under:.1f}pp
    
    Total Tests:
      Before: {len(before_success)}
      After:  {len(after_success)}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'before_after_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def create_cache_effectiveness_plot(cache_df, output_dir):
    """Create cache effectiveness visualization"""
    if cache_df is None or not cache_df.exists():
        print("⚠️  Skipping cache plot (no cache test data)")
        return
    
    df = pd.read_csv(cache_df)
    successful = df[df['success'] == True]
    
    if len(successful) == 0:
        print("⚠️  No successful cache tests")
        return
    
    first_run = successful[successful['run'] == 1]
    second_run = successful[successful['run'] == 2]
    
    if len(first_run) == 0 or len(second_run) == 0:
        print("⚠️  Not enough cache test data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Approach 3.5: Cache Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Latency comparison
    ax1 = axes[0]
    first_lat = first_run['total_latency'].mean()
    second_lat = second_run['total_latency'].mean()
    cached_lat = second_run[second_run['cache_hit'] == True]['total_latency'].mean() if (second_run['cache_hit'] == True).any() else None
    
    categories = ['First Run\n(No Cache)', 'Second Run\n(With Cache)']
    latencies = [first_lat, second_lat]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax1.bar(categories, latencies, color=colors)
    ax1.set_ylabel('Mean Latency (seconds)')
    ax1.set_title('Cache Latency Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{lat:.3f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if cached_lat is not None and not pd.isna(cached_lat):
        ax1.axhline(cached_lat, color='blue', linestyle='--', linewidth=2, label=f'Cached only: {cached_lat:.3f}s')
        ax1.legend()
    
    # Cache hit rate
    ax2 = axes[1]
    cache_hits = (second_run['cache_hit'] == True).sum()
    cache_misses = len(second_run) - cache_hits
    hit_rate = cache_hits / len(second_run) * 100 if len(second_run) > 0 else 0
    
    ax2.pie([cache_hits, cache_misses], labels=[f'Cache Hits ({cache_hits})', f'Cache Misses ({cache_misses})'],
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
    ax2.set_title(f'Cache Hit Rate\n({hit_rate:.1f}%)')
    
    plt.tight_layout()
    output_path = output_dir / 'cache_effectiveness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("APPROACH 3.5: CREATING FINAL VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    after_df = load_results()
    if after_df is None:
        return
    
    # Load before results if available
    results_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    before_file = results_dir / 'batch_results.csv'
    before_df = pd.read_csv(before_file) if before_file.exists() else None
    
    # Create visualizations
    print("Creating latency distribution plot...")
    create_latency_distribution_plot(after_df, output_dir)
    
    if before_df is not None:
        print("Creating before/after comparison plot...")
        create_comparison_plot(before_df, after_df, output_dir)
    
    # Cache effectiveness plot
    cache_file = results_dir / 'cache_effectiveness_test.csv'
    if cache_file.exists():
        print("Creating cache effectiveness plot...")
        create_cache_effectiveness_plot(cache_file, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ All visualizations created successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

