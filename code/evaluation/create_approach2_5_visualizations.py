#!/usr/bin/env python3
"""
Create visualizations for Approach 2.5 optimized results
"""
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    if not csv_path.exists():
        print(f"⚠️  Results file not found: {csv_path}")
        return results
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def plot_latency_distribution(results, output_dir):
    """Plot latency distribution histogram"""
    latencies = []
    for r in results:
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                latencies.append(latency)
        except (ValueError, TypeError):
            continue
    
    if not latencies:
        print("⚠️  No latency data for distribution plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(latencies, bins=20, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.2f}s')
    ax.axvline(np.median(latencies), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(latencies):.2f}s')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='Target: 2.0s')
    ax.set_xlabel('Total Latency (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Approach 2.5: Latency Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def plot_latency_by_category(results, output_dir):
    """Plot latency by category"""
    category_data = {}
    for r in results:
        category = r.get('category', 'unknown')
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                if category not in category_data:
                    category_data[category] = []
                category_data[category].append(latency)
        except (ValueError, TypeError):
            continue
    
    if not category_data:
        print("⚠️  No category data for plot")
        return
    
    categories = sorted(category_data.keys())
    means = [np.mean(category_data[cat]) for cat in categories]
    stds = [np.std(category_data[cat]) for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axhline(2.0, color='orange', linestyle='--', linewidth=2, label='Target: 2.0s')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('Approach 2.5: Mean Latency by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.05,
                f'{mean:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'latency_by_category.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def plot_cache_performance(results, output_dir):
    """Plot cache hit vs miss latency comparison"""
    cache_hit_latencies = []
    cache_miss_latencies = []
    
    for r in results:
        cache_hit = r.get('cache_hit', 'False') == 'True'
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                if cache_hit:
                    cache_hit_latencies.append(latency)
                else:
                    cache_miss_latencies.append(latency)
        except (ValueError, TypeError):
            continue
    
    if not cache_hit_latencies and not cache_miss_latencies:
        print("⚠️  No cache data for plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    
    if cache_hit_latencies:
        data_to_plot.append(cache_hit_latencies)
        labels.append(f'Cache Hit\n(n={len(cache_hit_latencies)})')
    
    if cache_miss_latencies:
        data_to_plot.append(cache_miss_latencies)
        labels.append(f'Cache Miss\n(n={len(cache_miss_latencies)})')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Latency (seconds)', fontsize=12)
        ax.set_title('Approach 2.5: Cache Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if cache_hit_latencies and cache_miss_latencies:
            speedup = np.mean(cache_miss_latencies) / np.mean(cache_hit_latencies)
            ax.text(0.5, 0.95, f'Speedup: {speedup:.1f}x', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'cache_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def plot_latency_breakdown(results, output_dir):
    """Plot detection vs generation latency breakdown"""
    detection_latencies = []
    generation_latencies = []
    
    for r in results:
        try:
            detection = float(r.get('detection_latency', 0))
            generation = float(r.get('generation_latency', 0))
            if detection > 0:
                detection_latencies.append(detection)
            if generation > 0:
                generation_latencies.append(generation)
        except (ValueError, TypeError):
            continue
    
    if not detection_latencies and not generation_latencies:
        print("⚠️  No breakdown data for plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = []
    means = []
    stds = []
    
    if detection_latencies:
        stages.append('Detection\n(YOLO)')
        means.append(np.mean(detection_latencies))
        stds.append(np.std(detection_latencies))
    
    if generation_latencies:
        stages.append('Generation\n(LLM)')
        means.append(np.mean(generation_latencies))
        stds.append(np.std(generation_latencies))
    
    x_pos = np.arange(len(stages))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['#9b59b6', '#e67e22'], edgecolor='black')
    ax.set_xlabel('Pipeline Stage', fontsize=12)
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('Approach 2.5: Latency Breakdown by Stage', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stages)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'latency_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def plot_comparison_with_baseline(results_25, results_2, output_dir):
    """Plot comparison with Approach 2 baseline"""
    latencies_25 = []
    latencies_2 = []
    
    for r in results_25:
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                latencies_25.append(latency)
        except (ValueError, TypeError):
            continue
    
    for r in results_2:
        if r.get('yolo_model') == 'yolov8n' and r.get('llm_model') == 'gpt-4o-mini':
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    latencies_2.append(latency)
            except (ValueError, TypeError):
                continue
    
    if not latencies_25 or not latencies_2:
        print("⚠️  Insufficient data for comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = [latencies_2, latencies_25]
    labels = ['Approach 2\n(GPT-4o-mini)', 'Approach 2.5\n(GPT-3.5-turbo)']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['#95a5a6', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(2.0, color='orange', linestyle='--', linewidth=2, label='Target: 2.0s')
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Approach 2 vs Approach 2.5: Latency Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotation
    speedup = np.mean(latencies_2) / np.mean(latencies_25)
    ax.text(0.5, 0.95, f'Speedup: {speedup:.1f}x ({((speedup-1)*100):.1f}% faster)',
           transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_with_baseline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path.name}")


def main():
    """Generate all visualizations"""
    project_root = Path(__file__).parent.parent.parent
    csv_path_25 = project_root / 'results' / 'approach_2_5_optimized' / 'raw' / 'batch_results.csv'
    csv_path_2 = project_root / 'results' / 'approach_2_yolo_llm' / 'raw' / 'batch_results.csv'
    output_dir = project_root / 'results' / 'approach_2_5_optimized' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("APPROACH 2.5 VISUALIZATION GENERATION")
    print("=" * 80)
    print()
    
    print("Loading Approach 2.5 results...")
    results_25 = load_results(csv_path_25)
    print(f"Loaded {len(results_25)} results")
    
    print("\nLoading Approach 2 baseline results...")
    results_2 = load_results(csv_path_2)
    print(f"Loaded {len(results_2)} baseline results")
    
    print("\nGenerating visualizations...")
    print()
    
    plot_latency_distribution(results_25, output_dir)
    plot_latency_by_category(results_25, output_dir)
    plot_cache_performance(results_25, output_dir)
    plot_latency_breakdown(results_25, output_dir)
    plot_comparison_with_baseline(results_25, results_2, output_dir)
    
    print()
    print("=" * 80)
    print("✅ All visualizations generated!")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

