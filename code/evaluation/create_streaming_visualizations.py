#!/usr/bin/env python3
"""
Create visualizations for Approach 5: Streaming/Progressive Models
"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def create_tier_latency_comparison(results, output_dir):
    """Create latency comparison chart: tier1 vs tier2"""
    tier1_latencies = []
    tier2_latencies = []
    total_latencies = []
    time_to_first = []
    
    for r in results:
        if r.get('tier1_latency'):
            try:
                tier1_latencies.append(float(r['tier1_latency']))
            except (ValueError, TypeError):
                pass
        
        if r.get('tier2_latency'):
            try:
                tier2_latencies.append(float(r['tier2_latency']))
            except (ValueError, TypeError):
                pass
        
        if r.get('total_latency'):
            try:
                total_latencies.append(float(r['total_latency']))
            except (ValueError, TypeError):
                pass
        
        if r.get('time_to_first_output'):
            try:
                time_to_first.append(float(r['time_to_first_output']))
            except (ValueError, TypeError):
                pass
    
    if not tier1_latencies and not tier2_latencies:
        print("  ⚠️  No latency data for comparison chart")
        return
    
    # Prepare data for box plot
    data = []
    labels = []
    
    if tier1_latencies:
        data.extend(tier1_latencies)
        labels.extend(['Tier1 (BLIP-2)'] * len(tier1_latencies))
    
    if tier2_latencies:
        data.extend(tier2_latencies)
        labels.extend(['Tier2 (GPT-4V)'] * len(tier2_latencies))
    
    if time_to_first:
        data.extend(time_to_first)
        labels.extend(['Time to First Output'] * len(time_to_first))
    
    if total_latencies:
        data.extend(total_latencies)
        labels.extend(['Total Latency'] * len(total_latencies))
    
    df = pd.DataFrame({'Tier': labels, 'Latency (s)': data})
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Tier', y='Latency (s)')
    plt.title('Latency Comparison: Tier1 vs Tier2 vs Total', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xlabel('Tier', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'tier_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: tier_latency_comparison.png")


def create_perceived_latency_improvement(results, output_dir):
    """Create perceived latency improvement chart"""
    improvements = []
    
    for r in results:
        if r.get('perceived_latency_improvement'):
            try:
                imp = float(r['perceived_latency_improvement'])
                improvements.append(imp)
            except (ValueError, TypeError):
                pass
    
    if not improvements:
        print("  ⚠️  No improvement data")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(improvements, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(improvements):.1f}%')
    plt.axvline(np.median(improvements), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(improvements):.1f}%')
    plt.xlabel('Perceived Latency Improvement (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Perceived Latency Improvement', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'perceived_latency_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: perceived_latency_improvement.png")


def create_description_length_comparison(results, output_dir):
    """Create description length comparison: tier1 vs tier2"""
    tier1_lengths = []
    tier2_lengths = []
    
    for r in results:
        if r.get('tier1_description'):
            desc = r['tier1_description']
            if desc and desc.strip():
                tier1_lengths.append(len(desc.split()))
        
        if r.get('tier2_description'):
            desc = r['tier2_description']
            if desc and desc.strip():
                tier2_lengths.append(len(desc.split()))
    
    if not tier1_lengths and not tier2_lengths:
        print("  ⚠️  No description length data")
        return
    
    data = []
    labels = []
    
    if tier1_lengths:
        data.extend(tier1_lengths)
        labels.extend(['Tier1 (BLIP-2)'] * len(tier1_lengths))
    
    if tier2_lengths:
        data.extend(tier2_lengths)
        labels.extend(['Tier2 (GPT-4V)'] * len(tier2_lengths))
    
    df = pd.DataFrame({'Tier': labels, 'Word Count': data})
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Tier', y='Word Count')
    plt.title('Description Length Comparison: Tier1 vs Tier2', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Words', fontsize=12)
    plt.xlabel('Tier', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'description_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: description_length_comparison.png")


def create_latency_by_category(results, output_dir):
    """Create latency by category chart"""
    categories = []
    tier1_latencies = []
    tier2_latencies = []
    
    for r in results:
        category = r.get('category', 'unknown')
        
        if r.get('tier1_latency'):
            try:
                categories.append(category)
                tier1_latencies.append(float(r['tier1_latency']))
                tier2_latencies.append(float(r['tier2_latency']) if r.get('tier2_latency') else None)
            except (ValueError, TypeError):
                pass
    
    if not categories:
        print("  ⚠️  No category data")
        return
    
    # Prepare data
    data = []
    tier_labels = []
    cat_labels = []
    
    for i, cat in enumerate(categories):
        if tier1_latencies[i]:
            data.append(tier1_latencies[i])
            tier_labels.append('Tier1')
            cat_labels.append(cat)
        if tier2_latencies[i] is not None:
            data.append(tier2_latencies[i])
            tier_labels.append('Tier2')
            cat_labels.append(cat)
    
    df = pd.DataFrame({
        'Category': cat_labels,
        'Tier': tier_labels,
        'Latency (s)': data
    })
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x='Category', y='Latency (s)', hue='Tier')
    plt.title('Latency by Category: Tier1 vs Tier2', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Tier')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: latency_by_category.png")


def create_success_rate_chart(results, output_dir):
    """Create success rate comparison chart"""
    total = len(results)
    tier1_success = sum(1 for r in results if r.get('tier1_success') == 'True')
    tier2_success = sum(1 for r in results if r.get('tier2_success') == 'True')
    both_success = sum(1 for r in results if r.get('tier1_success') == 'True' and r.get('tier2_success') == 'True')
    
    if total == 0:
        print("  ⚠️  No results for success rate chart")
        return
    
    tiers = ['Tier1\n(BLIP-2)', 'Tier2\n(GPT-4V)', 'Both\nTiers']
    success_counts = [tier1_success, tier2_success, both_success]
    success_rates = [s/total*100 for s in success_counts]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(tiers, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xlabel('Tier', fontsize=12)
    plt.title('Success Rate by Tier', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, count, rate) in enumerate(zip(bars, success_counts, success_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}/{total}\n({rate:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: success_rate.png")


def create_cost_analysis(results, output_dir):
    """Create cost analysis chart"""
    costs = []
    
    for r in results:
        if r.get('tier2_cost'):
            try:
                cost = float(r['tier2_cost'])
                if cost >= 0:
                    costs.append(cost)
            except (ValueError, TypeError):
                pass
    
    if not costs:
        print("  ⚠️  No cost data")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(costs, bins=20, edgecolor='black', alpha=0.7, color='green')
    plt.axvline(np.mean(costs), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(costs):.4f}')
    plt.xlabel('Cost per Query ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Cost Distribution (Tier2 - GPT-4V only)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: cost_analysis.png")


def create_time_to_first_vs_tier2(results, output_dir):
    """Create scatter plot: time to first output vs tier2 latency"""
    time_to_first = []
    tier2_latencies = []
    
    for r in results:
        if r.get('time_to_first_output') and r.get('tier2_latency'):
            try:
                ttf = float(r['time_to_first_output'])
                t2 = float(r['tier2_latency'])
                if ttf > 0 and t2 > 0:
                    time_to_first.append(ttf)
                    tier2_latencies.append(t2)
            except (ValueError, TypeError):
                pass
    
    if not time_to_first:
        print("  ⚠️  No data for time-to-first comparison")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tier2_latencies, time_to_first, alpha=0.6, s=50)
    
    # Add diagonal line (y=x) for reference
    max_val = max(max(tier2_latencies), max(time_to_first))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (no improvement)')
    
    plt.xlabel('Tier2 Latency (GPT-4V) (seconds)', fontsize=12)
    plt.ylabel('Time to First Output (Tier1) (seconds)', fontsize=12)
    plt.title('Perceived Latency Improvement\n(Points below diagonal show improvement)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_to_first_vs_tier2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: time_to_first_vs_tier2.png")


def main():
    """Generate all visualizations"""
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / 'results' / 'approach_5_streaming' / 'raw' / 'batch_results.csv'
    output_dir = project_root / 'results' / 'approach_5_streaming' / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Please run batch_test_streaming.py first")
        return
    
    print("Creating visualizations for Approach 5: Streaming/Progressive Models")
    print("=" * 60)
    
    results = load_results(results_path)
    
    if not results:
        print("No results found")
        return
    
    create_tier_latency_comparison(results, output_dir)
    create_perceived_latency_improvement(results, output_dir)
    create_description_length_comparison(results, output_dir)
    create_latency_by_category(results, output_dir)
    create_success_rate_chart(results, output_dir)
    create_cost_analysis(results, output_dir)
    create_time_to_first_vs_tier2(results, output_dir)
    
    print("=" * 60)
    print("All visualizations created successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

