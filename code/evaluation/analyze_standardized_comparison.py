#!/usr/bin/env python3
"""
Analyze Standardized Comparison Results
Processes CSV results and generates statistics and visualizations
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path: Path) -> list:
    """Load results from CSV file"""
    results = []
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return results
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    return results


def calculate_statistics(results: list, approach: str) -> dict:
    """Calculate statistics for an approach"""
    latencies = []
    success_count = 0
    total_count = len(results)
    
    for result in results:
        success_key = f'approach_{approach}_success'
        latency_key = f'approach_{approach}_latency'
        
        if result.get(success_key, '').lower() == 'true':
            success_count += 1
            try:
                latency = float(result.get(latency_key, 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                pass
    
    if not latencies:
        return {
            'success_rate': 0,
            'mean_latency': 0,
            'median_latency': 0,
            'min_latency': 0,
            'max_latency': 0,
            'std_dev': 0,
            'total_tests': total_count,
            'successful_tests': success_count
        }
    
    return {
        'success_rate': success_count / total_count if total_count > 0 else 0,
        'mean_latency': statistics.mean(latencies),
        'median_latency': statistics.median(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'total_tests': total_count,
        'successful_tests': success_count,
        'latencies': latencies
    }


def calculate_category_stats(results: list, approach: str) -> dict:
    """Calculate statistics by category"""
    category_stats = defaultdict(lambda: {'latencies': [], 'success': 0, 'total': 0})
    
    for result in results:
        category = result.get('category', 'unknown')
        success_key = f'approach_{approach}_success'
        latency_key = f'approach_{approach}_latency'
        
        category_stats[category]['total'] += 1
        if result.get(success_key, '').lower() == 'true':
            category_stats[category]['success'] += 1
            try:
                latency = float(result.get(latency_key, 0))
                if latency > 0:
                    category_stats[category]['latencies'].append(latency)
            except (ValueError, TypeError):
                pass
    
    stats_by_category = {}
    for category, data in category_stats.items():
        if data['latencies']:
            stats_by_category[category] = {
                'mean_latency': statistics.mean(data['latencies']),
                'median_latency': statistics.median(data['latencies']),
                'success_rate': data['success'] / data['total'] if data['total'] > 0 else 0,
                'count': data['total']
            }
        else:
            stats_by_category[category] = {
                'mean_latency': 0,
                'median_latency': 0,
                'success_rate': 0,
                'count': data['total']
            }
    
    return stats_by_category


def create_latency_comparison_chart(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict, output_path: Path):
    """Create bar chart comparing mean latencies"""
    approaches = ['Approach 1.5', 'Approach 2.5', 'Approach 3.5']
    mean_latencies = [
        stats_1_5['mean_latency'],
        stats_2_5['mean_latency'],
        stats_3_5['mean_latency']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(approaches, mean_latencies, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # Add value labels on bars
    for i, (bar, latency) in enumerate(zip(bars, mean_latencies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.2f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('Standardized Comparison: Mean Latency by Approach', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(mean_latencies) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_latency_distribution_chart(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict, output_path: Path):
    """Create box plot comparing latency distributions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [
        stats_1_5.get('latencies', []),
        stats_2_5.get('latencies', []),
        stats_3_5.get('latencies', [])
    ]
    
    bp = ax.boxplot(data, labels=['Approach 1.5', 'Approach 2.5', 'Approach 3.5'], 
                     patch_artist=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Standardized Comparison: Latency Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_category_comparison_chart(stats_by_category: dict, output_path: Path):
    """Create grouped bar chart comparing approaches by category"""
    categories = sorted(stats_by_category.keys())
    approaches = ['Approach 1.5', 'Approach 2.5', 'Approach 3.5']
    
    # Prepare data
    data = {approach: [] for approach in approaches}
    for category in categories:
        # We'll need to calculate this per approach per category
        # For now, use overall stats (this would need to be enhanced)
        data['Approach 1.5'].append(stats_by_category[category].get('mean_latency', 0))
        data['Approach 2.5'].append(stats_by_category[category].get('mean_latency', 0))
        data['Approach 3.5'].append(stats_by_category[category].get('mean_latency', 0))
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, approach in enumerate(approaches):
        offset = (i - 1) * width
        ax.bar(x + offset, data[approach], width, label=approach, alpha=0.8)
    
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('Standardized Comparison: Latency by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def generate_report(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict, output_path: Path):
    """Generate markdown report"""
    report = f"""# Standardized Comparison Results

## Overview

This report compares Approaches 1.5, 2.5, and 3.5 using **identical parameters** to isolate architectural differences.

### Standardized Parameters

- **max_tokens**: 100
- **temperature**: 0.7
- **top_p**: 1.0
- **cache**: DISABLED
- **adaptive parameters**: DISABLED
- **image preprocessing**: DISABLED
- **prompts**: Neutral, standardized (same style/length)

---

## Overall Statistics

### Approach 1.5: Optimized Pure VLM (GPT-4V only)

- **Success Rate**: {stats_1_5['success_rate']*100:.1f}% ({stats_1_5['successful_tests']}/{stats_1_5['total_tests']})
- **Mean Latency**: {stats_1_5['mean_latency']:.2f}s
- **Median Latency**: {stats_1_5['median_latency']:.2f}s
- **Min Latency**: {stats_1_5['min_latency']:.2f}s
- **Max Latency**: {stats_1_5['max_latency']:.2f}s
- **Std Deviation**: {stats_1_5['std_dev']:.2f}s

### Approach 2.5: Optimized YOLO + LLM

- **Success Rate**: {stats_2_5['success_rate']*100:.1f}% ({stats_2_5['successful_tests']}/{stats_2_5['total_tests']})
- **Mean Latency**: {stats_2_5['mean_latency']:.2f}s
- **Median Latency**: {stats_2_5['median_latency']:.2f}s
- **Min Latency**: {stats_2_5['min_latency']:.2f}s
- **Max Latency**: {stats_2_5['max_latency']:.2f}s
- **Std Deviation**: {stats_2_5['std_dev']:.2f}s

### Approach 3.5: Optimized Specialized Multi-Model

- **Success Rate**: {stats_3_5['success_rate']*100:.1f}% ({stats_3_5['successful_tests']}/{stats_3_5['total_tests']})
- **Mean Latency**: {stats_3_5['mean_latency']:.2f}s
- **Median Latency**: {stats_3_5['median_latency']:.2f}s
- **Min Latency**: {stats_3_5['min_latency']:.2f}s
- **Max Latency**: {stats_3_5['max_latency']:.2f}s
- **Std Deviation**: {stats_3_5['std_dev']:.2f}s

---

## Comparison Table

| Metric | Approach 1.5 | Approach 2.5 | Approach 3.5 | Winner |
|--------|--------------|--------------|--------------|--------|
| **Mean Latency** | {stats_1_5['mean_latency']:.2f}s | {stats_2_5['mean_latency']:.2f}s | {stats_3_5['mean_latency']:.2f}s | {"Approach 2.5" if stats_2_5['mean_latency'] < min(stats_1_5['mean_latency'], stats_3_5['mean_latency']) else ("Approach 1.5" if stats_1_5['mean_latency'] < stats_3_5['mean_latency'] else "Approach 3.5")} |
| **Median Latency** | {stats_1_5['median_latency']:.2f}s | {stats_2_5['median_latency']:.2f}s | {stats_3_5['median_latency']:.2f}s | {"Approach 2.5" if stats_2_5['median_latency'] < min(stats_1_5['median_latency'], stats_3_5['median_latency']) else ("Approach 1.5" if stats_1_5['median_latency'] < stats_3_5['median_latency'] else "Approach 3.5")} |
| **Success Rate** | {stats_1_5['success_rate']*100:.1f}% | {stats_2_5['success_rate']*100:.1f}% | {stats_3_5['success_rate']*100:.1f}% | {"Approach 1.5" if stats_1_5['success_rate'] > max(stats_2_5['success_rate'], stats_3_5['success_rate']) else ("Approach 2.5" if stats_2_5['success_rate'] > stats_3_5['success_rate'] else "Approach 3.5")} |
| **Consistency (Std Dev)** | {stats_1_5['std_dev']:.2f}s | {stats_2_5['std_dev']:.2f}s | {stats_3_5['std_dev']:.2f}s | {"Approach 2.5" if stats_2_5['std_dev'] < min(stats_1_5['std_dev'], stats_3_5['std_dev']) else ("Approach 1.5" if stats_1_5['std_dev'] < stats_3_5['std_dev'] else "Approach 3.5")} |

---

## Key Findings

### Architectural Differences

1. **Approach 1.5 (Pure VLM)**:
   - Direct GPT-4V analysis
   - No preprocessing steps
   - Single API call

2. **Approach 2.5 (YOLO + LLM)**:
   - YOLO object detection first (~0.1s)
   - Then GPT-3.5-turbo generation
   - Two-stage pipeline

3. **Approach 3.5 (Specialized)**:
   - YOLO detection + OCR/Depth estimation
   - Then GPT-3.5-turbo generation
   - Three-stage pipeline (most complex)

### Speed Analysis

With identical parameters:
- **Fastest**: {"Approach 2.5" if stats_2_5['mean_latency'] < min(stats_1_5['mean_latency'], stats_3_5['mean_latency']) else ("Approach 1.5" if stats_1_5['mean_latency'] < stats_3_5['mean_latency'] else "Approach 3.5")} ({min(stats_1_5['mean_latency'], stats_2_5['mean_latency'], stats_3_5['mean_latency']):.2f}s)
- **Most Consistent**: {"Approach 2.5" if stats_2_5['std_dev'] < min(stats_1_5['std_dev'], stats_3_5['std_dev']) else ("Approach 1.5" if stats_1_5['std_dev'] < stats_3_5['std_dev'] else "Approach 3.5")} (std dev: {min(stats_1_5['std_dev'], stats_2_5['std_dev'], stats_3_5['std_dev']):.2f}s)

### Comparison with Optimized Results

**Note**: These standardized results show **architectural differences** only. In practice, each approach would be optimized with different parameters, which may change the rankings.

---

## Methodology

- **Test Images**: All images from `data/images/` (gaming, indoor, outdoor, text)
- **Parameters**: Identical across all approaches (see above)
- **Measurements**: Latency from start to description received
- **Repetitions**: Single run per image

---

*Generated from standardized comparison test results*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_path}")


def main():
    """Main analysis execution"""
    print("="*80)
    print("STANDARDIZED COMPARISON ANALYSIS")
    print("="*80)
    
    # Load results
    csv_path = project_root / "results" / "standardized_comparison" / "raw" / "batch_results.csv"
    results = load_results(csv_path)
    
    if not results:
        print("❌ No results found. Run standardized_comparison_test.py first.")
        return
    
    print(f"\nLoaded {len(results)} test results")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_1_5 = calculate_statistics(results, '1_5')
    stats_2_5 = calculate_statistics(results, '2_5')
    stats_3_5 = calculate_statistics(results, '3_5')
    
    # Create output directory
    output_dir = project_root / "results" / "standardized_comparison" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_latency_comparison_chart(
        stats_1_5, stats_2_5, stats_3_5,
        output_dir / "latency_comparison.png"
    )
    
    create_latency_distribution_chart(
        stats_1_5, stats_2_5, stats_3_5,
        output_dir / "latency_distribution.png"
    )
    
    # Generate report
    print("\nGenerating report...")
    generate_report(
        stats_1_5, stats_2_5, stats_3_5,
        output_dir / "comparison_report.md"
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

