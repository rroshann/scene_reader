#!/usr/bin/env python3
"""
Analyze Mode-Specific Comparison Results
Processes CSV results from mode_specific_comparison_test.py and generates
comprehensive statistics, visualizations, and analysis reports for both
real_world and gaming modes, including sub-category analysis.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not installed. Statistical tests will be skipped. Install with: pip install scipy")

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


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
    costs = []
    tokens_list = []
    success_count = 0
    total_count = len(results)
    
    for result in results:
        success_key = f'approach_{approach}_success'
        latency_key = f'approach_{approach}_latency'
        cost_key = f'approach_{approach}_cost'
        tokens_key = f'approach_{approach}_tokens'
        
        if result.get(success_key, '').lower() == 'true':
            success_count += 1
            try:
                latency = float(result.get(latency_key, 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                pass
            
            try:
                cost = float(result.get(cost_key, 0) or 0)
                if cost > 0:
                    costs.append(cost)
            except (ValueError, TypeError):
                pass
            
            try:
                tokens = int(result.get(tokens_key, 0) or 0)
                if tokens > 0:
                    tokens_list.append(tokens)
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
            'p95_latency': 0,
            'mean_cost': 0,
            'mean_tokens': 0,
            'total_tests': total_count,
            'successful_tests': success_count,
            'latencies': []
        }
    
    # Calculate p95
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
    
    return {
        'success_rate': success_count / total_count if total_count > 0 else 0,
        'mean_latency': statistics.mean(latencies),
        'median_latency': statistics.median(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'p95_latency': p95_latency,
        'mean_cost': statistics.mean(costs) if costs else 0,
        'mean_tokens': statistics.mean(tokens_list) if tokens_list else 0,
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
                'std_dev': statistics.stdev(data['latencies']) if len(data['latencies']) > 1 else 0,
                'success_rate': data['success'] / data['total'] if data['total'] > 0 else 0,
                'count': data['total'],
                'latencies': data['latencies']
            }
        else:
            stats_by_category[category] = {
                'mean_latency': 0,
                'median_latency': 0,
                'std_dev': 0,
                'success_rate': 0,
                'count': data['total'],
                'latencies': []
            }
    
    return stats_by_category


def calculate_statistical_tests(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict) -> dict:
    """Calculate statistical significance tests between approaches"""
    tests = {}
    
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}
    
    # Get latencies
    latencies_1_5 = stats_1_5.get('latencies', [])
    latencies_2_5 = stats_2_5.get('latencies', [])
    latencies_3_5 = stats_3_5.get('latencies', [])
    
    # Paired t-tests (assuming same images tested)
    if len(latencies_1_5) == len(latencies_2_5) and len(latencies_1_5) > 1:
        try:
            t_stat, p_value = scipy_stats.ttest_rel(latencies_1_5, latencies_2_5)
            mean_diff = statistics.mean([a - b for a, b in zip(latencies_1_5, latencies_2_5)])
            tests['1_5_vs_2_5'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < 0.05
            }
        except Exception as e:
            tests['1_5_vs_2_5'] = {'error': f'Could not calculate: {str(e)}'}
    
    if len(latencies_1_5) == len(latencies_3_5) and len(latencies_1_5) > 1:
        try:
            t_stat, p_value = scipy_stats.ttest_rel(latencies_1_5, latencies_3_5)
            mean_diff = statistics.mean([a - b for a, b in zip(latencies_1_5, latencies_3_5)])
            tests['1_5_vs_3_5'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < 0.05
            }
        except Exception as e:
            tests['1_5_vs_3_5'] = {'error': f'Could not calculate: {str(e)}'}
    
    if len(latencies_2_5) == len(latencies_3_5) and len(latencies_2_5) > 1:
        try:
            t_stat, p_value = scipy_stats.ttest_rel(latencies_2_5, latencies_3_5)
            mean_diff = statistics.mean([a - b for a, b in zip(latencies_2_5, latencies_3_5)])
            tests['2_5_vs_3_5'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'significant': p_value < 0.05
            }
        except Exception as e:
            tests['2_5_vs_3_5'] = {'error': f'Could not calculate: {str(e)}'}
    
    return tests


def create_latency_comparison_boxplot(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict, 
                                     mode: str, output_path: Path):
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
    ax.set_title(f'Latency Distribution Comparison ({mode.upper()} Mode)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_latency_comparison_barchart(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict,
                                      mode: str, output_path: Path):
    """Create bar chart comparing mean and median latencies"""
    approaches = ['Approach 1.5', 'Approach 2.5', 'Approach 3.5']
    mean_latencies = [
        stats_1_5['mean_latency'],
        stats_2_5['mean_latency'],
        stats_3_5['mean_latency']
    ]
    median_latencies = [
        stats_1_5['median_latency'],
        stats_2_5['median_latency'],
        stats_3_5['median_latency']
    ]
    
    x = np.arange(len(approaches))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, mean_latencies, width, label='Mean', color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    bars2 = ax.bar(x + width/2, median_latencies, width, label='Median', color=['#2980b9', '#27ae60', '#c0392b'], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title(f'Mean vs Median Latency Comparison ({mode.upper()} Mode)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_category_breakdown_chart(category_stats: dict, mode: str, output_path: Path):
    """Create grouped bar chart comparing approaches by category"""
    categories = sorted(category_stats.keys())
    approaches = ['Approach 1.5', 'Approach 2.5', 'Approach 3.5']
    
    # Prepare data - need to extract from nested structure
    # This assumes category_stats has structure: {category: {approach: stats}}
    # We'll need to reorganize the data
    
    # For now, create a simpler version showing mean latency per category
    # This will need to be enhanced if we want per-approach breakdown
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract mean latencies per category (aggregated across approaches)
    category_means = []
    for category in categories:
        cat_data = category_stats[category]
        if isinstance(cat_data, dict) and 'mean_latency' in cat_data:
            category_means.append(cat_data['mean_latency'])
        else:
            # If it's a dict of approaches, calculate overall mean
            if isinstance(cat_data, dict):
                means = [v.get('mean_latency', 0) for v in cat_data.values() if isinstance(v, dict)]
                category_means.append(statistics.mean(means) if means else 0)
            else:
                category_means.append(0)
    
    bars = ax.bar(categories, category_means, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(categories)], alpha=0.8)
    
    # Add value labels
    for bar, mean in zip(bars, category_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title(f'Latency by Category ({mode.upper()} Mode)', fontsize=14, fontweight='bold')
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_cost_comparison_chart(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict,
                                 mode: str, output_path: Path):
    """Create bar chart comparing costs"""
    approaches = ['Approach 1.5', 'Approach 2.5', 'Approach 3.5']
    costs = [
        stats_1_5['mean_cost'],
        stats_2_5['mean_cost'],
        stats_3_5['mean_cost']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(approaches, costs, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Cost per Query ($)', fontsize=12)
    ax.set_title(f'Cost Comparison ({mode.upper()} Mode)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(costs) * 1.2 if costs else 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def create_latency_distributions(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict,
                                mode: str, output_path: Path):
    """Create histogram showing latency distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    data_list = [
        (stats_1_5.get('latencies', []), 'Approach 1.5', '#3498db'),
        (stats_2_5.get('latencies', []), 'Approach 2.5', '#2ecc71'),
        (stats_3_5.get('latencies', []), 'Approach 3.5', '#e74c3c')
    ]
    
    for ax, (data, title, color) in zip(axes, data_list):
        if data:
            ax.hist(data, bins=15, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(statistics.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {statistics.mean(data):.2f}s')
            ax.axvline(statistics.median(data), color='orange', linestyle='--', linewidth=2, label=f'Median: {statistics.median(data):.2f}s')
            ax.set_xlabel('Latency (seconds)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Latency Distributions ({mode.upper()} Mode)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path.name}")


def generate_report(stats_1_5: dict, stats_2_5: dict, stats_3_5: dict,
                   category_stats: dict, statistical_tests: dict,
                   mode: str, output_path: Path):
    """Generate comprehensive markdown report"""
    
    # Determine winners
    fastest = min([('1.5', stats_1_5['mean_latency']), ('2.5', stats_2_5['mean_latency']), ('3.5', stats_3_5['mean_latency'])], key=lambda x: x[1])[0]
    most_consistent = min([('1.5', stats_1_5['std_dev']), ('2.5', stats_2_5['std_dev']), ('3.5', stats_3_5['std_dev'])], key=lambda x: x[1])[0]
    cheapest = min([('1.5', stats_1_5['mean_cost']), ('2.5', stats_2_5['mean_cost']), ('3.5', stats_3_5['mean_cost'])], key=lambda x: x[1])[0]
    
    # Format statistical tests
    test_text = ""
    for test_name, test_result in statistical_tests.items():
        if 'error' not in test_result:
            sig = "✅ Significant" if test_result['significant'] else "❌ Not Significant"
            test_text += f"\n- **{test_name.replace('_', ' ').title()}**: t={test_result['t_statistic']:.3f}, p={test_result['p_value']:.4f}, mean_diff={test_result['mean_difference']:.3f}s ({sig})"
        else:
            test_text += f"\n- **{test_name.replace('_', ' ').title()}**: {test_result['error']}"
    
    # Format category stats
    category_text = ""
    for category, cat_stats in sorted(category_stats.items()):
        if isinstance(cat_stats, dict) and 'mean_latency' in cat_stats:
            category_text += f"\n- **{category.title()}**: Mean={cat_stats['mean_latency']:.2f}s, Median={cat_stats['median_latency']:.2f}s, Count={cat_stats['count']}"
    
    report = f"""# Mode-Specific Comparison Results: {mode.upper()} Mode

## Overview

This report compares Approaches 1.5, 2.5, and 3.5 using their **actual prompt_mode parameters** ({mode} mode).

### Test Configuration

- **Mode**: {mode}
- **Cache**: ENABLED (real-world usage)
- **Adaptive Parameters**: ENABLED for Approach 3.5
- **Prompts**: Mode-specific (not standardized)
- **Test Images**: {"Gaming images (12)" if mode == "gaming" else "Indoor + Outdoor + Text images (30)"}

---

## Overall Statistics

### Approach 1.5: Optimized Pure VLM (GPT-4V only)

- **Success Rate**: {stats_1_5['success_rate']*100:.1f}% ({stats_1_5['successful_tests']}/{stats_1_5['total_tests']})
- **Mean Latency**: {stats_1_5['mean_latency']:.2f}s
- **Median Latency**: {stats_1_5['median_latency']:.2f}s
- **Min Latency**: {stats_1_5['min_latency']:.2f}s
- **Max Latency**: {stats_1_5['max_latency']:.2f}s
- **P95 Latency**: {stats_1_5['p95_latency']:.2f}s
- **Std Deviation**: {stats_1_5['std_dev']:.2f}s
- **Mean Cost**: ${stats_1_5['mean_cost']:.4f}/query
- **Mean Tokens**: {stats_1_5['mean_tokens']:.0f}

### Approach 2.5: Optimized YOLO + LLM

- **Success Rate**: {stats_2_5['success_rate']*100:.1f}% ({stats_2_5['successful_tests']}/{stats_2_5['total_tests']})
- **Mean Latency**: {stats_2_5['mean_latency']:.2f}s
- **Median Latency**: {stats_2_5['median_latency']:.2f}s
- **Min Latency**: {stats_2_5['min_latency']:.2f}s
- **Max Latency**: {stats_2_5['max_latency']:.2f}s
- **P95 Latency**: {stats_2_5['p95_latency']:.2f}s
- **Std Deviation**: {stats_2_5['std_dev']:.2f}s
- **Mean Cost**: ${stats_2_5['mean_cost']:.4f}/query
- **Mean Tokens**: {stats_2_5['mean_tokens']:.0f}

### Approach 3.5: Optimized Specialized Multi-Model

- **Success Rate**: {stats_3_5['success_rate']*100:.1f}% ({stats_3_5['successful_tests']}/{stats_3_5['total_tests']})
- **Mean Latency**: {stats_3_5['mean_latency']:.2f}s
- **Median Latency**: {stats_3_5['median_latency']:.2f}s
- **Min Latency**: {stats_3_5['min_latency']:.2f}s
- **Max Latency**: {stats_3_5['max_latency']:.2f}s
- **P95 Latency**: {stats_3_5['p95_latency']:.2f}s
- **Std Deviation**: {stats_3_5['std_dev']:.2f}s
- **Mean Cost**: ${stats_3_5['mean_cost']:.4f}/query
- **Mean Tokens**: {stats_3_5['mean_tokens']:.0f}

---

## Comparison Table

| Metric | Approach 1.5 | Approach 2.5 | Approach 3.5 | Winner |
|--------|--------------|--------------|--------------|--------|
| **Mean Latency** | {stats_1_5['mean_latency']:.2f}s | {stats_2_5['mean_latency']:.2f}s | {stats_3_5['mean_latency']:.2f}s | Approach {fastest} |
| **Median Latency** | {stats_1_5['median_latency']:.2f}s | {stats_2_5['median_latency']:.2f}s | {stats_3_5['median_latency']:.2f}s | Approach {fastest} |
| **P95 Latency** | {stats_1_5['p95_latency']:.2f}s | {stats_2_5['p95_latency']:.2f}s | {stats_3_5['p95_latency']:.2f}s | Approach {fastest} |
| **Success Rate** | {stats_1_5['success_rate']*100:.1f}% | {stats_2_5['success_rate']*100:.1f}% | {stats_3_5['success_rate']*100:.1f}% | {"All tied" if stats_1_5['success_rate'] == stats_2_5['success_rate'] == stats_3_5['success_rate'] else "Varies"} |
| **Consistency (Std Dev)** | {stats_1_5['std_dev']:.2f}s | {stats_2_5['std_dev']:.2f}s | {stats_3_5['std_dev']:.2f}s | Approach {most_consistent} |
| **Cost per Query** | ${stats_1_5['mean_cost']:.4f} | ${stats_2_5['mean_cost']:.4f} | ${stats_3_5['mean_cost']:.4f} | Approach {cheapest} |

---

## Category-Specific Statistics

{category_text if category_text else "No category breakdown available"}

---

## Statistical Significance Tests

{test_text if test_text else "Statistical tests could not be calculated"}

---

## Key Findings

### Speed Analysis

- **Fastest**: Approach {fastest} ({min(stats_1_5['mean_latency'], stats_2_5['mean_latency'], stats_3_5['mean_latency']):.2f}s mean latency)
- **Most Consistent**: Approach {most_consistent} (std dev: {min(stats_1_5['std_dev'], stats_2_5['std_dev'], stats_3_5['std_dev']):.2f}s)
- **Cheapest**: Approach {cheapest} (${min(stats_1_5['mean_cost'], stats_2_5['mean_cost'], stats_3_5['mean_cost']):.4f}/query)

### Performance Characteristics

1. **Approach 1.5 (Pure VLM)**:
   - Direct GPT-4V analysis
   - Highest quality but slower
   - Most expensive

2. **Approach 2.5 (YOLO + LLM)**:
   - Fast two-stage pipeline
   - Good balance of speed and cost
   - YOLO detection + GPT-3.5-turbo

3. **Approach 3.5 (Specialized)**:
   - Most versatile (OCR/Depth + YOLO)
   - Good for text-heavy or spatial scenarios
   - Moderate speed and cost

---

## Methodology

- **Test Images**: {"12 gaming images" if mode == "gaming" else "30 real-world images (10 indoor + 10 outdoor + 10 text)"}
- **Mode**: {mode}
- **Parameters**: Mode-specific prompts, cache enabled, adaptive enabled for Approach 3.5
- **Measurements**: Latency from start to description received
- **Repetitions**: Single run per image

---

*Generated from mode-specific comparison test results*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_path}")


def analyze_mode(mode: str, output_dir: Path):
    """Analyze results for a specific mode"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {mode.upper()} MODE")
    print(f"{'='*80}")
    
    # Load results
    csv_path = project_root / "results" / "mode_specific_comparison" / "raw" / f"{mode}_results.csv"
    results = load_results(csv_path)
    
    if not results:
        print(f"❌ No results found for {mode} mode. Run mode_specific_comparison_test.py first.")
        return
    
    print(f"\nLoaded {len(results)} test results for {mode} mode")
    
    # Calculate overall statistics
    print("\nCalculating overall statistics...")
    stats_1_5 = calculate_statistics(results, '1_5')
    stats_2_5 = calculate_statistics(results, '2_5')
    stats_3_5 = calculate_statistics(results, '3_5')
    
    # Calculate category statistics
    print("Calculating category statistics...")
    category_stats_1_5 = calculate_category_stats(results, '1_5')
    category_stats_2_5 = calculate_category_stats(results, '2_5')
    category_stats_3_5 = calculate_category_stats(results, '3_5')
    
    # Combine category stats
    all_categories = set(category_stats_1_5.keys()) | set(category_stats_2_5.keys()) | set(category_stats_3_5.keys())
    combined_category_stats = {}
    for category in all_categories:
        combined_category_stats[category] = {
            '1_5': category_stats_1_5.get(category, {}),
            '2_5': category_stats_2_5.get(category, {}),
            '3_5': category_stats_3_5.get(category, {})
        }
    
    # Calculate statistical tests
    print("Calculating statistical tests...")
    statistical_tests = calculate_statistical_tests(stats_1_5, stats_2_5, stats_3_5)
    
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_latency_comparison_boxplot(
        stats_1_5, stats_2_5, stats_3_5,
        mode, figures_dir / f"latency_comparison_boxplot_{mode}.png"
    )
    
    create_latency_comparison_barchart(
        stats_1_5, stats_2_5, stats_3_5,
        mode, figures_dir / f"latency_comparison_barchart_{mode}.png"
    )
    
    if mode == 'real_world':
        # Create category breakdown for real_world
        create_category_breakdown_chart(
            combined_category_stats, mode,
            figures_dir / f"category_breakdown_{mode}.png"
        )
    
    create_cost_comparison_chart(
        stats_1_5, stats_2_5, stats_3_5,
        mode, figures_dir / f"cost_comparison_{mode}.png"
    )
    
    create_latency_distributions(
        stats_1_5, stats_2_5, stats_3_5,
        mode, figures_dir / f"latency_distributions_{mode}.png"
    )
    
    # Generate report
    print("Generating report...")
    generate_report(
        stats_1_5, stats_2_5, stats_3_5,
        combined_category_stats, statistical_tests,
        mode, output_dir / f"comparison_report_{mode}.md"
    )
    
    # Generate statistics summary
    print("Generating statistics summary...")
    summary_path = output_dir / f"statistics_summary_{mode}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Mode-Specific Comparison Statistics: {mode.upper()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Approach 1.5: Mean={stats_1_5['mean_latency']:.2f}s, Median={stats_1_5['median_latency']:.2f}s, StdDev={stats_1_5['std_dev']:.2f}s\n")
        f.write(f"Approach 2.5: Mean={stats_2_5['mean_latency']:.2f}s, Median={stats_2_5['median_latency']:.2f}s, StdDev={stats_2_5['std_dev']:.2f}s\n")
        f.write(f"Approach 3.5: Mean={stats_3_5['mean_latency']:.2f}s, Median={stats_3_5['median_latency']:.2f}s, StdDev={stats_3_5['std_dev']:.2f}s\n")
    
    print(f"✅ Analysis complete for {mode} mode")


def main():
    """Main analysis execution"""
    print("="*80)
    print("MODE-SPECIFIC COMPARISON ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = project_root / "results" / "mode_specific_comparison" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze both modes
    analyze_mode('real_world', output_dir)
    analyze_mode('gaming', output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - comparison_report_real_world.md")
    print("  - comparison_report_gaming.md")
    print("  - statistics_summary_real_world.txt")
    print("  - statistics_summary_gaming.txt")
    print("  - figures/ (visualizations)")


if __name__ == "__main__":
    main()

