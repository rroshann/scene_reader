#!/usr/bin/env python3
"""
Comprehensive analysis of Approach 5: Streaming/Progressive Models results
Calculates quantitative metrics for tier1, tier2, and overall performance
"""
import csv
import statistics
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def calculate_latency_stats(results, tier='tier1'):
    """Calculate comprehensive latency statistics for a tier"""
    latencies = []
    latency_key = f'{tier}_latency'
    
    for r in results:
        if r.get(latency_key):
            try:
                lat = float(r[latency_key])
                if lat > 0:  # Valid latency
                    latencies.append(lat)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    return {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p75': statistics.quantiles(latencies, n=4)[2] if len(latencies) >= 4 else None,
        'p90': statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else None,
        'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None,
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def calculate_latency_by_category(results, tier='tier1'):
    """Calculate latency statistics by category"""
    by_category = defaultdict(list)
    latency_key = f'{tier}_latency'
    
    for r in results:
        if r.get(latency_key):
            category = r.get('category', 'unknown')
            try:
                lat = float(r[latency_key])
                if lat > 0:
                    by_category[category].append(lat)
            except (ValueError, TypeError):
                continue
    
    stats_by_category = {}
    for category, latencies in by_category.items():
        if latencies:
            stats_by_category[category] = {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies)
            }
    
    return stats_by_category


def analyze_description_length(results, tier='tier1'):
    """Analyze description length metrics"""
    word_counts = []
    char_counts = []
    desc_key = f'{tier}_description'
    
    for r in results:
        if r.get(desc_key):
            desc = r[desc_key]
            if desc and desc.strip():
                word_count = len(desc.split())
                char_count = len(desc)
                word_counts.append(word_count)
                char_counts.append(char_count)
    
    if not word_counts:
        return None
    
    return {
        'word_count': {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts),
            'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        },
        'char_count': {
            'mean': statistics.mean(char_counts),
            'median': statistics.median(char_counts),
            'min': min(char_counts),
            'max': max(char_counts)
        }
    }


def calculate_success_rates(results):
    """Calculate success rates for each tier"""
    total = len(results)
    
    tier1_success = sum(1 for r in results if r.get('tier1_success') == 'True')
    tier2_success = sum(1 for r in results if r.get('tier2_success') == 'True')
    both_success = sum(1 for r in results if r.get('tier1_success') == 'True' and r.get('tier2_success') == 'True')
    either_success = sum(1 for r in results if r.get('tier1_success') == 'True' or r.get('tier2_success') == 'True')
    
    return {
        'total': total,
        'tier1_success': tier1_success,
        'tier1_rate': (tier1_success / total * 100) if total > 0 else 0,
        'tier2_success': tier2_success,
        'tier2_rate': (tier2_success / total * 100) if total > 0 else 0,
        'both_success': both_success,
        'both_rate': (both_success / total * 100) if total > 0 else 0,
        'either_success': either_success,
        'either_rate': (either_success / total * 100) if total > 0 else 0
    }


def calculate_cost_stats(results):
    """Calculate cost statistics"""
    costs = []
    tokens = []
    
    for r in results:
        if r.get('tier2_cost'):
            try:
                cost = float(r['tier2_cost'])
                if cost >= 0:
                    costs.append(cost)
            except (ValueError, TypeError):
                pass
        
        if r.get('tier2_tokens'):
            try:
                token_count = int(r['tier2_tokens'])
                if token_count > 0:
                    tokens.append(token_count)
            except (ValueError, TypeError):
                pass
    
    if not costs:
        return None
    
    return {
        'total_cost': sum(costs),
        'mean_cost_per_query': statistics.mean(costs),
        'median_cost_per_query': statistics.median(costs),
        'min_cost': min(costs),
        'max_cost': max(costs),
        'cost_per_1000_queries': statistics.mean(costs) * 1000,
        'mean_tokens': statistics.mean(tokens) if tokens else None,
        'total_tokens': sum(tokens) if tokens else None
    }


def calculate_perceived_latency_improvement(results):
    """Calculate perceived latency improvement metrics"""
    improvements = []
    time_to_first = []
    tier2_only_latencies = []
    
    for r in results:
        if r.get('perceived_latency_improvement'):
            try:
                imp = float(r['perceived_latency_improvement'])
                improvements.append(imp)
            except (ValueError, TypeError):
                pass
        
        if r.get('time_to_first_output'):
            try:
                ttf = float(r['time_to_first_output'])
                if ttf > 0:
                    time_to_first.append(ttf)
            except (ValueError, TypeError):
                pass
        
        if r.get('tier2_latency'):
            try:
                t2_lat = float(r['tier2_latency'])
                if t2_lat > 0:
                    tier2_only_latencies.append(t2_lat)
            except (ValueError, TypeError):
                pass
    
    if not improvements:
        return None
    
    return {
        'mean_improvement_pct': statistics.mean(improvements),
        'median_improvement_pct': statistics.median(improvements),
        'min_improvement_pct': min(improvements),
        'max_improvement_pct': max(improvements),
        'mean_time_to_first': statistics.mean(time_to_first) if time_to_first else None,
        'mean_tier2_only_latency': statistics.mean(tier2_only_latencies) if tier2_only_latencies else None,
        'latency_reduction': statistics.mean(tier2_only_latencies) - statistics.mean(time_to_first) if time_to_first and tier2_only_latencies else None
    }


def generate_analysis_report(results_path, output_path):
    """Generate comprehensive analysis report"""
    results = load_results(results_path)
    
    if not results:
        print(f"No results found in {results_path}")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 5: STREAMING/PROGRESSIVE MODELS - ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Success Rates
    report_lines.append("SUCCESS RATES")
    report_lines.append("-" * 80)
    success_rates = calculate_success_rates(results)
    report_lines.append(f"Total tests: {success_rates['total']}")
    report_lines.append(f"Tier1 (BLIP-2) successful: {success_rates['tier1_success']}/{success_rates['total']} ({success_rates['tier1_rate']:.1f}%)")
    report_lines.append(f"Tier2 (GPT-4V) successful: {success_rates['tier2_success']}/{success_rates['total']} ({success_rates['tier2_rate']:.1f}%)")
    report_lines.append(f"Both tiers successful: {success_rates['both_success']}/{success_rates['total']} ({success_rates['both_rate']:.1f}%)")
    report_lines.append(f"Either tier successful: {success_rates['either_success']}/{success_rates['total']} ({success_rates['either_rate']:.1f}%)")
    report_lines.append("")
    
    # Tier1 Latency Stats
    report_lines.append("TIER1 (BLIP-2) LATENCY STATISTICS")
    report_lines.append("-" * 80)
    tier1_stats = calculate_latency_stats(results, 'tier1')
    if tier1_stats:
        report_lines.append(f"Count: {tier1_stats['count']}")
        report_lines.append(f"Mean: {tier1_stats['mean']:.3f}s")
        report_lines.append(f"Median: {tier1_stats['median']:.3f}s")
        report_lines.append(f"P75: {tier1_stats['p75']:.3f}s" if tier1_stats['p75'] else "P75: N/A")
        report_lines.append(f"P90: {tier1_stats['p90']:.3f}s" if tier1_stats['p90'] else "P90: N/A")
        report_lines.append(f"P95: {tier1_stats['p95']:.3f}s" if tier1_stats['p95'] else "P95: N/A")
        report_lines.append(f"Min: {tier1_stats['min']:.3f}s")
        report_lines.append(f"Max: {tier1_stats['max']:.3f}s")
        report_lines.append(f"Std Dev: {tier1_stats['stdev']:.3f}s")
    else:
        report_lines.append("No valid tier1 latency data")
    report_lines.append("")
    
    # Tier2 Latency Stats
    report_lines.append("TIER2 (GPT-4V) LATENCY STATISTICS")
    report_lines.append("-" * 80)
    tier2_stats = calculate_latency_stats(results, 'tier2')
    if tier2_stats:
        report_lines.append(f"Count: {tier2_stats['count']}")
        report_lines.append(f"Mean: {tier2_stats['mean']:.3f}s")
        report_lines.append(f"Median: {tier2_stats['median']:.3f}s")
        report_lines.append(f"P75: {tier2_stats['p75']:.3f}s" if tier2_stats['p75'] else "P75: N/A")
        report_lines.append(f"P90: {tier2_stats['p90']:.3f}s" if tier2_stats['p90'] else "P90: N/A")
        report_lines.append(f"P95: {tier2_stats['p95']:.3f}s" if tier2_stats['p95'] else "P95: N/A")
        report_lines.append(f"Min: {tier2_stats['min']:.3f}s")
        report_lines.append(f"Max: {tier2_stats['max']:.3f}s")
        report_lines.append(f"Std Dev: {tier2_stats['stdev']:.3f}s")
    else:
        report_lines.append("No valid tier2 latency data")
    report_lines.append("")
    
    # Total Latency Stats
    report_lines.append("TOTAL LATENCY STATISTICS (Max of tier1 and tier2)")
    report_lines.append("-" * 80)
    total_stats = []
    for r in results:
        if r.get('total_latency'):
            try:
                total_stats.append(float(r['total_latency']))
            except (ValueError, TypeError):
                pass
    if total_stats:
        report_lines.append(f"Count: {len(total_stats)}")
        report_lines.append(f"Mean: {statistics.mean(total_stats):.3f}s")
        report_lines.append(f"Median: {statistics.median(total_stats):.3f}s")
        report_lines.append(f"Min: {min(total_stats):.3f}s")
        report_lines.append(f"Max: {max(total_stats):.3f}s")
    else:
        report_lines.append("No valid total latency data")
    report_lines.append("")
    
    # Perceived Latency Improvement
    report_lines.append("PERCEIVED LATENCY IMPROVEMENT")
    report_lines.append("-" * 80)
    improvement_stats = calculate_perceived_latency_improvement(results)
    if improvement_stats:
        report_lines.append(f"Mean improvement: {improvement_stats['mean_improvement_pct']:.1f}%")
        report_lines.append(f"Median improvement: {improvement_stats['median_improvement_pct']:.1f}%")
        report_lines.append(f"Min improvement: {improvement_stats['min_improvement_pct']:.1f}%")
        report_lines.append(f"Max improvement: {improvement_stats['max_improvement_pct']:.1f}%")
        if improvement_stats['mean_time_to_first']:
            report_lines.append(f"Mean time to first output: {improvement_stats['mean_time_to_first']:.3f}s")
        if improvement_stats['mean_tier2_only_latency']:
            report_lines.append(f"Mean tier2-only latency: {improvement_stats['mean_tier2_only_latency']:.3f}s")
        if improvement_stats['latency_reduction']:
            report_lines.append(f"Average latency reduction: {improvement_stats['latency_reduction']:.3f}s")
    else:
        report_lines.append("No valid improvement data")
    report_lines.append("")
    
    # Description Length Analysis
    report_lines.append("DESCRIPTION LENGTH ANALYSIS")
    report_lines.append("-" * 80)
    tier1_length = analyze_description_length(results, 'tier1')
    tier2_length = analyze_description_length(results, 'tier2')
    
    if tier1_length:
        report_lines.append("Tier1 (BLIP-2):")
        report_lines.append(f"  Mean words: {tier1_length['word_count']['mean']:.1f}")
        report_lines.append(f"  Median words: {tier1_length['word_count']['median']:.1f}")
        report_lines.append(f"  Mean chars: {tier1_length['char_count']['mean']:.1f}")
    
    if tier2_length:
        report_lines.append("Tier2 (GPT-4V):")
        report_lines.append(f"  Mean words: {tier2_length['word_count']['mean']:.1f}")
        report_lines.append(f"  Median words: {tier2_length['word_count']['median']:.1f}")
        report_lines.append(f"  Mean chars: {tier2_length['char_count']['mean']:.1f}")
    
    report_lines.append("")
    
    # Cost Analysis
    report_lines.append("COST ANALYSIS")
    report_lines.append("-" * 80)
    cost_stats = calculate_cost_stats(results)
    if cost_stats:
        report_lines.append(f"Total cost (Tier2 only): ${cost_stats['total_cost']:.4f}")
        report_lines.append(f"Mean cost per query: ${cost_stats['mean_cost_per_query']:.4f}")
        report_lines.append(f"Cost per 1000 queries: ${cost_stats['cost_per_1000_queries']:.2f}")
        if cost_stats['mean_tokens']:
            report_lines.append(f"Mean tokens per query: {cost_stats['mean_tokens']:.0f}")
        if cost_stats['total_tokens']:
            report_lines.append(f"Total tokens: {cost_stats['total_tokens']}")
    else:
        report_lines.append("No cost data available")
    report_lines.append("")
    
    # Latency by Category
    report_lines.append("TIER1 LATENCY BY CATEGORY")
    report_lines.append("-" * 80)
    tier1_by_cat = calculate_latency_by_category(results, 'tier1')
    for category, stats in sorted(tier1_by_cat.items()):
        report_lines.append(f"{category}: mean={stats['mean']:.3f}s, median={stats['median']:.3f}s, n={stats['count']}")
    report_lines.append("")
    
    report_lines.append("TIER2 LATENCY BY CATEGORY")
    report_lines.append("-" * 80)
    tier2_by_cat = calculate_latency_by_category(results, 'tier2')
    for category, stats in sorted(tier2_by_cat.items()):
        report_lines.append(f"{category}: mean={stats['mean']:.3f}s, median={stats['median']:.3f}s, n={stats['count']}")
    report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    """Main analysis function"""
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / 'results' / 'approach_5_streaming' / 'raw' / 'batch_results.csv'
    output_path = project_root / 'results' / 'approach_5_streaming' / 'analysis' / 'streaming_analysis.txt'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Please run batch_test_streaming.py first")
        return
    
    generate_analysis_report(results_path, output_path)


if __name__ == "__main__":
    main()

