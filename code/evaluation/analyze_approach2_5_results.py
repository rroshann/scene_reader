#!/usr/bin/env python3
"""
Comprehensive analysis of Approach 2.5 optimized results
Calculates all quantitative metrics automatically
"""
import csv
import statistics
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    if not csv_path.exists():
        print(f"⚠️  Results file not found: {csv_path}")
        return results
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def calculate_latency_stats(results):
    """Calculate comprehensive latency statistics"""
    latencies = []
    detection_latencies = []
    generation_latencies = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                total = float(r.get('total_latency', 0))
                detection = float(r.get('detection_latency', 0))
                generation = float(r.get('generation_latency', 0))
                
                if total > 0:
                    latencies.append(total)
                if detection > 0:
                    detection_latencies.append(detection)
                if generation > 0:
                    generation_latencies.append(generation)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    stats = {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'min': min(latencies),
        'max': max(latencies),
        'p25': sorted(latencies)[len(latencies) // 4] if latencies else 0,
        'p75': sorted(latencies)[len(latencies) * 3 // 4] if latencies else 0,
        'p95': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    }
    
    if detection_latencies:
        stats['detection_mean'] = statistics.mean(detection_latencies)
        stats['detection_median'] = statistics.median(detection_latencies)
    
    if generation_latencies:
        stats['generation_mean'] = statistics.mean(generation_latencies)
        stats['generation_median'] = statistics.median(generation_latencies)
    
    return stats


def calculate_word_count_stats(results):
    """Calculate word count statistics"""
    word_counts = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            description = r.get('description', '')
            if description:
                word_count = len(description.split())
                word_counts.append(word_count)
    
    if not word_counts:
        return None
    
    return {
        'mean': statistics.mean(word_counts),
        'median': statistics.median(word_counts),
        'std_dev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0.0,
        'min': min(word_counts),
        'max': max(word_counts)
    }


def analyze_by_category(results):
    """Analyze results by category"""
    category_stats = defaultdict(list)
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            category = r.get('category', 'unknown')
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    category_stats[category].append(latency)
            except (ValueError, TypeError):
                continue
    
    analysis = {}
    for category, latencies in category_stats.items():
        if latencies:
            analysis[category] = {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }
    
    return analysis


def analyze_cache_performance(results):
    """Analyze cache performance"""
    cache_hits = 0
    cache_misses = 0
    cache_hit_latencies = []
    cache_miss_latencies = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            cache_hit = r.get('cache_hit', 'False') == 'True'
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    if cache_hit:
                        cache_hits += 1
                        cache_hit_latencies.append(latency)
                    else:
                        cache_misses += 1
                        cache_miss_latencies.append(latency)
            except (ValueError, TypeError):
                continue
    
    total_requests = cache_hits + cache_misses
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    cache_stats = {
        'hits': cache_hits,
        'misses': cache_misses,
        'total': total_requests,
        'hit_rate': hit_rate
    }
    
    if cache_hit_latencies:
        cache_stats['hit_mean_latency'] = statistics.mean(cache_hit_latencies)
        cache_stats['hit_median_latency'] = statistics.median(cache_hit_latencies)
    
    if cache_miss_latencies:
        cache_stats['miss_mean_latency'] = statistics.mean(cache_miss_latencies)
        cache_stats['miss_median_latency'] = statistics.median(cache_miss_latencies)
    
    if cache_hit_latencies and cache_miss_latencies:
        speedup = statistics.mean(cache_miss_latencies) / statistics.mean(cache_hit_latencies)
        cache_stats['speedup'] = speedup
    
    return cache_stats


def analyze_target_achievement(results, target=2.0):
    """Analyze <2s target achievement"""
    under_target = 0
    over_target = 0
    latencies = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    latencies.append(latency)
                    if latency < target:
                        under_target += 1
                    else:
                        over_target += 1
            except (ValueError, TypeError):
                continue
    
    total = under_target + over_target
    achievement_rate = (under_target / total * 100) if total > 0 else 0.0
    
    return {
        'target': target,
        'under_target': under_target,
        'over_target': over_target,
        'total': total,
        'achievement_rate': achievement_rate,
        'mean_latency': statistics.mean(latencies) if latencies else 0.0
    }


def generate_report(results, output_path):
    """Generate comprehensive analysis report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 2.5 COMPREHENSIVE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Results: {len(results)}")
    
    successful = [r for r in results if r.get('success') == 'True' or r.get('success') is True]
    report_lines.append(f"Successful: {len(successful)}")
    report_lines.append(f"Failed: {len(results) - len(successful)}")
    report_lines.append(f"Success Rate: {len(successful)/len(results)*100:.1f}%" if results else "N/A")
    report_lines.append("")
    
    # Latency Statistics
    latency_stats = calculate_latency_stats(successful)
    if latency_stats:
        report_lines.append("LATENCY STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Latency:")
        report_lines.append(f"  Mean: {latency_stats['mean']:.2f}s")
        report_lines.append(f"  Median: {latency_stats['median']:.2f}s")
        report_lines.append(f"  Std Dev: {latency_stats['std_dev']:.2f}s")
        report_lines.append(f"  Min: {latency_stats['min']:.2f}s")
        report_lines.append(f"  Max: {latency_stats['max']:.2f}s")
        report_lines.append(f"  P25: {latency_stats['p25']:.2f}s")
        report_lines.append(f"  P75: {latency_stats['p75']:.2f}s")
        report_lines.append(f"  P95: {latency_stats['p95']:.2f}s")
        
        if 'detection_mean' in latency_stats:
            report_lines.append(f"\nDetection Latency:")
            report_lines.append(f"  Mean: {latency_stats['detection_mean']:.2f}s")
            report_lines.append(f"  Median: {latency_stats['detection_median']:.2f}s")
        
        if 'generation_mean' in latency_stats:
            report_lines.append(f"\nGeneration Latency:")
            report_lines.append(f"  Mean: {latency_stats['generation_mean']:.2f}s")
            report_lines.append(f"  Median: {latency_stats['generation_median']:.2f}s")
        report_lines.append("")
    
    # Word Count Statistics
    word_stats = calculate_word_count_stats(successful)
    if word_stats:
        report_lines.append("WORD COUNT STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Mean: {word_stats['mean']:.1f} words")
        report_lines.append(f"Median: {word_stats['median']:.1f} words")
        report_lines.append(f"Std Dev: {word_stats['std_dev']:.1f} words")
        report_lines.append(f"Range: {word_stats['min']} - {word_stats['max']} words")
        report_lines.append("")
    
    # Category Analysis
    category_analysis = analyze_by_category(successful)
    if category_analysis:
        report_lines.append("CATEGORY ANALYSIS")
        report_lines.append("-" * 80)
        for category, stats in sorted(category_analysis.items()):
            report_lines.append(f"{category.capitalize()}:")
            report_lines.append(f"  Count: {stats['count']}")
            report_lines.append(f"  Mean Latency: {stats['mean']:.2f}s")
            report_lines.append(f"  Median Latency: {stats['median']:.2f}s")
            report_lines.append(f"  Std Dev: {stats['std_dev']:.2f}s")
            report_lines.append("")
    
    # Cache Performance
    cache_stats = analyze_cache_performance(successful)
    if cache_stats:
        report_lines.append("CACHE PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"Cache Hits: {cache_stats['hits']}")
        report_lines.append(f"Cache Misses: {cache_stats['misses']}")
        report_lines.append(f"Total Requests: {cache_stats['total']}")
        report_lines.append(f"Hit Rate: {cache_stats['hit_rate']:.1f}%")
        
        if 'hit_mean_latency' in cache_stats:
            report_lines.append(f"\nCache Hit Latency:")
            report_lines.append(f"  Mean: {cache_stats['hit_mean_latency']:.2f}s")
            report_lines.append(f"  Median: {cache_stats['hit_median_latency']:.2f}s")
        
        if 'miss_mean_latency' in cache_stats:
            report_lines.append(f"\nCache Miss Latency:")
            report_lines.append(f"  Mean: {cache_stats['miss_mean_latency']:.2f}s")
            report_lines.append(f"  Median: {cache_stats['miss_median_latency']:.2f}s")
        
        if 'speedup' in cache_stats:
            report_lines.append(f"\nCache Speedup: {cache_stats['speedup']:.1f}x")
        report_lines.append("")
    
    # Target Achievement
    target_analysis = analyze_target_achievement(successful, target=2.0)
    if target_analysis:
        report_lines.append("TARGET ACHIEVEMENT (<2 SECONDS)")
        report_lines.append("-" * 80)
        report_lines.append(f"Under {target_analysis['target']}s: {target_analysis['under_target']}/{target_analysis['total']} ({target_analysis['achievement_rate']:.1f}%)")
        report_lines.append(f"Over {target_analysis['target']}s: {target_analysis['over_target']}/{target_analysis['total']} ({100-target_analysis['achievement_rate']:.1f}%)")
        report_lines.append(f"Mean Latency: {target_analysis['mean_latency']:.2f}s")
        report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    if latency_stats:
        report_lines.append(f"✅ Mean Latency: {latency_stats['mean']:.2f}s")
        if latency_stats['mean'] < 2.0:
            report_lines.append(f"✅ <2s Target: ACHIEVED ({target_analysis['achievement_rate']:.1f}% of tests)")
        else:
            report_lines.append(f"❌ <2s Target: NOT ACHIEVED ({target_analysis['achievement_rate']:.1f}% of tests)")
    
    if cache_stats and cache_stats['total'] > 0:
        report_lines.append(f"✅ Cache Hit Rate: {cache_stats['hit_rate']:.1f}%")
        if 'speedup' in cache_stats:
            report_lines.append(f"✅ Cache Speedup: {cache_stats['speedup']:.1f}x")
    
    if word_stats:
        report_lines.append(f"✅ Mean Word Count: {word_stats['mean']:.1f} words")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))


def main():
    """Main analysis function"""
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / 'results' / 'approach_2_5_optimized' / 'raw' / 'batch_results.csv'
    output_path = project_root / 'results' / 'approach_2_5_optimized' / 'analysis' / 'comprehensive_analysis.txt'
    
    print("Loading results...")
    results = load_results(csv_path)
    
    if not results:
        print("⚠️  No results found. Run batch test first.")
        return
    
    print(f"Loaded {len(results)} results")
    print("Generating comprehensive analysis...")
    
    generate_report(results, output_path)
    
    print(f"\n✅ Analysis complete! Report saved to: {output_path}")


if __name__ == "__main__":
    main()

