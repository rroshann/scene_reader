#!/usr/bin/env python3
"""
Compare Approach 5 (Streaming) with Approach 1 (Baseline GPT-4V)
Analyzes perceived latency improvement and tradeoffs
"""
import csv
import statistics
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_streaming_results(csv_path):
    """Load streaming results"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def load_baseline_results(csv_path):
    """Load baseline GPT-4V results from Approach 1"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only get GPT-4V results
            if row.get('model') == 'GPT-4V' and row.get('success') == 'True':
                results.append(row)
    return results


def match_results_by_filename(streaming_results, baseline_results):
    """Match streaming and baseline results by filename"""
    baseline_dict = {r['filename']: r for r in baseline_results}
    matched = []
    
    for s_result in streaming_results:
        filename = s_result.get('filename')
        if filename in baseline_dict:
            matched.append({
                'filename': filename,
                'category': s_result.get('category'),
                'streaming': s_result,
                'baseline': baseline_dict[filename]
            })
    
    return matched


def calculate_comparison_stats(matched_results):
    """Calculate comparison statistics"""
    streaming_latencies = []
    baseline_latencies = []
    time_to_first = []
    improvements = []
    
    for match in matched_results:
        streaming = match['streaming']
        baseline = match['baseline']
        
        # Baseline latency
        if baseline.get('latency_seconds'):
            try:
                baseline_lat = float(baseline['latency_seconds'])
                baseline_latencies.append(baseline_lat)
            except (ValueError, TypeError):
                pass
        
        # Streaming tier2 latency (should be similar to baseline)
        if streaming.get('tier2_latency'):
            try:
                streaming_tier2 = float(streaming['tier2_latency'])
                streaming_latencies.append(streaming_tier2)
            except (ValueError, TypeError):
                pass
        
        # Time to first output (perceived latency)
        if streaming.get('time_to_first_output'):
            try:
                ttf = float(streaming['time_to_first_output'])
                time_to_first.append(ttf)
            except (ValueError, TypeError):
                pass
        
        # Improvement
        if streaming.get('perceived_latency_improvement'):
            try:
                imp = float(streaming['perceived_latency_improvement'])
                improvements.append(imp)
            except (ValueError, TypeError):
                pass
    
    stats = {}
    
    if baseline_latencies:
        stats['baseline'] = {
            'mean': statistics.mean(baseline_latencies),
            'median': statistics.median(baseline_latencies),
            'count': len(baseline_latencies)
        }
    
    if streaming_latencies:
        stats['streaming_tier2'] = {
            'mean': statistics.mean(streaming_latencies),
            'median': statistics.median(streaming_latencies),
            'count': len(streaming_latencies)
        }
    
    if time_to_first:
        stats['time_to_first'] = {
            'mean': statistics.mean(time_to_first),
            'median': statistics.median(time_to_first),
            'count': len(time_to_first)
        }
    
    if improvements:
        stats['improvement'] = {
            'mean': statistics.mean(improvements),
            'median': statistics.median(improvements),
            'count': len(improvements)
        }
    
    # Calculate latency reduction
    if baseline_latencies and time_to_first:
        baseline_mean = statistics.mean(baseline_latencies)
        ttf_mean = statistics.mean(time_to_first)
        stats['latency_reduction'] = baseline_mean - ttf_mean
        stats['latency_reduction_pct'] = ((baseline_mean - ttf_mean) / baseline_mean) * 100
    
    return stats


def create_comparison_visualization(matched_results, output_dir):
    """Create comparison visualization"""
    baseline_latencies = []
    streaming_tier2_latencies = []
    time_to_first = []
    
    for match in matched_results:
        baseline = match['baseline']
        streaming = match['streaming']
        
        if baseline.get('latency_seconds'):
            try:
                baseline_latencies.append(float(baseline['latency_seconds']))
            except (ValueError, TypeError):
                pass
        
        if streaming.get('tier2_latency'):
            try:
                streaming_tier2_latencies.append(float(streaming['tier2_latency']))
            except (ValueError, TypeError):
                pass
        
        if streaming.get('time_to_first_output'):
            try:
                time_to_first.append(float(streaming['time_to_first_output']))
            except (ValueError, TypeError):
                pass
    
    if not baseline_latencies or not time_to_first:
        print("  ⚠️  Insufficient data for comparison visualization")
        return
    
    # Prepare data
    data = []
    labels = []
    
    data.extend(baseline_latencies)
    labels.extend(['Baseline\n(GPT-4V)'] * len(baseline_latencies))
    
    if streaming_tier2_latencies:
        data.extend(streaming_tier2_latencies)
        labels.extend(['Streaming\nTier2'] * len(streaming_tier2_latencies))
    
    data.extend(time_to_first)
    labels.extend(['Streaming\nTime to First'] * len(time_to_first))
    
    df = pd.DataFrame({'Approach': labels, 'Latency (s)': data})
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Approach', y='Latency (s)')
    plt.title('Latency Comparison: Baseline vs Streaming', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xlabel('Approach', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'streaming_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: streaming_vs_baseline_comparison.png")


def generate_comparison_report(matched_results, stats, output_path):
    """Generate comparison report"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("APPROACH 5 (STREAMING) vs APPROACH 1 (BASELINE) COMPARISON")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("LATENCY COMPARISON")
    report_lines.append("-" * 80)
    
    if 'baseline' in stats:
        b = stats['baseline']
        report_lines.append(f"Baseline (GPT-4V only):")
        report_lines.append(f"  Mean: {b['mean']:.3f}s")
        report_lines.append(f"  Median: {b['median']:.3f}s")
        report_lines.append(f"  Count: {b['count']}")
        report_lines.append("")
    
    if 'streaming_tier2' in stats:
        s2 = stats['streaming_tier2']
        report_lines.append(f"Streaming Tier2 (GPT-4V):")
        report_lines.append(f"  Mean: {s2['mean']:.3f}s")
        report_lines.append(f"  Median: {s2['median']:.3f}s")
        report_lines.append(f"  Count: {s2['count']}")
        report_lines.append("")
    
    if 'time_to_first' in stats:
        ttf = stats['time_to_first']
        report_lines.append(f"Streaming Time to First Output (Tier1):")
        report_lines.append(f"  Mean: {ttf['mean']:.3f}s")
        report_lines.append(f"  Median: {ttf['median']:.3f}s")
        report_lines.append(f"  Count: {ttf['count']}")
        report_lines.append("")
    
    if 'latency_reduction' in stats:
        report_lines.append("PERCEIVED LATENCY IMPROVEMENT")
        report_lines.append("-" * 80)
        report_lines.append(f"Average latency reduction: {stats['latency_reduction']:.3f}s")
        report_lines.append(f"Percentage improvement: {stats['latency_reduction_pct']:.1f}%")
        report_lines.append("")
    
    if 'improvement' in stats:
        imp = stats['improvement']
        report_lines.append("IMPROVEMENT STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Mean improvement: {imp['mean']:.1f}%")
        report_lines.append(f"Median improvement: {imp['median']:.1f}%")
        report_lines.append(f"Count: {imp['count']}")
        report_lines.append("")
    
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 80)
    if 'latency_reduction' in stats:
        report_lines.append(f"✓ Streaming provides {stats['latency_reduction']:.2f}s faster perceived latency")
        report_lines.append(f"✓ This represents a {stats['latency_reduction_pct']:.1f}% improvement over baseline")
    report_lines.append("✓ Users get immediate feedback from Tier1 (BLIP-2) while waiting for detailed description")
    report_lines.append("✓ Total latency is similar (max of tier1 and tier2, run in parallel)")
    report_lines.append("✓ Cost is the same (only Tier2 uses GPT-4V API)")
    report_lines.append("")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    """Main comparison function"""
    project_root = Path(__file__).parent.parent.parent
    
    streaming_results_path = project_root / 'results' / 'approach_5_streaming' / 'raw' / 'batch_results.csv'
    baseline_results_path = project_root / 'results' / 'approach_1_vlm' / 'raw' / 'batch_results.csv'
    output_dir = project_root / 'results' / 'approach_5_streaming' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not streaming_results_path.exists():
        print(f"Streaming results not found: {streaming_results_path}")
        print("Please run batch_test_streaming.py first")
        return
    
    if not baseline_results_path.exists():
        print(f"Baseline results not found: {baseline_results_path}")
        print("Please ensure Approach 1 results are available")
        return
    
    print("Comparing Approach 5 (Streaming) with Approach 1 (Baseline)")
    print("=" * 60)
    
    streaming_results = load_streaming_results(streaming_results_path)
    baseline_results = load_baseline_results(baseline_results_path)
    
    matched_results = match_results_by_filename(streaming_results, baseline_results)
    
    if not matched_results:
        print("No matching results found between streaming and baseline")
        return
    
    print(f"Matched {len(matched_results)} results")
    
    stats = calculate_comparison_stats(matched_results)
    
    create_comparison_visualization(matched_results, output_dir)
    
    report_path = output_dir / 'streaming_vs_baseline_comparison.txt'
    generate_comparison_report(matched_results, stats, report_path)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

