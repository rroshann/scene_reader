#!/usr/bin/env python3
"""
Comprehensive analysis of Local Models results (Approach 4)
Calculates all quantitative metrics automatically
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


def calculate_latency_stats(results, filter_key=None, filter_value=None):
    """
    Calculate comprehensive latency statistics
    
    Args:
        results: List of result dicts
        filter_key: Optional key to filter by (e.g., 'model', 'device')
        filter_value: Optional value to filter by
    """
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                latency = float(r.get('total_latency', 0) or r.get('latency', 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'min': min(latencies),
        'max': max(latencies),
        'count': len(latencies)
    }


def calculate_response_length_stats(results, filter_key=None, filter_value=None):
    """Calculate response length statistics"""
    lengths = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            desc = r.get('description', '')
            if desc:
                lengths.append(len(desc.split()))
    
    if not lengths:
        return None
    
    return {
        'mean': statistics.mean(lengths),
        'median': statistics.median(lengths),
        'std_dev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        'min': min(lengths),
        'max': max(lengths),
        'count': len(lengths)
    }


def analyze_device_usage(results):
    """Analyze device usage (MPS vs CPU)"""
    device_counts = defaultdict(int)
    device_latencies = defaultdict(list)
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            device = r.get('device', 'unknown')
            device_counts[device] += 1
            try:
                latency = float(r.get('total_latency', 0) or r.get('latency', 0))
                if latency > 0:
                    device_latencies[device].append(latency)
            except (ValueError, TypeError):
                continue
    
    device_stats = {}
    for device, latencies in device_latencies.items():
        if latencies:
            device_stats[device] = {
                'count': device_counts[device],
                'mean_latency': statistics.mean(latencies),
                'median_latency': statistics.median(latencies)
            }
    
    return device_stats


def main():
    """Run analysis"""
    print("=" * 60)
    print("Local Models Analysis (Approach 4)")
    print("=" * 60)
    print()
    
    # Load results
    csv_path = Path('results/approach_4_local/raw/batch_results.csv')
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} results")
    
    successful = [r for r in results if r.get('success') == 'True' or r.get('success') is True]
    print(f"Successful: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)")
    print()
    
    # Output directory
    output_dir = Path('results/approach_4_local/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'local_analysis_summary.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Local Models Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Results: {len(results)}\n")
        f.write(f"Successful: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)\n\n")
        
        # Overall latency
        overall_latency = calculate_latency_stats(successful)
        if overall_latency:
            f.write("Overall Total Latency:\n")
            f.write(f"  Mean: {overall_latency['mean']:.3f}s\n")
            f.write(f"  Median: {overall_latency['median']:.3f}s\n")
            f.write(f"  Std Dev: {overall_latency['std_dev']:.3f}s\n")
            f.write(f"  Min: {overall_latency['min']:.3f}s\n")
            f.write(f"  Max: {overall_latency['max']:.3f}s\n\n")
        
        # Latency by model
        f.write("Latency by Model:\n")
        for model in ['blip2']:
            model_latency = calculate_latency_stats(successful, 'model', model)
            if model_latency:
                f.write(f"  {model}: {model_latency['mean']:.3f}s mean\n")
        f.write("\n")
        
        # Device usage
        device_stats = analyze_device_usage(successful)
        if device_stats:
            f.write("Device Usage:\n")
            for device, stats in device_stats.items():
                f.write(f"  {device}: {stats['count']} tests, {stats['mean_latency']:.3f}s mean\n")
            f.write("\n")
        
        # Response length
        overall_length = calculate_response_length_stats(successful)
        if overall_length:
            f.write("Response Lengths:\n")
            f.write(f"  Mean: {overall_length['mean']:.1f} words\n")
            f.write(f"  Median: {overall_length['median']:.1f} words\n")
            f.write(f"  Std Dev: {overall_length['std_dev']:.1f} words\n\n")
        
        # Response length by model
        f.write("Response Length by Model:\n")
        for model in ['blip2']:
            model_length = calculate_response_length_stats(successful, 'model', model)
            if model_length:
                f.write(f"  {model}: {model_length['mean']:.1f} words avg\n")
        f.write("\n")
        
        # Cost analysis (zero cost for local models)
        f.write("Cost Analysis:\n")
        f.write("  Total Cost: $0.00 (local models, no API calls)\n")
        f.write("  Cost per Query: $0.00\n")
        f.write("  Cost per 1000 Queries: $0.00\n\n")
    
    print(f"✅ Analysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

