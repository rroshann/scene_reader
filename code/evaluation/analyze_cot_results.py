#!/usr/bin/env python3
"""
Comprehensive analysis of Chain-of-Thought (CoT) results (Approach 7)
Calculates all quantitative metrics and compares to baseline GPT-4V
"""
import csv
import statistics
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_cot_results(csv_path):
    """Load CoT results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('prompt_type') == 'CoT' and row['success'] == 'True':
                results.append(row)
    return results

def load_baseline_results(csv_path):
    """Load baseline GPT-4V results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include system prompt test results (same as CoT)
            if row['model'] == 'GPT-4V' and '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True':
                    results.append(row)
    return results

def calculate_latency_stats(results):
    """Calculate comprehensive latency statistics"""
    latencies = []
    for r in results:
        if r.get('latency_seconds'):
            try:
                latencies.append(float(r['latency_seconds']))
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

def calculate_latency_by_scenario(results):
    """Calculate latency statistics by scenario"""
    by_scenario = defaultdict(list)
    
    for r in results:
        if r.get('latency_seconds'):
            category = r.get('category', 'unknown')
            try:
                by_scenario[category].append(float(r['latency_seconds']))
            except (ValueError, TypeError):
                continue
    
    stats_by_scenario = {}
    for scenario, latencies in by_scenario.items():
        if latencies:
            stats_by_scenario[scenario] = {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies)
            }
    
    return stats_by_scenario

def analyze_response_length(results):
    """Analyze response length metrics"""
    word_counts = []
    char_counts = []
    token_counts = []
    
    for r in results:
        if r.get('description'):
            desc = r['description']
            word_count = len(desc.split())
            char_count = len(desc)
            
            word_counts.append(word_count)
            char_counts.append(char_count)
            
            # Try to get token count if available
            if r.get('tokens_used'):
                try:
                    token_counts.append(int(r['tokens_used']))
                except (ValueError, TypeError):
                    pass
    
    if not word_counts:
        return None
    
    stats = {
        'word_count': {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts)
        },
        'char_count': {
            'mean': statistics.mean(char_counts),
            'median': statistics.median(char_counts),
            'min': min(char_counts),
            'max': max(char_counts)
        }
    }
    
    if token_counts:
        stats['token_count'] = {
            'mean': statistics.mean(token_counts),
            'median': statistics.median(token_counts),
            'min': min(token_counts),
            'max': max(token_counts)
        }
    
    return stats

def calculate_costs(results):
    """Calculate cost estimates for CoT (using GPT-4V pricing)"""
    PRICING = {
        'input_per_1k': 0.0025,  # $0.0025 per 1K input tokens
        'output_per_1k': 0.01,   # $0.01 per 1K output tokens
        'image': 0.00765         # $0.00765 per image (1024x1024)
    }
    
    total_cost = 0
    total_queries = 0
    token_data = []
    
    for r in results:
        total_queries += 1
        cost = PRICING['image']  # Base image cost
        
        # Add token costs if available
        if r.get('tokens_used'):
            try:
                tokens = int(r['tokens_used'])
                # Estimate: 70% input, 30% output (rough estimate)
                input_tokens = int(tokens * 0.7)
                output_tokens = int(tokens * 0.3)
                cost += (input_tokens / 1000) * PRICING['input_per_1k']
                cost += (output_tokens / 1000) * PRICING['output_per_1k']
                token_data.append(tokens)
            except (ValueError, TypeError):
                pass
        
        total_cost += cost
    
    if total_queries == 0:
        return None
    
    return {
        'total_queries': total_queries,
        'total_cost': total_cost,
        'cost_per_query': total_cost / total_queries,
        'cost_per_1000': (total_cost / total_queries) * 1000,
        'avg_tokens': statistics.mean(token_data) if token_data else None
    }

def compare_cot_baseline(cot_results, baseline_results):
    """Compare CoT vs baseline metrics"""
    # Match by filename
    cot_by_file = {r['filename']: r for r in cot_results}
    baseline_by_file = {r['filename']: r for r in baseline_results}
    
    # Find common files
    common_files = set(cot_by_file.keys()) & set(baseline_by_file.keys())
    
    if not common_files:
        return None
    
    # Calculate differences
    latency_diff = []
    length_diff = []
    token_diff = []
    
    for filename in common_files:
        cot = cot_by_file[filename]
        baseline = baseline_by_file[filename]
        
        # Latency difference
        if cot.get('latency_seconds') and baseline.get('latency_seconds'):
            try:
                cot_lat = float(cot['latency_seconds'])
                base_lat = float(baseline['latency_seconds'])
                latency_diff.append(cot_lat - base_lat)
            except (ValueError, TypeError):
                pass
        
        # Length difference
        if cot.get('description') and baseline.get('description'):
            cot_words = len(cot['description'].split())
            base_words = len(baseline['description'].split())
            length_diff.append(cot_words - base_words)
        
        # Token difference
        if cot.get('tokens_used') and baseline.get('tokens_used'):
            try:
                cot_tokens = int(cot['tokens_used'])
                base_tokens = int(baseline['tokens_used'])
                token_diff.append(cot_tokens - base_tokens)
            except (ValueError, TypeError):
                pass
    
    comparison = {
        'common_files': len(common_files),
        'latency': {
            'mean_diff': statistics.mean(latency_diff) if latency_diff else None,
            'pct_change': (statistics.mean(latency_diff) / statistics.mean([float(baseline_by_file[f]['latency_seconds']) for f in common_files if baseline_by_file[f].get('latency_seconds')])) * 100 if latency_diff else None
        } if latency_diff else None,
        'length': {
            'mean_diff': statistics.mean(length_diff) if length_diff else None,
            'pct_change': (statistics.mean(length_diff) / statistics.mean([len(baseline_by_file[f]['description'].split()) for f in common_files if baseline_by_file[f].get('description')])) * 100 if length_diff else None
        } if length_diff else None,
        'tokens': {
            'mean_diff': statistics.mean(token_diff) if token_diff else None,
            'pct_change': (statistics.mean(token_diff) / statistics.mean([int(baseline_by_file[f]['tokens_used']) for f in common_files if baseline_by_file[f].get('tokens_used')])) * 100 if token_diff else None
        } if token_diff else None
    }
    
    return comparison

def main():
    """Run all analyses"""
    cot_csv_path = Path('results/approach_7_cot/raw/batch_results.csv')
    baseline_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    
    if not cot_csv_path.exists():
        print(f"❌ CoT results file not found: {cot_csv_path}")
        return
    
    if not baseline_csv_path.exists():
        print(f"⚠️  Baseline results file not found: {baseline_csv_path}")
        print("   Continuing with CoT-only analysis...")
        baseline_results = []
    else:
        baseline_results = load_baseline_results(baseline_csv_path)
    
    print("=" * 60)
    print("CHAIN-OF-THOUGHT (CoT) RESULTS ANALYSIS - Approach 7")
    print("=" * 60)
    print()
    
    cot_results = load_cot_results(cot_csv_path)
    print(f"📊 Loaded {len(cot_results)} CoT results")
    if baseline_results:
        print(f"📊 Loaded {len(baseline_results)} baseline GPT-4V results")
    print()
    
    all_stats = {}
    
    # 1. Latency Analysis
    print("1️⃣  LATENCY ANALYSIS (CoT)")
    print("-" * 60)
    cot_latency = calculate_latency_stats(cot_results)
    if cot_latency:
        all_stats['cot'] = {'latency': cot_latency}
        print(f"\nCoT (GPT-4V):")
        print(f"  Count: {cot_latency['count']}")
        print(f"  Mean: {cot_latency['mean']:.2f}s")
        print(f"  Median (p50): {cot_latency['median']:.2f}s")
        if cot_latency['p95']:
            print(f"  p95: {cot_latency['p95']:.2f}s")
        print(f"  Min: {cot_latency['min']:.2f}s")
        print(f"  Max: {cot_latency['max']:.2f}s")
        print(f"  Std Dev: {cot_latency['stdev']:.2f}s")
    
    # 2. Latency by Scenario
    print("\n\n2️⃣  LATENCY BY SCENARIO (CoT)")
    print("-" * 60)
    cot_scenario_stats = calculate_latency_by_scenario(cot_results)
    if cot_scenario_stats:
        print(f"\nCoT (GPT-4V):")
        for scenario, stats in cot_scenario_stats.items():
            print(f"  {scenario.capitalize()}: {stats['mean']:.2f}s (median: {stats['median']:.2f}s, n={stats['count']})")
        all_stats['cot']['latency_by_scenario'] = cot_scenario_stats
    
    # 3. Response Length Analysis
    print("\n\n3️⃣  RESPONSE LENGTH ANALYSIS (CoT)")
    print("-" * 60)
    cot_length = analyze_response_length(cot_results)
    if cot_length:
        all_stats['cot']['response_length'] = cot_length
        print(f"\nCoT (GPT-4V):")
        print(f"  Word Count: {cot_length['word_count']['mean']:.1f} avg (median: {cot_length['word_count']['median']:.1f})")
        print(f"  Character Count: {cot_length['char_count']['mean']:.0f} avg (median: {cot_length['char_count']['median']:.0f})")
        if 'token_count' in cot_length:
            print(f"  Token Count: {cot_length['token_count']['mean']:.0f} avg (median: {cot_length['token_count']['median']:.0f})")
    
    # 4. Cost Analysis
    print("\n\n4️⃣  COST ANALYSIS (CoT)")
    print("-" * 60)
    cot_cost = calculate_costs(cot_results)
    if cot_cost:
        all_stats['cot']['cost'] = cot_cost
        print(f"\nCoT (GPT-4V):")
        print(f"  Total Queries: {cot_cost['total_queries']}")
        print(f"  Total Cost: ${cot_cost['total_cost']:.2f}")
        print(f"  Cost per Query: ${cot_cost['cost_per_query']:.4f}")
        print(f"  Cost per 1000 Queries: ${cot_cost['cost_per_1000']:.2f}")
        if cot_cost['avg_tokens']:
            print(f"  Avg Tokens per Query: {cot_cost['avg_tokens']:.0f}")
    
    # 5. Comparison to Baseline
    if baseline_results:
        print("\n\n5️⃣  COMPARISON TO BASELINE (CoT vs Standard Prompt)")
        print("-" * 60)
        comparison = compare_cot_baseline(cot_results, baseline_results)
        if comparison:
            print(f"\nCommon Images: {comparison['common_files']}")
            
            if comparison.get('latency'):
                print(f"\nLatency:")
                print(f"  Mean Difference: {comparison['latency']['mean_diff']:+.2f}s")
                if comparison['latency']['pct_change']:
                    print(f"  Percent Change: {comparison['latency']['pct_change']:+.1f}%")
            
            if comparison.get('length'):
                print(f"\nResponse Length:")
                print(f"  Mean Difference: {comparison['length']['mean_diff']:+.1f} words")
                if comparison['length']['pct_change']:
                    print(f"  Percent Change: {comparison['length']['pct_change']:+.1f}%")
            
            if comparison.get('tokens'):
                print(f"\nToken Usage:")
                print(f"  Mean Difference: {comparison['tokens']['mean_diff']:+.0f} tokens")
                if comparison['tokens']['pct_change']:
                    print(f"  Percent Change: {comparison['tokens']['pct_change']:+.1f}%")
            
            all_stats['comparison'] = comparison
    
    # Save summary to file
    summary_path = Path('results/approach_7_cot/analysis/cot_analysis_summary.txt')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("CHAIN-OF-THOUGHT (CoT) RESULTS ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Total CoT Results Analyzed: {len(cot_results)}\n\n")
        
        if 'cot' in all_stats:
            f.write("CoT (GPT-4V) Metrics:\n")
            f.write("-" * 40 + "\n")
            if 'latency' in all_stats['cot']:
                f.write(f"Latency: {all_stats['cot']['latency']['mean']:.2f}s mean, {all_stats['cot']['latency']['median']:.2f}s median\n")
            if 'response_length' in all_stats['cot']:
                f.write(f"Response Length: {all_stats['cot']['response_length']['word_count']['mean']:.1f} words avg\n")
            if 'cost' in all_stats['cot']:
                f.write(f"Cost: ${all_stats['cot']['cost']['cost_per_query']:.4f} per query\n")
        
        if 'comparison' in all_stats:
            f.write("\nComparison to Baseline:\n")
            f.write("-" * 40 + "\n")
            comp = all_stats['comparison']
            if comp.get('latency'):
                f.write(f"Latency Change: {comp['latency']['mean_diff']:+.2f}s ({comp['latency']['pct_change']:+.1f}%)\n")
            if comp.get('length'):
                f.write(f"Length Change: {comp['length']['mean_diff']:+.1f} words ({comp['length']['pct_change']:+.1f}%)\n")
    
    print(f"\n\n✅ Analysis complete! Summary saved to: {summary_path}")
    
    # Save comparison details
    if baseline_results and comparison:
        comparison_path = Path('results/approach_7_cot/analysis/cot_vs_baseline_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write("CoT vs Baseline GPT-4V Comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Common Images: {comparison['common_files']}\n\n")
            
            if comparison.get('latency'):
                f.write(f"Latency:\n")
                f.write(f"  Mean Difference: {comparison['latency']['mean_diff']:+.2f}s\n")
                if comparison['latency']['pct_change']:
                    f.write(f"  Percent Change: {comparison['latency']['pct_change']:+.1f}%\n")
            
            if comparison.get('length'):
                f.write(f"\nResponse Length:\n")
                f.write(f"  Mean Difference: {comparison['length']['mean_diff']:+.1f} words\n")
                if comparison['length']['pct_change']:
                    f.write(f"  Percent Change: {comparison['length']['pct_change']:+.1f}%\n")
            
            if comparison.get('tokens'):
                f.write(f"\nToken Usage:\n")
                f.write(f"  Mean Difference: {comparison['tokens']['mean_diff']:+.0f} tokens\n")
                if comparison['tokens']['pct_change']:
                    f.write(f"  Percent Change: {comparison['tokens']['pct_change']:+.1f}%\n")
        
        print(f"✅ Comparison saved to: {comparison_path}")
    
    return all_stats

if __name__ == '__main__':
    main()

