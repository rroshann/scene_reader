#!/usr/bin/env python3
"""
Comprehensive analysis of VLM results (Approach 1)
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
            # Only include recent results (system prompt test)
            if '2025-11-22T20:' in row.get('timestamp', ''):
                results.append(row)
    return results

def calculate_latency_stats(results, model_name):
    """Calculate comprehensive latency statistics"""
    latencies = []
    for r in results:
        if r['model'] == model_name and r['success'] == 'True' and r.get('latency_seconds'):
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
        'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else None,
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }

def calculate_latency_by_scenario(results, model_name):
    """Calculate latency statistics by scenario"""
    by_scenario = defaultdict(list)
    
    for r in results:
        if r['model'] == model_name and r['success'] == 'True' and r.get('latency_seconds'):
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

def analyze_response_length(results, model_name):
    """Analyze response length metrics"""
    word_counts = []
    char_counts = []
    token_counts = []
    
    for r in results:
        if r['model'] == model_name and r['success'] == 'True' and r.get('description'):
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

def calculate_costs(results, model_name):
    """Calculate cost estimates based on token usage"""
    # API pricing (as of Nov 2025)
    PRICING = {
        'GPT-4V': {
            'input_per_1k': 0.0025,  # $0.0025 per 1K input tokens
            'output_per_1k': 0.01,   # $0.01 per 1K output tokens
            'image': 0.00765         # $0.00765 per image (1024x1024)
        },
        'Gemini': {
            'input_per_1k': 0.00125,
            'output_per_1k': 0.005,
            'image': 0.00315
        },
        'Claude': {
            'input_per_1k': 0.003,
            'output_per_1k': 0.015,
            'image': 0.024
        }
    }
    
    if model_name not in PRICING:
        return None
    
    pricing = PRICING[model_name]
    total_cost = 0
    total_queries = 0
    token_data = []
    
    for r in results:
        if r['model'] == model_name and r['success'] == 'True':
            total_queries += 1
            cost = pricing['image']  # Base image cost
            
            # Add token costs if available
            if r.get('tokens_used'):
                try:
                    tokens = int(r['tokens_used'])
                    # Estimate: 70% input, 30% output (rough estimate)
                    input_tokens = int(tokens * 0.7)
                    output_tokens = int(tokens * 0.3)
                    cost += (input_tokens / 1000) * pricing['input_per_1k']
                    cost += (output_tokens / 1000) * pricing['output_per_1k']
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

def main():
    """Run all analyses"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    print("=" * 60)
    print("VLM RESULTS ANALYSIS - Approach 1")
    print("=" * 60)
    print()
    
    results = load_results(csv_path)
    print(f"📊 Loaded {len(results)} results from system prompt test")
    print()
    
    models = ['GPT-4V', 'Gemini', 'Claude']
    all_stats = {}
    
    # 1. Latency Analysis
    print("1️⃣  LATENCY ANALYSIS")
    print("-" * 60)
    for model in models:
        stats = calculate_latency_stats(results, model)
        if stats:
            all_stats[model] = {'latency': stats}
            print(f"\n{model}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f}s")
            print(f"  Median (p50): {stats['median']:.2f}s")
            if stats['p95']:
                print(f"  p95: {stats['p95']:.2f}s")
            if stats['p99']:
                print(f"  p99: {stats['p99']:.2f}s")
            print(f"  Min: {stats['min']:.2f}s")
            print(f"  Max: {stats['max']:.2f}s")
            print(f"  Std Dev: {stats['stdev']:.2f}s")
    
    # 2. Latency by Scenario
    print("\n\n2️⃣  LATENCY BY SCENARIO")
    print("-" * 60)
    for model in models:
        scenario_stats = calculate_latency_by_scenario(results, model)
        if scenario_stats:
            print(f"\n{model}:")
            for scenario, stats in scenario_stats.items():
                print(f"  {scenario.capitalize()}: {stats['mean']:.2f}s (median: {stats['median']:.2f}s, n={stats['count']})")
            all_stats[model]['latency_by_scenario'] = scenario_stats
    
    # 3. Response Length Analysis
    print("\n\n3️⃣  RESPONSE LENGTH ANALYSIS")
    print("-" * 60)
    for model in models:
        length_stats = analyze_response_length(results, model)
        if length_stats:
            all_stats[model]['response_length'] = length_stats
            print(f"\n{model}:")
            print(f"  Word Count: {length_stats['word_count']['mean']:.1f} avg (median: {length_stats['word_count']['median']:.1f})")
            print(f"  Character Count: {length_stats['char_count']['mean']:.0f} avg (median: {length_stats['char_count']['median']:.0f})")
            if 'token_count' in length_stats:
                print(f"  Token Count: {length_stats['token_count']['mean']:.0f} avg (median: {length_stats['token_count']['median']:.0f})")
    
    # 4. Cost Analysis
    print("\n\n4️⃣  COST ANALYSIS")
    print("-" * 60)
    for model in models:
        cost_stats = calculate_costs(results, model)
        if cost_stats:
            all_stats[model]['cost'] = cost_stats
            print(f"\n{model}:")
            print(f"  Total Queries: {cost_stats['total_queries']}")
            print(f"  Total Cost: ${cost_stats['total_cost']:.2f}")
            print(f"  Cost per Query: ${cost_stats['cost_per_query']:.4f}")
            print(f"  Cost per 1000 Queries: ${cost_stats['cost_per_1000']:.2f}")
            if cost_stats['avg_tokens']:
                print(f"  Avg Tokens per Query: {cost_stats['avg_tokens']:.0f}")
    
    # Save summary to file
    summary_path = Path('results/approach_1_vlm/analysis/vlm_analysis_summary.txt')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("VLM RESULTS ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Total Results Analyzed: {len(results)}\n\n")
        
        for model, stats in all_stats.items():
            f.write(f"\n{model}:\n")
            f.write("-" * 40 + "\n")
            if 'latency' in stats:
                f.write(f"Latency: {stats['latency']['mean']:.2f}s mean, {stats['latency']['median']:.2f}s median\n")
            if 'response_length' in stats:
                f.write(f"Response Length: {stats['response_length']['word_count']['mean']:.1f} words avg\n")
            if 'cost' in stats:
                f.write(f"Cost: ${stats['cost']['cost_per_query']:.4f} per query\n")
    
    print(f"\n\n✅ Analysis complete! Summary saved to: {summary_path}")
    
    return all_stats

if __name__ == '__main__':
    main()

