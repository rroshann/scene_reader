#!/usr/bin/env python3
"""
Comprehensive analysis of RAG-Enhanced Vision results (Approach 6)
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
    """Calculate comprehensive latency statistics"""
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    latencies.append(latency)
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


def calculate_stage_latencies(results, filter_key=None, filter_value=None):
    """Calculate latency breakdown by stage"""
    stages = {
        'base': [],
        'entity_extraction': [],
        'retrieval': [],
        'enhancement': []
    }
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            try:
                if r.get('base_latency'):
                    stages['base'].append(float(r['base_latency']))
                if r.get('entity_extraction_latency'):
                    stages['entity_extraction'].append(float(r['entity_extraction_latency']))
                if r.get('retrieval_latency'):
                    stages['retrieval'].append(float(r['retrieval_latency']))
                if r.get('enhancement_latency'):
                    stages['enhancement'].append(float(r['enhancement_latency']))
            except (ValueError, TypeError):
                continue
    
    stats = {}
    for stage, latencies in stages.items():
        if latencies:
            stats[stage] = {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
    
    return stats if stats else None


def calculate_cost_estimates(results, filter_key=None, filter_value=None):
    """Estimate costs based on token usage"""
    # Pricing (per 1K tokens)
    gpt4o_input_rate = 0.005 / 1000  # $5 per 1M input tokens
    gpt4o_output_rate = 0.015 / 1000  # $15 per 1M output tokens
    gpt4o_mini_input_rate = 0.00015 / 1000
    gpt4o_mini_output_rate = 0.0006 / 1000
    gemini_input_rate = 0.0 / 1000  # Free for Gemini 2.5 Flash
    gemini_output_rate = 0.0 / 1000
    claude_input_rate = 0.00025 / 1000
    claude_output_rate = 0.00125 / 1000
    
    total_cost = 0.0
    model_counts = defaultdict(int)
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            vlm_model = r.get('vlm_model', '')
            base_tokens = r.get('base_tokens')
            enhancement_tokens = r.get('enhancement_tokens')
            use_rag = r.get('use_rag') == 'True' or r.get('use_rag') is True
            
            # Base VLM cost
            if base_tokens:
                try:
                    tokens_int = int(base_tokens)
                    if 'gpt-4o' in vlm_model.lower() and 'mini' not in vlm_model.lower():
                        input_tokens = int(tokens_int * 0.7)
                        output_tokens = int(tokens_int * 0.3)
                        cost = (input_tokens * gpt4o_input_rate) + (output_tokens * gpt4o_output_rate)
                        total_cost += cost
                        model_counts['gpt-4o'] += 1
                    elif 'gpt' in vlm_model.lower():
                        input_tokens = int(tokens_int * 0.7)
                        output_tokens = int(tokens_int * 0.3)
                        cost = (input_tokens * gpt4o_mini_input_rate) + (output_tokens * gpt4o_mini_output_rate)
                        total_cost += cost
                        model_counts['gpt-4o-mini'] += 1
                    elif 'gemini' in vlm_model.lower():
                        # Gemini 2.5 Flash is free
                        model_counts['gemini'] += 1
                    elif 'claude' in vlm_model.lower():
                        input_tokens = int(tokens_int * 0.7)
                        output_tokens = int(tokens_int * 0.3)
                        cost = (input_tokens * claude_input_rate) + (output_tokens * claude_output_rate)
                        total_cost += cost
                        model_counts['claude'] += 1
                except (ValueError, TypeError):
                    pass
            
            # Enhancement cost (RAG only, uses gpt-4o-mini)
            if use_rag and enhancement_tokens:
                try:
                    tokens_int = int(enhancement_tokens)
                    input_tokens = int(tokens_int * 0.7)
                    output_tokens = int(tokens_int * 0.3)
                    cost = (input_tokens * gpt4o_mini_input_rate) + (output_tokens * gpt4o_mini_output_rate)
                    total_cost += cost
                    model_counts['enhancement'] += 1
                except (ValueError, TypeError):
                    pass
    
    return {
        'total_cost': total_cost,
        'cost_per_query': total_cost / len(results) if results else 0,
        'model_counts': dict(model_counts)
    }


def calculate_retrieval_stats(results, filter_key=None, filter_value=None):
    """Calculate retrieval quality statistics"""
    num_chunks = []
    retrieval_latencies = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            use_rag = r.get('use_rag') == 'True' or r.get('use_rag') is True
            if not use_rag:
                continue
            
            try:
                if r.get('num_retrieved_chunks'):
                    num_chunks.append(int(r['num_retrieved_chunks']))
                if r.get('retrieval_latency'):
                    retrieval_latencies.append(float(r['retrieval_latency']))
            except (ValueError, TypeError):
                continue
    
    stats = {}
    if num_chunks:
        stats['num_chunks'] = {
            'mean': statistics.mean(num_chunks),
            'median': statistics.median(num_chunks),
            'min': min(num_chunks),
            'max': max(num_chunks)
        }
    
    if retrieval_latencies:
        stats['retrieval_latency'] = {
            'mean': statistics.mean(retrieval_latencies),
            'median': statistics.median(retrieval_latencies),
            'min': min(retrieval_latencies),
            'max': max(retrieval_latencies)
        }
    
    return stats if stats else None


def calculate_response_lengths(results, filter_key=None, filter_value=None):
    """Calculate response length statistics"""
    base_lengths = []
    enhanced_lengths = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            if r.get('base_description'):
                base_lengths.append(len(r['base_description'].split()))
            if r.get('enhanced_description'):
                enhanced_lengths.append(len(r['enhanced_description'].split()))
    
    stats = {}
    if base_lengths:
        stats['base'] = {
            'mean': statistics.mean(base_lengths),
            'median': statistics.median(base_lengths),
            'min': min(base_lengths),
            'max': max(base_lengths)
        }
    
    if enhanced_lengths:
        stats['enhanced'] = {
            'mean': statistics.mean(enhanced_lengths),
            'median': statistics.median(enhanced_lengths),
            'min': min(enhanced_lengths),
            'max': max(enhanced_lengths)
        }
    
    return stats if stats else None


def main():
    """Main analysis function"""
    # Load results
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    print(f"Loaded {len(results)} results")
    
    # Output directory
    output_dir = Path('results/approach_6_rag/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall statistics
    successful = [r for r in results if r.get('success') == 'True' or r.get('success') is True]
    print(f"Successful: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)")
    
    # Overall latency
    overall_latency = calculate_latency_stats(results)
    
    # Latency by RAG vs Base
    base_latency = calculate_latency_stats(results, filter_key='use_rag', filter_value=False)
    rag_latency = calculate_latency_stats(results, filter_key='use_rag', filter_value=True)
    
    # Latency by VLM model
    vlm_latencies = {}
    for vlm in ['gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku']:
        vlm_latencies[vlm] = calculate_latency_stats(results, filter_key='vlm_model', filter_value=vlm)
    
    # Stage latencies (RAG only)
    rag_results = [r for r in results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
    stage_latencies = calculate_stage_latencies(rag_results)
    
    # Cost analysis
    cost_stats = calculate_cost_estimates(results)
    
    # Retrieval stats
    retrieval_stats = calculate_retrieval_stats(results)
    
    # Response lengths
    length_stats = calculate_response_lengths(results)
    
    # Write summary
    output_file = output_dir / 'rag_analysis_summary.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RAG-Enhanced Vision Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total Results: {len(results)}\n")
        f.write(f"Successful: {len(successful)}/{len(results)} ({len(successful)*100/len(results):.1f}%)\n\n")
        
        f.write("Overall Total Latency:\n")
        if overall_latency:
            f.write(f"  Mean: {overall_latency['mean']:.3f}s\n")
            f.write(f"  Median: {overall_latency['median']:.3f}s\n")
            f.write(f"  Std Dev: {overall_latency['stdev']:.3f}s\n\n")
        
        f.write("Latency: Base vs RAG-Enhanced\n")
        if base_latency:
            f.write(f"  Base VLM Mean: {base_latency['mean']:.3f}s\n")
        if rag_latency:
            f.write(f"  RAG-Enhanced Mean: {rag_latency['mean']:.3f}s\n")
            if base_latency:
                overhead = ((rag_latency['mean'] / base_latency['mean']) - 1) * 100
                f.write(f"  Overhead: {overhead:.1f}%\n")
        f.write("\n")
        
        f.write("Latency by VLM Model:\n")
        for vlm, stats in vlm_latencies.items():
            if stats:
                f.write(f"  {vlm}: {stats['mean']:.3f}s mean\n")
        f.write("\n")
        
        f.write("Stage Latencies (RAG-Enhanced):\n")
        if stage_latencies:
            for stage, stats in stage_latencies.items():
                f.write(f"  {stage}: {stats['mean']:.3f}s mean\n")
        f.write("\n")
        
        f.write("Cost Analysis:\n")
        if cost_stats:
            f.write(f"  Total Cost: ${cost_stats['total_cost']:.4f}\n")
            f.write(f"  Cost per Query: ${cost_stats['cost_per_query']:.6f}\n")
            f.write(f"  Cost per 1000 Queries: ${cost_stats['cost_per_query'] * 1000:.2f}\n")
        f.write("\n")
        
        f.write("Retrieval Statistics:\n")
        if retrieval_stats:
            if 'num_chunks' in retrieval_stats:
                f.write(f"  Avg Chunks Retrieved: {retrieval_stats['num_chunks']['mean']:.1f}\n")
            if 'retrieval_latency' in retrieval_stats:
                f.write(f"  Avg Retrieval Latency: {retrieval_stats['retrieval_latency']['mean']:.3f}s\n")
        f.write("\n")
        
        f.write("Response Lengths:\n")
        if length_stats:
            if 'base' in length_stats:
                f.write(f"  Base Mean: {length_stats['base']['mean']:.1f} words\n")
            if 'enhanced' in length_stats:
                f.write(f"  Enhanced Mean: {length_stats['enhanced']['mean']:.1f} words\n")
                if 'base' in length_stats:
                    increase = ((length_stats['enhanced']['mean'] / length_stats['base']['mean']) - 1) * 100
                    f.write(f"  Increase: {increase:.1f}%\n")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

