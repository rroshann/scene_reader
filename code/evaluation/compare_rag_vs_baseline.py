#!/usr/bin/env python3
"""
Compare Approach 6 (RAG) with Approach 1 (Pure VLM) on gaming subset
"""
import csv
from pathlib import Path
from collections import defaultdict


def load_approach1_results():
    """Load Approach 1 results (gaming subset only)"""
    results_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    if not results_path.exists():
        return []
    
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter to gaming images only
            if row.get('category') == 'gaming' and (row.get('success') == 'True' or row.get('success') is True):
                results.append(row)
    
    return results


def load_approach6_results():
    """Load Approach 6 results"""
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    if not results_path.exists():
        return []
    
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    
    return results


def calculate_stats(results, latency_key='total_latency'):
    """Calculate basic statistics"""
    latencies = []
    for r in results:
        try:
            latency = float(r.get(latency_key, 0))
            if latency > 0:
                latencies.append(latency)
        except (ValueError, TypeError):
            continue
    
    if not latencies:
        return None
    
    return {
        'mean': sum(latencies) / len(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'count': len(latencies)
    }


def main():
    """Main comparison function"""
    approach1_results = load_approach1_results()
    approach6_results = load_approach6_results()
    
    if not approach1_results:
        print("⚠️  Approach 1 results not found or no gaming results")
        return
    
    if not approach6_results:
        print("⚠️  Approach 6 results not found")
        return
    
    output_dir = Path('results/approach_6_rag/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'rag_vs_baseline_comparison.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Approach 6 (RAG-Enhanced) vs Approach 1 (Pure VLM) Comparison\n")
        f.write("=" * 60 + "\n\n")
        f.write("Note: Comparison on gaming images only (12 images)\n\n")
        
        # Approach 1 stats
        approach1_stats = calculate_stats(approach1_results)
        if approach1_stats:
            f.write("Approach 1 (Pure VLM) - Gaming Subset:\n")
            f.write(f"  Mean latency: {approach1_stats['mean']:.3f}s\n")
            f.write(f"  Count: {approach1_stats['count']}\n\n")
        
        # Approach 6 - Base VLM (for fair comparison)
        approach6_base = [r for r in approach6_results if r.get('use_rag') == 'False' or r.get('use_rag') is False]
        approach6_base_stats = calculate_stats(approach6_base)
        if approach6_base_stats:
            f.write("Approach 6 - Base VLM (no RAG):\n")
            f.write(f"  Mean latency: {approach6_base_stats['mean']:.3f}s\n")
            f.write(f"  Count: {approach6_base_stats['count']}\n\n")
        
        # Approach 6 - RAG-Enhanced
        approach6_rag = [r for r in approach6_results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
        approach6_rag_stats = calculate_stats(approach6_rag)
        if approach6_rag_stats:
            f.write("Approach 6 - RAG-Enhanced:\n")
            f.write(f"  Mean latency: {approach6_rag_stats['mean']:.3f}s\n")
            f.write(f"  Count: {approach6_rag_stats['count']}\n\n")
        
        # Comparison
        if approach1_stats and approach6_rag_stats:
            latency_increase = ((approach6_rag_stats['mean'] / approach1_stats['mean']) - 1) * 100
            f.write("Comparison:\n")
            f.write(f"  Latency increase (RAG vs Pure VLM): {latency_increase:.1f}%\n")
            f.write(f"  RAG is {approach6_rag_stats['mean'] / approach1_stats['mean']:.2f}x slower\n")
            f.write(f"  But provides context-aware, educational descriptions\n")
    
    print(f"✅ Comparison complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

