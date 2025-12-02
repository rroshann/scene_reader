#!/usr/bin/env python3
"""
Analyze retrieval quality for RAG-Enhanced Vision
"""
import csv
from pathlib import Path
from collections import defaultdict


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def analyze_retrieval_quality(results):
    """Analyze retrieval quality metrics"""
    rag_results = [r for r in results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
    
    num_chunks = []
    games_identified = defaultdict(int)
    retrieval_success = 0
    
    for r in rag_results:
        # Count chunks retrieved
        try:
            if r.get('num_retrieved_chunks'):
                num_chunks.append(int(r['num_retrieved_chunks']))
                retrieval_success += 1
        except (ValueError, TypeError):
            continue
        
        # Count games identified
        game = r.get('game_name')
        if game:
            games_identified[game] += 1
    
    stats = {
        'total_rag_tests': len(rag_results),
        'retrieval_success_count': retrieval_success,
        'retrieval_success_rate': retrieval_success / len(rag_results) if rag_results else 0,
        'avg_chunks_retrieved': sum(num_chunks) / len(num_chunks) if num_chunks else 0,
        'games_identified': dict(games_identified)
    }
    
    return stats


def main():
    """Main analysis function"""
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    stats = analyze_retrieval_quality(results)
    
    output_dir = Path('results/approach_6_rag/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'retrieval_quality.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Retrieval Quality Analysis - Approach 6\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total RAG tests: {stats['total_rag_tests']}\n")
        f.write(f"Retrieval success count: {stats['retrieval_success_count']}\n")
        f.write(f"Retrieval success rate: {stats['retrieval_success_rate']*100:.1f}%\n")
        f.write(f"Average chunks retrieved: {stats['avg_chunks_retrieved']:.1f}\n\n")
        
        f.write("Games identified:\n")
        for game, count in stats['games_identified'].items():
            f.write(f"  {game}: {count} times\n")
    
    print(f"✅ Retrieval quality analysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

