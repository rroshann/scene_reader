#!/usr/bin/env python3
"""
Helper script to create qualitative evaluation template for Approach 4
Generates CSV template for manual scoring
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


def main():
    """Create evaluation template"""
    print("=" * 60)
    print("Qualitative Evaluation Helper - Approach 4 (Local Models)")
    print("=" * 60)
    print()
    
    # Load results
    csv_path = Path('results/approach_4_local/raw/batch_results.csv')
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    
    # Sample results (one per model per category for manageable evaluation)
    sampled = []
    seen = set()
    
    for r in results:
        key = (r.get('model'), r.get('category'))
        if key not in seen:
            seen.add(key)
            sampled.append(r)
    
    # Output directory
    output_dir = Path('results/approach_4_local/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'qualitative_scores.csv'
    
    # Create template
    fieldnames = [
        'filename',
        'category',
        'model',
        'description',
        'accuracy',
        'completeness',
        'clarity',
        'actionability',
        'notes'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in sampled:
            writer.writerow({
                'filename': r.get('filename', ''),
                'category': r.get('category', ''),
                'model': r.get('model', ''),
                'description': r.get('description', '')[:500],  # Truncate for CSV
                'accuracy': '',  # 1-5: How correct is the description?
                'completeness': '',  # 1-5: Does it mention all important elements?
                'clarity': '',  # 1-5: How easy to understand?
                'actionability': '',  # 1-5: How useful for decision-making?
                'notes': ''  # Any additional observations
            })
    
    print(f"✅ Created evaluation template: {output_file}")
    print(f"   Total entries: {len(sampled)}")
    print()
    print("Scoring guide:")
    print("  - Accuracy: How correct is the description? (1=Many errors, 5=Perfect)")
    print("  - Completeness: Does it mention all important elements? (1=Missing key info, 5=Complete)")
    print("  - Clarity: How easy to understand? (1=Confusing, 5=Crystal clear)")
    print("  - Actionability: How useful for gameplay decisions? (1=Not useful, 5=Highly actionable)")
    print()
    print("Fill in the scores and notes, then save the file.")


if __name__ == "__main__":
    main()

