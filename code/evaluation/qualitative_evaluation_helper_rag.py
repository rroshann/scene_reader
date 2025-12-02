#!/usr/bin/env python3
"""
Create qualitative evaluation template for Approach 6 RAG-Enhanced Vision
"""
import csv
from pathlib import Path


def create_evaluation_template():
    """Create CSV template for manual qualitative evaluation"""
    # Load results to get all configurations
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("Run batch testing first!")
        return
    
    results = []
    with open(results_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    
    # Filter to RAG-enhanced only (most interesting for comparison)
    rag_results = [r for r in results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
    
    # Sample subset for evaluation (or use all)
    # For now, use all RAG results
    evaluation_results = rag_results
    
    output_dir = Path('results/approach_6_rag/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'qualitative_scores.csv'
    
    fieldnames = [
        'filename',
        'vlm_model',
        'configuration',
        'base_description',
        'enhanced_description',
        'game_name',
        'accuracy',
        'completeness',
        'clarity',
        'actionability',
        'context_relevance',
        'educational_value',
        'notes'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in evaluation_results:
            writer.writerow({
                'filename': r.get('filename', ''),
                'vlm_model': r.get('vlm_model', ''),
                'configuration': r.get('configuration', ''),
                'base_description': r.get('base_description', '')[:500],  # Truncate for readability
                'enhanced_description': r.get('enhanced_description', '')[:500],
                'game_name': r.get('game_name', ''),
                'accuracy': '',  # To be filled manually (1-5)
                'completeness': '',  # To be filled manually (1-5)
                'clarity': '',  # To be filled manually (1-5)
                'actionability': '',  # To be filled manually (1-5)
                'context_relevance': '',  # RAG-specific (1-5)
                'educational_value': '',  # RAG-specific (1-5)
                'notes': ''  # Optional notes
            })
    
    print(f"✅ Created evaluation template: {output_file}")
    print(f"   Total entries: {len(evaluation_results)}")
    print("\nScoring guide:")
    print("  - Accuracy: How correct is the description? (1=Many errors, 5=Perfect)")
    print("  - Completeness: Does it mention all important elements? (1=Missing key info, 5=Complete)")
    print("  - Clarity: How easy to understand? (1=Confusing, 5=Crystal clear)")
    print("  - Actionability: How useful for gameplay decisions? (1=Not useful, 5=Highly actionable)")
    print("  - Context Relevance: How relevant is retrieved context? (1=Irrelevant, 5=Highly relevant)")
    print("  - Educational Value: Does it teach game mechanics? (1=No learning, 5=Very educational)")


if __name__ == "__main__":
    create_evaluation_template()

