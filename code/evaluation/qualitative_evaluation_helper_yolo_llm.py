#!/usr/bin/env python3
"""
Helper tool for manual qualitative evaluation of YOLO+LLM results
Creates CSV template for scoring descriptions
"""
import csv
from pathlib import Path


def create_evaluation_template():
    """Create CSV template for qualitative scoring"""
    csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    output_path = Path('results/approach_2_yolo_llm/evaluation/qualitative_scores.csv')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_yolo_llm.py first.")
        return
    
    # Load results
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    
    # Create template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename', 'category', 'configuration', 'yolo_model', 'llm_model',
            'description', 'completeness', 'clarity', 'conciseness', 
            'actionability', 'safety_focus', 'overall_score', 'notes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'filename': r.get('filename', ''),
                'category': r.get('category', ''),
                'configuration': r.get('configuration', ''),
                'yolo_model': r.get('yolo_model', ''),
                'llm_model': r.get('llm_model', ''),
                'description': r.get('description', ''),
                'completeness': '',  # 1-5 scale
                'clarity': '',  # 1-5 scale
                'conciseness': '',  # 1-5 scale
                'actionability': '',  # 1-5 scale
                'safety_focus': '',  # 1-5 scale
                'overall_score': '',  # 1-5 scale
                'notes': ''
            })
    
    print(f"✅ Created evaluation template: {output_path}")
    print(f"   Total descriptions to score: {len(results)}")
    print("\nScoring Guide (1-5 scale):")
    print("  Completeness: Does it cover all important elements?")
    print("  Clarity: Is it easy to understand?")
    print("  Conciseness: Is it brief and to the point?")
    print("  Actionability: Can the user act on this information?")
    print("  Safety Focus: Does it prioritize safety-critical info?")
    print("  Overall: Overall quality score")


if __name__ == "__main__":
    create_evaluation_template()

