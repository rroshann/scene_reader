#!/usr/bin/env python3
"""
Automated qualitative evaluation of CoT descriptions
Scores descriptions on 5 dimensions and compares to baseline GPT-4V
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

# Import scoring functions from qualitative_scorer
import sys
sys.path.append(str(Path(__file__).parent))
from qualitative_scorer import (
    score_completeness, score_clarity, score_conciseness,
    score_actionability, score_safety_focus, calculate_overall_score
)

def evaluate_cot_descriptions():
    """Evaluate all CoT descriptions"""
    cot_csv_path = Path('results/approach_7_cot/raw/batch_results.csv')
    baseline_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    output_path = Path('results/approach_7_cot/evaluation/qualitative_scores.csv')
    
    # Load CoT results
    cot_results = []
    with open(cot_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('prompt_type') == 'CoT' and row['success'] == 'True' and row.get('description'):
                cot_results.append(row)
    
    # Load baseline results
    baseline_results = []
    baseline_by_file = {}
    if baseline_csv_path.exists():
        with open(baseline_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['model'] == 'GPT-4V' and '2025-11-22T20:' in row.get('timestamp', ''):
                    if row['success'] == 'True' and row.get('description'):
                        baseline_results.append(row)
                        baseline_by_file[row['filename']] = row
    
    print(f"Evaluating {len(cot_results)} CoT descriptions...")
    if baseline_results:
        print(f"Comparing to {len(baseline_results)} baseline GPT-4V descriptions...")
    
    scored_results = []
    comparison_data = []
    
    for r in cot_results:
        description = r['description']
        category = r['category']
        filename = r['filename']
        
        # Score CoT description
        scores = {
            'completeness': score_completeness(description, category, filename),
            'clarity': score_clarity(description),
            'conciseness': score_conciseness(description),
            'actionability': score_actionability(description, category),
            'safety_focus': score_safety_focus(description, category)
        }
        
        overall = calculate_overall_score(scores)
        
        scored_results.append({
            'filename': filename,
            'category': category,
            'model': 'GPT-4V-CoT',
            'prompt_type': 'CoT',
            'completeness': scores['completeness'],
            'clarity': scores['clarity'],
            'conciseness': scores['conciseness'],
            'actionability': scores['actionability'],
            'safety_focus': scores['safety_focus'],
            'overall_score': overall,
            'notes': ''
        })
        
        # Compare to baseline if available
        if filename in baseline_by_file:
            baseline = baseline_by_file[filename]
            baseline_desc = baseline['description']
            
            baseline_scores = {
                'completeness': score_completeness(baseline_desc, category, filename),
                'clarity': score_clarity(baseline_desc),
                'conciseness': score_conciseness(baseline_desc),
                'actionability': score_actionability(baseline_desc, category),
                'safety_focus': score_safety_focus(baseline_desc, category)
            }
            
            baseline_overall = calculate_overall_score(baseline_scores)
            
            comparison_data.append({
                'filename': filename,
                'category': category,
                'cot_completeness': scores['completeness'],
                'baseline_completeness': baseline_scores['completeness'],
                'cot_clarity': scores['clarity'],
                'baseline_clarity': baseline_scores['clarity'],
                'cot_conciseness': scores['conciseness'],
                'baseline_conciseness': baseline_scores['conciseness'],
                'cot_actionability': scores['actionability'],
                'baseline_actionability': baseline_scores['actionability'],
                'cot_safety_focus': scores['safety_focus'],
                'baseline_safety_focus': baseline_scores['safety_focus'],
                'cot_overall': overall,
                'baseline_overall': baseline_overall,
                'improvement': overall - baseline_overall
            })
    
    # Save CoT scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'category', 'model', 'prompt_type', 'completeness', 'clarity', 
                     'conciseness', 'actionability', 'safety_focus', 'overall_score', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_results)
    
    print(f"✅ Saved {len(scored_results)} scored CoT descriptions to {output_path}")
    
    # Print summary statistics
    print("\n📊 CoT Summary Statistics:")
    cot_scores = [r['overall_score'] for r in scored_results]
    avg_cot = sum(cot_scores) / len(cot_scores) if cot_scores else 0
    print(f"  CoT (GPT-4V): {avg_cot:.2f} avg overall score (n={len(cot_scores)})")
    
    # Print by category
    print("\n📊 CoT Scores by Category:")
    for category in ['gaming', 'indoor', 'outdoor', 'text']:
        cat_scores = [r['overall_score'] for r in scored_results if r['category'] == category]
        if cat_scores:
            avg = sum(cat_scores) / len(cat_scores)
            print(f"  {category.capitalize()}: {avg:.2f} avg (n={len(cat_scores)})")
    
    # Comparison summary
    if comparison_data:
        print("\n📊 Comparison to Baseline:")
        avg_cot_overall = sum(c['cot_overall'] for c in comparison_data) / len(comparison_data)
        avg_baseline_overall = sum(c['baseline_overall'] for c in comparison_data) / len(comparison_data)
        avg_improvement = sum(c['improvement'] for c in comparison_data) / len(comparison_data)
        
        print(f"  CoT Overall: {avg_cot_overall:.2f}")
        print(f"  Baseline Overall: {avg_baseline_overall:.2f}")
        print(f"  Improvement: {avg_improvement:+.2f} ({avg_improvement/avg_baseline_overall*100:+.1f}%)")
        
        # By dimension
        print("\n  Improvement by Dimension:")
        dimensions = ['completeness', 'clarity', 'conciseness', 'actionability', 'safety_focus']
        for dim in dimensions:
            cot_avg = sum(c[f'cot_{dim}'] for c in comparison_data) / len(comparison_data)
            base_avg = sum(c[f'baseline_{dim}'] for c in comparison_data) / len(comparison_data)
            improvement = cot_avg - base_avg
            print(f"    {dim.capitalize()}: {improvement:+.2f} ({cot_avg:.2f} vs {base_avg:.2f})")
        
        # Save comparison
        comparison_path = Path('results/approach_7_cot/evaluation/cot_vs_baseline_quality.txt')
        with open(comparison_path, 'w') as f:
            f.write("CoT vs Baseline GPT-4V Quality Comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Common Images: {len(comparison_data)}\n\n")
            f.write(f"Overall Scores:\n")
            f.write(f"  CoT: {avg_cot_overall:.2f}\n")
            f.write(f"  Baseline: {avg_baseline_overall:.2f}\n")
            f.write(f"  Improvement: {avg_improvement:+.2f} ({avg_improvement/avg_baseline_overall*100:+.1f}%)\n\n")
            
            f.write("By Dimension:\n")
            for dim in dimensions:
                cot_avg = sum(c[f'cot_{dim}'] for c in comparison_data) / len(comparison_data)
                base_avg = sum(c[f'baseline_{dim}'] for c in comparison_data) / len(comparison_data)
                improvement = cot_avg - base_avg
                f.write(f"  {dim.capitalize()}: {improvement:+.2f} ({cot_avg:.2f} vs {base_avg:.2f})\n")
            
            f.write("\nBy Category:\n")
            for category in ['gaming', 'indoor', 'outdoor', 'text']:
                cat_comps = [c for c in comparison_data if c['category'] == category]
                if cat_comps:
                    cat_cot = sum(c['cot_overall'] for c in cat_comps) / len(cat_comps)
                    cat_base = sum(c['baseline_overall'] for c in cat_comps) / len(cat_comps)
                    cat_imp = cat_cot - cat_base
                    f.write(f"  {category.capitalize()}: {cat_imp:+.2f} ({cat_cot:.2f} vs {cat_base:.2f}, n={len(cat_comps)})\n")
        
        print(f"\n✅ Comparison saved to: {comparison_path}")
    
    return scored_results, comparison_data

if __name__ == '__main__':
    evaluate_cot_descriptions()

