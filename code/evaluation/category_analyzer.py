#!/usr/bin/env python3
"""
Category-specific performance analysis
Identifies best performers per category
"""
import csv
from pathlib import Path
from collections import defaultdict

def analyze_by_category():
    """Analyze performance by category"""
    csv_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    
    if not csv_path.exists():
        print("❌ Qualitative scores not found. Run qualitative_scorer.py first.")
        return
    
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    print("=" * 60)
    print("CATEGORY-SPECIFIC PERFORMANCE ANALYSIS")
    print("=" * 60)
    print()
    
    categories = ['gaming', 'indoor', 'outdoor', 'text']
    models = ['GPT-4V', 'Gemini', 'Claude']
    
    category_summary = {}
    
    for category in categories:
        print(f"📊 {category.upper()} SCENARIO")
        print("-" * 60)
        
        category_results = [r for r in results if r['category'] == category]
        
        model_scores = defaultdict(list)
        model_metrics = defaultdict(lambda: {
            'completeness': [], 'clarity': [], 'conciseness': [],
            'actionability': [], 'safety_focus': [], 'overall': []
        })
        
        for r in category_results:
            model = r['model']
            model_scores[model].append(float(r['overall_score']))
            for metric in ['completeness', 'clarity', 'conciseness', 'actionability', 'safety_focus', 'overall_score']:
                model_metrics[model][metric.replace('_score', '')].append(float(r[metric]))
        
        category_best = {}
        
        for model in models:
            if model_scores[model]:
                avg_overall = sum(model_scores[model]) / len(model_scores[model])
                avg_completeness = sum(model_metrics[model]['completeness']) / len(model_metrics[model]['completeness'])
                avg_clarity = sum(model_metrics[model]['clarity']) / len(model_metrics[model]['clarity'])
                avg_actionability = sum(model_metrics[model]['actionability']) / len(model_metrics[model]['actionability'])
                
                print(f"\n{model}:")
                print(f"  Overall Score: {avg_overall:.2f} (n={len(model_scores[model])})")
                print(f"  Completeness: {avg_completeness:.2f}")
                print(f"  Clarity: {avg_clarity:.2f}")
                print(f"  Actionability: {avg_actionability:.2f}")
                
                category_best[model] = avg_overall
        
        # Find best performer
        if category_best:
            best_model = max(category_best.items(), key=lambda x: x[1])
            print(f"\n🏆 Best Performer: {best_model[0]} ({best_model[1]:.2f})")
            category_summary[category] = {
                'best_model': best_model[0],
                'best_score': best_model[1],
                'scores': category_best
            }
        
        print()
    
    # Summary table
    print("=" * 60)
    print("SUMMARY BY CATEGORY")
    print("=" * 60)
    print()
    print(f"{'Category':<15} {'Best Model':<15} {'Score':<10} {'GPT-4V':<10} {'Gemini':<10} {'Claude':<10}")
    print("-" * 70)
    
    for category in categories:
        if category in category_summary:
            summary = category_summary[category]
            scores = summary['scores']
            print(f"{category.capitalize():<15} {summary['best_model']:<15} {summary['best_score']:<10.2f} "
                  f"{scores.get('GPT-4V', 0):<10.2f} {scores.get('Gemini', 0):<10.2f} {scores.get('Claude', 0):<10.2f}")
    
    # Save summary
    summary_path = Path('results/approach_1_vlm/analysis/category_performance.txt')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("CATEGORY-SPECIFIC PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for category, summary in category_summary.items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Best Model: {summary['best_model']} ({summary['best_score']:.2f})\n")
            f.write(f"  Scores: {summary['scores']}\n\n")
    
    print(f"\n✅ Summary saved to: {summary_path}")
    
    return category_summary

if __name__ == '__main__':
    analyze_by_category()

