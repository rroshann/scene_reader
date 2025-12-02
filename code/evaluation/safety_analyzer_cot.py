#!/usr/bin/env python3
"""
Automated safety-critical error analysis for CoT descriptions
Compares CoT vs baseline GPT-4V on navigation images
"""
import csv
from pathlib import Path
from collections import defaultdict

# Import safety analysis function from safety_analyzer
import sys
sys.path.append(str(Path(__file__).parent))
from safety_analyzer import analyze_safety

def analyze_cot_safety():
    """Analyze safety for CoT descriptions and compare to baseline"""
    cot_csv_path = Path('results/approach_7_cot/raw/batch_results.csv')
    baseline_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    output_path = Path('results/approach_7_cot/evaluation/safety_analysis.csv')
    
    # Load CoT results (navigation only)
    cot_results = []
    with open(cot_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('prompt_type') == 'CoT' and row['success'] == 'True' and 
                row.get('description') and row['category'] in ['indoor', 'outdoor']):
                cot_results.append(row)
    
    # Load baseline results
    baseline_results = []
    baseline_by_file = {}
    if baseline_csv_path.exists():
        with open(baseline_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row['model'] == 'GPT-4V' and '2025-11-22T20:' in row.get('timestamp', '') and
                    row['success'] == 'True' and row.get('description') and 
                    row['category'] in ['indoor', 'outdoor']):
                    baseline_results.append(row)
                    baseline_by_file[row['filename']] = row
    
    print(f"Analyzing {len(cot_results)} CoT navigation descriptions...")
    if baseline_results:
        print(f"Comparing to {len(baseline_results)} baseline GPT-4V descriptions...")
    
    safety_results = []
    comparison_data = []
    
    for r in cot_results:
        description = r['description']
        category = r['category']
        filename = r['filename']
        
        # Analyze CoT safety
        safety_data = analyze_safety(description, category, filename)
        
        safety_results.append({
            'filename': filename,
            'category': category,
            'model': 'GPT-4V-CoT',
            'prompt_type': 'CoT',
            'description': description[:200] + '...' if len(description) > 200 else description,
            'hazards_mentioned': safety_data['hazards_mentioned'],
            'hazards_detected_count': safety_data['hazards_detected_count'],
            'false_negative': safety_data['false_negative'],
            'safety_score': safety_data['safety_score'],
            'notes': safety_data['notes']
        })
        
        # Compare to baseline if available
        if filename in baseline_by_file:
            baseline = baseline_by_file[filename]
            baseline_desc = baseline['description']
            
            baseline_safety = analyze_safety(baseline_desc, category, filename)
            
            comparison_data.append({
                'filename': filename,
                'category': category,
                'cot_hazards_count': safety_data['hazards_detected_count'],
                'baseline_hazards_count': baseline_safety['hazards_detected_count'],
                'cot_safety_score': safety_data['safety_score'],
                'baseline_safety_score': baseline_safety['safety_score'],
                'cot_false_negative': safety_data['false_negative'],
                'baseline_false_negative': baseline_safety['false_negative'],
                'hazards_improvement': safety_data['hazards_detected_count'] - baseline_safety['hazards_detected_count'],
                'safety_score_improvement': safety_data['safety_score'] - baseline_safety['safety_score']
            })
    
    # Save CoT safety results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'category', 'model', 'prompt_type', 'description', 'hazards_mentioned', 
                     'hazards_detected_count', 'false_negative', 'safety_score', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safety_results)
    
    print(f"✅ Saved {len(safety_results)} CoT safety analyses to {output_path}")
    
    # Print CoT summary
    print("\n📊 CoT Safety Summary:")
    avg_score = sum(r['safety_score'] for r in safety_results) / len(safety_results)
    false_negatives = sum(1 for r in safety_results if r['false_negative'] == 'Y')
    avg_hazards = sum(r['hazards_detected_count'] for r in safety_results) / len(safety_results)
    print(f"  Avg Safety Score: {avg_score:.2f}")
    print(f"  False Negatives: {false_negatives}/{len(safety_results)} ({false_negatives/len(safety_results)*100:.1f}%)")
    print(f"  Avg Hazards Detected: {avg_hazards:.1f}")
    
    # By category
    print("\n📊 CoT Safety by Category:")
    for category in ['indoor', 'outdoor']:
        cat_results = [r for r in safety_results if r['category'] == category]
        if cat_results:
            cat_score = sum(r['safety_score'] for r in cat_results) / len(cat_results)
            cat_fn = sum(1 for r in cat_results if r['false_negative'] == 'Y')
            cat_hazards = sum(r['hazards_detected_count'] for r in cat_results) / len(cat_results)
            print(f"  {category.capitalize()}: Score {cat_score:.2f}, FN {cat_fn}/{len(cat_results)}, Hazards {cat_hazards:.1f}")
    
    # Comparison summary
    if comparison_data:
        print("\n📊 Comparison to Baseline:")
        avg_cot_score = sum(c['cot_safety_score'] for c in comparison_data) / len(comparison_data)
        avg_baseline_score = sum(c['baseline_safety_score'] for c in comparison_data) / len(comparison_data)
        avg_score_improvement = sum(c['safety_score_improvement'] for c in comparison_data) / len(comparison_data)
        
        avg_cot_hazards = sum(c['cot_hazards_count'] for c in comparison_data) / len(comparison_data)
        avg_baseline_hazards = sum(c['baseline_hazards_count'] for c in comparison_data) / len(comparison_data)
        avg_hazards_improvement = sum(c['hazards_improvement'] for c in comparison_data) / len(comparison_data)
        
        cot_fn = sum(1 for c in comparison_data if c['cot_false_negative'] == 'Y')
        baseline_fn = sum(1 for c in comparison_data if c['baseline_false_negative'] == 'Y')
        
        print(f"  Safety Score:")
        print(f"    CoT: {avg_cot_score:.2f}")
        print(f"    Baseline: {avg_baseline_score:.2f}")
        print(f"    Improvement: {avg_score_improvement:+.2f} ({avg_score_improvement/avg_baseline_score*100:+.1f}%)")
        
        print(f"  Hazards Detected:")
        print(f"    CoT: {avg_cot_hazards:.1f}")
        print(f"    Baseline: {avg_baseline_hazards:.1f}")
        print(f"    Improvement: {avg_hazards_improvement:+.1f} ({avg_hazards_improvement/avg_baseline_hazards*100:+.1f}%)")
        
        print(f"  False Negatives:")
        print(f"    CoT: {cot_fn}/{len(comparison_data)} ({cot_fn/len(comparison_data)*100:.1f}%)")
        print(f"    Baseline: {baseline_fn}/{len(comparison_data)} ({baseline_fn/len(comparison_data)*100:.1f}%)")
        print(f"    Improvement: {baseline_fn - cot_fn} fewer false negatives")
        
        # Save comparison
        comparison_path = Path('results/approach_7_cot/evaluation/safety_comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write("CoT vs Baseline GPT-4V Safety Comparison\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Common Navigation Images: {len(comparison_data)}\n\n")
            
            f.write("Overall Safety Metrics:\n")
            f.write(f"  CoT Safety Score: {avg_cot_score:.2f}\n")
            f.write(f"  Baseline Safety Score: {avg_baseline_score:.2f}\n")
            f.write(f"  Improvement: {avg_score_improvement:+.2f} ({avg_score_improvement/avg_baseline_score*100:+.1f}%)\n\n")
            
            f.write("Hazard Detection:\n")
            f.write(f"  CoT Avg Hazards: {avg_cot_hazards:.1f}\n")
            f.write(f"  Baseline Avg Hazards: {avg_baseline_hazards:.1f}\n")
            f.write(f"  Improvement: {avg_hazards_improvement:+.1f} ({avg_hazards_improvement/avg_baseline_hazards*100:+.1f}%)\n\n")
            
            f.write("False Negatives:\n")
            f.write(f"  CoT: {cot_fn}/{len(comparison_data)} ({cot_fn/len(comparison_data)*100:.1f}%)\n")
            f.write(f"  Baseline: {baseline_fn}/{len(comparison_data)} ({baseline_fn/len(comparison_data)*100:.1f}%)\n")
            f.write(f"  Improvement: {baseline_fn - cot_fn} fewer false negatives\n\n")
            
            f.write("By Category:\n")
            for category in ['indoor', 'outdoor']:
                cat_comps = [c for c in comparison_data if c['category'] == category]
                if cat_comps:
                    cat_cot_score = sum(c['cot_safety_score'] for c in cat_comps) / len(cat_comps)
                    cat_base_score = sum(c['baseline_safety_score'] for c in cat_comps) / len(cat_comps)
                    cat_cot_hazards = sum(c['cot_hazards_count'] for c in cat_comps) / len(cat_comps)
                    cat_base_hazards = sum(c['baseline_hazards_count'] for c in cat_comps) / len(cat_comps)
                    cat_cot_fn = sum(1 for c in cat_comps if c['cot_false_negative'] == 'Y')
                    cat_base_fn = sum(1 for c in cat_comps if c['baseline_false_negative'] == 'Y')
                    
                    f.write(f"  {category.capitalize()}:\n")
                    f.write(f"    Safety Score: {cat_cot_score:.2f} vs {cat_base_score:.2f} ({cat_cot_score - cat_base_score:+.2f})\n")
                    f.write(f"    Hazards: {cat_cot_hazards:.1f} vs {cat_base_hazards:.1f} ({cat_cot_hazards - cat_base_hazards:+.1f})\n")
                    f.write(f"    False Negatives: {cat_cot_fn}/{len(cat_comps)} vs {cat_base_fn}/{len(cat_comps)}\n")
        
        print(f"\n✅ Comparison saved to: {comparison_path}")
    
    return safety_results, comparison_data

if __name__ == '__main__':
    analyze_cot_safety()

