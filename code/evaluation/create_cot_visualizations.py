#!/usr/bin/env python3
"""
Create visualizations for CoT results and comparison to baseline
"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import statistics
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_cot_results(csv_path):
    """Load CoT results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('prompt_type') == 'CoT' and row['success'] == 'True':
                results.append(row)
    return results

def load_baseline_results(csv_path):
    """Load baseline GPT-4V results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == 'GPT-4V' and '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True':
                    results.append(row)
    return results

def create_latency_comparison_boxplot(cot_results, baseline_results, output_path):
    """Create box plot comparing CoT vs baseline latency"""
    cot_latencies = []
    baseline_latencies = []
    
    # Match by filename
    baseline_by_file = {r['filename']: r for r in baseline_results}
    
    for r in cot_results:
        if r.get('latency_seconds'):
            try:
                lat = float(r['latency_seconds'])
                if lat < 30:  # Filter outliers
                    cot_latencies.append(lat)
                    
                    # Get matching baseline
                    if r['filename'] in baseline_by_file:
                        base = baseline_by_file[r['filename']]
                        if base.get('latency_seconds'):
                            try:
                                base_lat = float(base['latency_seconds'])
                                if base_lat < 30:
                                    baseline_latencies.append(base_lat)
                            except (ValueError, TypeError):
                                pass
            except (ValueError, TypeError):
                pass
    
    if not cot_latencies:
        print(f"⚠️  No latency data for CoT")
        return
    
    plt.figure(figsize=(10, 6))
    data = [baseline_latencies if baseline_latencies else [], cot_latencies]
    labels = ['Baseline GPT-4V', 'CoT GPT-4V']
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#e0e0e0')
    bp['boxes'][1].set_facecolor('#4285f4')
    
    plt.ylabel('Latency (seconds)')
    plt.title('Latency Comparison: CoT vs Baseline GPT-4V')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_response_length_comparison(cot_results, baseline_results, output_path):
    """Create bar chart comparing response lengths"""
    cot_words = []
    baseline_words = []
    
    baseline_by_file = {r['filename']: r for r in baseline_results}
    
    for r in cot_results:
        if r.get('description'):
            cot_words.append(len(r['description'].split()))
            
            if r['filename'] in baseline_by_file:
                base = baseline_by_file[r['filename']]
                if base.get('description'):
                    baseline_words.append(len(base['description'].split()))
    
    if not cot_words:
        print(f"⚠️  No description data for CoT")
        return
    
    avg_cot = statistics.mean(cot_words) if cot_words else 0
    avg_baseline = statistics.mean(baseline_words) if baseline_words else 0
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Baseline GPT-4V', 'CoT GPT-4V'], 
                   [avg_baseline, avg_cot],
                   color=['#e0e0e0', '#4285f4'])
    plt.ylabel('Average Word Count')
    plt.title('Response Length Comparison: CoT vs Baseline')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_quality_scores_comparison(output_path):
    """Create grouped bar chart comparing quality scores"""
    # Load qualitative scores if available
    cot_scores_path = Path('results/approach_7_cot/evaluation/qualitative_scores.csv')
    baseline_scores_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    
    if not cot_scores_path.exists() or not baseline_scores_path.exists():
        print(f"⚠️  Qualitative scores not found, skipping quality comparison")
        return
    
    cot_scores = pd.read_csv(cot_scores_path)
    baseline_scores = pd.read_csv(baseline_scores_path)
    baseline_gpt4v = baseline_scores[baseline_scores['model'] == 'GPT-4V']
    
    dimensions = ['completeness', 'clarity', 'conciseness', 'actionability', 'safety_focus']
    
    cot_avgs = [cot_scores[dim].mean() for dim in dimensions]
    baseline_avgs = [baseline_gpt4v[dim].mean() for dim in dimensions]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_avgs, width, label='Baseline GPT-4V', color='#e0e0e0')
    bars2 = ax.bar(x + width/2, cot_avgs, width, label='CoT GPT-4V', color='#4285f4')
    
    ax.set_xlabel('Quality Dimension')
    ax.set_ylabel('Average Score (1-5)')
    ax.set_title('Quality Scores Comparison: CoT vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions])
    ax.legend()
    ax.set_ylim([0, 5.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_safety_detection_comparison(output_path):
    """Create bar chart comparing safety detection"""
    # Load safety analysis if available
    cot_safety_path = Path('results/approach_7_cot/evaluation/safety_analysis.csv')
    baseline_safety_path = Path('results/approach_1_vlm/evaluation/safety_analysis.csv')
    
    if not cot_safety_path.exists() or not baseline_safety_path.exists():
        print(f"⚠️  Safety analysis not found, skipping safety comparison")
        return
    
    cot_safety = pd.read_csv(cot_safety_path)
    baseline_safety = pd.read_csv(baseline_safety_path)
    baseline_gpt4v = baseline_safety[baseline_safety['model'] == 'GPT-4V']
    
    metrics = {
        'Safety Score': {
            'cot': cot_safety['safety_score'].mean(),
            'baseline': baseline_gpt4v['safety_score'].mean()
        },
        'Hazards Detected': {
            'cot': cot_safety['hazards_detected_count'].mean(),
            'baseline': baseline_gpt4v['hazards_detected_count'].mean()
        }
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Safety Score
    ax1.bar(['Baseline', 'CoT'], 
            [metrics['Safety Score']['baseline'], metrics['Safety Score']['cot']],
            color=['#e0e0e0', '#4285f4'])
    ax1.set_ylabel('Average Safety Score (1-5)')
    ax1.set_title('Safety Score Comparison')
    ax1.set_ylim([0, 5.5])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(0, metrics['Safety Score']['baseline'], 
            f"{metrics['Safety Score']['baseline']:.2f}",
            ha='center', va='bottom')
    ax1.text(1, metrics['Safety Score']['cot'], 
            f"{metrics['Safety Score']['cot']:.2f}",
            ha='center', va='bottom')
    
    # Hazards Detected
    ax2.bar(['Baseline', 'CoT'], 
            [metrics['Hazards Detected']['baseline'], metrics['Hazards Detected']['cot']],
            color=['#e0e0e0', '#4285f4'])
    ax2.set_ylabel('Average Hazards Detected')
    ax2.set_title('Hazard Detection Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0, metrics['Hazards Detected']['baseline'], 
            f"{metrics['Hazards Detected']['baseline']:.1f}",
            ha='center', va='bottom')
    ax2.text(1, metrics['Hazards Detected']['cot'], 
            f"{metrics['Hazards Detected']['cot']:.1f}",
            ha='center', va='bottom')
    
    plt.suptitle('Safety Detection Comparison: CoT vs Baseline', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_latency_vs_quality_tradeoff(cot_results, baseline_results, output_path):
    """Create scatter plot: latency vs quality tradeoff"""
    # Load quality scores
    cot_scores_path = Path('results/approach_7_cot/evaluation/qualitative_scores.csv')
    baseline_scores_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    
    if not cot_scores_path.exists() or not baseline_scores_path.exists():
        print(f"⚠️  Quality scores not found, skipping tradeoff plot")
        return
    
    cot_scores = pd.read_csv(cot_scores_path)
    baseline_scores = pd.read_csv(baseline_scores_path)
    baseline_gpt4v = baseline_scores[baseline_scores['model'] == 'GPT-4V']
    
    # Match by filename
    cot_by_file = {r['filename']: r for r in cot_results}
    baseline_by_file = {r['filename']: r for r in baseline_results}
    
    cot_data = []
    baseline_data = []
    
    for _, row in cot_scores.iterrows():
        filename = row['filename']
        if filename in cot_by_file:
            cot = cot_by_file[filename]
            if cot.get('latency_seconds'):
                try:
                    lat = float(cot['latency_seconds'])
                    if lat < 30:
                        cot_data.append({
                            'latency': lat,
                            'quality': row['overall_score']
                        })
                except (ValueError, TypeError):
                    pass
    
    for _, row in baseline_gpt4v.iterrows():
        filename = row['filename']
        if filename in baseline_by_file:
            base = baseline_by_file[filename]
            if base.get('latency_seconds'):
                try:
                    lat = float(base['latency_seconds'])
                    if lat < 30:
                        baseline_data.append({
                            'latency': lat,
                            'quality': row['overall_score']
                        })
                except (ValueError, TypeError):
                    pass
    
    if not cot_data or not baseline_data:
        print(f"⚠️  Insufficient data for tradeoff plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    baseline_lats = [d['latency'] for d in baseline_data]
    baseline_quals = [d['quality'] for d in baseline_data]
    cot_lats = [d['latency'] for d in cot_data]
    cot_quals = [d['quality'] for d in cot_data]
    
    plt.scatter(baseline_lats, baseline_quals, alpha=0.6, label='Baseline GPT-4V', 
               color='#e0e0e0', s=50, edgecolors='black', linewidth=0.5)
    plt.scatter(cot_lats, cot_quals, alpha=0.6, label='CoT GPT-4V', 
               color='#4285f4', s=50, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Overall Quality Score')
    plt.title('Latency vs Quality Tradeoff: CoT vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def main():
    """Create all CoT visualizations"""
    cot_csv_path = Path('results/approach_7_cot/raw/batch_results.csv')
    baseline_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    figures_dir = Path('results/approach_7_cot/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not cot_csv_path.exists():
        print(f"❌ CoT results file not found: {cot_csv_path}")
        return
    
    if not baseline_csv_path.exists():
        print(f"⚠️  Baseline results file not found: {baseline_csv_path}")
        print("   Creating CoT-only visualizations...")
        baseline_results = []
    else:
        baseline_results = load_baseline_results(baseline_csv_path)
    
    print("=" * 60)
    print("CREATING CoT VISUALIZATIONS")
    print("=" * 60)
    print()
    
    cot_results = load_cot_results(cot_csv_path)
    print(f"📊 Loaded {len(cot_results)} CoT results")
    if baseline_results:
        print(f"📊 Loaded {len(baseline_results)} baseline results")
    print()
    
    # Create visualizations
    if baseline_results:
        create_latency_comparison_boxplot(cot_results, baseline_results, 
                                         figures_dir / 'latency_comparison.png')
        create_response_length_comparison(cot_results, baseline_results,
                                         figures_dir / 'response_length_comparison.png')
        create_quality_scores_comparison(figures_dir / 'quality_scores_comparison.png')
        create_safety_detection_comparison(figures_dir / 'safety_comparison.png')
        create_latency_vs_quality_tradeoff(cot_results, baseline_results,
                                          figures_dir / 'latency_vs_quality_tradeoff.png')
    else:
        print("⚠️  Skipping comparison visualizations (no baseline data)")
    
    print()
    print("✅ All CoT visualizations created!")

if __name__ == '__main__':
    main()

