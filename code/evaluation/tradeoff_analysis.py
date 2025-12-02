#!/usr/bin/env python3
"""
Tradeoff analysis: Latency vs Quality, Cost vs Quality
Creates scatter plots and identifies Pareto frontier
"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import statistics

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load latency, cost, and quality data"""
    # Load qualitative scores
    scores_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    raw_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    
    scores_data = {}
    with open(scores_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['filename'], row['model'])
            scores_data[key] = {
                'overall_score': float(row['overall_score']),
                'category': row['category']
            }
    
    # Load latency and cost data
    results = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True':
                    key = (row['filename'], row['model'])
                    if key in scores_data:
                        try:
                            latency = float(row['latency_seconds'])
                            results.append({
                                'model': row['model'],
                                'latency': latency,
                                'quality': scores_data[key]['overall_score'],
                                'category': scores_data[key]['category']
                            })
                        except (ValueError, TypeError):
                            continue
    
    return results

def calculate_costs():
    """Calculate cost per query for each model"""
    PRICING = {
        'GPT-4V': 0.0124,
        'Gemini': 0.0031,
        'Claude': 0.0240
    }
    return PRICING

def create_latency_vs_quality_plot(results, output_path):
    """Create latency vs quality scatter plot"""
    df = pd.DataFrame(results)
    
    # Calculate average latency and quality per model
    model_avg = df.groupby('model').agg({
        'latency': 'mean',
        'quality': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with different colors per model
    colors = {'GPT-4V': '#10a37f', 'Gemini': '#4285f4', 'Claude': '#d97757'}
    
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        model_data = df[df['model'] == model]
        plt.scatter(model_data['latency'], model_data['quality'], 
                   alpha=0.6, label=model, color=colors[model], s=50)
    
    # Add average points
    for _, row in model_avg.iterrows():
        plt.scatter(row['latency'], row['quality'], 
                   color=colors[row['model']], s=200, marker='*', 
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Quality Score (1-5)', fontsize=12)
    plt.title('Latency vs Quality Tradeoff - Approach 1 (VLMs)', fontsize=14, fontweight='bold')
    plt.legend(title='Model', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add efficiency lines (quality/latency)
    for _, row in model_avg.iterrows():
        efficiency = row['quality'] / row['latency']
        plt.text(row['latency'] + 0.3, row['quality'], 
                f'Efficiency: {efficiency:.2f}', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_cost_vs_quality_plot(results, output_path):
    """Create cost vs quality scatter plot"""
    df = pd.DataFrame(results)
    costs = calculate_costs()
    
    # Add cost column
    df['cost'] = df['model'].map(costs)
    
    # Calculate average cost and quality per model
    model_avg = df.groupby('model').agg({
        'cost': 'mean',
        'quality': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    colors = {'GPT-4V': '#10a37f', 'Gemini': '#4285f4', 'Claude': '#d97757'}
    
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        model_data = df[df['model'] == model]
        plt.scatter(model_data['cost'], model_data['quality'], 
                   alpha=0.6, label=model, color=colors[model], s=50)
    
    # Add average points
    for _, row in model_avg.iterrows():
        plt.scatter(row['cost'], row['quality'], 
                   color=colors[row['model']], s=200, marker='*', 
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    plt.xlabel('Cost per Query ($)', fontsize=12)
    plt.ylabel('Quality Score (1-5)', fontsize=12)
    plt.title('Cost vs Quality Tradeoff - Approach 1 (VLMs)', fontsize=14, fontweight='bold')
    plt.legend(title='Model', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add cost-effectiveness (quality/cost)
    for _, row in model_avg.iterrows():
        cost_effectiveness = row['quality'] / row['cost']
        plt.text(row['cost'] + 0.001, row['quality'], 
                f'Cost-Eff: {cost_effectiveness:.0f}', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_efficiency_comparison(results, output_path):
    """Create efficiency comparison (quality/latency and quality/cost)"""
    df = pd.DataFrame(results)
    costs = calculate_costs()
    df['cost'] = df['model'].map(costs)
    
    # Calculate efficiency metrics
    df['latency_efficiency'] = df['quality'] / df['latency']
    df['cost_efficiency'] = df['quality'] / df['cost']
    
    model_avg = df.groupby('model').agg({
        'latency_efficiency': 'mean',
        'cost_efficiency': 'mean',
        'quality': 'mean',
        'latency': 'mean',
        'cost': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'GPT-4V': '#10a37f', 'Gemini': '#4285f4', 'Claude': '#d97757'}
    
    # Latency efficiency
    models = model_avg['model'].tolist()
    latency_eff = model_avg['latency_efficiency'].tolist()
    ax1.bar(models, latency_eff, color=[colors[m] for m in models])
    ax1.set_ylabel('Quality / Latency (Efficiency)', fontsize=12)
    ax1.set_title('Latency Efficiency (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cost efficiency
    cost_eff = model_avg['cost_efficiency'].tolist()
    ax2.bar(models, cost_eff, color=[colors[m] for m in models])
    ax2.set_ylabel('Quality / Cost (Cost-Effectiveness)', fontsize=12)
    ax2.set_title('Cost Efficiency (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")
    
    # Print summary
    print("\n📊 Efficiency Summary:")
    for _, row in model_avg.iterrows():
        print(f"  {row['model']}:")
        print(f"    Latency Efficiency: {row['latency_efficiency']:.3f} (Quality/Latency)")
        print(f"    Cost Efficiency: {row['cost_efficiency']:.1f} (Quality/Cost)")

def main():
    """Run tradeoff analysis"""
    figures_dir = Path('results/approach_1_vlm/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRADEOFF ANALYSIS - Approach 1")
    print("=" * 60)
    print()
    
    results = load_data()
    print(f"📊 Loaded {len(results)} data points")
    print()
    
    create_latency_vs_quality_plot(results, figures_dir / 'latency_vs_quality_tradeoff.png')
    create_cost_vs_quality_plot(results, figures_dir / 'cost_vs_quality_tradeoff.png')
    create_efficiency_comparison(results, figures_dir / 'efficiency_comparison.png')
    
    print("\n✅ All tradeoff visualizations created!")

if __name__ == '__main__':
    main()

