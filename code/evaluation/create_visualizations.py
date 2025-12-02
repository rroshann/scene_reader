#!/usr/bin/env python3
"""
Create visualizations for VLM results
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

def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '2025-11-22T20:' in row.get('timestamp', ''):
                results.append(row)
    return results

def create_latency_boxplot(results, output_path):
    """Create box plot of latency distributions"""
    data = []
    labels = []
    
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        latencies = []
        for r in results:
            if r['model'] == model and r['success'] == 'True' and r.get('latency_seconds'):
                try:
                    lat = float(r['latency_seconds'])
                    # Filter outliers for better visualization (keep < 20s)
                    if lat < 20:
                        latencies.append(lat)
                except (ValueError, TypeError):
                    continue
        if latencies:
            data.append(latencies)
            labels.append(model)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Latency (seconds)')
    plt.title('Latency Distribution by Model (Approach 1: VLMs)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_latency_by_scenario(results, output_path):
    """Create bar chart of latency by scenario"""
    models = ['GPT-4V', 'Gemini', 'Claude']
    scenarios = ['gaming', 'indoor', 'outdoor', 'text']
    
    data = {model: [] for model in models}
    
    for scenario in scenarios:
        for model in models:
            latencies = []
            for r in results:
                if (r['model'] == model and r['category'] == scenario and 
                    r['success'] == 'True' and r.get('latency_seconds')):
                    try:
                        lat = float(r['latency_seconds'])
                        if lat < 20:  # Filter outliers
                            latencies.append(lat)
                    except (ValueError, TypeError):
                        continue
            if latencies:
                data[model].append(statistics.mean(latencies))
            else:
                data[model].append(0)
    
    x = range(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        ax.bar([xi + i*width for xi in x], data[model], width, label=model)
    
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Average Latency (seconds)')
    ax.set_title('Average Latency by Scenario and Model')
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_response_length_comparison(results, output_path):
    """Create bar chart comparing response lengths"""
    models = ['GPT-4V', 'Gemini', 'Claude']
    word_counts = {model: [] for model in models}
    
    for model in models:
        for r in results:
            if r['model'] == model and r['success'] == 'True' and r.get('description'):
                word_count = len(r['description'].split())
                word_counts[model].append(word_count)
    
    avg_words = {model: statistics.mean(word_counts[model]) if word_counts[model] else 0 
                 for model in models}
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_words.keys(), avg_words.values(), color=['#10a37f', '#4285f4', '#d97757'])
    plt.ylabel('Average Word Count')
    plt.title('Average Response Length by Model')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def create_cost_comparison(results, output_path):
    """Create bar chart comparing costs"""
    # Pricing
    PRICING = {
        'GPT-4V': {'image': 0.00765, 'input_per_1k': 0.0025, 'output_per_1k': 0.01},
        'Gemini': {'image': 0.00315, 'input_per_1k': 0.00125, 'output_per_1k': 0.005},
        'Claude': {'image': 0.024, 'input_per_1k': 0.003, 'output_per_1k': 0.015}
    }
    
    models = ['GPT-4V', 'Gemini', 'Claude']
    costs_per_query = {}
    
    for model in models:
        total_cost = 0
        count = 0
        for r in results:
            if r['model'] == model and r['success'] == 'True':
                count += 1
                cost = PRICING[model]['image']
                if r.get('tokens_used'):
                    try:
                        tokens = int(r['tokens_used'])
                        input_tokens = int(tokens * 0.7)
                        output_tokens = int(tokens * 0.3)
                        cost += (input_tokens / 1000) * PRICING[model]['input_per_1k']
                        cost += (output_tokens / 1000) * PRICING[model]['output_per_1k']
                    except (ValueError, TypeError):
                        pass
                total_cost += cost
        costs_per_query[model] = total_cost / count if count > 0 else 0
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(costs_per_query.keys(), costs_per_query.values(), 
                   color=['#10a37f', '#4285f4', '#d97757'])
    plt.ylabel('Cost per Query ($)')
    plt.title('Cost per Query by Model')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created: {output_path}")

def main():
    """Create all visualizations"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    figures_dir = Path('results/approach_1_vlm/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    print()
    
    results = load_results(csv_path)
    print(f"📊 Loaded {len(results)} results")
    print()
    
    # Create visualizations
    create_latency_boxplot(results, figures_dir / 'latency_comparison.png')
    create_latency_by_scenario(results, figures_dir / 'latency_by_scenario.png')
    create_response_length_comparison(results, figures_dir / 'response_length_comparison.png')
    create_cost_comparison(results, figures_dir / 'cost_comparison.png')
    
    print()
    print("✅ All visualizations created!")

if __name__ == '__main__':
    main()

