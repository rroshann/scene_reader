#!/usr/bin/env python3
"""
Create visualizations for YOLO+LLM results (Approach 2)
"""
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def plot_latency_comparison(results, output_dir):
    """Plot latency comparison across configurations"""
    configs = []
    total_latencies = []
    detection_latencies = []
    generation_latencies = []
    
    for r in results:
        config = r.get('configuration', 'Unknown')
        try:
            total = float(r.get('total_latency', 0))
            detection = float(r.get('detection_latency', 0))
            generation = float(r.get('generation_latency', 0))
            
            if total > 0:
                configs.append(config)
                total_latencies.append(total)
                detection_latencies.append(detection)
                generation_latencies.append(generation)
        except (ValueError, TypeError):
            continue
    
    if not configs:
        print("No valid latency data found")
        return
    
    df = pd.DataFrame({
        'Configuration': configs,
        'Total Latency': total_latencies,
        'Detection': detection_latencies,
        'Generation': generation_latencies
    })
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df['Configuration'].unique()))
    width = 0.25
    
    configs_unique = sorted(df['Configuration'].unique())
    detection_means = [df[df['Configuration'] == c]['Detection'].mean() for c in configs_unique]
    generation_means = [df[df['Configuration'] == c]['Generation'].mean() for c in configs_unique]
    total_means = [df[df['Configuration'] == c]['Total Latency'].mean() for c in configs_unique]
    
    ax.bar(x - width, detection_means, width, label='Detection (YOLO)', color='#3498db')
    ax.bar(x, generation_means, width, label='Generation (LLM)', color='#e74c3c')
    ax.bar(x + width, total_means, width, label='Total', color='#2ecc71')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency Breakdown by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs_unique, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_latency_by_yolo_variant(results, output_dir):
    """Plot latency by YOLO variant"""
    yolo_variants = []
    detection_latencies = []
    
    for r in results:
        yolo_model = r.get('yolo_model', '')
        try:
            detection = float(r.get('detection_latency', 0))
            if detection > 0:
                yolo_variants.append(yolo_model)
                detection_latencies.append(detection)
        except (ValueError, TypeError):
            continue
    
    if not yolo_variants:
        return
    
    df = pd.DataFrame({
        'YOLO Variant': yolo_variants,
        'Detection Latency': detection_latencies
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='YOLO Variant', y='Detection Latency', ax=ax)
    ax.set_title('Detection Latency by YOLO Variant', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_by_yolo_variant.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_latency_by_llm_model(results, output_dir):
    """Plot latency by LLM model"""
    llm_models = []
    generation_latencies = []
    
    for r in results:
        llm_model = r.get('llm_model', '')
        try:
            generation = float(r.get('generation_latency', 0))
            if generation > 0:
                llm_models.append(llm_model)
                generation_latencies.append(generation)
        except (ValueError, TypeError):
            continue
    
    if not llm_models:
        return
    
    df = pd.DataFrame({
        'LLM Model': llm_models,
        'Generation Latency': generation_latencies
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='LLM Model', y='Generation Latency', ax=ax)
    ax.set_title('Generation Latency by LLM Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_by_llm_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_object_detection_stats(results, output_dir):
    """Plot object detection statistics"""
    configs = []
    num_objects = []
    confidences = []
    
    for r in results:
        config = r.get('configuration', 'Unknown')
        try:
            num_obj = int(r.get('num_objects_detected', 0))
            conf = float(r.get('avg_confidence', 0))
            
            if num_obj >= 0:
                configs.append(config)
                num_objects.append(num_obj)
                confidences.append(conf)
        except (ValueError, TypeError):
            continue
    
    if not configs:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    df = pd.DataFrame({
        'Configuration': configs,
        'Objects Detected': num_objects,
        'Confidence': confidences
    })
    
    # Objects detected
    configs_unique = sorted(df['Configuration'].unique())
    obj_means = [df[df['Configuration'] == c]['Objects Detected'].mean() for c in configs_unique]
    
    ax1.bar(configs_unique, obj_means, color='#9b59b6')
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Average Objects Detected', fontsize=12)
    ax1.set_title('Average Objects Detected by Configuration', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(configs_unique, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Confidence scores
    conf_means = [df[df['Configuration'] == c]['Confidence'].mean() for c in configs_unique]
    
    ax2.bar(configs_unique, conf_means, color='#f39c12')
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Average Confidence', fontsize=12)
    ax2.set_title('Average Detection Confidence by Configuration', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(configs_unique, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = output_dir / 'object_detection_stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_cost_comparison(results, output_dir):
    """Plot cost comparison"""
    llm_models = []
    costs = []
    
    # Estimate costs
    gpt4o_mini_rate = 0.00075  # per query estimate
    claude_rate = 0.0015  # per query estimate
    
    for r in results:
        llm_model = r.get('llm_model', '')
        if 'gpt' in llm_model.lower():
            llm_models.append('GPT-4o-mini')
            costs.append(gpt4o_mini_rate)
        elif 'claude' in llm_model.lower():
            llm_models.append('Claude Haiku')
            costs.append(claude_rate)
    
    if not llm_models:
        return
    
    df = pd.DataFrame({
        'LLM Model': llm_models,
        'Cost per Query': costs
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    model_counts = df['LLM Model'].value_counts()
    model_costs = df.groupby('LLM Model')['Cost per Query'].first()
    
    bars = ax.bar(model_costs.index, model_costs.values, color=['#3498db', '#e74c3c'])
    ax.set_xlabel('LLM Model', fontsize=12)
    ax.set_ylabel('Cost per Query ($)', fontsize=12)
    ax.set_title('Cost Comparison by LLM Model', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / 'cost_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_latency_by_category(results, output_dir):
    """Plot latency by image category"""
    categories = []
    total_latencies = []
    
    for r in results:
        category = r.get('category', 'Unknown')
        try:
            total = float(r.get('total_latency', 0))
            if total > 0:
                categories.append(category)
                total_latencies.append(total)
        except (ValueError, TypeError):
            continue
    
    if not categories:
        return
    
    df = pd.DataFrame({
        'Category': categories,
        'Total Latency': total_latencies
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Category', y='Total Latency', ax=ax)
    ax.set_title('Total Latency by Image Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_xlabel('Category', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_by_category.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    figures_dir = Path('results/approach_2_yolo_llm/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_yolo_llm.py first.")
        return
    
    print("=" * 60)
    print("Creating YOLO+LLM Visualizations")
    print("=" * 60)
    print()
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    print()
    
    plot_latency_comparison(results, figures_dir)
    plot_latency_by_yolo_variant(results, figures_dir)
    plot_latency_by_llm_model(results, figures_dir)
    plot_object_detection_stats(results, figures_dir)
    plot_cost_comparison(results, figures_dir)
    plot_latency_by_category(results, figures_dir)
    
    print(f"\n✅ All visualizations saved to: {figures_dir}")


if __name__ == "__main__":
    main()

