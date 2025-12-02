#!/usr/bin/env python3
"""
Create visualizations for Approach 3.5: Optimized Specialized Multi-Model System
High-quality, publication-ready plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style for publication-ready plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_results():
    """Load Approach 3.5 results"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_5_optimized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    return df[df['success'] == True].copy()


def load_approach_3_results():
    """Load Approach 3 results for comparison"""
    project_root = Path(__file__).parent.parent.parent
    results_file = project_root / 'results' / 'approach_3_specialized' / 'raw' / 'batch_results.csv'
    
    if not results_file.exists():
        return None
    
    df = pd.read_csv(results_file)
    return df[df['success'] == True].copy()


def plot_latency_comparison(df_3_5, df_3, output_dir):
    """Plot 1: Latency comparison (Approach 3.5 vs Approach 3)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter depth mode only (for fair comparison)
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    depth_3 = df_3[df_3['mode'] == 'depth'] if df_3 is not None else None
    
    data = []
    if len(depth_3_5) > 0:
        data.append({'Approach': '3.5 Optimized', 'Latency': depth_3_5['total_latency'].mean()})
        data.append({'Approach': '3.5 GPT-3.5-turbo', 'Latency': depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']['total_latency'].mean()})
        data.append({'Approach': '3.5 GPT-4o-mini', 'Latency': depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']['total_latency'].mean()})
    
    if depth_3 is not None and len(depth_3) > 0:
        data.append({'Approach': '3 Baseline', 'Latency': depth_3['total_latency'].mean()})
    
    if data:
        comparison_df = pd.DataFrame(data)
        bars = ax.bar(comparison_df['Approach'], comparison_df['Latency'], 
                     color=['#2ecc71', '#3498db', '#e74c3c', '#95a5a6'][:len(data)])
        ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
        ax.set_title('Approach 3.5 vs Approach 3: Latency Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Created: latency_comparison.png")


def plot_component_breakdown(df_3_5, output_dir):
    """Plot 2: Component latency breakdown (stacked bar)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        return
    
    # Group by configuration
    configs = depth_3_5['configuration'].unique()
    
    detection_means = []
    depth_means = []
    generation_means = []
    config_labels = []
    
    for config in configs:
        config_df = depth_3_5[depth_3_5['configuration'] == config]
        if len(config_df) > 0:
            detection_means.append(config_df['detection_latency'].mean())
            depth_means.append(config_df['depth_latency'].mean())
            generation_means.append(config_df['generation_latency'].mean())
            config_labels.append(config.replace('YOLOv8N+Depth-Anything+', '').replace('+Cache+Adaptive', ''))
    
    x = np.arange(len(config_labels))
    width = 0.6
    
    p1 = ax.bar(x, detection_means, width, label='Detection', color='#3498db')
    p2 = ax.bar(x, depth_means, width, bottom=detection_means, label='Depth', color='#2ecc71')
    p3 = ax.bar(x, generation_means, width, 
               bottom=np.array(detection_means) + np.array(depth_means), 
               label='Generation', color='#e74c3c')
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Component Latency Breakdown by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: component_breakdown.png")


def plot_latency_distribution(df_3_5, output_dir):
    """Plot 3: Latency distribution histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        return
    
    # Separate by LLM model
    gpt35 = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']['total_latency']
    gpt4mini = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']['total_latency']
    
    ax.hist(gpt35, bins=15, alpha=0.7, label='GPT-3.5-turbo', color='#3498db', edgecolor='black')
    ax.hist(gpt4mini, bins=15, alpha=0.7, label='GPT-4o-mini', color='#e74c3c', edgecolor='black')
    
    ax.axvline(gpt35.mean(), color='#3498db', linestyle='--', linewidth=2, label=f'GPT-3.5-turbo mean: {gpt35.mean():.2f}s')
    ax.axvline(gpt4mini.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'GPT-4o-mini mean: {gpt4mini.mean():.2f}s')
    
    ax.set_xlabel('Total Latency (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Latency Distribution: GPT-3.5-turbo vs GPT-4o-mini', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: latency_distribution.png")


def plot_complexity_distribution(df_3_5, output_dir):
    """Plot 4: Complexity distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if 'complexity' not in depth_3_5.columns or depth_3_5['complexity'].isna().all():
        return
    
    complexity_counts = depth_3_5['complexity'].value_counts()
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(complexity_counts.index, complexity_counts.values, color=colors[:len(complexity_counts)])
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Complexity Level', fontsize=12)
    ax.set_title('Scene Complexity Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: complexity_distribution.png")


def plot_generation_comparison(df_3_5, output_dir):
    """Plot 5: Generation latency comparison (box plot)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        return
    
    gpt35_gen = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']['generation_latency']
    gpt4mini_gen = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']['generation_latency']
    
    data_to_plot = [gpt35_gen.dropna(), gpt4mini_gen.dropna()]
    labels = ['GPT-3.5-turbo', 'GPT-4o-mini']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_ylabel('Generation Latency (seconds)', fontsize=12)
    ax.set_title('Generation Latency Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: generation_comparison.png")


def plot_mode_performance(df_3_5, output_dir):
    """Plot 6: Mode-specific performance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(df_3_5) == 0:
        return
    
    # Group by mode and LLM model
    mode_data = []
    for mode in df_3_5['mode'].unique():
        mode_df = df_3_5[df_3_5['mode'] == mode]
        for llm in mode_df['llm_model'].unique():
            llm_df = mode_df[mode_df['llm_model'] == llm]
            if len(llm_df) > 0:
                mode_data.append({
                    'Mode': mode.upper(),
                    'LLM': llm,
                    'Mean Latency': llm_df['total_latency'].mean()
                })
    
    if mode_data:
        mode_df = pd.DataFrame(mode_data)
        pivot_df = mode_df.pivot(index='Mode', columns='LLM', values='Mean Latency')
        
        pivot_df.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], width=0.8)
        ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
        ax.set_title('Mode-Specific Performance by LLM Model', fontsize=14, fontweight='bold')
        ax.legend(title='LLM Model')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'mode_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Created: mode_performance.png")


def plot_improvement_percentage(df_3_5, output_dir):
    """Plot 7: Improvement percentage"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        return
    
    gpt35 = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']
    gpt4mini = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']
    
    if len(gpt35) == 0 or len(gpt4mini) == 0:
        return
    
    # Calculate improvements
    total_improvement = ((gpt4mini['total_latency'].mean() - gpt35['total_latency'].mean()) / gpt4mini['total_latency'].mean()) * 100
    gen_improvement = ((gpt4mini['generation_latency'].mean() - gpt35['generation_latency'].mean()) / gpt4mini['generation_latency'].mean()) * 100
    
    categories = ['Total Latency', 'Generation Latency']
    improvements = [total_improvement, gen_improvement]
    
    bars = ax.bar(categories, improvements, color=['#2ecc71', '#3498db'])
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Approach 3.5 Optimization Impact', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: improvement_percentage.png")


def plot_target_analysis(df_3_5, output_dir):
    """Plot 8: Target analysis (<2 seconds)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depth_3_5 = df_3_5[df_3_5['mode'] == 'depth']
    
    if len(depth_3_5) == 0:
        return
    
    gpt35 = depth_3_5[depth_3_5['llm_model'] == 'gpt-3.5-turbo']
    gpt4mini = depth_3_5[depth_3_5['llm_model'] == 'gpt-4o-mini']
    
    gpt35_under_2s = len(gpt35[gpt35['total_latency'] < 2.0])
    gpt4mini_under_2s = len(gpt4mini[gpt4mini['total_latency'] < 2.0])
    
    gpt35_total = len(gpt35)
    gpt4mini_total = len(gpt4mini)
    
    categories = ['GPT-3.5-turbo', 'GPT-4o-mini']
    under_2s = [gpt35_under_2s, gpt4mini_under_2s]
    total = [gpt35_total, gpt4mini_total]
    percentages = [(u/t)*100 if t > 0 else 0 for u, t in zip(under_2s, total)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, under_2s, width, label='Under 2s', color='#2ecc71')
    bars2 = ax.bar(x + width/2, [t - u for u, t in zip(under_2s, total)], width, 
                   label='2s or more', color='#e74c3c', bottom=under_2s)
    
    # Add percentage labels
    for i, (bar1, bar2, pct) in enumerate(zip(bars1, bars2, percentages)):
        ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height()/2,
               f'{pct:.1f}%',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Analysis: <2 Seconds Latency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'target_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: target_analysis.png")


def main():
    """Main function to create all visualizations"""
    print("=" * 80)
    print("CREATING APPROACH 3.5 VISUALIZATIONS")
    print("=" * 80)
    print()
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_3_5 = load_results()
    if df_3_5 is None or len(df_3_5) == 0:
        print("No successful results found. Cannot create visualizations.")
        return
    
    df_3 = load_approach_3_results()
    
    print("Creating visualizations...")
    print()
    
    # Create all plots
    plot_latency_comparison(df_3_5, df_3, output_dir)
    plot_component_breakdown(df_3_5, output_dir)
    plot_latency_distribution(df_3_5, output_dir)
    plot_complexity_distribution(df_3_5, output_dir)
    plot_generation_comparison(df_3_5, output_dir)
    plot_mode_performance(df_3_5, output_dir)
    plot_improvement_percentage(df_3_5, output_dir)
    plot_target_analysis(df_3_5, output_dir)
    
    print()
    print("=" * 80)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}")
    print(f"Total plots created: 8")


if __name__ == '__main__':
    main()

