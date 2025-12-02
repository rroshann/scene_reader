#!/usr/bin/env python3
"""
Create visualizations for RAG-Enhanced Vision results (Approach 6)
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
    """Plot latency comparison: Base vs RAG-Enhanced"""
    base_latencies = []
    rag_latencies = []
    
    for r in results:
        try:
            if r.get('use_rag') == 'True' or r.get('use_rag') is True:
                rag_latencies.append(float(r.get('total_latency', 0)))
            else:
                base_latencies.append(float(r.get('total_latency', 0)))
        except (ValueError, TypeError):
            continue
    
    if not base_latencies or not rag_latencies:
        print("⚠️  Not enough data for latency comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [base_latencies, rag_latencies]
    labels = ['Base VLM', 'RAG-Enhanced']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency Comparison: Base VLM vs RAG-Enhanced', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: latency_comparison.png")


def plot_latency_by_stage(results, output_dir):
    """Plot latency breakdown by stage for RAG-Enhanced"""
    rag_results = [r for r in results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
    
    stages = {
        'Base VLM': [],
        'Entity Extraction': [],
        'Retrieval': [],
        'Enhancement': []
    }
    
    for r in rag_results:
        try:
            if r.get('base_latency'):
                stages['Base VLM'].append(float(r['base_latency']))
            if r.get('entity_extraction_latency'):
                stages['Entity Extraction'].append(float(r['entity_extraction_latency']))
            if r.get('retrieval_latency'):
                stages['Retrieval'].append(float(r['retrieval_latency']))
            if r.get('enhancement_latency'):
                stages['Enhancement'].append(float(r['enhancement_latency']))
        except (ValueError, TypeError):
            continue
    
    # Calculate means
    means = {stage: np.mean(vals) if vals else 0 for stage, vals in stages.items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages_list = list(means.keys())
    means_list = [means[s] for s in stages_list]
    
    bars = ax.bar(stages_list, means_list, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('Latency Breakdown by Stage (RAG-Enhanced)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_stage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: latency_by_stage.png")


def plot_latency_by_vlm(results, output_dir):
    """Plot latency by VLM model"""
    vlm_latencies = {}
    
    for r in results:
        vlm = r.get('vlm_model', 'Unknown')
        try:
            latency = float(r.get('total_latency', 0))
            if vlm not in vlm_latencies:
                vlm_latencies[vlm] = []
            vlm_latencies[vlm].append(latency)
        except (ValueError, TypeError):
            continue
    
    if not vlm_latencies:
        print("⚠️  Not enough data for VLM comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [vlm_latencies[vlm] for vlm in sorted(vlm_latencies.keys())]
    labels = sorted(vlm_latencies.keys())
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency by VLM Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_vlm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: latency_by_vlm.png")


def plot_response_length_comparison(results, output_dir):
    """Plot response length comparison: Base vs Enhanced"""
    base_lengths = []
    enhanced_lengths = []
    
    for r in results:
        try:
            if r.get('base_description'):
                base_lengths.append(len(r['base_description'].split()))
            if r.get('enhanced_description'):
                enhanced_lengths.append(len(r['enhanced_description'].split()))
        except (ValueError, TypeError):
            continue
    
    if not base_lengths or not enhanced_lengths:
        print("⚠️  Not enough data for response length comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [base_lengths, enhanced_lengths]
    labels = ['Base Description', 'Enhanced Description']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Word Count', fontsize=12)
    ax.set_title('Response Length Comparison: Base vs Enhanced', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'response_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: response_length_comparison.png")


def plot_retrieval_stats(results, output_dir):
    """Plot retrieval statistics"""
    rag_results = [r for r in results if r.get('use_rag') == 'True' or r.get('use_rag') is True]
    
    num_chunks = []
    retrieval_latencies = []
    
    for r in rag_results:
        try:
            if r.get('num_retrieved_chunks'):
                num_chunks.append(int(r['num_retrieved_chunks']))
            if r.get('retrieval_latency'):
                retrieval_latencies.append(float(r['retrieval_latency']))
        except (ValueError, TypeError):
            continue
    
    if not num_chunks:
        print("⚠️  Not enough data for retrieval stats")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Number of chunks
    ax1.hist(num_chunks, bins=range(min(num_chunks), max(num_chunks)+2), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Retrieved Chunks', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Retrieved Chunks', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Retrieval latency
    if retrieval_latencies:
        ax2.hist(retrieval_latencies, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Retrieval Latency (seconds)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Retrieval Latency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: retrieval_stats.png")


def main():
    """Main visualization function"""
    results_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    print(f"Loaded {len(results)} successful results")
    
    output_dir = Path('results/approach_6_rag/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating visualizations...")
    plot_latency_comparison(results, output_dir)
    plot_latency_by_stage(results, output_dir)
    plot_latency_by_vlm(results, output_dir)
    plot_response_length_comparison(results, output_dir)
    plot_retrieval_stats(results, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

