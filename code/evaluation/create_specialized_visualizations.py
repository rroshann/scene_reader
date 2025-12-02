#!/usr/bin/env python3
"""
Create visualizations for Approach 3: Specialized Multi-Model System results
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


def plot_latency_breakdown(results, output_dir):
    """Plot latency breakdown by component (stacked bar)"""
    modes = []
    detection_latencies = []
    ocr_latencies = []
    depth_latencies = []
    generation_latencies = []
    
    for r in results:
        mode = r.get('mode', 'unknown')
        try:
            detection = float(r.get('detection_latency', 0))
            generation = float(r.get('generation_latency', 0))
            ocr = float(r.get('ocr_latency', 0)) if r.get('ocr_latency') else 0
            depth = float(r.get('depth_latency', 0)) if r.get('depth_latency') else 0
            
            if detection > 0 and generation > 0:
                modes.append(mode)
                detection_latencies.append(detection)
                ocr_latencies.append(ocr)
                depth_latencies.append(depth)
                generation_latencies.append(generation)
        except (ValueError, TypeError):
            continue
    
    if not modes:
        print("No valid latency data for breakdown plot")
        return
    
    df = pd.DataFrame({
        'Mode': modes,
        'Detection': detection_latencies,
        'OCR': ocr_latencies,
        'Depth': depth_latencies,
        'Generation': generation_latencies
    })
    
    # Group by mode and calculate means
    mode_means = df.groupby('Mode').mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(mode_means.index))
    width = 0.6
    
    detection_means = mode_means['Detection'].values
    ocr_means = mode_means['OCR'].values
    depth_means = mode_means['Depth'].values
    generation_means = mode_means['Generation'].values
    
    # Stack bars
    bottom = detection_means
    ax.bar(x, detection_means, width, label='Detection (YOLO)', color='#3498db')
    ax.bar(x, ocr_means, width, bottom=bottom, label='OCR', color='#9b59b6')
    bottom = bottom + ocr_means
    ax.bar(x, depth_means, width, bottom=bottom, label='Depth', color='#16a085')
    bottom = bottom + depth_means
    ax.bar(x, generation_means, width, bottom=bottom, label='Generation (LLM)', color='#e74c3c')
    
    ax.set_xlabel('Mode', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency Breakdown by Component (Approach 3)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(mode_means.index, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_ocr_vs_depth_latency(results, output_dir):
    """Plot OCR vs Depth latency comparison (box plot)"""
    ocr_latencies = []
    depth_latencies = []
    
    for r in results:
        mode = r.get('mode', '')
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                if mode == 'ocr':
                    ocr_latencies.append(latency)
                elif mode == 'depth':
                    depth_latencies.append(latency)
        except (ValueError, TypeError):
            continue
    
    if not ocr_latencies and not depth_latencies:
        print("No valid latency data for OCR vs Depth comparison")
        return
    
    data = []
    labels = []
    
    if ocr_latencies:
        data.extend(ocr_latencies)
        labels.extend(['OCR Mode'] * len(ocr_latencies))
    
    if depth_latencies:
        data.extend(depth_latencies)
        labels.extend(['Depth Mode'] * len(depth_latencies))
    
    df = pd.DataFrame({
        'Mode': labels,
        'Total Latency': data
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='Mode', y='Total Latency', ax=ax, palette='muted')
    ax.set_title('OCR vs Depth Mode Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Latency (seconds)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'ocr_vs_depth_latency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_latency_by_category(results, output_dir):
    """Plot category-specific performance (bar chart)"""
    categories = []
    latencies = []
    modes = []
    
    for r in results:
        category = r.get('category', '')
        mode = r.get('mode', '')
        try:
            latency = float(r.get('total_latency', 0))
            if latency > 0:
                categories.append(category)
                latencies.append(latency)
                modes.append(mode)
        except (ValueError, TypeError):
            continue
    
    if not categories:
        print("No valid latency data for category plot")
        return
    
    df = pd.DataFrame({
        'Category': categories,
        'Latency': latencies,
        'Mode': modes
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Category', y='Latency', hue='Mode', ax=ax, palette='viridis')
    ax.set_title('Latency by Category (Approach 3)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Latency (seconds)', fontsize=12)
    ax.set_xlabel('Image Category', fontsize=12)
    ax.legend(title='Mode')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_by_category.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_comparison_with_baseline(results, output_dir):
    """Plot comparison with Approach 2 baseline"""
    # Load Approach 2 results
    a2_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    if not a2_path.exists():
        print("Approach 2 results not found, skipping baseline comparison")
        return
    
    a2_results = load_results(a2_path)
    
    # Filter Approach 2: YOLOv8N + GPT-4o-mini
    a2_filtered = [r for r in a2_results 
                   if r.get('yolo_model') == 'yolov8n' 
                   and r.get('llm_model') == 'gpt-4o-mini']
    
    # Separate Approach 3 by mode
    a3_ocr = [r for r in results if r.get('mode') == 'ocr']
    a3_depth = [r for r in results if r.get('mode') == 'depth']
    
    # Get common images for fair comparison
    a2_filenames = {r['filename'] for r in a2_filtered}
    a3_ocr_filenames = {r['filename'] for r in a3_ocr}
    a3_depth_filenames = {r['filename'] for r in a3_depth}
    
    # Text images comparison (3A vs 2)
    text_common = list(a2_filenames.intersection(a3_ocr_filenames))
    a2_text = [r for r in a2_filtered if r['filename'] in text_common]
    a3_text = [r for r in a3_ocr if r['filename'] in text_common]
    
    # Navigation images comparison (3B vs 2)
    nav_common = list(a2_filenames.intersection(a3_depth_filenames))
    a2_nav = [r for r in a2_filtered if r['filename'] in nav_common]
    a3_nav = [r for r in a3_depth if r['filename'] in nav_common]
    
    data = []
    labels = []
    
    # Text images
    if a2_text:
        for r in a2_text:
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    data.append(latency)
                    labels.append('Approach 2\n(Text)')
            except (ValueError, TypeError):
                continue
    
    if a3_text:
        for r in a3_text:
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    data.append(latency)
                    labels.append('Approach 3A\n(OCR)')
            except (ValueError, TypeError):
                continue
    
    # Navigation images
    if a2_nav:
        for r in a2_nav:
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    data.append(latency)
                    labels.append('Approach 2\n(Navigation)')
            except (ValueError, TypeError):
                continue
    
    if a3_nav:
        for r in a3_nav:
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    data.append(latency)
                    labels.append('Approach 3B\n(Depth)')
            except (ValueError, TypeError):
                continue
    
    if not data:
        print("No valid data for baseline comparison")
        return
    
    df = pd.DataFrame({
        'Approach': labels,
        'Latency': data
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Approach', y='Latency', ax=ax, palette='pastel')
    ax.set_title('Latency Comparison: Approach 3 vs Approach 2 Baseline', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Latency (seconds)', fontsize=12)
    ax.set_xlabel('Approach', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_with_baseline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_component_contribution(results, output_dir):
    """Plot component contribution analysis (pie chart)"""
    # Separate by mode
    ocr_results = [r for r in results if r.get('mode') == 'ocr']
    depth_results = [r for r in results if r.get('mode') == 'depth']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # OCR mode
    if ocr_results:
        ocr_detection = []
        ocr_ocr = []
        ocr_generation = []
        
        for r in ocr_results:
            try:
                detection = float(r.get('detection_latency', 0))
                ocr = float(r.get('ocr_latency', 0)) if r.get('ocr_latency') else 0
                generation = float(r.get('generation_latency', 0))
                
                if detection > 0 and generation > 0:
                    total = detection + ocr + generation
                    if total > 0:
                        ocr_detection.append(detection / total * 100)
                        ocr_ocr.append(ocr / total * 100)
                        ocr_generation.append(generation / total * 100)
            except (ValueError, TypeError):
                continue
        
        if ocr_detection:
            ocr_means = [
                np.mean(ocr_detection),
                np.mean(ocr_ocr),
                np.mean(ocr_generation)
            ]
            axes[0].pie(ocr_means, labels=['Detection', 'OCR', 'Generation'], 
                       autopct='%1.1f%%', startangle=90, colors=['#3498db', '#9b59b6', '#e74c3c'])
            axes[0].set_title('OCR Mode Component Contribution', fontsize=12, fontweight='bold')
    
    # Depth mode
    if depth_results:
        depth_detection = []
        depth_depth = []
        depth_generation = []
        
        for r in depth_results:
            try:
                detection = float(r.get('detection_latency', 0))
                depth = float(r.get('depth_latency', 0)) if r.get('depth_latency') else 0
                generation = float(r.get('generation_latency', 0))
                
                if detection > 0 and generation > 0:
                    total = detection + depth + generation
                    if total > 0:
                        depth_detection.append(detection / total * 100)
                        depth_depth.append(depth / total * 100)
                        depth_generation.append(generation / total * 100)
            except (ValueError, TypeError):
                continue
        
        if depth_detection:
            depth_means = [
                np.mean(depth_detection),
                np.mean(depth_depth),
                np.mean(depth_generation)
            ]
            axes[1].pie(depth_means, labels=['Detection', 'Depth', 'Generation'], 
                       autopct='%1.1f%%', startangle=90, colors=['#3498db', '#16a085', '#e74c3c'])
            axes[1].set_title('Depth Mode Component Contribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'component_contribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_response_length_comparison(results, output_dir):
    """Plot response length comparison (box plot)"""
    ocr_word_counts = []
    depth_word_counts = []
    
    for r in results:
        mode = r.get('mode', '')
        try:
            if r.get('word_count'):
                word_count = int(r['word_count'])
            elif r.get('description'):
                word_count = len(str(r['description']).split())
            else:
                continue
            
            if mode == 'ocr':
                ocr_word_counts.append(word_count)
            elif mode == 'depth':
                depth_word_counts.append(word_count)
        except (ValueError, TypeError):
            continue
    
    if not ocr_word_counts and not depth_word_counts:
        print("No valid word count data for response length comparison")
        return
    
    data = []
    labels = []
    
    if ocr_word_counts:
        data.extend(ocr_word_counts)
        labels.extend(['OCR Mode'] * len(ocr_word_counts))
    
    if depth_word_counts:
        data.extend(depth_word_counts)
        labels.extend(['Depth Mode'] * len(depth_word_counts))
    
    df = pd.DataFrame({
        'Mode': labels,
        'Word Count': data
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='Mode', y='Word Count', ax=ax, palette='coolwarm')
    ax.set_title('Response Length Comparison (Approach 3)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Word Count', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'response_length_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def main():
    csv_path = Path('results/approach_3_specialized/raw/batch_results.csv')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_specialized.py first.")
        return
    
    print("Loading results...")
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    
    output_dir = Path('results/approach_3_specialized/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    plot_latency_breakdown(results, output_dir)
    plot_ocr_vs_depth_latency(results, output_dir)
    plot_latency_by_category(results, output_dir)
    plot_comparison_with_baseline(results, output_dir)
    plot_component_contribution(results, output_dir)
    plot_response_length_comparison(results, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

