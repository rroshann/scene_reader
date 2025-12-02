#!/usr/bin/env python3
"""
Create visualizations for Approach 4 (Local Models)
"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
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


def create_latency_comparison(results, output_dir):
    """Create latency comparison chart"""
    models = []
    latencies = []
    
    for r in results:
        try:
            latency = float(r.get('total_latency', 0) or r.get('latency', 0))
            if latency > 0:
                models.append(r.get('model', 'unknown'))
                latencies.append(latency)
        except (ValueError, TypeError):
            continue
    
    if not latencies:
        print("  ⚠️  No latency data for comparison chart")
        return
    
    df = pd.DataFrame({'Model': models, 'Latency (s)': latencies})
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Model', y='Latency (s)')
    plt.title('Latency by Model: BLIP-2', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: latency_comparison.png")


def create_latency_by_category(results, output_dir):
    """Create latency by category chart"""
    categories = []
    latencies = []
    models = []
    
    for r in results:
        try:
            latency = float(r.get('total_latency', 0) or r.get('latency', 0))
            if latency > 0:
                categories.append(r.get('category', 'unknown'))
                latencies.append(latency)
                models.append(r.get('model', 'unknown'))
        except (ValueError, TypeError):
            continue
    
    if not latencies:
        print("  ⚠️  No latency data for category chart")
        return
    
    df = pd.DataFrame({
        'Category': categories,
        'Latency (s)': latencies,
        'Model': models
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Category', y='Latency (s)', hue='Model')
    plt.title('Latency by Category and Model', fontsize=14, fontweight='bold')
    plt.ylabel('Latency (seconds)', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: latency_by_category.png")


def create_device_usage_chart(results, output_dir):
    """Create device usage chart"""
    devices = []
    counts = []
    
    device_counts = {}
    for r in results:
        device = r.get('device', 'unknown')
        device_counts[device] = device_counts.get(device, 0) + 1
    
    if not device_counts:
        print("  ⚠️  No device data for device usage chart")
        return
    
    for device, count in device_counts.items():
        devices.append(device)
        counts.append(count)
    
    plt.figure(figsize=(8, 6))
    plt.bar(devices, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Device Usage Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Tests', fontsize=12)
    plt.xlabel('Device', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'device_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: device_usage.png")


def create_response_length_comparison(results, output_dir):
    """Create response length comparison"""
    models = []
    lengths = []
    
    for r in results:
        desc = r.get('description', '')
        if desc:
            models.append(r.get('model', 'unknown'))
            lengths.append(len(desc.split()))
    
    if not lengths:
        print("  ⚠️  No response length data")
        return
    
    df = pd.DataFrame({'Model': models, 'Words': lengths})
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Model', y='Words')
    plt.title('Response Length by Model: BLIP-2', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Words', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'response_length_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Created: response_length_comparison.png")


def main():
    """Create all visualizations"""
    print("=" * 60)
    print("Creating Local Models Visualizations")
    print("=" * 60)
    print()
    
    # Load results
    csv_path = Path('results/approach_4_local/raw/batch_results.csv')
    if not csv_path.exists():
        print(f"❌ Results file not found: {csv_path}")
        return
    
    results = load_results(csv_path)
    print(f"Loaded {len(results)} successful results")
    print()
    
    # Output directory
    output_dir = Path('results/approach_4_local/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualizations...")
    create_latency_comparison(results, output_dir)
    create_latency_by_category(results, output_dir)
    create_device_usage_chart(results, output_dir)
    create_response_length_comparison(results, output_dir)
    
    print()
    print(f"✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

