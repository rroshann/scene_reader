#!/usr/bin/env python3
"""
Compare Approach 2 (YOLO+LLM) with Approach 1 (Pure VLMs)
"""
import csv
import statistics
from pathlib import Path
from collections import defaultdict


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def calculate_avg_latency(results, latency_key='latency_seconds'):
    """Calculate average latency"""
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                latency = float(r.get(latency_key, 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                continue
    return statistics.mean(latencies) if latencies else None


def calculate_avg_cost(results):
    """Calculate average cost per query"""
    # Approach 1 costs (from FINDINGS.md)
    costs_vlm = {
        'GPT-4V': 0.0124,
        'Gemini': 0.0031,
        'Claude': 0.0240
    }
    
    # Approach 2 costs (LLM only, YOLO is free)
    costs_yolo_llm = {
        'gpt-4o-mini': 0.00075,
        'claude-haiku': 0.0015
    }
    
    total_cost = 0
    count = 0
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            # For Approach 1
            if 'model' in r:
                model = r['model']
                if model in costs_vlm:
                    total_cost += costs_vlm[model]
                    count += 1
            # For Approach 2
            elif 'llm_model' in r:
                llm_model = r['llm_model'].lower()
                for model_key, cost in costs_yolo_llm.items():
                    if model_key in llm_model:
                        total_cost += cost
                        count += 1
                        break
    
    return total_cost / count if count > 0 else None


def main():
    vlm_csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    yolo_llm_csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    output_path = Path('results/approach_2_yolo_llm/analysis/yolo_llm_vs_vlm_comparison.txt')
    
    if not vlm_csv_path.exists():
        print(f"Error: Approach 1 results not found: {vlm_csv_path}")
        return
    
    if not yolo_llm_csv_path.exists():
        print(f"Error: Approach 2 results not found: {yolo_llm_csv_path}")
        print("Please run batch_test_yolo_llm.py first.")
        return
    
    print("=" * 60)
    print("Comparing Approach 2 (YOLO+LLM) vs Approach 1 (Pure VLMs)")
    print("=" * 60)
    print()
    
    # Load results
    vlm_results = load_results(vlm_csv_path)
    yolo_llm_results = load_results(yolo_llm_csv_path)
    
    # Filter VLM results to recent ones (system prompt test)
    vlm_results = [r for r in vlm_results if '2025-11-22T20:' in r.get('timestamp', '')]
    
    print(f"Approach 1 (VLMs): {len(vlm_results)} results")
    print(f"Approach 2 (YOLO+LLM): {len(yolo_llm_results)} results")
    print()
    
    # Latency comparison
    print("=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)
    
    vlm_latencies = []
    for r in vlm_results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                lat = float(r.get('latency_seconds', 0))
                if lat > 0:
                    vlm_latencies.append(lat)
            except (ValueError, TypeError):
                continue
    
    yolo_llm_latencies = []
    for r in yolo_llm_results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                lat = float(r.get('total_latency', 0))
                if lat > 0:
                    yolo_llm_latencies.append(lat)
            except (ValueError, TypeError):
                continue
    
    if vlm_latencies and yolo_llm_latencies:
        vlm_mean = statistics.mean(vlm_latencies)
        yolo_llm_mean = statistics.mean(yolo_llm_latencies)
        
        print(f"\nApproach 1 (Pure VLMs):")
        print(f"  Mean Latency: {vlm_mean:.3f}s")
        print(f"  Median: {statistics.median(vlm_latencies):.3f}s")
        print(f"  Min: {min(vlm_latencies):.3f}s")
        print(f"  Max: {max(vlm_latencies):.3f}s")
        
        print(f"\nApproach 2 (YOLO+LLM):")
        print(f"  Mean Latency: {yolo_llm_mean:.3f}s")
        print(f"  Median: {statistics.median(yolo_llm_latencies):.3f}s")
        print(f"  Min: {min(yolo_llm_latencies):.3f}s")
        print(f"  Max: {max(yolo_llm_latencies):.3f}s")
        
        speedup = vlm_mean / yolo_llm_mean if yolo_llm_mean > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    # Cost comparison
    print("\n" + "=" * 60)
    print("COST COMPARISON")
    print("=" * 60)
    
    vlm_cost = calculate_avg_cost(vlm_results)
    yolo_llm_cost = calculate_avg_cost(yolo_llm_results)
    
    if vlm_cost and yolo_llm_cost:
        print(f"\nApproach 1 (Pure VLMs):")
        print(f"  Average Cost per Query: ${vlm_cost:.6f}")
        print(f"  Cost per 1000 Queries: ${vlm_cost * 1000:.2f}")
        
        print(f"\nApproach 2 (YOLO+LLM):")
        print(f"  Average Cost per Query: ${yolo_llm_cost:.6f}")
        print(f"  Cost per 1000 Queries: ${yolo_llm_cost * 1000:.2f}")
        
        cost_savings = ((vlm_cost - yolo_llm_cost) / vlm_cost) * 100 if vlm_cost > 0 else 0
        print(f"\nCost Savings: {cost_savings:.1f}% cheaper with Approach 2")
    
    # Breakdown for Approach 2
    print("\n" + "=" * 60)
    print("APPROACH 2 BREAKDOWN")
    print("=" * 60)
    
    detection_latencies = []
    generation_latencies = []
    
    for r in yolo_llm_results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                det_lat = float(r.get('detection_latency', 0))
                gen_lat = float(r.get('generation_latency', 0))
                if det_lat > 0:
                    detection_latencies.append(det_lat)
                if gen_lat > 0:
                    generation_latencies.append(gen_lat)
            except (ValueError, TypeError):
                continue
    
    if detection_latencies and generation_latencies:
        print(f"\nDetection Stage (YOLO):")
        print(f"  Mean: {statistics.mean(detection_latencies):.3f}s")
        print(f"  Median: {statistics.median(detection_latencies):.3f}s")
        print(f"  Percentage of Total: {(statistics.mean(detection_latencies) / yolo_llm_mean * 100):.1f}%")
        
        print(f"\nGeneration Stage (LLM):")
        print(f"  Mean: {statistics.mean(generation_latencies):.3f}s")
        print(f"  Median: {statistics.median(generation_latencies):.3f}s")
        print(f"  Percentage of Total: {(statistics.mean(generation_latencies) / yolo_llm_mean * 100):.1f}%")
    
    # Object detection stats
    print("\n" + "=" * 60)
    print("OBJECT DETECTION STATISTICS (Approach 2)")
    print("=" * 60)
    
    num_objects = []
    for r in yolo_llm_results:
        if r.get('success') == 'True' or r.get('success') is True:
            try:
                num_obj = int(r.get('num_objects_detected', 0))
                if num_obj >= 0:
                    num_objects.append(num_obj)
            except (ValueError, TypeError):
                continue
    
    if num_objects:
        print(f"\nObjects Detected per Image:")
        print(f"  Mean: {statistics.mean(num_objects):.1f}")
        print(f"  Median: {statistics.median(num_objects):.1f}")
        print(f"  Min: {min(num_objects)}")
        print(f"  Max: {max(num_objects)}")
    
    # Save comparison
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Approach 2 (YOLO+LLM) vs Approach 1 (Pure VLMs) Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        if vlm_latencies and yolo_llm_latencies:
            f.write("Latency Comparison:\n")
            f.write(f"  Approach 1 Mean: {vlm_mean:.3f}s\n")
            f.write(f"  Approach 2 Mean: {yolo_llm_mean:.3f}s\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
        
        if vlm_cost and yolo_llm_cost:
            f.write("Cost Comparison:\n")
            f.write(f"  Approach 1: ${vlm_cost:.6f} per query\n")
            f.write(f"  Approach 2: ${yolo_llm_cost:.6f} per query\n")
            f.write(f"  Cost Savings: {cost_savings:.1f}%\n\n")
        
        if detection_latencies and generation_latencies:
            f.write("Approach 2 Breakdown:\n")
            f.write(f"  Detection: {statistics.mean(detection_latencies):.3f}s\n")
            f.write(f"  Generation: {statistics.mean(generation_latencies):.3f}s\n")
    
    print(f"\n✅ Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()

