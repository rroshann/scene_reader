#!/usr/bin/env python3
"""
Comprehensive analysis of YOLO+LLM results (Approach 2)
Calculates all quantitative metrics automatically
"""
import csv
import statistics
import pandas as pd
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


def calculate_latency_stats(results, filter_key=None, filter_value=None):
    """
    Calculate comprehensive latency statistics
    
    Args:
        results: List of result dicts
        filter_key: Optional key to filter by (e.g., 'yolo_model', 'llm_model')
        filter_value: Optional value to filter by
    """
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                latency = float(r.get('total_latency', 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    return {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'p75': statistics.quantiles(latencies, n=4)[2] if len(latencies) >= 4 else None,
        'p90': statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else None,
        'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None,
        'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else None,
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def calculate_detection_stats(results, filter_key=None, filter_value=None):
    """Calculate detection latency statistics"""
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                latency = float(r.get('detection_latency', 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    return {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def calculate_generation_stats(results, filter_key=None, filter_value=None):
    """Calculate generation latency statistics"""
    latencies = []
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                latency = float(r.get('generation_latency', 0))
                if latency > 0:
                    latencies.append(latency)
            except (ValueError, TypeError):
                continue
    
    if not latencies:
        return None
    
    return {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def calculate_object_detection_stats(results, filter_key=None, filter_value=None):
    """Calculate object detection statistics"""
    num_objects = []
    confidences = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            try:
                num_obj = int(r.get('num_objects_detected', 0))
                if num_obj >= 0:
                    num_objects.append(num_obj)
                
                conf = float(r.get('avg_confidence', 0))
                if conf > 0:
                    confidences.append(conf)
            except (ValueError, TypeError):
                continue
    
    stats = {}
    if num_objects:
        stats['num_objects'] = {
            'mean': statistics.mean(num_objects),
            'median': statistics.median(num_objects),
            'min': min(num_objects),
            'max': max(num_objects),
            'stdev': statistics.stdev(num_objects) if len(num_objects) > 1 else 0
        }
    
    if confidences:
        stats['confidence'] = {
            'mean': statistics.mean(confidences),
            'median': statistics.median(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'stdev': statistics.stdev(confidences) if len(confidences) > 1 else 0
        }
    
    return stats if stats else None


def calculate_response_length_stats(results, filter_key=None, filter_value=None):
    """Calculate response length statistics"""
    word_counts = []
    char_counts = []
    token_counts = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            desc = r.get('description', '')
            if desc:
                words = desc.split()
                word_counts.append(len(words))
                char_counts.append(len(desc))
            
            tokens = r.get('tokens_used')
            if tokens:
                try:
                    token_counts.append(int(tokens))
                except (ValueError, TypeError):
                    pass
    
    stats = {}
    if word_counts:
        stats['word_count'] = {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts),
            'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        }
    
    if char_counts:
        stats['char_count'] = {
            'mean': statistics.mean(char_counts),
            'median': statistics.median(char_counts),
            'min': min(char_counts),
            'max': max(char_counts),
            'stdev': statistics.stdev(char_counts) if len(char_counts) > 1 else 0
        }
    
    if token_counts:
        stats['token_count'] = {
            'mean': statistics.mean(token_counts),
            'median': statistics.median(token_counts),
            'min': min(token_counts),
            'max': max(token_counts),
            'stdev': statistics.stdev(token_counts) if len(token_counts) > 1 else 0
        }
    
    return stats if stats else None


def calculate_cost_estimates(results, filter_key=None, filter_value=None):
    """
    Estimate costs based on token usage
    GPT-4o-mini: $0.00015/1K input tokens, $0.0006/1K output tokens
    Claude Haiku: $0.00025/1K input tokens, $0.00125/1K output tokens
    """
    gpt4o_mini_input_rate = 0.00015 / 1000
    gpt4o_mini_output_rate = 0.0006 / 1000
    claude_input_rate = 0.00025 / 1000
    claude_output_rate = 0.00125 / 1000
    
    total_cost = 0.0
    gpt_calls = 0
    claude_calls = 0
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            llm_model = r.get('llm_model', '')
            tokens = r.get('tokens_used')
            
            if 'gpt' in llm_model.lower() or 'openai' in llm_model.lower():
                gpt_calls += 1
                # Estimate: 70% input, 30% output tokens
                if tokens:
                    try:
                        tokens_int = int(tokens)
                        input_tokens = int(tokens_int * 0.7)
                        output_tokens = int(tokens_int * 0.3)
                        cost = (input_tokens * gpt4o_mini_input_rate) + (output_tokens * gpt4o_mini_output_rate)
                        total_cost += cost
                    except (ValueError, TypeError):
                        # Fallback estimate
                        total_cost += 0.00075  # Average per query
            elif 'claude' in llm_model.lower() or 'anthropic' in llm_model.lower():
                claude_calls += 1
                # Estimate: 70% input, 30% output tokens
                if tokens:
                    try:
                        tokens_int = int(tokens)
                        input_tokens = int(tokens_int * 0.7)
                        output_tokens = int(tokens_int * 0.3)
                        cost = (input_tokens * claude_input_rate) + (output_tokens * claude_output_rate)
                        total_cost += cost
                    except (ValueError, TypeError):
                        # Fallback estimate
                        total_cost += 0.0015  # Average per query
    
    return {
        'total_cost': total_cost,
        'gpt_calls': gpt_calls,
        'claude_calls': claude_calls,
        'cost_per_query': total_cost / len([r for r in results if r.get('success') == 'True' or r.get('success') is True]) if results else 0
    }


def analyze_by_configuration(results):
    """Analyze results by configuration"""
    configs = defaultdict(list)
    for r in results:
        config = r.get('configuration', 'Unknown')
        configs[config].append(r)
    
    analysis = {}
    for config, config_results in sorted(configs.items()):
        analysis[config] = {
            'total_tests': len(config_results),
            'successful': sum(1 for r in config_results if r.get('success') == 'True' or r.get('success') is True),
            'total_latency': calculate_latency_stats(config_results),
            'detection_latency': calculate_detection_stats(config_results),
            'generation_latency': calculate_generation_stats(config_results),
            'object_stats': calculate_object_detection_stats(config_results),
            'response_length': calculate_response_length_stats(config_results),
            'cost': calculate_cost_estimates(config_results)
        }
    
    return analysis


def analyze_by_category(results):
    """Analyze results by image category"""
    categories = defaultdict(list)
    for r in results:
        cat = r.get('category', 'Unknown')
        categories[cat].append(r)
    
    analysis = {}
    for cat, cat_results in sorted(categories.items()):
        analysis[cat] = {
            'total_tests': len(cat_results),
            'successful': sum(1 for r in cat_results if r.get('success') == 'True' or r.get('success') is True),
            'total_latency': calculate_latency_stats(cat_results),
            'detection_latency': calculate_detection_stats(cat_results),
            'generation_latency': calculate_generation_stats(cat_results),
            'object_stats': calculate_object_detection_stats(cat_results)
        }
    
    return analysis


def main():
    csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_yolo_llm.py first.")
        return
    
    print("=" * 60)
    print("YOLO + LLM Results Analysis (Approach 2)")
    print("=" * 60)
    print()
    
    # Load results
    results = load_results(csv_path)
    print(f"Loaded {len(results)} results")
    
    # Overall statistics
    successful = sum(1 for r in results if r.get('success') == 'True' or r.get('success') is True)
    print(f"Successful: {successful}/{len(results)} ({successful*100/len(results):.1f}%)")
    print()
    
    # Overall latency statistics
    print("=" * 60)
    print("OVERALL LATENCY STATISTICS")
    print("=" * 60)
    
    total_latency = calculate_latency_stats(results)
    if total_latency:
        print(f"\nTotal Latency (detection + generation):")
        print(f"  Count: {total_latency['count']}")
        print(f"  Mean: {total_latency['mean']:.3f}s")
        print(f"  Median: {total_latency['median']:.3f}s")
        print(f"  p75: {total_latency['p75']:.3f}s" if total_latency['p75'] else "  p75: N/A")
        print(f"  p90: {total_latency['p90']:.3f}s" if total_latency['p90'] else "  p90: N/A")
        print(f"  p95: {total_latency['p95']:.3f}s" if total_latency['p95'] else "  p95: N/A")
        print(f"  Min: {total_latency['min']:.3f}s")
        print(f"  Max: {total_latency['max']:.3f}s")
        print(f"  Std Dev: {total_latency['stdev']:.3f}s")
    
    detection_latency = calculate_detection_stats(results)
    if detection_latency:
        print(f"\nDetection Latency (YOLO):")
        print(f"  Count: {detection_latency['count']}")
        print(f"  Mean: {detection_latency['mean']:.3f}s")
        print(f"  Median: {detection_latency['median']:.3f}s")
        print(f"  Min: {detection_latency['min']:.3f}s")
        print(f"  Max: {detection_latency['max']:.3f}s")
        print(f"  Std Dev: {detection_latency['stdev']:.3f}s")
    
    generation_latency = calculate_generation_stats(results)
    if generation_latency:
        print(f"\nGeneration Latency (LLM):")
        print(f"  Count: {generation_latency['count']}")
        print(f"  Mean: {generation_latency['mean']:.3f}s")
        print(f"  Median: {generation_latency['median']:.3f}s")
        print(f"  Min: {generation_latency['min']:.3f}s")
        print(f"  Max: {generation_latency['max']:.3f}s")
        print(f"  Std Dev: {generation_latency['stdev']:.3f}s")
    
    # Object detection statistics
    print("\n" + "=" * 60)
    print("OBJECT DETECTION STATISTICS")
    print("=" * 60)
    
    obj_stats = calculate_object_detection_stats(results)
    if obj_stats:
        if 'num_objects' in obj_stats:
            print(f"\nObjects Detected per Image:")
            print(f"  Mean: {obj_stats['num_objects']['mean']:.1f}")
            print(f"  Median: {obj_stats['num_objects']['median']:.1f}")
            print(f"  Min: {obj_stats['num_objects']['min']}")
            print(f"  Max: {obj_stats['num_objects']['max']}")
            print(f"  Std Dev: {obj_stats['num_objects']['stdev']:.2f}")
        
        if 'confidence' in obj_stats:
            print(f"\nAverage Confidence:")
            print(f"  Mean: {obj_stats['confidence']['mean']:.3f}")
            print(f"  Median: {obj_stats['confidence']['median']:.3f}")
            print(f"  Min: {obj_stats['confidence']['min']:.3f}")
            print(f"  Max: {obj_stats['confidence']['max']:.3f}")
    
    # Response length statistics
    print("\n" + "=" * 60)
    print("RESPONSE LENGTH STATISTICS")
    print("=" * 60)
    
    resp_stats = calculate_response_length_stats(results)
    if resp_stats:
        if 'word_count' in resp_stats:
            print(f"\nWord Count:")
            print(f"  Mean: {resp_stats['word_count']['mean']:.1f}")
            print(f"  Median: {resp_stats['word_count']['median']:.1f}")
            print(f"  Min: {resp_stats['word_count']['min']}")
            print(f"  Max: {resp_stats['word_count']['max']}")
        
        if 'token_count' in resp_stats:
            print(f"\nToken Count:")
            print(f"  Mean: {resp_stats['token_count']['mean']:.1f}")
            print(f"  Median: {resp_stats['token_count']['median']:.1f}")
    
    # Cost analysis
    print("\n" + "=" * 60)
    print("COST ANALYSIS")
    print("=" * 60)
    
    cost_stats = calculate_cost_estimates(results)
    print(f"\nTotal Cost: ${cost_stats['total_cost']:.4f}")
    print(f"GPT-4o-mini calls: {cost_stats['gpt_calls']}")
    print(f"Claude Haiku calls: {cost_stats['claude_calls']}")
    print(f"Cost per query: ${cost_stats['cost_per_query']:.6f}")
    print(f"Cost per 1000 queries: ${cost_stats['cost_per_query'] * 1000:.2f}")
    
    # By configuration
    print("\n" + "=" * 60)
    print("ANALYSIS BY CONFIGURATION")
    print("=" * 60)
    
    config_analysis = analyze_by_configuration(results)
    for config, stats in sorted(config_analysis.items()):
        print(f"\n{config}:")
        print(f"  Successful: {stats['successful']}/{stats['total_tests']}")
        if stats['total_latency']:
            print(f"  Mean Total Latency: {stats['total_latency']['mean']:.3f}s")
        if stats['detection_latency']:
            print(f"  Mean Detection Latency: {stats['detection_latency']['mean']:.3f}s")
        if stats['generation_latency']:
            print(f"  Mean Generation Latency: {stats['generation_latency']['mean']:.3f}s")
        if stats['cost']:
            print(f"  Estimated Cost: ${stats['cost']['total_cost']:.4f}")
    
    # By YOLO variant
    print("\n" + "=" * 60)
    print("ANALYSIS BY YOLO VARIANT")
    print("=" * 60)
    
    for yolo_size in ['n', 'm', 'x']:
        yolo_results = [r for r in results if f'yolov8{yolo_size}' in r.get('yolo_model', '')]
        if yolo_results:
            yolo_latency = calculate_detection_stats(yolo_results)
            yolo_obj_stats = calculate_object_detection_stats(yolo_results)
            print(f"\nYOLOv8{yolo_size.upper()}:")
            print(f"  Tests: {len(yolo_results)}")
            if yolo_latency:
                print(f"  Mean Detection Latency: {yolo_latency['mean']:.3f}s")
            if yolo_obj_stats and 'num_objects' in yolo_obj_stats:
                print(f"  Mean Objects Detected: {yolo_obj_stats['num_objects']['mean']:.1f}")
    
    # By LLM model
    print("\n" + "=" * 60)
    print("ANALYSIS BY LLM MODEL")
    print("=" * 60)
    
    for llm_model in ['gpt-4o-mini', 'claude-haiku']:
        llm_results = [r for r in results if llm_model in r.get('llm_model', '').lower()]
        if llm_results:
            llm_latency = calculate_generation_stats(llm_results)
            llm_resp_stats = calculate_response_length_stats(llm_results)
            print(f"\n{llm_model}:")
            print(f"  Tests: {len(llm_results)}")
            if llm_latency:
                print(f"  Mean Generation Latency: {llm_latency['mean']:.3f}s")
            if llm_resp_stats and 'word_count' in llm_resp_stats:
                print(f"  Mean Word Count: {llm_resp_stats['word_count']['mean']:.1f}")
    
    # Save summary
    summary_path = Path('results/approach_2_yolo_llm/analysis/yolo_llm_analysis_summary.txt')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("YOLO + LLM Hybrid Pipeline Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Results: {len(results)}\n")
        f.write(f"Successful: {successful}/{len(results)} ({successful*100/len(results):.1f}%)\n\n")
        
        if total_latency:
            f.write("Overall Total Latency:\n")
            f.write(f"  Mean: {total_latency['mean']:.3f}s\n")
            f.write(f"  Median: {total_latency['median']:.3f}s\n")
            f.write(f"  Std Dev: {total_latency['stdev']:.3f}s\n\n")
        
        if detection_latency:
            f.write("Detection Latency (YOLO):\n")
            f.write(f"  Mean: {detection_latency['mean']:.3f}s\n")
            f.write(f"  Median: {detection_latency['median']:.3f}s\n\n")
        
        if generation_latency:
            f.write("Generation Latency (LLM):\n")
            f.write(f"  Mean: {generation_latency['mean']:.3f}s\n")
            f.write(f"  Median: {generation_latency['median']:.3f}s\n\n")
        
        f.write(f"Cost Analysis:\n")
        f.write(f"  Total Cost: ${cost_stats['total_cost']:.4f}\n")
        f.write(f"  Cost per Query: ${cost_stats['cost_per_query']:.6f}\n")
        f.write(f"  Cost per 1000 Queries: ${cost_stats['cost_per_query'] * 1000:.2f}\n")
    
    print(f"\n✅ Analysis complete! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

