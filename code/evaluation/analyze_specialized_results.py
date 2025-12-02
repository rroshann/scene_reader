#!/usr/bin/env python3
"""
Comprehensive analysis of Approach 3: Specialized Multi-Model System results
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
        filter_key: Optional key to filter by (e.g., 'mode', 'category')
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
        'min': min(latencies),
        'max': max(latencies),
        'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0
    }


def calculate_component_latencies(results, filter_key=None, filter_value=None):
    """Calculate latency breakdown by component"""
    components = {
        'detection': [],
        'ocr': [],
        'depth': [],
        'generation': []
    }
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            try:
                if r.get('detection_latency'):
                    components['detection'].append(float(r['detection_latency']))
                if r.get('ocr_latency'):
                    components['ocr'].append(float(r['ocr_latency']))
                if r.get('depth_latency'):
                    components['depth'].append(float(r['depth_latency']))
                if r.get('generation_latency'):
                    components['generation'].append(float(r['generation_latency']))
            except (ValueError, TypeError):
                continue
    
    stats = {}
    for component, latencies in components.items():
        if latencies:
            stats[component] = {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'count': len(latencies)
            }
    
    return stats


def calculate_response_length_stats(results, filter_key=None, filter_value=None):
    """Calculate response length statistics"""
    word_counts = []
    token_counts = []
    
    for r in results:
        if r.get('success') == 'True' or r.get('success') is True:
            if filter_key and r.get(filter_key) != filter_value:
                continue
            
            try:
                if r.get('word_count'):
                    word_counts.append(int(r['word_count']))
                elif r.get('description'):
                    words = str(r['description']).split()
                    word_counts.append(len(words))
                
                if r.get('tokens_used'):
                    token_counts.append(int(r['tokens_used']))
            except (ValueError, TypeError):
                continue
    
    stats = {}
    if word_counts:
        stats['word_count'] = {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts),
            'stdev': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
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
    """
    gpt4o_mini_input_rate = 0.00015 / 1000
    gpt4o_mini_output_rate = 0.0006 / 1000
    
    total_cost = 0.0
    gpt_calls = 0
    
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
    
    successful = len([r for r in results if (r.get('success') == 'True' or r.get('success') is True) and (not filter_key or r.get(filter_key) == filter_value)])
    
    return {
        'total_cost': total_cost,
        'gpt_calls': gpt_calls,
        'cost_per_query': total_cost / successful if successful > 0 else 0
    }


def analyze_by_mode(results):
    """Analyze results by mode (OCR vs Depth)"""
    modes = defaultdict(list)
    for r in results:
        mode = r.get('mode', 'unknown')
        modes[mode].append(r)
    
    analysis = {}
    for mode, mode_results in sorted(modes.items()):
        successful = [r for r in mode_results if r.get('success') == 'True' or r.get('success') is True]
        
        analysis[mode] = {
            'total_tests': len(mode_results),
            'successful': len(successful),
            'failed': len(mode_results) - len(successful),
            'success_rate': len(successful) / len(mode_results) * 100 if mode_results else 0,
            'total_latency': calculate_latency_stats(mode_results),
            'component_latencies': calculate_component_latencies(mode_results),
            'response_length': calculate_response_length_stats(mode_results),
            'cost': calculate_cost_estimates(mode_results)
        }
        
        # Mode-specific metrics
        if mode == 'ocr':
            ocr_texts = [r.get('ocr_num_texts') for r in successful if r.get('ocr_num_texts')]
            if ocr_texts:
                analysis[mode]['ocr_stats'] = {
                    'mean_texts_per_image': statistics.mean([int(t) for t in ocr_texts if t]),
                    'total_texts_extracted': sum([int(t) for t in ocr_texts if t])
                }
        
        if mode == 'depth':
            depth_values = [r.get('depth_mean') for r in successful if r.get('depth_mean')]
            if depth_values:
                analysis[mode]['depth_stats'] = {
                    'mean_depth': statistics.mean([float(d) for d in depth_values if d]),
                    'min_depth': min([float(d) for d in depth_values if d]),
                    'max_depth': max([float(d) for d in depth_values if d])
                }
    
    return analysis


def analyze_by_category(results):
    """Analyze results by image category"""
    categories = defaultdict(list)
    for r in results:
        cat = r.get('category', 'Unknown')
        categories[cat].append(r)
    
    analysis = {}
    for category, cat_results in sorted(categories.items()):
        successful = [r for r in cat_results if r.get('success') == 'True' or r.get('success') is True]
        
        analysis[category] = {
            'total_tests': len(cat_results),
            'successful': len(successful),
            'failed': len(cat_results) - len(successful),
            'success_rate': len(successful) / len(cat_results) * 100 if cat_results else 0,
            'total_latency': calculate_latency_stats(cat_results),
            'component_latencies': calculate_component_latencies(cat_results),
            'response_length': calculate_response_length_stats(cat_results)
        }
    
    return analysis


def analyze_failures(results):
    """Analyze failure cases"""
    failures = [r for r in results if r.get('success') != 'True' and r.get('success') is not True]
    
    failure_analysis = {
        'total_failures': len(failures),
        'failure_rate': len(failures) / len(results) * 100 if results else 0,
        'errors_by_type': defaultdict(int),
        'failures_by_mode': defaultdict(int),
        'failures_by_category': defaultdict(int)
    }
    
    for failure in failures:
        error = failure.get('error', 'Unknown error')
        mode = failure.get('mode', 'unknown')
        category = failure.get('category', 'unknown')
        
        # Categorize errors
        if 'SSL' in error or 'certificate' in error.lower():
            failure_analysis['errors_by_type']['SSL/Certificate'] += 1
        elif 'API' in error or 'openai' in error.lower() or 'anthropic' in error.lower():
            failure_analysis['errors_by_type']['API Error'] += 1
        elif 'timeout' in error.lower():
            failure_analysis['errors_by_type']['Timeout'] += 1
        else:
            failure_analysis['errors_by_type']['Other'] += 1
        
        failure_analysis['failures_by_mode'][mode] += 1
        failure_analysis['failures_by_category'][category] += 1
    
    return failure_analysis


def main():
    csv_path = Path('results/approach_3_specialized/raw/batch_results.csv')
    
    if not csv_path.exists():
        print(f"Error: Results file not found: {csv_path}")
        print("Please run batch_test_specialized.py first.")
        return
    
    print("Loading results...")
    results = load_results(csv_path)
    print(f"Loaded {len(results)} results")
    
    # Overall statistics
    successful = [r for r in results if r.get('success') == 'True' or r.get('success') is True]
    failed = [r for r in results if r.get('success') != 'True' and r.get('success') is not True]
    
    print(f"\nOverall Statistics:")
    print(f"  Total tests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    # Overall latency
    overall_latency = calculate_latency_stats(results)
    if overall_latency:
        print(f"\nOverall Latency Statistics:")
        print(f"  Mean: {overall_latency['mean']:.2f}s")
        print(f"  Median: {overall_latency['median']:.2f}s")
        print(f"  Std Dev: {overall_latency['stdev']:.2f}s")
        print(f"  Min: {overall_latency['min']:.2f}s")
        print(f"  Max: {overall_latency['max']:.2f}s")
        if overall_latency.get('p75'):
            print(f"  P75: {overall_latency['p75']:.2f}s")
        if overall_latency.get('p95'):
            print(f"  P95: {overall_latency['p95']:.2f}s")
    
    # Component breakdown
    component_stats = calculate_component_latencies(results)
    if component_stats:
        print(f"\nComponent Latency Breakdown:")
        for component, stats in component_stats.items():
            print(f"  {component.capitalize()}:")
            print(f"    Mean: {stats['mean']:.3f}s")
            print(f"    Median: {stats['median']:.3f}s")
            print(f"    Range: {stats['min']:.3f}s - {stats['max']:.3f}s")
    
    # Response length
    resp_stats = calculate_response_length_stats(results)
    if resp_stats:
        print(f"\nResponse Length Statistics:")
        if 'word_count' in resp_stats:
            print(f"  Word Count:")
            print(f"    Mean: {resp_stats['word_count']['mean']:.1f}")
            print(f"    Median: {resp_stats['word_count']['median']:.1f}")
            print(f"    Min: {resp_stats['word_count']['min']}")
            print(f"    Max: {resp_stats['word_count']['max']}")
        if 'token_count' in resp_stats:
            print(f"  Token Count:")
            print(f"    Mean: {resp_stats['token_count']['mean']:.1f}")
            print(f"    Median: {resp_stats['token_count']['median']:.1f}")
    
    # Cost analysis
    print(f"\nCost Analysis:")
    cost_stats = calculate_cost_estimates(results)
    print(f"  Total Cost: ${cost_stats['total_cost']:.4f}")
    print(f"  GPT-4o-mini calls: {cost_stats['gpt_calls']}")
    print(f"  Cost per query: ${cost_stats['cost_per_query']:.6f}")
    print(f"  Cost per 1000 queries: ${cost_stats['cost_per_query'] * 1000:.2f}")
    
    # Mode-specific analysis
    mode_analysis = analyze_by_mode(results)
    if mode_analysis:
        print(f"\nMode-Specific Analysis:")
        for mode, stats in sorted(mode_analysis.items()):
            print(f"\n  {mode.upper()} Mode:")
            print(f"    Total tests: {stats['total_tests']}")
            print(f"    Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
            if stats['total_latency']:
                print(f"    Mean latency: {stats['total_latency']['mean']:.2f}s")
            if 'ocr_stats' in stats:
                print(f"    OCR Stats:")
                print(f"      Mean texts per image: {stats['ocr_stats']['mean_texts_per_image']:.1f}")
                print(f"      Total texts extracted: {stats['ocr_stats']['total_texts_extracted']}")
            if 'depth_stats' in stats:
                print(f"    Depth Stats:")
                print(f"      Mean depth: {stats['depth_stats']['mean_depth']:.2f}")
                print(f"      Depth range: {stats['depth_stats']['min_depth']:.2f} - {stats['depth_stats']['max_depth']:.2f}")
    
    # Category analysis
    category_analysis = analyze_by_category(results)
    if category_analysis:
        print(f"\nCategory-Specific Analysis:")
        for category, stats in sorted(category_analysis.items()):
            print(f"\n  {category.capitalize()}:")
            print(f"    Total tests: {stats['total_tests']}")
            print(f"    Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
            if stats['total_latency']:
                print(f"    Mean latency: {stats['total_latency']['mean']:.2f}s")
    
    # Failure analysis
    failure_analysis = analyze_failures(results)
    if failure_analysis['total_failures'] > 0:
        print(f"\nFailure Analysis:")
        print(f"  Total failures: {failure_analysis['total_failures']}")
        print(f"  Failure rate: {failure_analysis['failure_rate']:.1f}%")
        print(f"  Errors by type:")
        for error_type, count in failure_analysis['errors_by_type'].items():
            print(f"    {error_type}: {count}")
        print(f"  Failures by mode:")
        for mode, count in failure_analysis['failures_by_mode'].items():
            print(f"    {mode}: {count}")
    
    # Write comprehensive report
    output_dir = Path('results/approach_3_specialized/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'comprehensive_analysis.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("APPROACH 3: SPECIALIZED MULTI-MODEL SYSTEM - COMPREHENSIVE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total tests: {len(results)}\n")
        f.write(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)\n")
        f.write(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)\n\n")
        
        if overall_latency:
            f.write("OVERALL LATENCY STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in overall_latency.items():
                if value is not None:
                    f.write(f"{key.capitalize()}: {value:.2f}s\n")
            f.write("\n")
        
        if component_stats:
            f.write("COMPONENT LATENCY BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for component, stats in component_stats.items():
                f.write(f"{component.capitalize()}:\n")
                for key, value in stats.items():
                    if value is not None:
                        f.write(f"  {key.capitalize()}: {value:.3f}s\n")
            f.write("\n")
        
        if resp_stats:
            f.write("RESPONSE LENGTH STATISTICS\n")
            f.write("-" * 80 + "\n")
            for metric, stats in resp_stats.items():
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                for key, value in stats.items():
                    if value is not None:
                        f.write(f"  {key.capitalize()}: {value:.1f}\n")
            f.write("\n")
        
        f.write("COST ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Cost: ${cost_stats['total_cost']:.4f}\n")
        f.write(f"Cost per Query: ${cost_stats['cost_per_query']:.6f}\n")
        f.write(f"Cost per 1000 Queries: ${cost_stats['cost_per_query'] * 1000:.2f}\n\n")
        
        if mode_analysis:
            f.write("MODE-SPECIFIC ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for mode, stats in sorted(mode_analysis.items()):
                f.write(f"{mode.upper()} Mode:\n")
                f.write(f"  Total tests: {stats['total_tests']}\n")
                f.write(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)\n")
                if stats['total_latency']:
                    f.write(f"  Mean latency: {stats['total_latency']['mean']:.2f}s\n")
                f.write("\n")
        
        if category_analysis:
            f.write("CATEGORY-SPECIFIC ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for category, stats in sorted(category_analysis.items()):
                f.write(f"{category.capitalize()}:\n")
                f.write(f"  Total tests: {stats['total_tests']}\n")
                f.write(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)\n")
                if stats['total_latency']:
                    f.write(f"  Mean latency: {stats['total_latency']['mean']:.2f}s\n")
                f.write("\n")
        
        if failure_analysis['total_failures'] > 0:
            f.write("FAILURE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total failures: {failure_analysis['total_failures']}\n")
            f.write(f"Failure rate: {failure_analysis['failure_rate']:.1f}%\n")
            f.write("Errors by type:\n")
            for error_type, count in failure_analysis['errors_by_type'].items():
                f.write(f"  {error_type}: {count}\n")
    
    print(f"\n✅ Comprehensive analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

