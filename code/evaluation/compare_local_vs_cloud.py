#!/usr/bin/env python3
"""
Compare Approach 4 (Local Models) with Approach 1 (Pure VLMs)
"""
import csv
import statistics
from pathlib import Path


def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') is True:
                results.append(row)
    return results


def calculate_mean_latency(results):
    """Calculate mean latency from results"""
    latencies = []
    for r in results:
        try:
            lat = float(r.get('total_latency', 0) or r.get('latency', 0))
            if lat > 0:
                latencies.append(lat)
        except (ValueError, TypeError):
            continue
    
    if not latencies:
        return None
    
    return statistics.mean(latencies)


def calculate_mean_response_length(results):
    """Calculate mean response length"""
    lengths = []
    for r in results:
        desc = r.get('description', '')
        if desc:
            lengths.append(len(desc.split()))
    
    if not lengths:
        return None
    
    return statistics.mean(lengths)


def main():
    """Compare local models with cloud VLMs"""
    print("=" * 60)
    print("Approach 4 (Local Models) vs Approach 1 (Pure VLMs) Comparison")
    print("=" * 60)
    print()
    
    # Load local results
    local_csv = Path('results/approach_4_local/raw/batch_results.csv')
    if not local_csv.exists():
        print(f"❌ Local results file not found: {local_csv}")
        return
    
    local_results = load_results(local_csv)
    print(f"Loaded {len(local_results)} local model results")
    
    # Load cloud results (Approach 1)
    cloud_csv = Path('results/approach_1_vlm/raw/batch_results.csv')
    if not cloud_csv.exists():
        print(f"⚠️  Cloud results file not found: {cloud_csv}")
        print("   Comparison will be limited to local models only")
        cloud_results = []
    else:
        cloud_results = load_results(cloud_csv)
        print(f"Loaded {len(cloud_results)} cloud VLM results")
    
    print()
    
    # Output directory
    output_dir = Path('results/approach_4_local/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'local_vs_cloud_comparison.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Approach 4 (Local Models) vs Approach 1 (Pure VLMs) Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        # Local models stats
        local_mean_latency = calculate_mean_latency(local_results)
        local_mean_length = calculate_mean_response_length(local_results)
        
        f.write("Approach 4 - Local Models:\n")
        if local_mean_latency:
            f.write(f"  Mean latency: {local_mean_latency:.3f}s\n")
        if local_mean_length:
            f.write(f"  Mean response length: {local_mean_length:.1f} words\n")
        f.write(f"  Count: {len(local_results)}\n")
        f.write(f"  Cost: $0.00 (local models, no API calls)\n\n")
        
        # Breakdown by model
        f.write("Local Models Breakdown:\n")
        for model in ['blip2']:
            model_results = [r for r in local_results if r.get('model') == model]
            if model_results:
                model_latency = calculate_mean_latency(model_results)
                model_length = calculate_mean_response_length(model_results)
                f.write(f"  {model.upper()}:\n")
                if model_latency:
                    f.write(f"    Mean latency: {model_latency:.3f}s\n")
                if model_length:
                    f.write(f"    Mean response length: {model_length:.1f} words\n")
                f.write(f"    Count: {len(model_results)}\n")
        f.write("\n")
        
        # Cloud VLMs comparison (if available)
        if cloud_results:
            cloud_mean_latency = calculate_mean_latency(cloud_results)
            cloud_mean_length = calculate_mean_response_length(cloud_results)
            
            f.write("Approach 1 - Pure VLMs (Cloud):\n")
            if cloud_mean_latency:
                f.write(f"  Mean latency: {cloud_mean_latency:.3f}s\n")
            if cloud_mean_length:
                f.write(f"  Mean response length: {cloud_mean_length:.1f} words\n")
            f.write(f"  Count: {len(cloud_results)}\n")
            f.write(f"  Cost: ~$0.013 per query (average across models)\n\n")
            
            # Comparison
            if local_mean_latency and cloud_mean_latency:
                latency_diff = local_mean_latency - cloud_mean_latency
                latency_ratio = local_mean_latency / cloud_mean_latency
                f.write("Comparison:\n")
                f.write(f"  Latency difference: {latency_diff:+.3f}s ({latency_ratio:.2f}x)\n")
                if latency_diff > 0:
                    f.write(f"    Local models are {latency_ratio:.2f}x slower\n")
                else:
                    f.write(f"    Local models are {1/latency_ratio:.2f}x faster\n")
            
            if local_mean_length and cloud_mean_length:
                length_diff = local_mean_length - cloud_mean_length
                f.write(f"  Response length difference: {length_diff:+.1f} words\n")
            
            f.write(f"  Cost difference: $0.00 vs ~$0.013 per query (100% savings with local)\n")
            f.write(f"  Privacy: Local models keep data on-device, cloud models send to API\n")
            f.write(f"  Offline capability: Local models work offline, cloud requires internet\n")
    
    print(f"✅ Comparison complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

