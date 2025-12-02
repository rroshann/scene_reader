#!/usr/bin/env python3
"""
Test Adaptive Parameters
Compares fixed vs adaptive max_tokens on subset
"""
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
approach25_dir = project_root / "code" / "approach_2_5_optimized"
sys.path.insert(0, str(approach25_dir))

load_dotenv()

from hybrid_pipeline_optimized import run_hybrid_pipeline_optimized


def get_subset_images():
    """Get 8-image subset (2 per category)"""
    images_dir = project_root / 'data/images'
    subset = []
    
    categories = ['gaming', 'indoor', 'outdoor', 'text']
    for category in categories:
        cat_dir = images_dir / category
        if cat_dir.exists():
            images = sorted(cat_dir.glob('*.png')) + sorted(cat_dir.glob('*.jpg')) + sorted(cat_dir.glob('*.jpeg'))
            for img_file in images[:2]:
                subset.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category
                })
    
    return subset


def main():
    """Test adaptive parameters"""
    print("=" * 80)
    print("ADAPTIVE PARAMETERS TEST")
    print("=" * 80)
    print()
    
    images = get_subset_images()
    print(f"Testing on {len(images)} images (2 per category)")
    print()
    
    results_fixed = []
    results_adaptive = []
    
    for img_info in images:
        filename = img_info['filename']
        category = img_info['category']
        print(f"\n📸 {filename} ({category})")
        
        # Test fixed max_tokens
        print("  Testing fixed max_tokens (200)...")
        result_fixed = run_hybrid_pipeline_optimized(
            img_info['path'],
            use_cache=False,  # Disable cache for fair comparison
            use_adaptive=False
        )
        if result_fixed['success']:
            results_fixed.append({
                'filename': filename,
                'category': category,
                'latency': result_fixed['total_latency'],
                'word_count': len(result_fixed['description'].split())
            })
            print(f"    ✅ Latency: {result_fixed['total_latency']:.2f}s, Words: {len(result_fixed['description'].split())}")
        
        time.sleep(0.5)
        
        # Test adaptive max_tokens
        print("  Testing adaptive max_tokens...")
        result_adaptive = run_hybrid_pipeline_optimized(
            img_info['path'],
            use_cache=False,  # Disable cache for fair comparison
            use_adaptive=True
        )
        if result_adaptive['success']:
            results_adaptive.append({
                'filename': filename,
                'category': category,
                'latency': result_adaptive['total_latency'],
                'word_count': len(result_adaptive['description'].split())
            })
            print(f"    ✅ Latency: {result_adaptive['total_latency']:.2f}s, Words: {len(result_adaptive['description'].split())}")
        
        time.sleep(0.5)
    
    # Summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    if results_fixed and results_adaptive:
        fixed_latencies = [r['latency'] for r in results_fixed]
        adaptive_latencies = [r['latency'] for r in results_adaptive]
        fixed_words = [r['word_count'] for r in results_fixed]
        adaptive_words = [r['word_count'] for r in results_adaptive]
        
        avg_fixed_latency = sum(fixed_latencies) / len(fixed_latencies)
        avg_adaptive_latency = sum(adaptive_latencies) / len(adaptive_latencies)
        avg_fixed_words = sum(fixed_words) / len(fixed_words)
        avg_adaptive_words = sum(adaptive_words) / len(adaptive_words)
        
        speedup = ((avg_fixed_latency - avg_adaptive_latency) / avg_fixed_latency) * 100 if avg_fixed_latency > 0 else 0
        
        print(f"Fixed max_tokens (200):")
        print(f"  Mean Latency: {avg_fixed_latency:.2f}s")
        print(f"  Mean Words: {avg_fixed_words:.1f}")
        print()
        print(f"Adaptive max_tokens:")
        print(f"  Mean Latency: {avg_adaptive_latency:.2f}s")
        print(f"  Mean Words: {avg_adaptive_words:.1f}")
        print()
        print(f"Speedup: {speedup:.1f}% ({'✅ Faster' if speedup > 0 else '❌ Slower'})")
        print(f"Word reduction: {avg_fixed_words - avg_adaptive_words:.1f} words ({'✅ Shorter' if avg_adaptive_words < avg_fixed_words else '❌ Longer'})")
    
    # Save report
    report_path = project_root / 'results' / 'approach_2_5_optimized' / 'analysis' / 'adaptive_params_test.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Adaptive Parameters Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Tested: {len(images)}\n\n")
        
        if results_fixed and results_adaptive:
            f.write(f"Fixed max_tokens: {avg_fixed_latency:.2f}s mean, {avg_fixed_words:.1f} words\n")
            f.write(f"Adaptive max_tokens: {avg_adaptive_latency:.2f}s mean, {avg_adaptive_words:.1f} words\n")
            f.write(f"Speedup: {speedup:.1f}%\n")
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()

