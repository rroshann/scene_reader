#!/usr/bin/env python3
"""
Test cache effectiveness for Approach 3.5
Runs same images twice to measure cache hit rate and speedup
"""
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
approach3_5_dir = project_root / "code" / "approach_3_5_optimized"
sys.path.insert(0, str(approach3_5_dir))

load_dotenv()

# Import pipeline
from specialized_pipeline_optimized import (
    run_specialized_pipeline_optimized,
    warmup_models
)
from ocr_processor_optimized import OCRProcessorOptimized
from cache_manager import get_cache_manager

# Import DepthEstimator from Approach 3
approach3_dir = project_root / "code" / "approach_3_specialized"
sys.path.insert(0, str(approach3_dir))
import importlib.util
depth_estimator_path = approach3_dir / 'depth_estimator.py'
spec = importlib.util.spec_from_file_location("depth_estimator", depth_estimator_path)
depth_estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_estimator_module)
DepthEstimator = depth_estimator_module.DepthEstimator


def get_test_images(n=10):
    """Get a subset of test images for cache testing"""
    images_dir = project_root / 'data/images'
    images = []
    
    # Get 5 text images
    text_dir = images_dir / 'text'
    if text_dir.exists():
        text_files = sorted(list(text_dir.glob('*.png')) + list(text_dir.glob('*.jpg')) + list(text_dir.glob('*.jpeg')))[:5]
        for img_file in text_files:
            images.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'text',
                'mode': 'ocr'
            })
    
    # Get 5 navigation images
    for category in ['indoor', 'outdoor']:
        cat_dir = images_dir / category
        if cat_dir.exists():
            cat_files = sorted(list(cat_dir.glob('*.png')) + list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.jpeg')))[:5]
            for img_file in cat_files:
                if len([img for img in images if img['mode'] == 'depth']) < 5:
                    images.append({
                        'path': img_file,
                        'filename': img_file.name,
                        'category': category,
                        'mode': 'depth'
                    })
                if len([img for img in images if img['mode'] == 'depth']) >= 5:
                    break
        if len([img for img in images if img['mode'] == 'depth']) >= 5:
            break
    
    return images[:n]


def test_cache_effectiveness():
    """Test cache effectiveness by running same images twice"""
    print("=" * 80)
    print("APPROACH 3.5: CACHE EFFECTIVENESS TESTING")
    print("=" * 80)
    print()
    
    # Get test images
    test_images = get_test_images(10)
    print(f"Selected {len(test_images)} images for cache testing")
    print(f"  - Text images (OCR mode): {sum(1 for img in test_images if img['mode'] == 'ocr')}")
    print(f"  - Navigation images (Depth mode): {sum(1 for img in test_images if img['mode'] == 'depth')}")
    print()
    
    # Clear cache before testing
    cache_manager = get_cache_manager()
    cache_manager.clear_cache()
    print("✅ Cache cleared before testing")
    print()
    
    # Warmup models
    print("=" * 80)
    print("WARMING UP MODELS")
    print("=" * 80)
    warmup_models(mode='both')
    print()
    
    # Initialize model instances
    ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
    depth_estimator = DepthEstimator()
    
    # Test results
    results = []
    
    print("=" * 80)
    print("FIRST RUN (No Cache)")
    print("=" * 80)
    
    # First run: no cache hits
    for i, img_info in enumerate(test_images, 1):
        image_path = img_info['path']
        category = img_info['category']
        mode = img_info['mode']
        
        print(f"\n[{i}/{len(test_images)}] First run: {image_path.name} ({category}, {mode})")
        
        result = run_specialized_pipeline_optimized(
            image_path,
            category=category,
            mode=mode,
            yolo_size='n',
            llm_model='gpt-3.5-turbo',
            use_cache=True,
            use_adaptive=True,
            quality_mode='balanced',
            ocr_processor=ocr_processor if mode == 'ocr' else None,
            depth_estimator=depth_estimator if mode == 'depth' else None
        )
        
        results.append({
            'filename': image_path.name,
            'category': category,
            'mode': mode,
            'run': 1,
            'cache_hit': result.get('cache_hit', False),
            'total_latency': result['total_latency'],
            'detection_latency': result.get('detection_latency'),
            'ocr_latency': result.get('ocr_latency') if mode == 'ocr' else None,
            'depth_latency': result.get('depth_latency') if mode == 'depth' else None,
            'generation_latency': result.get('generation_latency'),
            'success': result['success']
        })
        
        if result['success']:
            print(f"  ✅ Success! Latency: {result['total_latency']:.3f}s (cache: {result.get('cache_hit', False)})")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("SECOND RUN (With Cache)")
    print("=" * 80)
    
    # Second run: should have cache hits
    for i, img_info in enumerate(test_images, 1):
        image_path = img_info['path']
        category = img_info['category']
        mode = img_info['mode']
        
        print(f"\n[{i}/{len(test_images)}] Second run: {image_path.name} ({category}, {mode})")
        
        result = run_specialized_pipeline_optimized(
            image_path,
            category=category,
            mode=mode,
            yolo_size='n',
            llm_model='gpt-3.5-turbo',
            use_cache=True,
            use_adaptive=True,
            quality_mode='balanced',
            ocr_processor=ocr_processor if mode == 'ocr' else None,
            depth_estimator=depth_estimator if mode == 'depth' else None
        )
        
        results.append({
            'filename': image_path.name,
            'category': category,
            'mode': mode,
            'run': 2,
            'cache_hit': result.get('cache_hit', False),
            'total_latency': result['total_latency'],
            'detection_latency': result.get('detection_latency'),
            'ocr_latency': result.get('ocr_latency') if mode == 'ocr' else None,
            'depth_latency': result.get('depth_latency') if mode == 'depth' else None,
            'generation_latency': result.get('generation_latency'),
            'success': result['success']
        })
        
        if result['success']:
            cache_status = "✅ CACHE HIT" if result.get('cache_hit', False) else "❌ Cache miss"
            print(f"  ✅ Success! Latency: {result['total_latency']:.3f}s ({cache_status})")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("CACHE EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    
    # Separate first and second runs
    first_run = [r for r in results if r['run'] == 1 and r['success']]
    second_run = [r for r in results if r['run'] == 2 and r['success']]
    
    if len(first_run) == 0 or len(second_run) == 0:
        print("❌ Not enough successful tests for analysis")
        return
    
    # Cache hit rate
    cache_hits = sum(1 for r in second_run if r['cache_hit'])
    cache_hit_rate = cache_hits / len(second_run) * 100
    print(f"\nCache Hit Rate: {cache_hits}/{len(second_run)} ({cache_hit_rate:.1f}%)")
    
    # Latency comparison
    first_latency = sum(r['total_latency'] for r in first_run) / len(first_run)
    second_latency = sum(r['total_latency'] for r in second_run) / len(second_run)
    speedup = first_latency / second_latency if second_latency > 0 else 0
    
    print(f"\nLatency Comparison:")
    print(f"  First run (no cache):  {first_latency:.3f}s")
    print(f"  Second run (cached):   {second_latency:.3f}s")
    print(f"  Speedup:               {speedup:.2f}x")
    
    # Cached vs non-cached in second run
    cached_second = [r for r in second_run if r['cache_hit']]
    non_cached_second = [r for r in second_run if not r['cache_hit']]
    
    if len(cached_second) > 0:
        cached_latency = sum(r['total_latency'] for r in cached_second) / len(cached_second)
        print(f"\nSecond Run Breakdown:")
        print(f"  Cached:     {cached_latency:.3f}s (n={len(cached_second)})")
        
        if len(non_cached_second) > 0:
            non_cached_latency = sum(r['total_latency'] for r in non_cached_second) / len(non_cached_second)
            print(f"  Non-cached: {non_cached_latency:.3f}s (n={len(non_cached_second)})")
            cache_speedup = non_cached_latency / cached_latency if cached_latency > 0 else 0
            print(f"  Cache speedup: {cache_speedup:.2f}x")
    
    # Save results
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / 'cache_effectiveness_test.csv'
    
    fieldnames = [
        'filename', 'category', 'mode', 'run', 'cache_hit', 'total_latency',
        'detection_latency', 'ocr_latency', 'depth_latency', 'generation_latency', 'success'
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {csv_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_cache_effectiveness()

