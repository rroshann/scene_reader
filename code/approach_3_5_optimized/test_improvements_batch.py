#!/usr/bin/env python3
"""
Subset test script for Approach 3.5 high-value improvements
Tests parallel execution, smart truncation, and cache collision prevention
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

# Import DepthEstimator from Approach 3
approach3_dir = project_root / "code" / "approach_3_specialized"
sys.path.insert(0, str(approach3_dir))
import importlib.util
depth_estimator_path = approach3_dir / 'depth_estimator.py'
spec = importlib.util.spec_from_file_location("depth_estimator", depth_estimator_path)
depth_estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_estimator_module)
DepthEstimator = depth_estimator_module.DepthEstimator


def get_subset_images():
    """Get a subset of 10-15 images for testing improvements"""
    images_dir = project_root / 'data/images'
    images = []
    
    # Select 5 text images for OCR mode
    text_dir = images_dir / 'text'
    if text_dir.exists():
        text_images = sorted(text_dir.glob('*.jpg'))[:5]
        for img_file in text_images:
            images.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'text',
                'mode': 'ocr'
            })
    
    # Select 5-10 navigation images for Depth mode
    for category in ['indoor', 'outdoor']:
        cat_dir = images_dir / category
        if cat_dir.exists():
            cat_images = sorted(cat_dir.glob('*.jpg'))[:5]
            for img_file in cat_images:
                images.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category,
                    'mode': 'depth'
                })
    
    return images


def test_image_with_improvements(
    image_path,
    category,
    mode,
    output_dir,
    ocr_processor=None,
    depth_estimator=None
):
    """
    Test a single image with all improvements active
    
    Args:
        image_path: Path to image
        category: Image category
        mode: 'ocr' or 'depth'
        output_dir: Directory to save results
        ocr_processor: Optional pre-initialized OCR processor
        depth_estimator: Optional pre-initialized depth estimator
    
    Returns:
        Result dict for CSV
    """
    config_name = "YOLOv8N+PaddleOCR/Depth-Anything+gpt-3.5-turbo+Cache+Adaptive+Improvements"
    
    print(f"\n  Testing: {image_path.name} ({mode} mode)")
    
    try:
        result = run_specialized_pipeline_optimized(
            image_path,
            category=category,
            mode=mode,
            yolo_size='n',
            llm_model='gpt-3.5-turbo',
            use_cache=True,
            use_adaptive=True,
            quality_mode='balanced',
            ocr_processor=ocr_processor,
            depth_estimator=depth_estimator
        )
        
        if result['success']:
            # Calculate word count
            word_count = len(result['description'].split()) if result.get('description') else 0
            
            # Extract OCR results if available
            ocr_text = None
            ocr_num_texts = None
            ocr_engine = None
            if result.get('ocr_results'):
                ocr_text = result['ocr_results'].get('full_text')
                ocr_num_texts = result['ocr_results'].get('num_texts', 0)
                ocr_engine = result['ocr_results'].get('engine')
            
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'mode': mode,
                'yolo_model': 'yolov8n',
                'llm_model': 'gpt-3.5-turbo',
                'configuration': config_name,
                'success': True,
                'total_latency': result['total_latency'],
                'detection_latency': result['detection_latency'],
                'ocr_latency': result.get('ocr_latency') if mode == 'ocr' else None,
                'depth_latency': result.get('depth_latency') if mode == 'depth' else None,
                'generation_latency': result['generation_latency'],
                'num_objects': result['num_objects'],
                'word_count': word_count,
                'tokens_used': result.get('tokens_used'),
                'description': result['description'],
                'ocr_text': ocr_text,
                'ocr_num_texts': ocr_num_texts,
                'ocr_engine': ocr_engine,
                'cache_hit': result.get('cache_hit', False),
                'complexity': result.get('complexity'),
                'max_tokens_used': result.get('max_tokens_used'),
                'quality_mode': result.get('quality_mode', 'balanced'),
                'approach_version': result.get('approach_version', '3.5'),
                'improvements_active': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"    ✅ Success! Total latency: {result['total_latency']:.3f}s")
            print(f"       Detection: {result['detection_latency']:.3f}s, "
                  f"{'OCR' if mode == 'ocr' else 'Depth'}: "
                  f"{result.get('ocr_latency' if mode == 'ocr' else 'depth_latency', 0.0):.3f}s, "
                  f"Generation: {result['generation_latency']:.3f}s")
            if result.get('cache_hit'):
                print(f"    💾 Cache HIT!")
            return csv_result
        else:
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'mode': mode,
                'yolo_model': 'yolov8n',
                'llm_model': 'gpt-3.5-turbo',
                'configuration': config_name,
                'success': False,
                'total_latency': result.get('total_latency', 0.0),
                'detection_latency': result.get('detection_latency'),
                'ocr_latency': result.get('ocr_latency') if mode == 'ocr' else None,
                'depth_latency': result.get('depth_latency') if mode == 'depth' else None,
                'generation_latency': result.get('generation_latency'),
                'num_objects': result.get('num_objects', 0),
                'word_count': 0,
                'tokens_used': None,
                'description': None,
                'ocr_text': None,
                'ocr_num_texts': None,
                'ocr_engine': None,
                'cache_hit': False,
                'complexity': result.get('complexity'),
                'max_tokens_used': result.get('max_tokens_used'),
                'quality_mode': result.get('quality_mode', 'balanced'),
                'approach_version': result.get('approach_version', '3.5'),
                'improvements_active': True,
                'error': result.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
            return csv_result
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            'filename': image_path.name,
            'category': category,
            'mode': mode,
            'yolo_model': 'yolov8n',
            'llm_model': 'gpt-3.5-turbo',
            'configuration': config_name,
            'success': False,
            'total_latency': 0.0,
            'detection_latency': None,
            'ocr_latency': None,
            'depth_latency': None,
            'generation_latency': None,
            'num_objects': 0,
            'word_count': 0,
            'tokens_used': None,
            'description': None,
            'ocr_text': None,
            'ocr_num_texts': None,
            'ocr_engine': None,
            'cache_hit': False,
            'complexity': None,
            'max_tokens_used': None,
            'quality_mode': 'balanced',
            'approach_version': '3.5',
            'improvements_active': True,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Run subset test with improvements"""
    print("=" * 70)
    print("Approach 3.5: Subset Test with High-Value Improvements")
    print("=" * 70)
    print("\nTesting improvements:")
    print("  1. Parallel execution for depth mode")
    print("  2. Smart prompt truncation")
    print("  3. Cache key collision prevention")
    print()
    
    # Get subset of images
    images = get_subset_images()
    print(f"Found {len(images)} images for testing")
    
    if not images:
        print("❌ No images found! Check data/images directory.")
        return
    
    # Create output directory
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Warmup models
    print("\n🔥 Warming up models...")
    warmup_models(mode='both')
    
    # Initialize processors (reuse across tests)
    ocr_processor = None
    depth_estimator = None
    
    try:
        ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
        print("✅ OCR processor initialized")
    except Exception as e:
        print(f"⚠️  OCR processor initialization failed: {e}")
    
    try:
        depth_estimator = DepthEstimator()
        print("✅ Depth estimator initialized")
    except Exception as e:
        print(f"⚠️  Depth estimator initialization failed: {e}")
    
    # Run tests
    print(f"\n📊 Running tests on {len(images)} images...")
    print("=" * 70)
    
    results = []
    successful = 0
    failed = 0
    
    for i, img_info in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_info['filename']}")
        
        result = test_image_with_improvements(
            img_info['path'],
            img_info['category'],
            img_info['mode'],
            output_dir,
            ocr_processor=ocr_processor,
            depth_estimator=depth_estimator
        )
        
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    # Save results
    output_file = output_dir / 'subset_test_improvements.csv'
    fieldnames = [
        'filename', 'category', 'mode', 'yolo_model', 'llm_model', 'configuration',
        'success', 'total_latency', 'detection_latency', 'ocr_latency', 'depth_latency',
        'generation_latency', 'num_objects', 'word_count', 'tokens_used', 'description',
        'ocr_text', 'ocr_num_texts', 'ocr_engine', 'cache_hit', 'complexity',
        'max_tokens_used', 'quality_mode', 'approach_version', 'improvements_active',
        'error', 'timestamp'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(results)*100:.1f}%)")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        avg_latency = sum(r['total_latency'] for r in successful_results) / len(successful_results)
        avg_detection = sum(r['detection_latency'] for r in successful_results if r['detection_latency']) / len([r for r in successful_results if r['detection_latency']])
        avg_generation = sum(r['generation_latency'] for r in successful_results if r['generation_latency']) / len([r for r in successful_results if r['generation_latency']])
        
        depth_results = [r for r in successful_results if r['mode'] == 'depth' and r['depth_latency']]
        ocr_results = [r for r in successful_results if r['mode'] == 'ocr' and r['ocr_latency']]
        
        print(f"\nAverage Latency:")
        print(f"  Total: {avg_latency:.3f}s")
        print(f"  Detection: {avg_detection:.3f}s")
        if depth_results:
            avg_depth = sum(r['depth_latency'] for r in depth_results) / len(depth_results)
            print(f"  Depth: {avg_depth:.3f}s ({len(depth_results)} tests)")
        if ocr_results:
            avg_ocr = sum(r['ocr_latency'] for r in ocr_results) / len(ocr_results)
            print(f"  OCR: {avg_ocr:.3f}s ({len(ocr_results)} tests)")
        print(f"  Generation: {avg_generation:.3f}s")
        
        cache_hits = sum(1 for r in successful_results if r.get('cache_hit'))
        print(f"\nCache hits: {cache_hits}/{successful} ({cache_hits/successful*100:.1f}%)")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

