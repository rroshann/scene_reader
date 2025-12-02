#!/usr/bin/env python3
"""
Batch test Approach 3.5: Optimized Specialized Multi-Model System
Tests OCR mode (3A) on text images and Depth mode (3B) on navigation images
With optimizations: GPT-3.5-turbo, caching, adaptive max_tokens, PaddleOCR
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


def get_all_images():
    """Get all images from data/images folder, filtered for Approach 3.5"""
    images_dir = project_root / 'data/images'
    images = []
    
    # Text images for OCR mode (3A)
    text_dir = images_dir / 'text'
    if text_dir.exists():
        for img_file in sorted(text_dir.glob('*.png')) + sorted(text_dir.glob('*.jpg')) + sorted(text_dir.glob('*.jpeg')):
            images.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'text',
                'mode': 'ocr'
            })
    
    # Navigation images for Depth mode (3B)
    for category in ['indoor', 'outdoor']:
        cat_dir = images_dir / category
        if cat_dir.exists():
            for img_file in sorted(cat_dir.glob('*.png')) + sorted(cat_dir.glob('*.jpg')) + sorted(cat_dir.glob('*.jpeg')):
                images.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category,
                    'mode': 'depth'
                })
    
    return images


def test_configuration_on_image(
    image_path,
    category,
    mode,
    yolo_size,
    llm_model,
    output_dir,
    use_cache=True,
    use_adaptive=True,
    quality_mode='balanced',
    improvements_active=True,
    ocr_processor=None,
    depth_estimator=None
):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category
        mode: 'ocr' or 'depth'
        yolo_size: 'n' (nano)
        llm_model: 'gpt-3.5-turbo' (optimized) or 'gpt-4o-mini' (baseline)
        output_dir: Directory to save results
        use_cache: Whether to use caching
        use_adaptive: Whether to use adaptive max_tokens
        quality_mode: Quality mode ('fast', 'balanced', 'quality')
        improvements_active: Whether high-value improvements are active
        ocr_processor: Optional pre-initialized OCR processor
        depth_estimator: Optional pre-initialized depth estimator
    
    Returns:
        Result dict for CSV
    """
    config_name = f"YOLOv8{yolo_size.upper()}+{'PaddleOCR' if mode == 'ocr' else 'Depth-Anything'}+{llm_model}"
    if use_cache:
        config_name += "+Cache"
    if use_adaptive:
        config_name += "+Adaptive"
    if improvements_active:
        config_name += "+Improvements"
    
    print(f"\n  Testing: {config_name}")
    
    try:
        result = run_specialized_pipeline_optimized(
            image_path,
            category=category,
            mode=mode,
            yolo_size=yolo_size,
            llm_model=llm_model,
            use_cache=use_cache,
            use_adaptive=use_adaptive,
            quality_mode=quality_mode,
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
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'configuration': config_name,
                'success': True,
                'total_latency': result['total_latency'],
                'detection_latency': result['detection_latency'],
                'ocr_latency': result.get('ocr_latency', 0.0) if mode == 'ocr' else None,
                'depth_latency': result.get('depth_latency', 0.0) if mode == 'depth' else None,
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
                'quality_mode': result.get('quality_mode', quality_mode),
                'improvements_active': improvements_active,
                'approach_version': result.get('approach_version', '3.5'),
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"    ✅ Success! Total latency: {result['total_latency']:.2f}s")
            if result.get('cache_hit'):
                print(f"    💾 Cache HIT!")
            return csv_result
        else:
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'mode': mode,
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
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
                'quality_mode': result.get('quality_mode', quality_mode),
                'improvements_active': improvements_active,
                'approach_version': result.get('approach_version', '3.5'),
                'error': result.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
            return csv_result
            
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        return {
            'filename': image_path.name,
            'category': category,
            'mode': mode,
            'yolo_model': f'yolov8{yolo_size}',
            'llm_model': llm_model,
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
            'quality_mode': quality_mode,
            'improvements_active': improvements_active,
            'approach_version': '3.5',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main batch testing function"""
    print("=" * 80)
    print("APPROACH 3.5: OPTIMIZED SPECIALIZED MULTI-MODEL SYSTEM - BATCH TESTING")
    print("=" * 80)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print(f"  - Text images (OCR mode): {sum(1 for img in images if img['mode'] == 'ocr')}")
    print(f"  - Navigation images (Depth mode): {sum(1 for img in images if img['mode'] == 'depth')}")
    print()
    
    # Create output directory
    output_dir = project_root / 'results' / 'approach_3_5_optimized' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV file path (with improvements)
    csv_file = output_dir / 'batch_results_with_improvements.csv'
    
    # Check if CSV exists and load existing results
    existing_results = {}
    if csv_file.exists():
        print(f"Loading existing results from {csv_file}...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['filename'], row['configuration'])
                existing_results[key] = row
        print(f"  Loaded {len(existing_results)} existing results")
    
    # Test configurations (optimized only with improvements)
    configurations = [
        {
            'yolo_size': 'n',
            'llm_model': 'gpt-3.5-turbo',
            'use_cache': True,
            'use_adaptive': True,
            'quality_mode': 'balanced',
            'improvements_active': True,
            'name': 'Optimized (GPT-3.5-turbo + Cache + Adaptive + Improvements)'
        }
    ]
    
    # Warmup models
    print("\n" + "=" * 80)
    print("WARMING UP MODELS")
    print("=" * 80)
    warmup_models(mode='both')
    print()
    
    # Initialize model instances for reuse
    ocr_processor = None
    depth_estimator = None
    
    try:
        ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
    except Exception as e:
        print(f"Warning: Could not initialize OCR processor: {e}")
    
    try:
        depth_estimator = DepthEstimator()
    except Exception as e:
        print(f"Warning: Could not initialize depth estimator: {e}")
    
    # Test each configuration
    all_results = []
    total_tests = len(images) * len(configurations)
    current_test = 0
    
    for config in configurations:
        print("\n" + "=" * 80)
        print(f"CONFIGURATION: {config['name']}")
        print("=" * 80)
        
        for img_info in images:
            current_test += 1
            image_path = img_info['path']
            category = img_info['category']
            mode = img_info['mode']
            
            # Check if already tested
            config_name = f"YOLOv8{config['yolo_size'].upper()}+{'PaddleOCR' if mode == 'ocr' else 'Depth-Anything'}+{config['llm_model']}"
            if config['use_cache']:
                config_name += "+Cache"
            if config['use_adaptive']:
                config_name += "+Adaptive"
            if config.get('improvements_active', False):
                config_name += "+Improvements"
            
            key = (image_path.name, config_name)
            if key in existing_results:
                print(f"\n[{current_test}/{total_tests}] Skipping {image_path.name} (already tested)")
                all_results.append(existing_results[key])
                continue
            
            print(f"\n[{current_test}/{total_tests}] Testing {image_path.name} ({category}, {mode})")
            
            result = test_configuration_on_image(
                image_path,
                category,
                mode,
                config['yolo_size'],
                config['llm_model'],
                output_dir,
                use_cache=config['use_cache'],
                use_adaptive=config['use_adaptive'],
                quality_mode=config.get('quality_mode', 'balanced'),
                improvements_active=config.get('improvements_active', True),
                ocr_processor=ocr_processor if mode == 'ocr' else None,
                depth_estimator=depth_estimator if mode == 'depth' else None
            )
            
            all_results.append(result)
            
            # Save incrementally
            fieldnames = [
                'filename', 'category', 'mode', 'yolo_model', 'llm_model', 'configuration',
                'success', 'total_latency', 'detection_latency', 'ocr_latency', 'depth_latency',
                'generation_latency', 'num_objects', 'word_count', 'tokens_used', 'description',
                'ocr_text', 'ocr_num_texts', 'ocr_engine', 'cache_hit', 'complexity',
                'max_tokens_used', 'quality_mode', 'improvements_active', 'approach_version', 'error', 'timestamp'
            ]
            
            file_exists = csv_file.exists()
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH TESTING COMPLETE")
    print("=" * 80)
    print(f"Total tests: {len(all_results)}")
    successful = sum(1 for r in all_results if r.get('success'))
    print(f"Successful: {successful} ({100 * successful / len(all_results):.1f}%)")
    print(f"Results saved to: {csv_file}")
    
    # Cache statistics
    cache_hits = sum(1 for r in all_results if r.get('cache_hit'))
    if cache_hits > 0:
        print(f"\nCache Performance:")
        print(f"  Cache hits: {cache_hits} ({100 * cache_hits / len(all_results):.1f}%)")
        cache_hit_latencies = [r['total_latency'] for r in all_results if r.get('cache_hit')]
        if cache_hit_latencies:
            print(f"  Mean cache hit latency: {sum(cache_hit_latencies) / len(cache_hit_latencies):.3f}s")


if __name__ == '__main__':
    main()

