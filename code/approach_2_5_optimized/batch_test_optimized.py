#!/usr/bin/env python3
"""
Batch test for Approach 2.5: Optimized YOLO+LLM Pipeline
Tests GPT-3.5-turbo on all 42 images
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

# Add approach_2_5_optimized to path
approach25_dir = project_root / "code" / "approach_2_5_optimized"
sys.path.insert(0, str(approach25_dir))

load_dotenv()

# Import optimized pipeline
from hybrid_pipeline_optimized import run_hybrid_pipeline_optimized


def get_all_images():
    """Get all images from data/images folder"""
    images_dir = project_root / 'data/images'
    images = []
    
    for category in ['gaming', 'indoor', 'outdoor', 'text']:
        cat_dir = images_dir / category
        if cat_dir.exists():
            for img_file in sorted(cat_dir.glob('*.png')) + sorted(cat_dir.glob('*.jpg')) + sorted(cat_dir.glob('*.jpeg')):
                images.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category
                })
    
    return images


def test_configuration_on_image(image_path, category, yolo_size, llm_model, output_dir):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category
        yolo_size: 'n' (nano) - only testing nano for speed
        llm_model: 'gpt-3.5-turbo' (optimized)
        output_dir: Directory to save results
    
    Returns:
        Result dict for CSV
    """
    config_name = f"YOLOv8{yolo_size.upper()}+{llm_model}"
    print(f"\n  Testing: {config_name}")
    
    try:
        result = run_hybrid_pipeline_optimized(
            image_path,
            yolo_size=yolo_size,
            llm_model=llm_model
        )
        
        if result['success']:
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'approach_version': '2.5',  # Identifier
                'configuration': config_name,
                'description': result['description'],
                'detection_latency': result['detection_latency'],
                'generation_latency': result['generation_latency'],
                'total_latency': result['total_latency'],
                'num_objects_detected': result['num_objects'],
                'objects_detected': str(result['objects_detected']),  # Convert to string for CSV
                'avg_confidence': result['detection_summary'].get('avg_confidence', 0.0),
                'tokens_used': result.get('tokens_used'),
                'cache_hit': result.get('cache_hit', False),  # Track cache hits
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            print(f"    ✅ Success! Total latency: {result['total_latency']:.2f}s")
            return csv_result
        else:
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'approach_version': '2.5',
                'configuration': config_name,
                'description': None,
                'detection_latency': result.get('detection_latency'),
                'generation_latency': result.get('generation_latency'),
                'total_latency': result.get('total_latency'),
                'num_objects_detected': result.get('num_objects', 0),
                'objects_detected': None,
                'avg_confidence': None,
                'tokens_used': None,
                'cache_hit': False,
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'timestamp': datetime.now().isoformat()
            }
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
            return csv_result
            
    except Exception as e:
        csv_result = {
            'filename': image_path.name,
            'category': category,
            'yolo_model': f'yolov8{yolo_size}',
            'llm_model': llm_model,
            'approach_version': '2.5',
            'configuration': config_name,
            'description': None,
            'detection_latency': None,
            'generation_latency': None,
            'total_latency': None,
            'num_objects_detected': 0,
            'objects_detected': None,
                'avg_confidence': None,
                'tokens_used': None,
                'cache_hit': False,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        print(f"    ❌ Exception: {e}")
        return csv_result


def save_results(results, output_dir):
    """Save results to CSV file with incremental saving"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'batch_results.csv'
    
    # Append to existing file or create new
    file_exists = output_file.exists()
    
    fieldnames = [
        'filename', 'category', 'yolo_model', 'llm_model', 'approach_version',
        'configuration', 'description', 'detection_latency', 'generation_latency',
        'total_latency', 'num_objects_detected', 'objects_detected', 'avg_confidence',
        'tokens_used', 'success', 'error', 'timestamp'
    ]
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"\n  💾 Saved {len(results)} result(s) to {output_file}")


def main():
    """Run batch test for Approach 2.5"""
    print("=" * 80)
    print("APPROACH 2.5: BATCH TEST - GPT-3.5-turbo Optimization")
    print("=" * 80)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print()
    
    # Configuration: YOLOv8N + GPT-3.5-turbo (optimized for speed)
    yolo_size = 'n'  # Nano - fastest
    llm_model = 'gpt-3.5-turbo'  # Optimized model
    
    # Output directory
    output_dir = project_root / 'results' / 'approach_2_5_optimized' / 'raw'
    
    print(f"Configuration: YOLOv8{yolo_size.upper()} + {llm_model}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Test each image
    all_results = []
    start_time = time.time()
    
    for i, img_info in enumerate(images, 1):
        filename = img_info['filename']
        category = img_info['category']
        image_path = img_info['path']
        
        print(f"[{i}/{len(images)}] 📸 {filename} ({category})")
        
        result = test_configuration_on_image(
            image_path,
            category,
            yolo_size,
            llm_model,
            output_dir
        )
        
        all_results.append(result)
        
        # Incremental save after each test (prevent data loss)
        save_results([result], output_dir)
        
        # Small delay to avoid rate limiting
        if i < len(images):
            time.sleep(0.5)
    
    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful
    
    print()
    print("=" * 80)
    print("BATCH TEST COMPLETE")
    print("=" * 80)
    print(f"Total images tested: {len(images)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per image: {total_time/len(images):.1f}s")
    
    if successful > 0:
        latencies = [r['total_latency'] for r in all_results if r['success']]
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print(f"\nLatency Statistics:")
        print(f"  Mean: {avg_latency:.2f}s")
        print(f"  Min: {min_latency:.2f}s")
        print(f"  Max: {max_latency:.2f}s")
        print(f"  <2s target: {'✅ ACHIEVED' if avg_latency < 2.0 else '❌ NOT MET'} (mean: {avg_latency:.2f}s)")
    
    print(f"\nResults saved to: {output_dir / 'batch_results.csv'}")


if __name__ == "__main__":
    main()

