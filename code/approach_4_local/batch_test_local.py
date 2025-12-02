#!/usr/bin/env python3
"""
Batch testing for Approach 4: Local Models
Tests BLIP-2 on all 42 images
"""
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent))

load_dotenv()

from blip2_model import BLIP2Model


def get_all_images():
    """Get all images from data/images folder"""
    images_dir = Path('data/images')
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


def test_configuration_on_image(image_path, category, model_type, model_instance, output_dir, num_beams=3):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category
        model_type: 'blip2'
        model_instance: Loaded model instance
        output_dir: Directory to save results
        num_beams: Number of beams for beam search (1 = greedy, 3 = default)
    
    Returns:
        dict: Test result
    """
    config_name = f"{model_type}-beams{num_beams}"
    result = {
        'filename': image_path.name,
        'category': category,
        'model': model_type,
        'configuration': config_name,
        'num_beams': num_beams,
        'success': False,
        'description': None,
        'latency': None,
        'total_latency': None,
        'device': None,
        'error': None,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Track model loading time separately (only first time)
        if not hasattr(model_instance, '_loaded'):
            load_start = time.time()
            # Model is already loaded in __init__, just mark as loaded
            model_instance._loaded = True
            load_time = time.time() - load_start
        else:
            load_time = 0.0
        
        # Generate description
        if model_type == 'blip2':
            desc_result, error = model_instance.describe_image(image_path, num_beams=num_beams)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if error:
            result['error'] = error
            return result
        
        # Extract results
        result['success'] = True
        result['description'] = desc_result['description']
        result['latency'] = desc_result['latency']
        result['total_latency'] = desc_result['latency'] + load_time
        result['device'] = desc_result['device']
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
    
    return result


def save_results(results, output_dir):
    """Save results to CSV incrementally"""
    csv_path = output_dir / 'batch_results.csv'
    
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(results)


def main():
    """Run batch testing"""
    print("=" * 60)
    print("BATCH TESTING - Approach 4: Local Models")
    print("=" * 60)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print()
    
    # Output directory
    output_dir = Path('results/approach_4_local/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Models and configurations to test
    # Format: (model_type, model_name, num_beams)
    configurations = [
        ('blip2', 'BLIP-2', 1),  # Optimized: beams=1
        # ('blip2', 'BLIP-2', 3),  # Baseline: beams=3 (already in CSV, skip to avoid duplicates)
    ]
    
    # Load models once (reuse for all images)
    print("Loading models...")
    model_instances = {}
    
    try:
        print("  Loading BLIP-2...")
        model_instances['blip2'] = BLIP2Model()
        print("  ✅ BLIP-2 loaded")
    except Exception as e:
        print(f"  ❌ Failed to load BLIP-2: {e}")
        model_instances['blip2'] = None
    
    print()
    
    # Test configurations
    total_tests = len(images) * len(configurations)
    completed = 0
    all_results = []
    
    start_time = time.time()
    
    for img_idx, img_info in enumerate(images, 1):
        for model_type, model_name, num_beams in configurations:
            if model_instances[model_type] is None:
                # Skip if model failed to load
                config_name = f"{model_type}-beams{num_beams}"
                result = {
                    'filename': img_info['filename'],
                    'category': img_info['category'],
                    'model': model_type,
                    'configuration': config_name,
                    'num_beams': num_beams,
                    'success': False,
                    'description': None,
                    'latency': None,
                    'total_latency': None,
                    'device': None,
                    'error': f'Model {model_name} failed to load',
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(result)
                save_results([result], output_dir)
                completed += 1
                continue
            
            config_name = f"{model_type}-beams{num_beams}"
            print(f"[{completed + 1}/{total_tests}] Testing {model_name} (beams={num_beams}) on {img_info['filename']} ({img_info['category']})")
            
            result = test_configuration_on_image(
                img_info['path'],
                img_info['category'],
                model_type,
                model_instances[model_type],
                output_dir,
                num_beams=num_beams
            )
            
            all_results.append(result)
            save_results([result], output_dir)
            completed += 1
            
            if result['success']:
                print(f"  ✅ Success ({result['latency']:.2f}s)")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Small delay to prevent overheating
            time.sleep(0.5)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r['success'])
    
    print()
    print("=" * 60)
    print("BATCH TESTING COMPLETE")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful}/{total_tests} ({successful*100/total_tests:.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir / 'batch_results.csv'}")
    print()


if __name__ == "__main__":
    main()

