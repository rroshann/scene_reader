#!/usr/bin/env python3
"""
Batch test Approach 3: Specialized Multi-Model System
Tests OCR mode (3A) on text images and Depth mode (3B) on navigation images
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
approach3_dir = project_root / "code" / "approach_3_specialized"
sys.path.insert(0, str(approach3_dir))

load_dotenv()

# Import pipeline
from specialized_pipeline import run_specialized_pipeline
from depth_estimator import DepthEstimator
from ocr_processor import OCRProcessor


def get_all_images():
    """Get all images from data/images folder, filtered for Approach 3"""
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


def test_configuration_on_image(image_path, category, mode, yolo_size, llm_model, output_dir, ocr_processor=None, depth_estimator=None):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category
        mode: 'ocr' or 'depth'
        yolo_size: 'n' (nano)
        llm_model: 'gpt-4o-mini'
        output_dir: Directory to save results
        ocr_processor: Optional pre-initialized OCR processor
        depth_estimator: Optional pre-initialized depth estimator
    
    Returns:
        Result dict for CSV
    """
    config_name = f"YOLOv8{yolo_size.upper()}+{'EasyOCR' if mode == 'ocr' else 'Depth-Anything'}+{llm_model}"
    print(f"\n  Testing: {config_name}")
    
    try:
        result = run_specialized_pipeline(
            image_path,
            category=category,
            mode=mode,
            yolo_size=yolo_size,
            llm_model=llm_model,
            ocr_processor=ocr_processor,
            depth_estimator=depth_estimator
        )
        
        if result['success']:
            # Calculate word count
            word_count = len(result['description'].split()) if result.get('description') else 0
            
            # Extract OCR results if available
            ocr_text = None
            ocr_num_texts = None
            if result.get('ocr_results') and result['ocr_results'].get('full_text'):
                ocr_text = result['ocr_results']['full_text']
                ocr_num_texts = result['ocr_results'].get('num_texts', 0)
            
            # Extract depth results if available
            depth_mean = None
            if result.get('depth_results') and result['depth_results'].get('mean_depth') is not None:
                depth_mean = result['depth_results']['mean_depth']
            
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'mode': mode,
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'configuration': config_name,
                'description': result['description'],
                'detection_latency': result.get('detection_latency'),
                'ocr_latency': result.get('ocr_latency'),
                'depth_latency': result.get('depth_latency'),
                'generation_latency': result.get('generation_latency'),
                'total_latency': result.get('total_latency'),
                'num_objects_detected': result.get('num_objects', 0),
                'objects_detected': str(result.get('objects_detected', [])),
                'ocr_text': ocr_text,
                'ocr_num_texts': ocr_num_texts,
                'depth_mean': depth_mean,
                'word_count': word_count,
                'tokens_used': result.get('tokens_used'),
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
                'mode': mode,
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'configuration': config_name,
                'description': None,
                'detection_latency': result.get('detection_latency'),
                'ocr_latency': result.get('ocr_latency'),
                'depth_latency': result.get('depth_latency'),
                'generation_latency': result.get('generation_latency'),
                'total_latency': result.get('total_latency'),
                'num_objects_detected': result.get('num_objects', 0),
                'objects_detected': None,
                'ocr_text': None,
                'ocr_num_texts': None,
                'depth_mean': None,
                'word_count': None,
                'tokens_used': None,
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
            'mode': mode,
            'yolo_model': f'yolov8{yolo_size}',
            'llm_model': llm_model,
            'configuration': config_name,
            'description': None,
            'detection_latency': None,
            'ocr_latency': None,
            'depth_latency': None,
            'generation_latency': None,
            'total_latency': None,
            'num_objects_detected': 0,
            'objects_detected': None,
            'ocr_text': None,
            'ocr_num_texts': None,
            'depth_mean': None,
            'word_count': None,
            'tokens_used': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(f"    ❌ Exception: {e}")
        return csv_result


def save_results(results, output_dir):
    """Save results to CSV file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'batch_results.csv'
    
    file_exists = output_file.exists()
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename', 'category', 'mode', 'yolo_model', 'llm_model', 'configuration',
            'description', 'detection_latency', 'ocr_latency', 'depth_latency',
            'generation_latency', 'total_latency', 'num_objects_detected', 'objects_detected',
            'ocr_text', 'ocr_num_texts', 'depth_mean', 'word_count', 'tokens_used',
            'success', 'error', 'timestamp'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)


def main():
    print("=" * 80)
    print("APPROACH 3: BATCH TEST - SPECIALIZED MULTI-MODEL SYSTEM")
    print("=" * 80)
    print()
    
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print(f"  - OCR mode (text): {sum(1 for img in images if img['mode'] == 'ocr')} images")
    print(f"  - Depth mode (navigation): {sum(1 for img in images if img['mode'] == 'depth')} images")
    
    yolo_size = 'n'  # Use nano for speed
    llm_model = 'gpt-4o-mini'
    
    print(f"\nConfiguration:")
    print(f"  YOLO: yolov8{yolo_size}")
    print(f"  LLM: {llm_model}")
    print(f"  OCR: EasyOCR (for text images)")
    print(f"  Depth: Depth-Anything-V2-Small (for navigation images)")
    
    output_dir = Path('results/approach_3_specialized/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize processors once (for reuse)
    print("\nInitializing processors...")
    depth_estimator = None
    ocr_processor = None
    
    try:
        depth_estimator = DepthEstimator()
        print("  ✅ Depth estimator initialized")
    except Exception as e:
        print(f"  ⚠️  Depth estimator initialization failed: {e}")
        print("  Will attempt to initialize per-image")
    
    try:
        ocr_processor = OCRProcessor(languages=['en'], gpu=True)
        print("  ✅ OCR processor initialized")
    except Exception as e:
        print(f"  ⚠️  OCR processor initialization failed (SSL issue): {e}")
        print("  Will attempt graceful degradation")
        ocr_processor = None
    
    print()
    
    all_results = []
    start_time = time.time()
    successful_tests = 0
    failed_tests = 0
    
    # Separate images by mode for better progress tracking
    ocr_images = [img for img in images if img['mode'] == 'ocr']
    depth_images = [img for img in images if img['mode'] == 'depth']
    
    # Test depth mode first (validated)
    print("=" * 80)
    print("TESTING DEPTH MODE (3B)")
    print("=" * 80)
    for img_idx, img_info in enumerate(depth_images, 1):
        print(f"\n[{img_idx}/{len(depth_images)}] 📸 {img_info['filename']} ({img_info['category']})")
        
        result = test_configuration_on_image(
            img_info['path'],
            img_info['category'],
            img_info['mode'],
            yolo_size,
            llm_model,
            output_dir,
            depth_estimator=depth_estimator
        )
        all_results.append(result)
        
        if result['success']:
            successful_tests += 1
        else:
            failed_tests += 1
        
        save_results([result], output_dir)
        
        # Small delay between API calls
        time.sleep(0.5)
    
    # Test OCR mode (may fail due to SSL)
    print("\n" + "=" * 80)
    print("TESTING OCR MODE (3A)")
    print("=" * 80)
    for img_idx, img_info in enumerate(ocr_images, 1):
        print(f"\n[{img_idx}/{len(ocr_images)}] 📸 {img_info['filename']} ({img_info['category']})")
        
        result = test_configuration_on_image(
            img_info['path'],
            img_info['category'],
            img_info['mode'],
            yolo_size,
            llm_model,
            output_dir,
            ocr_processor=ocr_processor
        )
        all_results.append(result)
        
        if result['success']:
            successful_tests += 1
        else:
            failed_tests += 1
        
        save_results([result], output_dir)
        
        # Small delay between API calls
        time.sleep(0.5)
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 80}")
    print("BATCH TEST COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total images tested: {len(images)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {successful_tests/len(images)*100:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per image: {total_time / len(images):.1f}s")
    
    # Calculate latency statistics from results
    successful_results = [r for r in all_results if r['success']]
    if successful_results:
        latencies = [float(r['total_latency']) for r in successful_results if r.get('total_latency')]
        if latencies:
            mean_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print("\nLatency Statistics:")
            print(f"  Mean: {mean_latency:.2f}s")
            print(f"  Min: {min_latency:.2f}s")
            print(f"  Max: {max_latency:.2f}s")
            
            # Mode-specific statistics
            depth_results = [r for r in successful_results if r.get('mode') == 'depth']
            ocr_results = [r for r in successful_results if r.get('mode') == 'ocr']
            
            if depth_results:
                depth_latencies = [float(r['total_latency']) for r in depth_results if r.get('total_latency')]
                if depth_latencies:
                    print(f"\nDepth Mode (3B):")
                    print(f"  Count: {len(depth_results)}")
                    print(f"  Mean latency: {sum(depth_latencies)/len(depth_latencies):.2f}s")
            
            if ocr_results:
                ocr_latencies = [float(r['total_latency']) for r in ocr_results if r.get('total_latency')]
                if ocr_latencies:
                    print(f"\nOCR Mode (3A):")
                    print(f"  Count: {len(ocr_results)}")
                    print(f"  Mean latency: {sum(ocr_latencies)/len(ocr_latencies):.2f}s")
    
    print(f"\nResults saved to: {output_dir / 'batch_results.csv'}")


if __name__ == "__main__":
    main()

