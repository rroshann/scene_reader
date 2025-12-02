#!/usr/bin/env python3
"""
Subset test for Approach 3
Quick validation on small subset before full batch test
"""
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
approach3_dir = project_root / "code" / "approach_3_specialized"
sys.path.insert(0, str(approach3_dir))

load_dotenv()

from specialized_pipeline import run_specialized_pipeline
from ocr_processor import OCRProcessor
from depth_estimator import DepthEstimator


def get_subset_images():
    """Get subset of images for testing"""
    images_dir = project_root / 'data/images'
    subset = []
    
    # 3-4 text images for OCR testing
    text_dir = images_dir / 'text'
    if text_dir.exists():
        text_images = sorted(text_dir.glob('*.jpg')) + sorted(text_dir.glob('*.png'))
        for img_file in text_images[:4]:  # First 4 text images
            subset.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'text',
                'mode': 'ocr'
            })
    
    # 4-6 navigation images for depth testing (2-3 indoor + 2-3 outdoor)
    indoor_dir = images_dir / 'indoor'
    if indoor_dir.exists():
        indoor_images = sorted(indoor_dir.glob('*.jpg')) + sorted(indoor_dir.glob('*.png'))
        for img_file in indoor_images[:3]:  # First 3 indoor images
            subset.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'indoor',
                'mode': 'depth'
            })
    
    outdoor_dir = images_dir / 'outdoor'
    if outdoor_dir.exists():
        outdoor_images = sorted(outdoor_dir.glob('*.jpg')) + sorted(outdoor_dir.glob('*.png'))
        for img_file in outdoor_images[:3]:  # First 3 outdoor images
            subset.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'outdoor',
                'mode': 'depth'
            })
    
    return subset


def main():
    """Run subset tests"""
    print("=" * 80)
    print("APPROACH 3: SUBSET TEST (VALIDATION)")
    print("=" * 80)
    print()
    
    images = get_subset_images()
    print(f"Testing on {len(images)} images:")
    print(f"  - OCR mode (text): {sum(1 for img in images if img['mode'] == 'ocr')} images")
    print(f"  - Depth mode (navigation): {sum(1 for img in images if img['mode'] == 'depth')} images")
    print()
    
    # Initialize processors once (for reuse)
    print("Initializing processors...")
    try:
        ocr_processor = OCRProcessor(languages=['en'], gpu=True)
        print("  ✅ OCR processor initialized")
    except Exception as e:
        print(f"  ⚠️  OCR processor initialization failed: {e}")
        ocr_processor = None
    
    try:
        depth_estimator = DepthEstimator()
        print("  ✅ Depth estimator initialized")
    except Exception as e:
        print(f"  ⚠️  Depth estimator initialization failed: {e}")
        depth_estimator = None
    
    print()
    
    results = []
    successful = 0
    failed = 0
    
    for i, img_info in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] 📸 {img_info['filename']} ({img_info['category']}, mode: {img_info['mode']})")
        
        try:
            result = run_specialized_pipeline(
                image_path=img_info['path'],
                category=img_info['category'],
                mode=img_info['mode'],
                yolo_size='n',
                llm_model='gpt-4o-mini',
                ocr_processor=ocr_processor if img_info['mode'] == 'ocr' else None,
                depth_estimator=depth_estimator if img_info['mode'] == 'depth' else None
            )
            
            if result['success']:
                successful += 1
                latencies = []
                if result.get('detection_latency'):
                    latencies.append(f"Detection: {result['detection_latency']:.3f}s")
                if result.get('ocr_latency'):
                    latencies.append(f"OCR: {result['ocr_latency']:.3f}s")
                if result.get('depth_latency'):
                    latencies.append(f"Depth: {result['depth_latency']:.3f}s")
                if result.get('generation_latency'):
                    latencies.append(f"Generation: {result['generation_latency']:.3f}s")
                
                print(f"  ✅ Success! Total: {result['total_latency']:.3f}s")
                print(f"     Breakdown: {', '.join(latencies)}")
                
                if result.get('ocr_results') and result['ocr_results'].get('num_texts', 0) > 0:
                    print(f"     OCR: Found {result['ocr_results']['num_texts']} text(s)")
                    if result['ocr_results'].get('full_text'):
                        print(f"     Text: \"{result['ocr_results']['full_text'][:50]}...\"")
                
                if result.get('description'):
                    word_count = len(result['description'].split())
                    print(f"     Description: {word_count} words")
            else:
                failed += 1
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            
            results.append(result)
            
        except Exception as e:
            failed += 1
            print(f"  ❌ Exception: {e}")
            results.append({
                'success': False,
                'error': str(e),
                'filename': img_info['filename']
            })
        
        # Small delay between API calls
        time.sleep(0.5)
    
    # Summary
    print()
    print("=" * 80)
    print("SUBSET TEST SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(images)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(images)*100:.1f}%")
    
    if successful > 0:
        successful_results = [r for r in results if r.get('success')]
        total_latencies = [r['total_latency'] for r in successful_results if r.get('total_latency')]
        
        if total_latencies:
            print()
            print("Latency Statistics:")
            print(f"  Mean: {sum(total_latencies)/len(total_latencies):.2f}s")
            print(f"  Min: {min(total_latencies):.2f}s")
            print(f"  Max: {max(total_latencies):.2f}s")
            
            # Breakdown by mode
            ocr_results = [r for r in successful_results if r.get('mode') == 'ocr']
            depth_results = [r for r in successful_results if r.get('mode') == 'depth']
            
            if ocr_results:
                ocr_latencies = [r['total_latency'] for r in ocr_results]
                print()
                print("OCR Mode (3A):")
                print(f"  Mean: {sum(ocr_latencies)/len(ocr_latencies):.2f}s")
            
            if depth_results:
                depth_latencies = [r['total_latency'] for r in depth_results]
                print()
                print("Depth Mode (3B):")
                print(f"  Mean: {sum(depth_latencies)/len(depth_latencies):.2f}s")
    
    print()
    print("=" * 80)
    if successful == len(images):
        print("✅ All tests passed! Ready for full batch test.")
    else:
        print("⚠️  Some tests failed. Review errors before full batch test.")
    print("=" * 80)


if __name__ == "__main__":
    main()

