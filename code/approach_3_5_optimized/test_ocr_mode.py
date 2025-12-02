#!/usr/bin/env python3
"""
Test OCR mode on text images for Approach 3.5
Quick test script to validate OCR functionality
"""
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "code" / "approach_3_5_optimized"))

load_dotenv()

from specialized_pipeline_optimized import run_specialized_pipeline_optimized, warmup_models
from ocr_processor_optimized import OCRProcessorOptimized

def main():
    print("=" * 80)
    print("APPROACH 3.5: OCR MODE TESTING")
    print("=" * 80)
    print()
    
    # Get text images
    images_dir = project_root / 'data/images/text'
    text_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        text_images.extend(sorted(images_dir.glob(ext)))
    
    print(f"Found {len(text_images)} text images")
    print()
    
    # Warmup OCR processor
    print("Warming up OCR processor...")
    warmup_models(mode='ocr')
    
    # Initialize OCR processor
    try:
        ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
        print("✅ OCR processor ready")
    except Exception as e:
        print(f"❌ Failed to initialize OCR: {e}")
        return
    
    # Test on first 3 images (quick validation)
    print("\n" + "=" * 80)
    print("TESTING OCR MODE (First 3 images)")
    print("=" * 80)
    
    for i, image_path in enumerate(text_images[:3], 1):
        print(f"\n[{i}/3] Testing: {image_path.name}")
        start = time.time()
        
        try:
            result = run_specialized_pipeline_optimized(
                image_path,
                category='text',
                mode='ocr',
                yolo_size='n',
                llm_model='gpt-3.5-turbo',
                use_cache=True,
                use_adaptive=True
            )
            
            elapsed = time.time() - start
            
            if result['success']:
                print(f"  ✅ Success!")
                print(f"  Total latency: {result['total_latency']:.3f}s")
                print(f"  OCR latency: {result.get('ocr_latency', 0):.3f}s")
                print(f"  Generation latency: {result.get('generation_latency', 0):.3f}s")
                print(f"  Texts found: {result.get('ocr_results', {}).get('num_texts', 0)}")
                print(f"  Description length: {len(result.get('description', ''))} chars")
            else:
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("OCR MODE TESTING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

