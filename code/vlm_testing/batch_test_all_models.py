#!/usr/bin/env python3
"""
Batch test all VLMs on all images in the dataset
Runs GPT-4V, Gemini, and Claude on all images and saves results
"""
import os
import sys
import time
import base64
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

# Import test functions from test_api.py
# Need to import from same directory
import importlib.util
spec = importlib.util.spec_from_file_location("test_api", Path(__file__).parent / "test_api.py")
test_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_api)
test_openai_real = test_api.test_openai_real
test_google_real = test_api.test_google_real
test_anthropic_real = test_api.test_anthropic_real

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

def test_all_models_on_image(image_path, category, output_dir):
    """Test all 3 models on a single image"""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name} ({category})")
    print(f"{'='*60}")
    
    # Test OpenAI GPT-4V
    print("\n1️⃣  Testing GPT-4V...")
    try:
        result, error = test_openai_real(image_path)
        if result:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'GPT-4V',
                'description': result['description'],
                'latency_seconds': result['latency'],
                'tokens_used': result.get('tokens', None),
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        else:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'GPT-4V',
                'description': None,
                'latency_seconds': None,
                'tokens_used': None,
                'success': False,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ❌ Failed: {error}")
    except Exception as e:
        results.append({
            'filename': image_path.name,
            'category': category,
            'model': 'GPT-4V',
            'description': None,
            'latency_seconds': None,
            'tokens_used': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        print(f"   ❌ Error: {e}")
    
    # Small delay between API calls
    time.sleep(1)
    
    # Test Google Gemini
    print("\n2️⃣  Testing Gemini...")
    try:
        result, error = test_google_real(image_path)
        if result:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'Gemini',
                'description': result['description'],
                'latency_seconds': result['latency'],
                'tokens_used': None,
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        else:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'Gemini',
                'description': None,
                'latency_seconds': None,
                'tokens_used': None,
                'success': False,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ❌ Failed: {error}")
    except Exception as e:
        results.append({
            'filename': image_path.name,
            'category': category,
            'model': 'Gemini',
            'description': None,
            'latency_seconds': None,
            'tokens_used': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        print(f"   ❌ Error: {e}")
    
    # Small delay between API calls
    time.sleep(1)
    
    # Test Anthropic Claude
    print("\n3️⃣  Testing Claude...")
    try:
        result, error = test_anthropic_real(image_path)
        if result:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'Claude',
                'description': result['description'],
                'latency_seconds': result['latency'],
                'tokens_used': None,
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        else:
            results.append({
                'filename': image_path.name,
                'category': category,
                'model': 'Claude',
                'description': None,
                'latency_seconds': None,
                'tokens_used': None,
                'success': False,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            print(f"   ❌ Failed: {error}")
    except Exception as e:
        results.append({
            'filename': image_path.name,
            'category': category,
            'model': 'Claude',
            'description': None,
            'latency_seconds': None,
            'tokens_used': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        print(f"   ❌ Error: {e}")
    
    # Save results incrementally
    save_results(results, output_dir)
    
    return results

def save_results(results, output_dir):
    """Save results to CSV file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'batch_results.csv'
    
    # Append to existing file or create new
    file_exists = output_file.exists()
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename', 'category', 'model', 'description', 
            'latency_seconds', 'tokens_used', 'success', 
            'error', 'timestamp'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)

def main():
    print("="*60)
    print("BATCH TESTING - All VLMs on All Images")
    print("="*60)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print(f"Total API calls: {len(images) * 3} (3 models × {len(images)} images)")
    print()
    
    # Estimate cost
    print("Estimated costs:")
    print(f"  GPT-4V: ~${len(images) * 0.05:.2f} ({len(images)} images × $0.05)")
    print(f"  Gemini: Free tier (50/day) or ~${len(images) * 0.01:.2f}")
    print(f"  Claude: ~${len(images) * 0.02:.2f} ({len(images)} images × $0.02)")
    print(f"  Total: ~${len(images) * 0.08:.2f}")
    print()
    
    # Confirm (auto-confirm if running non-interactively)
    try:
        response = input("Continue with batch testing? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    except (EOFError, KeyboardInterrupt):
        print("⚠️  Running non-interactively. Auto-confirming batch test...")
        # Continue automatically
    
    # Output directory
    output_dir = Path('results/approach_1_vlm/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test all images
    all_results = []
    start_time = time.time()
    
    for i, img_info in enumerate(images, 1):
        print(f"\n{'#'*60}")
        print(f"Progress: {i}/{len(images)}")
        print(f"{'#'*60}")
        
        results = test_all_models_on_image(
            img_info['path'], 
            img_info['category'],
            output_dir
        )
        all_results.extend(results)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(images) - i)
        print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
        
        # Rate limiting - wait between images
        if i < len(images):
            print("   Waiting 2 seconds before next image...")
            time.sleep(2)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("BATCH TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {output_dir / 'batch_results.csv'}")
    
    # Success rate
    successful = sum(1 for r in all_results if r['success'])
    print(f"Success rate: {successful}/{len(all_results)} ({successful*100/len(all_results):.1f}%)")
    
    # By model
    print("\nBy model:")
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        model_results = [r for r in all_results if r['model'] == model]
        model_success = sum(1 for r in model_results if r['success'])
        if model_results:
            avg_latency = sum(r['latency_seconds'] for r in model_results if r['latency_seconds']) / len([r for r in model_results if r['latency_seconds']])
            print(f"  {model}: {model_success}/{len(model_results)} successful, avg latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    main()

