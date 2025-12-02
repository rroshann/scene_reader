#!/usr/bin/env python3
"""
Batch test GPT-4V with Chain-of-Thought (CoT) prompts on all images
Compares CoT prompting strategy to baseline standard prompting
"""
import os
import sys
import time
import base64
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory and current directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

load_dotenv()

# Import test function and CoT prompts
from test_api import test_openai_real
from prompts_cot import SYSTEM_PROMPT_COT, USER_PROMPT_COT

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

def test_cot_on_image(image_path, category):
    """Test GPT-4V with CoT prompts on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name} ({category})")
    print(f"{'='*60}")
    
    print("\n🧠 Testing GPT-4V with Chain-of-Thought prompts...")
    try:
        result, error = test_openai_real(
            image_path,
            system_prompt=SYSTEM_PROMPT_COT,
            user_prompt=USER_PROMPT_COT
        )
        
        if result:
            return {
                'filename': image_path.name,
                'category': category,
                'model': 'GPT-4V',
                'prompt_type': 'CoT',
                'description': result['description'],
                'latency_seconds': result['latency'],
                'tokens_used': result.get('tokens', None),
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }, None
        else:
            return {
                'filename': image_path.name,
                'category': category,
                'model': 'GPT-4V',
                'prompt_type': 'CoT',
                'description': None,
                'latency_seconds': None,
                'tokens_used': None,
                'success': False,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }, error
    except Exception as e:
        return {
            'filename': image_path.name,
            'category': category,
            'model': 'GPT-4V',
            'prompt_type': 'CoT',
            'description': None,
            'latency_seconds': None,
            'tokens_used': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, str(e)

def save_results(results, output_dir):
    """Save results to CSV file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'batch_results.csv'
    
    # Append to existing file or create new
    file_exists = output_file.exists()
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename', 'category', 'model', 'prompt_type', 'description', 
            'latency_seconds', 'tokens_used', 'success', 
            'error', 'timestamp'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)

def main():
    print("="*60)
    print("BATCH TESTING - GPT-4V with Chain-of-Thought Prompts")
    print("="*60)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"📸 Found {len(images)} images to test")
    print()
    
    # Confirm before starting
    try:
        confirm = input(f"Ready to test {len(images)} images with CoT prompts. Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    except (EOFError, KeyboardInterrupt):
        # Auto-confirm in non-interactive environments
        print("Auto-confirming in non-interactive mode...")
    
    # Setup output directory
    output_dir = Path('results/approach_7_cot/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Results will be saved to: {output_dir / 'batch_results.csv'}")
    print()
    
    # Test each image
    all_results = []
    successful = 0
    failed = 0
    
    for i, img_info in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing...")
        
        result, error = test_cot_on_image(img_info['path'], img_info['category'])
        all_results.append(result)
        
        if result['success']:
            successful += 1
            latency_str = f"{result['latency_seconds']:.2f}s" if result.get('latency_seconds') else "N/A"
            print(f"   ✅ Success! Latency: {latency_str}")
            if result.get('tokens_used'):
                print(f"   📊 Tokens: {result['tokens_used']}")
        else:
            failed += 1
            print(f"   ❌ Failed: {error}")
        
        # Save incrementally
        save_results([result], output_dir)
        
        # Small delay between API calls to avoid rate limiting
        if i < len(images):
            time.sleep(1)
    
    # Final summary
    print("\n" + "="*60)
    print("BATCH TESTING COMPLETE")
    print("="*60)
    print(f"✅ Successful: {successful}/{len(images)}")
    print(f"❌ Failed: {failed}/{len(images)}")
    print(f"📁 Results saved to: {output_dir / 'batch_results.csv'}")
    print()
    
    if successful > 0:
        avg_latency = sum(r['latency_seconds'] for r in all_results if r['success']) / successful
        print(f"📊 Average latency: {avg_latency:.2f}s")
    
    print("\n✅ Ready for analysis! Compare to baseline (Approach 1 GPT-4V results)")

if __name__ == '__main__':
    main()

