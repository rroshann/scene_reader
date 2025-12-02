#!/usr/bin/env python3
"""
Retest failed images for specific models
"""
import csv
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

# Import test functions
import importlib.util
spec = importlib.util.spec_from_file_location("test_api", Path(__file__).parent / "test_api.py")
test_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_api)
test_openai_real = test_api.test_openai_real
test_google_real = test_api.test_google_real
test_anthropic_real = test_api.test_anthropic_real

def get_failed_images(model_name):
    """Get list of images that failed for a specific model"""
    results_file = Path('results/approach_1_vlm/raw/batch_results.csv')
    if not results_file.exists():
        print("❌ No batch results found")
        return []
    
    failed = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['model'] == model_name and row['success'] == 'False':
                failed.append({
                    'filename': row['filename'],
                    'category': row['category']
                })
    
    return failed

def find_image_path(filename, category):
    """Find the full path to an image"""
    image_path = Path(f'data/images/{category}/{filename}')
    if image_path.exists():
        return image_path
    return None

def retest_model(model_name, test_func):
    """Retest failed images for a specific model"""
    failed = get_failed_images(model_name)
    
    if not failed:
        print(f"✅ No failed images for {model_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Retesting {model_name} on {len(failed)} failed images")
    print(f"{'='*60}\n")
    
    results = []
    output_dir = Path('results/approach_1_vlm/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_info in enumerate(failed, 1):
        print(f"\n[{i}/{len(failed)}] Testing: {img_info['filename']}")
        
        image_path = find_image_path(img_info['filename'], img_info['category'])
        if not image_path:
            print(f"  ❌ Image not found: {img_info['filename']}")
            continue
        
        try:
            result, error = test_func(image_path)
            if result:
                results.append({
                    'filename': img_info['filename'],
                    'category': img_info['category'],
                    'model': model_name,
                    'description': result['description'],
                    'latency_seconds': result['latency'],
                    'tokens_used': result.get('tokens', None),
                    'success': True,
                    'error': None,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"  ✅ Success! Latency: {result['latency']:.2f}s")
            else:
                results.append({
                    'filename': img_info['filename'],
                    'category': img_info['category'],
                    'model': model_name,
                    'description': None,
                    'latency_seconds': None,
                    'tokens_used': None,
                    'success': False,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"  ❌ Failed: {error}")
        except Exception as e:
            results.append({
                'filename': img_info['filename'],
                'category': img_info['category'],
                'model': model_name,
                'description': None,
                'latency_seconds': None,
                'tokens_used': None,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            print(f"  ❌ Error: {e}")
        
        # Small delay
        import time
        time.sleep(1)
    
    # Update results file
    update_results_file(results, output_dir)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\n✅ Retest complete: {successful}/{len(results)} successful")

def update_results_file(new_results, output_dir):
    """Update the batch_results.csv file with new results"""
    results_file = output_dir / 'batch_results.csv'
    
    # Read existing results
    existing = []
    if results_file.exists():
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    
    # Replace failed entries with new results
    updated = []
    for row in existing:
        # Check if this row should be replaced
        replaced = False
        for new_row in new_results:
            if row['filename'] == new_row['filename'] and row['model'] == new_row['model']:
                updated.append(new_row)
                replaced = True
                break
        
        if not replaced:
            updated.append(row)
    
    # Add any completely new results
    for new_row in new_results:
        found = False
        for row in updated:
            if row['filename'] == new_row['filename'] and row['model'] == new_row['model']:
                found = True
                break
        if not found:
            updated.append(new_row)
    
    # Write back
    fieldnames = ['filename', 'category', 'model', 'description', 'latency_seconds', 
                  'tokens_used', 'success', 'error', 'timestamp']
    
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    
    print(f"\n✅ Updated {results_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python retest_failed.py <model>")
        print("Models: Claude, GPT-4V, Gemini")
        return
    
    model = sys.argv[1]
    
    if model == 'Claude':
        retest_model('Claude', test_anthropic_real)
    elif model == 'GPT-4V':
        retest_model('GPT-4V', test_openai_real)
    elif model == 'Gemini':
        retest_model('Gemini', test_google_real)
    else:
        print(f"Unknown model: {model}")
        print("Use: Claude, GPT-4V, or Gemini")

if __name__ == "__main__":
    main()

