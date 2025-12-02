#!/usr/bin/env python3
"""
Subset test for faster LLM models in Approach 2
Tests GPT-3.5-turbo and Gemini Flash on 8-image subset
Compares against existing GPT-4o-mini and Claude Haiku results
"""
import os
import sys
import csv
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

# Import pipeline
from hybrid_pipeline import run_hybrid_pipeline


def get_subset_images():
    """Get 8-image subset (2 per category)"""
    images_dir = Path('data/images')
    subset = []
    
    categories = ['gaming', 'indoor', 'outdoor', 'text']
    for category in categories:
        cat_dir = images_dir / category
        if cat_dir.exists():
            # Get first 2 images from each category
            images = sorted(cat_dir.glob('*.png')) + sorted(cat_dir.glob('*.jpg')) + sorted(cat_dir.glob('*.jpeg'))
            for img_file in images[:2]:
                subset.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category
                })
    
    return subset


def load_existing_results():
    """Load existing Approach 2 results from CSV"""
    csv_path = Path('results/approach_2_yolo_llm/raw/batch_results.csv')
    existing = {}
    
    if not csv_path.exists():
        print("⚠️  No existing results found. Will test all configurations.")
        return existing
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('success') == 'True' or row.get('success') == 'true':
                # Match CSV format: filename, yolo_model (e.g., "yolov8n"), llm_model
                key = f"{row['filename']}_{row['yolo_model']}_{row['llm_model']}"
                try:
                    existing[key] = {
                        'total_latency': float(row['total_latency']),
                        'generation_latency': float(row['generation_latency']),
                        'detection_latency': float(row['detection_latency']),
                        'description': row.get('description', ''),
                        'word_count': len(row.get('description', '').split()) if row.get('description') else 0
                    }
                except (ValueError, KeyError) as e:
                    # Skip rows with missing or invalid data
                    continue
    
    print(f"✅ Loaded {len(existing)} existing results")
    return existing


def test_configuration(image_path, category, yolo_size, llm_model):
    """Test single configuration on single image"""
    try:
        result = run_hybrid_pipeline(
            image_path,
            yolo_size=yolo_size,
            llm_model=llm_model
        )
        return result, None
    except Exception as e:
        return None, str(e)


def main():
    """Run subset comparison test"""
    print("=" * 80)
    print("APPROACH 2: FASTER LLM MODELS - SUBSET TEST")
    print("=" * 80)
    print()
    
    # Get subset images
    images = get_subset_images()
    print(f"Testing on {len(images)} images (2 per category)")
    print()
    
    # Load existing results
    existing_results = load_existing_results()
    
    # Test configurations
    # YOLO: Use 'n' (fastest) for all tests
    # LLMs: Test faster models + compare with existing
    configurations = [
        ('n', 'gpt-3.5-turbo', 'GPT-3.5-turbo (NEW)'),
        ('n', 'gemini-flash', 'Gemini Flash (NEW)'),
        # Existing models for comparison
        ('n', 'gpt-4o-mini', 'GPT-4o-mini (EXISTING)'),
        ('n', 'claude-haiku', 'Claude Haiku (EXISTING)'),
    ]
    
    print("=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    print()
    
    results = {}
    
    for img_info in images:
        filename = img_info['filename']
        category = img_info['category']
        print(f"\n📸 {filename} ({category})")
        
        for yolo_size, llm_model, config_name in configurations:
            config_key = f"{yolo_size}_{llm_model}"
            
            # Check if we have existing result
            # Match CSV format: filename_yolov8n_llm_model
            existing_key = f"{filename}_yolov8{yolo_size}_{llm_model}"
            # Also try with different llm_model formats (handle aliases)
            alt_keys = []
            if llm_model == 'gpt-4o-mini':
                alt_keys = [f"{filename}_yolov8{yolo_size}_gpt-4o-mini"]
            elif llm_model == 'claude-haiku':
                alt_keys = [f"{filename}_yolov8{yolo_size}_claude-haiku", f"{filename}_yolov8{yolo_size}_claude-3-5-haiku-20241022"]
            
            found_existing = False
            existing_data = None
            
            if existing_key in existing_results:
                found_existing = True
                existing_data = existing_results[existing_key]
            else:
                # Try alternative keys
                for alt_key in alt_keys:
                    if alt_key in existing_results:
                        found_existing = True
                        existing_data = existing_results[alt_key]
                        break
            
            if found_existing and existing_data:
                print(f"  ✅ {config_name}: Using existing result ({existing_data['total_latency']:.2f}s)")
                results[config_key] = results.get(config_key, [])
                results[config_key].append({
                    'filename': filename,
                    'category': category,
                    'total_latency': existing_data['total_latency'],
                    'generation_latency': existing_data['generation_latency'],
                    'detection_latency': existing_data['detection_latency'],
                    'word_count': existing_data['word_count']
                })
            else:
                print(f"  Testing {config_name}...")
                result, error = test_configuration(img_info['path'], category, yolo_size, llm_model)
                
                if error:
                    print(f"  ❌ Error: {error}")
                    continue
                
                if not result['success']:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
                    continue
                
                word_count = len(result['description'].split()) if result.get('description') else 0
                results[config_key] = results.get(config_key, [])
                results[config_key].append({
                    'filename': filename,
                    'category': category,
                    'total_latency': result['total_latency'],
                    'generation_latency': result['generation_latency'],
                    'detection_latency': result['detection_latency'],
                    'word_count': word_count
                })
                
                print(f"  ✅ {config_name}: {result['total_latency']:.2f}s (gen: {result['generation_latency']:.2f}s, det: {result['detection_latency']:.2f}s)")
        
        # Small delay between images
        time.sleep(0.5)
    
    # Summary statistics
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    summary = []
    for yolo_size, llm_model, config_name in configurations:
        config_key = f"{yolo_size}_{llm_model}"
        if config_key not in results or not results[config_key]:
            continue
        
        latencies = [r['total_latency'] for r in results[config_key]]
        gen_latencies = [r['generation_latency'] for r in results[config_key]]
        word_counts = [r['word_count'] for r in results[config_key]]
        
        avg_latency = sum(latencies) / len(latencies)
        avg_gen = sum(gen_latencies) / len(gen_latencies)
        avg_words = sum(word_counts) / len(word_counts)
        
        summary.append({
            'config': config_name,
            'mean_latency': avg_latency,
            'mean_gen': avg_gen,
            'mean_words': avg_words,
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        })
        
        print(f"{config_name}:")
        print(f"  Mean Total Latency: {avg_latency:.2f}s")
        print(f"  Mean Generation Latency: {avg_gen:.2f}s")
        print(f"  Mean Word Count: {avg_words:.1f} words")
        print(f"  Latency Range: {min(latencies):.2f}s - {max(latencies):.2f}s")
        print()
    
    # Compare against <2s target
    print("=" * 80)
    print("TARGET ANALYSIS (<2 SECONDS)")
    print("=" * 80)
    print()
    
    for s in summary:
        under_2s = "✅" if s['mean_latency'] < 2.0 else "❌"
        print(f"{under_2s} {s['config']}: {s['mean_latency']:.2f}s {'(MEETS TARGET)' if s['mean_latency'] < 2.0 else '(ABOVE TARGET)'}")
    
    # Find fastest
    if summary:
        fastest = min(summary, key=lambda x: x['mean_latency'])
        print()
        print(f"🏆 FASTEST: {fastest['config']} - {fastest['mean_latency']:.2f}s")
        if fastest['mean_latency'] < 2.0:
            print("   ✅ MEETS <2 SECOND TARGET!")
        else:
            gap = fastest['mean_latency'] - 2.0
            print(f"   ⚠️  {gap:.2f}s above target")
    
    # Save comparison report
    report_path = Path('results/approach_2_yolo_llm/analysis/faster_models_subset_comparison.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Approach 2: Faster LLM Models - Subset Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Tested: {len(images)}\n")
        f.write(f"Target Latency: <2 seconds\n\n")
        
        f.write("Configuration Comparison:\n")
        f.write("-" * 80 + "\n\n")
        
        for s in summary:
            f.write(f"{s['config']}:\n")
            f.write(f"  Mean Total Latency: {s['mean_latency']:.2f}s\n")
            f.write(f"  Mean Generation Latency: {s['mean_gen']:.2f}s\n")
            f.write(f"  Mean Word Count: {s['mean_words']:.1f} words\n")
            f.write(f"  Latency Range: {s['min_latency']:.2f}s - {s['max_latency']:.2f}s\n")
            f.write(f"  Meets <2s Target: {'YES' if s['mean_latency'] < 2.0 else 'NO'}\n\n")
        
        f.write("Per-Image Results:\n")
        f.write("-" * 80 + "\n\n")
        for img_info in images:
            f.write(f"{img_info['filename']} ({img_info['category']}):\n")
            for yolo_size, llm_model, config_name in configurations:
                config_key = f"{yolo_size}_{llm_model}"
                if config_key in results:
                    img_results = [r for r in results[config_key] if r['filename'] == img_info['filename']]
                    if img_results:
                        r = img_results[0]
                        f.write(f"  {config_name}: {r['total_latency']:.2f}s (gen: {r['generation_latency']:.2f}s)\n")
            f.write("\n")
    
    print(f"📄 Detailed report saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()

