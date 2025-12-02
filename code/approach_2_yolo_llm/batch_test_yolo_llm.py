#!/usr/bin/env python3
"""
Batch test all YOLO+LLM configurations on all images
Tests 6 configurations: 3 YOLO variants (n, m, x) × 2 LLMs (GPT-4o-mini, Claude Haiku)
"""
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

# Import pipeline
from hybrid_pipeline import run_hybrid_pipeline


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


def test_configuration_on_image(image_path, category, yolo_size, llm_model, output_dir):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category
        yolo_size: 'n', 'm', or 'x'
        llm_model: 'gpt-4o-mini' or 'claude-haiku'
        output_dir: Directory to save results
    
    Returns:
        Result dict for CSV
    """
    config_name = f"YOLOv8{yolo_size.upper()}+{llm_model}"
    print(f"\n  Testing: {config_name}")
    
    try:
        result = run_hybrid_pipeline(
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
                'configuration': config_name,
                'description': result['description'],
                'detection_latency': result['detection_latency'],
                'generation_latency': result['generation_latency'],
                'total_latency': result['total_latency'],
                'num_objects_detected': result['num_objects'],
                'objects_detected': str(result['objects_detected']),  # Convert to string for CSV
                'avg_confidence': result['detection_summary'].get('avg_confidence', 0.0),
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
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model,
                'configuration': config_name,
                'description': None,
                'detection_latency': result.get('detection_latency'),
                'generation_latency': result.get('generation_latency'),
                'total_latency': result.get('total_latency'),
                'num_objects_detected': result.get('num_objects', 0),
                'objects_detected': None,
                'avg_confidence': None,
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
            'yolo_model': f'yolov8{yolo_size}',
            'llm_model': llm_model,
            'configuration': config_name,
            'description': None,
            'detection_latency': None,
            'generation_latency': None,
            'total_latency': None,
            'num_objects_detected': 0,
            'objects_detected': None,
            'avg_confidence': None,
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
    
    # Append to existing file or create new
    file_exists = output_file.exists()
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        fieldnames = [
            'filename', 'category', 'yolo_model', 'llm_model', 'configuration',
            'description', 'detection_latency', 'generation_latency', 'total_latency',
            'num_objects_detected', 'objects_detected', 'avg_confidence',
            'tokens_used', 'success', 'error', 'timestamp'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)


def main():
    print("=" * 60)
    print("BATCH TESTING - YOLO + LLM Hybrid Pipeline")
    print("=" * 60)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    
    # Define configurations
    yolo_sizes = ['n', 'm', 'x']
    llm_models = ['gpt-4o-mini', 'claude-haiku']
    configurations = [(y, l) for y in yolo_sizes for l in llm_models]
    
    total_tests = len(images) * len(configurations)
    print(f"Total configurations: {len(configurations)}")
    print(f"  YOLO variants: {', '.join([f'yolov8{s}' for s in yolo_sizes])}")
    print(f"  LLM models: {', '.join(llm_models)}")
    print(f"Total API calls: {total_tests} ({len(images)} images × {len(configurations)} configurations)")
    print()
    
    # Estimate cost (LLM only, YOLO is free)
    print("Estimated costs (LLM API calls only, YOLO is free):")
    gpt4o_mini_cost_per_query = 0.00015 + 0.0006  # ~$0.00075 per query (input + output tokens)
    claude_haiku_cost_per_query = 0.00025 + 0.00125  # ~$0.0015 per query
    avg_cost = (gpt4o_mini_cost_per_query + claude_haiku_cost_per_query) / 2
    
    total_gpt_calls = len(images) * len(yolo_sizes)  # 3 YOLO variants × images
    total_claude_calls = len(images) * len(yolo_sizes)
    
    estimated_cost = (total_gpt_calls * gpt4o_mini_cost_per_query) + (total_claude_calls * claude_haiku_cost_per_query)
    print(f"  GPT-4o-mini: ~${total_gpt_calls * gpt4o_mini_cost_per_query:.2f} ({total_gpt_calls} calls)")
    print(f"  Claude Haiku: ~${total_claude_calls * claude_haiku_cost_per_query:.2f} ({total_claude_calls} calls)")
    print(f"  Total: ~${estimated_cost:.2f}")
    print()
    
    # Confirm (auto-confirm if non-interactive or --yes flag)
    import sys
    auto_confirm = '--yes' in sys.argv or '--y' in sys.argv
    if not auto_confirm:
        try:
            response = input("Continue with batch testing? (y/N): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except (EOFError, KeyboardInterrupt):
            print("⚠️  Running non-interactively. Auto-confirming batch test...")
    else:
        print("⚠️  Auto-confirming batch test (--yes flag)...")
    
    # Output directory
    output_dir = Path('results/approach_2_yolo_llm/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test all configurations on all images
    all_results = []
    start_time = time.time()
    test_count = 0
    
    for img_idx, img_info in enumerate(images, 1):
        print(f"\n{'#' * 60}")
        print(f"Image {img_idx}/{len(images)}: {img_info['filename']} ({img_info['category']})")
        print(f"{'#' * 60}")
        
        image_results = []
        
        for config_idx, (yolo_size, llm_model) in enumerate(configurations, 1):
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] Configuration {config_idx}/{len(configurations)}")
            
            result = test_configuration_on_image(
                img_info['path'],
                img_info['category'],
                yolo_size,
                llm_model,
                output_dir
            )
            image_results.append(result)
            all_results.append(result)
            
            # Save incrementally
            save_results([result], output_dir)
            
            # Rate limiting - wait between API calls
            if test_count < total_tests:
                time.sleep(1)  # 1 second between API calls
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / test_count
        remaining = avg_time * (total_tests - test_count)
        print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
        
        # Wait between images
        if img_idx < len(images):
            print("   Waiting 2 seconds before next image...")
            time.sleep(2)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("BATCH TESTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {output_dir / 'batch_results.csv'}")
    
    # Success rate
    successful = sum(1 for r in all_results if r['success'])
    print(f"Success rate: {successful}/{len(all_results)} ({successful*100/len(all_results):.1f}%)")
    
    # By configuration
    print("\nBy configuration:")
    for yolo_size in yolo_sizes:
        for llm_model in llm_models:
            config_results = [r for r in all_results if r['yolo_model'] == f'yolov8{yolo_size}' and r['llm_model'] == llm_model]
            if config_results:
                config_success = sum(1 for r in config_results if r['success'])
                avg_latency = sum(float(r['total_latency']) for r in config_results if r['success'] and r['total_latency']) / config_success if config_success > 0 else 0
                print(f"  YOLOv8{yolo_size.upper()}+{llm_model}: {config_success}/{len(config_results)} successful, avg latency: {avg_latency:.2f}s")
    
    # By YOLO variant
    print("\nBy YOLO variant:")
    for yolo_size in yolo_sizes:
        yolo_results = [r for r in all_results if r['yolo_model'] == f'yolov8{yolo_size}']
        if yolo_results:
            yolo_success = sum(1 for r in yolo_results if r['success'])
            avg_detection_latency = sum(float(r['detection_latency']) for r in yolo_results if r['success'] and r['detection_latency']) / yolo_success if yolo_success > 0 else 0
            print(f"  YOLOv8{yolo_size.upper()}: {yolo_success}/{len(yolo_results)} successful, avg detection latency: {avg_detection_latency:.3f}s")
    
    # By LLM model
    print("\nBy LLM model:")
    for llm_model in llm_models:
        llm_results = [r for r in all_results if r['llm_model'] == llm_model]
        if llm_results:
            llm_success = sum(1 for r in llm_results if r['success'])
            avg_generation_latency = sum(float(r['generation_latency']) for r in llm_results if r['success'] and r['generation_latency']) / llm_success if llm_success > 0 else 0
            print(f"  {llm_model}: {llm_success}/{len(llm_results)} successful, avg generation latency: {avg_generation_latency:.2f}s")


if __name__ == "__main__":
    main()

