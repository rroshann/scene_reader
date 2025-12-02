#!/usr/bin/env python3
"""
Batch testing for Approach 5: Streaming/Progressive Models
Tests two-tier streaming architecture on all images
"""
import os
import sys
import asyncio
import csv
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent))

load_dotenv()

from streaming_pipeline import StreamingPipeline


def get_all_images():
    """Get all images from data/images folder"""
    images_dir = project_root / 'data' / 'images'
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


async def test_image(image_path: Path, category: str, pipeline: StreamingPipeline):
    """
    Test streaming pipeline on a single image
    
    Args:
        image_path: Path to image file
        category: Image category
        pipeline: StreamingPipeline instance
    
    Returns:
        dict: Test result
    """
    result = {
        'filename': image_path.name,
        'category': category,
        'success': False,
        'tier1_description': None,
        'tier1_latency': None,
        'tier1_success': False,
        'tier1_error': None,
        'tier2_description': None,
        'tier2_latency': None,
        'tier2_tokens': None,
        'tier2_cost': None,
        'tier2_success': False,
        'tier2_error': None,
        'total_latency': None,
        'time_to_first_output': None,
        'perceived_latency_improvement': None,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Run streaming pipeline
        streaming_result = await pipeline.describe_image(image_path)
        
        # Extract results
        result['success'] = streaming_result['success']
        result['tier1_description'] = streaming_result['tier1']['description']
        result['tier1_latency'] = streaming_result['tier1']['latency']
        result['tier1_success'] = streaming_result['tier1']['success']
        result['tier1_error'] = streaming_result['tier1']['error']
        
        result['tier2_description'] = streaming_result['tier2']['description']
        result['tier2_latency'] = streaming_result['tier2']['latency']
        result['tier2_tokens'] = streaming_result['tier2']['tokens']
        result['tier2_cost'] = streaming_result['tier2']['cost']
        result['tier2_success'] = streaming_result['tier2']['success']
        result['tier2_error'] = streaming_result['tier2']['error']
        
        result['total_latency'] = streaming_result['total_latency']
        result['time_to_first_output'] = streaming_result['time_to_first_output']
        result['perceived_latency_improvement'] = streaming_result.get('perceived_latency_improvement')
        
    except Exception as e:
        result['tier1_error'] = f"Pipeline error: {str(e)}"
        result['tier2_error'] = f"Pipeline error: {str(e)}"
    
    return result


def save_results(results: list, output_dir: Path):
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


async def main():
    """Run batch testing"""
    print("=" * 60)
    print("BATCH TESTING - Approach 5: Streaming/Progressive Models")
    print("=" * 60)
    print()
    
    # Get all images
    images = get_all_images()
    print(f"Found {len(images)} images to test")
    print()
    
    # Output directory
    output_dir = project_root / 'results' / 'approach_5_streaming' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print("Initializing streaming pipeline...")
    pipeline = StreamingPipeline()
    
    # Pre-load BLIP-2 model (if available)
    try:
        from model_wrappers import get_blip2_model
        blip2_model = get_blip2_model()
        if blip2_model:
            print("✅ BLIP-2 model ready")
        else:
            print("⚠️  BLIP-2 model not available - tier1 will fail")
    except Exception as e:
        print(f"⚠️  BLIP-2 initialization warning: {e}")
    
    print()
    
    # Test all images
    all_results = []
    start_time = time.time()
    
    for i, img_info in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Testing {img_info['filename']} ({img_info['category']})")
        
        result = await test_image(
            img_info['path'],
            img_info['category'],
            pipeline
        )
        
        all_results.append(result)
        save_results([result], output_dir)
        
        # Print status
        if result['success']:
            tier1_status = "✅" if result['tier1_success'] else "❌"
            tier2_status = "✅" if result['tier2_success'] else "❌"
            print(f"  Tier1: {tier1_status} ({result['tier1_latency']:.2f}s)" if result['tier1_latency'] else "  Tier1: ❌")
            print(f"  Tier2: {tier2_status} ({result['tier2_latency']:.2f}s)" if result['tier2_latency'] else "  Tier2: ❌")
            if result['time_to_first_output']:
                print(f"  Time to first output: {result['time_to_first_output']:.2f}s")
            if result['perceived_latency_improvement']:
                print(f"  Perceived latency improvement: {result['perceived_latency_improvement']:.1f}%")
        else:
            print(f"  ❌ Failed: {result.get('tier1_error') or result.get('tier2_error')}")
        
        print()
        
        # Small delay to prevent rate limiting
        if i < len(images):
            await asyncio.sleep(1)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r['success'])
    tier1_successful = sum(1 for r in all_results if r['tier1_success'])
    tier2_successful = sum(1 for r in all_results if r['tier2_success'])
    
    print("=" * 60)
    print("BATCH TESTING COMPLETE")
    print("=" * 60)
    print(f"Total tests: {len(images)}")
    print(f"Successful (at least one tier): {successful}/{len(images)} ({successful*100/len(images):.1f}%)")
    print(f"Tier1 successful: {tier1_successful}/{len(images)} ({tier1_successful*100/len(images):.1f}%)")
    print(f"Tier2 successful: {tier2_successful}/{len(images)} ({tier2_successful*100/len(images):.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir / 'batch_results.csv'}")
    
    # Calculate average latencies
    tier1_latencies = [r['tier1_latency'] for r in all_results if r['tier1_latency']]
    tier2_latencies = [r['tier2_latency'] for r in all_results if r['tier2_latency']]
    time_to_first = [r['time_to_first_output'] for r in all_results if r['time_to_first_output']]
    
    if tier1_latencies:
        print(f"\nTier1 (BLIP-2) average latency: {sum(tier1_latencies)/len(tier1_latencies):.2f}s")
    if tier2_latencies:
        print(f"Tier2 (GPT-4V) average latency: {sum(tier2_latencies)/len(tier2_latencies):.2f}s")
    if time_to_first:
        print(f"Average time to first output: {sum(time_to_first)/len(time_to_first):.2f}s")
    
    # Calculate total cost
    total_cost = sum(r['tier2_cost'] for r in all_results if r['tier2_cost'])
    print(f"\nTotal cost (Tier2 only): ${total_cost:.4f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

