#!/usr/bin/env python3
"""
Standardized Comparison Test
Runs Approaches 1.5, 2.5, and 3.5 with identical parameters for fair comparison
"""
import sys
import csv
import time
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import standardized prompts
from code.evaluation.standardized_prompts import (
    STANDARDIZED_SYSTEM_PROMPT,
    STANDARDIZED_USER_PROMPT,
    create_standardized_user_prompt_with_objects,
    create_standardized_ocr_fusion_prompt,
    create_standardized_depth_fusion_prompt
)

# Import approach pipelines
from code.approach_1_5_optimized.streaming_pipeline import StreamingPipeline
from code.approach_2_5_optimized.hybrid_pipeline_optimized import run_hybrid_pipeline_optimized
from code.approach_3_5_optimized.specialized_pipeline_optimized import run_specialized_pipeline_optimized

# Standardized parameters
STANDARDIZED_MAX_TOKENS = 100
STANDARDIZED_TEMPERATURE = 0.7
STANDARDIZED_TOP_P = 1.0


def get_all_test_images() -> List[Path]:
    """Get all test images from data/images/"""
    images_dir = project_root / "data" / "images"
    images = []
    
    for category in ['gaming', 'indoor', 'outdoor', 'text']:
        category_dir = images_dir / category
        if category_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                images.extend(category_dir.glob(ext))
    
    return sorted(images)


def determine_category(image_path: Path) -> str:
    """Determine image category from path"""
    path_str = str(image_path).lower()
    if 'gaming' in path_str:
        return 'gaming'
    elif 'indoor' in path_str:
        return 'indoor'
    elif 'outdoor' in path_str:
        return 'outdoor'
    elif 'text' in path_str:
        return 'text'
    else:
        return 'unknown'


async def test_approach_1_5(image_path: Path) -> Dict:
    """Test Approach 1.5 with standardized parameters"""
    try:
        pipeline = StreamingPipeline()
        result = await pipeline.describe_image(
            image_path=image_path,
            tier2_system_prompt=STANDARDIZED_SYSTEM_PROMPT,
            tier2_user_prompt=STANDARDIZED_USER_PROMPT,
            max_tokens=STANDARDIZED_MAX_TOKENS,
            temperature=STANDARDIZED_TEMPERATURE,
            top_p=STANDARDIZED_TOP_P,
            disable_image_resize=True  # No preprocessing for standardized test
        )
        
        if result.get('success'):
            tier2 = result.get('tier2', {})
            return {
                'success': True,
                'description': tier2.get('description', ''),
                'latency': result.get('total_latency', 0),
                'tokens': tier2.get('tokens'),
                'cost': tier2.get('cost', 0),
                'error': None
            }
        else:
            return {
                'success': False,
                'description': '',
                'latency': result.get('total_latency', 0),
                'tokens': None,
                'cost': 0,
                'error': result.get('tier2', {}).get('error', 'Unknown error')
            }
    except Exception as e:
        return {
            'success': False,
            'description': '',
            'latency': 0,
            'tokens': None,
            'cost': 0,
            'error': str(e)
        }


def test_approach_2_5(image_path: Path) -> Dict:
    """Test Approach 2.5 with standardized parameters"""
    try:
        # Override user prompt function to use standardized prompt
        from code.approach_2_5_optimized import prompts_optimized
        
        # Create standardized user prompt function
        def standardized_create_user_prompt(objects_text: str) -> str:
            return create_standardized_user_prompt_with_objects(objects_text)
        
        # Temporarily override get_user_prompt_function
        original_get_user_prompt_function = prompts_optimized.get_user_prompt_function
        prompts_optimized.get_user_prompt_function = lambda mode: standardized_create_user_prompt
        
        try:
            result = run_hybrid_pipeline_optimized(
                image_path=image_path,
                yolo_size='n',
                llm_model='gpt-3.5-turbo',
                confidence_threshold=0.25,
                system_prompt=STANDARDIZED_SYSTEM_PROMPT,
                use_cache=False,  # Disable cache for standardized test
                use_adaptive=False,  # Disable adaptive for standardized test
                prompt_mode='real_world',  # Use neutral mode
                max_tokens_override=STANDARDIZED_MAX_TOKENS,
                temperature_override=STANDARDIZED_TEMPERATURE
            )
        finally:
            # Restore original function
            prompts_optimized.get_user_prompt_function = original_get_user_prompt_function
        
        if result.get('success'):
            return {
                'success': True,
                'description': result.get('description', ''),
                'latency': result.get('total_latency', 0),
                'detection_latency': result.get('detection_latency', 0),
                'generation_latency': result.get('generation_latency', 0),
                'tokens': result.get('tokens_used'),
                'num_objects': result.get('num_objects', 0),
                'error': None
            }
        else:
            return {
                'success': False,
                'description': '',
                'latency': result.get('total_latency', 0),
                'detection_latency': result.get('detection_latency', 0),
                'generation_latency': result.get('generation_latency', 0),
                'tokens': None,
                'num_objects': result.get('num_objects', 0),
                'error': result.get('error', 'Unknown error')
            }
    except Exception as e:
        return {
            'success': False,
            'description': '',
            'latency': 0,
            'detection_latency': 0,
            'generation_latency': 0,
            'tokens': None,
            'num_objects': 0,
            'error': str(e)
        }


def test_approach_3_5(image_path: Path, category: str) -> Dict:
    """Test Approach 3.5 with standardized parameters"""
    try:
        # Override fusion prompt functions to use standardized prompts
        from code.approach_3_5_optimized import prompts_optimized
        
        # Import standardized fusion prompt functions
        from code.evaluation.standardized_prompts import (
            create_standardized_ocr_fusion_prompt,
            create_standardized_depth_fusion_prompt
        )
        
        # Temporarily override fusion prompt functions
        original_ocr_fusion = prompts_optimized.create_ocr_fusion_prompt
        original_depth_fusion = prompts_optimized.create_depth_fusion_prompt
        original_get_ocr_system = prompts_optimized.get_ocr_system_prompt
        original_get_depth_system = prompts_optimized.get_depth_system_prompt
        original_get_base_system = prompts_optimized.get_base_system_prompt
        
        prompts_optimized.create_ocr_fusion_prompt = create_standardized_ocr_fusion_prompt
        prompts_optimized.create_depth_fusion_prompt = create_standardized_depth_fusion_prompt
        prompts_optimized.get_ocr_system_prompt = lambda mode: STANDARDIZED_SYSTEM_PROMPT
        prompts_optimized.get_depth_system_prompt = lambda mode: STANDARDIZED_SYSTEM_PROMPT
        prompts_optimized.get_base_system_prompt = lambda mode: STANDARDIZED_SYSTEM_PROMPT
        
        try:
            # Determine mode based on category (use depth for most, OCR for text)
            mode = 'ocr' if category == 'text' else 'depth'
            
            result = run_specialized_pipeline_optimized(
                image_path=image_path,
                category=category,
                mode=mode,
                yolo_size='n',
                llm_model='gpt-3.5-turbo',
                confidence_threshold=0.25,
                system_prompt=STANDARDIZED_SYSTEM_PROMPT,
                use_cache=False,  # Disable cache for standardized test
                use_adaptive=False,  # Disable adaptive for standardized test
                quality_mode='balanced',
                prompt_mode='real_world',  # Use neutral mode
                max_tokens_override=STANDARDIZED_MAX_TOKENS,
                temperature_override=STANDARDIZED_TEMPERATURE
            )
        finally:
            # Restore original functions
            prompts_optimized.create_ocr_fusion_prompt = original_ocr_fusion
            prompts_optimized.create_depth_fusion_prompt = original_depth_fusion
            prompts_optimized.get_ocr_system_prompt = original_get_ocr_system
            prompts_optimized.get_depth_system_prompt = original_get_depth_system
            prompts_optimized.get_base_system_prompt = original_get_base_system
        
        if result.get('success'):
            return {
                'success': True,
                'description': result.get('description', ''),
                'latency': result.get('total_latency', 0),
                'detection_latency': result.get('detection_latency', 0),
                'ocr_latency': result.get('ocr_latency', 0) if mode == 'ocr' else None,
                'depth_latency': result.get('depth_latency', 0) if mode == 'depth' else None,
                'generation_latency': result.get('generation_latency', 0),
                'tokens': result.get('tokens_used'),
                'num_objects': result.get('num_objects', 0),
                'mode': result.get('mode', mode),
                'error': None
            }
        else:
            return {
                'success': False,
                'description': '',
                'latency': result.get('total_latency', 0),
                'detection_latency': result.get('detection_latency', 0),
                'ocr_latency': result.get('ocr_latency', 0) if mode == 'ocr' else None,
                'depth_latency': result.get('depth_latency', 0) if mode == 'depth' else None,
                'generation_latency': result.get('generation_latency', 0),
                'tokens': None,
                'num_objects': result.get('num_objects', 0),
                'mode': result.get('mode', mode),
                'error': result.get('error', 'Unknown error')
            }
    except Exception as e:
        return {
            'success': False,
            'description': '',
            'latency': 0,
            'detection_latency': 0,
            'ocr_latency': None,
            'depth_latency': None,
            'generation_latency': 0,
            'tokens': None,
            'num_objects': 0,
            'mode': 'unknown',
            'error': str(e)
        }


async def test_all_approaches(image_path: Path) -> Dict:
    """Test all 3 approaches on a single image"""
    category = determine_category(image_path)
    filename = image_path.name
    
    print(f"\n{'='*80}")
    print(f"Testing: {filename} ({category})")
    print(f"{'='*80}")
    
    results = {
        'filename': filename,
        'category': category,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test Approach 1.5
    print(f"\n[1/3] Testing Approach 1.5...")
    start_time = time.time()
    approach_1_5_result = await test_approach_1_5(image_path)
    results['approach_1_5'] = approach_1_5_result
    print(f"  {'✅' if approach_1_5_result['success'] else '❌'} Latency: {approach_1_5_result['latency']:.2f}s")
    
    # Test Approach 2.5
    print(f"\n[2/3] Testing Approach 2.5...")
    start_time = time.time()
    approach_2_5_result = test_approach_2_5(image_path)
    results['approach_2_5'] = approach_2_5_result
    print(f"  {'✅' if approach_2_5_result['success'] else '❌'} Latency: {approach_2_5_result['latency']:.2f}s")
    
    # Test Approach 3.5
    print(f"\n[3/3] Testing Approach 3.5...")
    start_time = time.time()
    approach_3_5_result = test_approach_3_5(image_path, category)
    results['approach_3_5'] = approach_3_5_result
    print(f"  {'✅' if approach_3_5_result['success'] else '❌'} Latency: {approach_3_5_result['latency']:.2f}s")
    
    return results


def save_results(results_list: List[Dict], output_path: Path):
    """Save results to CSV file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    fieldnames = [
        'filename', 'category', 'timestamp',
        # Approach 1.5
        'approach_1_5_success', 'approach_1_5_description', 'approach_1_5_latency', 
        'approach_1_5_tokens', 'approach_1_5_cost', 'approach_1_5_error',
        # Approach 2.5
        'approach_2_5_success', 'approach_2_5_description', 'approach_2_5_latency',
        'approach_2_5_detection_latency', 'approach_2_5_generation_latency',
        'approach_2_5_tokens', 'approach_2_5_num_objects', 'approach_2_5_error',
        # Approach 3.5
        'approach_3_5_success', 'approach_3_5_description', 'approach_3_5_latency',
        'approach_3_5_detection_latency', 'approach_3_5_ocr_latency', 'approach_3_5_depth_latency',
        'approach_3_5_generation_latency', 'approach_3_5_tokens', 'approach_3_5_num_objects',
        'approach_3_5_mode', 'approach_3_5_error'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results_list:
            row = {
                'filename': result['filename'],
                'category': result['category'],
                'timestamp': result['timestamp'],
                # Approach 1.5
                'approach_1_5_success': result['approach_1_5']['success'],
                'approach_1_5_description': result['approach_1_5']['description'],
                'approach_1_5_latency': result['approach_1_5']['latency'],
                'approach_1_5_tokens': result['approach_1_5']['tokens'],
                'approach_1_5_cost': result['approach_1_5']['cost'],
                'approach_1_5_error': result['approach_1_5']['error'],
                # Approach 2.5
                'approach_2_5_success': result['approach_2_5']['success'],
                'approach_2_5_description': result['approach_2_5']['description'],
                'approach_2_5_latency': result['approach_2_5']['latency'],
                'approach_2_5_detection_latency': result['approach_2_5'].get('detection_latency'),
                'approach_2_5_generation_latency': result['approach_2_5'].get('generation_latency'),
                'approach_2_5_tokens': result['approach_2_5']['tokens'],
                'approach_2_5_num_objects': result['approach_2_5']['num_objects'],
                'approach_2_5_error': result['approach_2_5']['error'],
                # Approach 3.5
                'approach_3_5_success': result['approach_3_5']['success'],
                'approach_3_5_description': result['approach_3_5']['description'],
                'approach_3_5_latency': result['approach_3_5']['latency'],
                'approach_3_5_detection_latency': result['approach_3_5'].get('detection_latency'),
                'approach_3_5_ocr_latency': result['approach_3_5'].get('ocr_latency'),
                'approach_3_5_depth_latency': result['approach_3_5'].get('depth_latency'),
                'approach_3_5_generation_latency': result['approach_3_5'].get('generation_latency'),
                'approach_3_5_tokens': result['approach_3_5']['tokens'],
                'approach_3_5_num_objects': result['approach_3_5']['num_objects'],
                'approach_3_5_mode': result['approach_3_5'].get('mode'),
                'approach_3_5_error': result['approach_3_5']['error']
            }
            writer.writerow(row)
    
    print(f"\n✅ Results saved to: {output_path}")


async def main():
    """Main test execution"""
    print("="*80)
    print("STANDARDIZED COMPARISON TEST")
    print("="*80)
    print(f"\nStandardized Parameters:")
    print(f"  max_tokens: {STANDARDIZED_MAX_TOKENS}")
    print(f"  temperature: {STANDARDIZED_TEMPERATURE}")
    print(f"  top_p: {STANDARDIZED_TOP_P}")
    print(f"  cache: DISABLED")
    print(f"  adaptive: DISABLED")
    print(f"  image preprocessing: DISABLED")
    print("="*80)
    
    # Get all test images
    images = get_all_test_images()
    print(f"\nFound {len(images)} test images")
    
    if len(images) == 0:
        print("❌ No test images found!")
        return
    
    # Randomize order to avoid bias
    random.shuffle(images)
    
    # Run tests
    results_list = []
    total_images = len(images)
    
    for i, image_path in enumerate(images, 1):
        print(f"\n\n[{i}/{total_images}] Processing: {image_path.name}")
        try:
            result = await test_all_approaches(image_path)
            results_list.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_dir = project_root / "results" / "standardized_comparison" / "raw"
    output_path = output_dir / "batch_results.csv"
    save_results(results_list, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    approach_1_5_success = sum(1 for r in results_list if r['approach_1_5']['success'])
    approach_2_5_success = sum(1 for r in results_list if r['approach_2_5']['success'])
    approach_3_5_success = sum(1 for r in results_list if r['approach_3_5']['success'])
    
    approach_1_5_latencies = [r['approach_1_5']['latency'] for r in results_list if r['approach_1_5']['success']]
    approach_2_5_latencies = [r['approach_2_5']['latency'] for r in results_list if r['approach_2_5']['success']]
    approach_3_5_latencies = [r['approach_3_5']['latency'] for r in results_list if r['approach_3_5']['success']]
    
    print(f"\nApproach 1.5: {approach_1_5_success}/{total_images} successful")
    if approach_1_5_latencies:
        print(f"  Mean latency: {sum(approach_1_5_latencies)/len(approach_1_5_latencies):.2f}s")
        print(f"  Min latency: {min(approach_1_5_latencies):.2f}s")
        print(f"  Max latency: {max(approach_1_5_latencies):.2f}s")
    
    print(f"\nApproach 2.5: {approach_2_5_success}/{total_images} successful")
    if approach_2_5_latencies:
        print(f"  Mean latency: {sum(approach_2_5_latencies)/len(approach_2_5_latencies):.2f}s")
        print(f"  Min latency: {min(approach_2_5_latencies):.2f}s")
        print(f"  Max latency: {max(approach_2_5_latencies):.2f}s")
    
    print(f"\nApproach 3.5: {approach_3_5_success}/{total_images} successful")
    if approach_3_5_latencies:
        print(f"  Mean latency: {sum(approach_3_5_latencies)/len(approach_3_5_latencies):.2f}s")
        print(f"  Min latency: {min(approach_3_5_latencies):.2f}s")
        print(f"  Max latency: {max(approach_3_5_latencies):.2f}s")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())

