#!/usr/bin/env python3
"""
Test Prompt Optimization Variants
Compares baseline vs optimized prompts on 8-image subset
"""
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
approach25_dir = project_root / "code" / "approach_2_5_optimized"
sys.path.insert(0, str(approach25_dir))

load_dotenv()

from hybrid_pipeline_optimized import run_hybrid_pipeline_optimized
import prompts_optimized


def get_subset_images():
    """Get 8-image subset (2 per category)"""
    images_dir = project_root / 'data/images'
    subset = []
    
    categories = ['gaming', 'indoor', 'outdoor', 'text']
    for category in categories:
        cat_dir = images_dir / category
        if cat_dir.exists():
            images = sorted(cat_dir.glob('*.png')) + sorted(cat_dir.glob('*.jpg')) + sorted(cat_dir.glob('*.jpeg'))
            for img_file in images[:2]:
                subset.append({
                    'path': img_file,
                    'filename': img_file.name,
                    'category': category
                })
    
    return subset


def test_prompt_variant(image_path, category, prompt_name, system_prompt_func, user_prompt_func):
    """Test a prompt variant on an image"""
    try:
        # Temporarily override prompts
        original_system = prompts_optimized.SYSTEM_PROMPT
        original_user = prompts_optimized.create_user_prompt
        
        prompts_optimized.SYSTEM_PROMPT = system_prompt_func
        prompts_optimized.create_user_prompt = user_prompt_func
        
        # Import and patch llm_generator
        approach2_dir = project_root / "code" / "approach_2_yolo_llm"
        sys.path.insert(0, str(approach2_dir))
        
        # We need to patch the prompts in the imported module
        # For now, pass system_prompt directly
        from yolo_detector import detect_objects, format_objects_for_prompt
        
        # Run detection
        detections, detection_latency, num_objects = detect_objects(
            image_path,
            model_size='n',
            confidence_threshold=0.25
        )
        
        objects_text = format_objects_for_prompt(detections, include_confidence=True)
        user_prompt = user_prompt_func(objects_text)
        
        # Run generation with custom prompt
        from llm_generator import generate_description_gpt35_turbo
        
        start_time = time.time()
        llm_result, error = generate_description_gpt35_turbo(
            objects_text,
            system_prompt=system_prompt_func
        )
        generation_latency = time.time() - start_time
        
        # Restore original prompts
        prompts_optimized.SYSTEM_PROMPT = original_system
        prompts_optimized.create_user_prompt = original_user
        
        if error or not llm_result:
            return None, error or "No result"
        
        total_latency = detection_latency + generation_latency
        word_count = len(llm_result['description'].split()) if llm_result.get('description') else 0
        
        return {
            'prompt_name': prompt_name,
            'total_latency': total_latency,
            'generation_latency': generation_latency,
            'detection_latency': detection_latency,
            'word_count': word_count,
            'description': llm_result.get('description', ''),
            'tokens_used': llm_result.get('tokens')
        }, None
        
    except Exception as e:
        return None, str(e)


def main():
    """Test prompt variants"""
    print("=" * 80)
    print("PROMPT OPTIMIZATION TEST")
    print("=" * 80)
    print()
    
    images = get_subset_images()
    print(f"Testing on {len(images)} images (2 per category)")
    print()
    
    # Prompt variants to test
    variants = [
        ('baseline', prompts_optimized.BASE_SYSTEM_PROMPT, prompts_optimized.BASE_CREATE_USER_PROMPT),
        ('minimal', prompts_optimized.SYSTEM_PROMPT_MINIMAL, prompts_optimized.create_user_prompt_minimal),
        ('structured', prompts_optimized.SYSTEM_PROMPT_STRUCTURED, prompts_optimized.create_user_prompt_structured),
        ('template', prompts_optimized.SYSTEM_PROMPT_MINIMAL, prompts_optimized.create_user_prompt_template),
    ]
    
    results = {}
    
    for img_info in images:
        filename = img_info['filename']
        category = img_info['category']
        print(f"\n📸 {filename} ({category})")
        
        for prompt_name, system_prompt, user_prompt_func in variants:
            print(f"  Testing {prompt_name}...")
            result, error = test_prompt_variant(
                img_info['path'],
                category,
                prompt_name,
                system_prompt,
                user_prompt_func
            )
            
            if error:
                print(f"    ❌ Error: {error}")
                continue
            
            if prompt_name not in results:
                results[prompt_name] = []
            
            results[prompt_name].append(result)
            print(f"    ✅ Latency: {result['total_latency']:.2f}s, Words: {result['word_count']}")
            
            time.sleep(0.5)  # Rate limiting
    
    # Summary
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    for prompt_name in variants:
        name = prompt_name[0]
        if name not in results:
            continue
        
        latencies = [r['total_latency'] for r in results[name]]
        word_counts = [r['word_count'] for r in results[name]]
        
        avg_latency = sum(latencies) / len(latencies)
        avg_words = sum(word_counts) / len(word_counts)
        
        print(f"{name.upper()}:")
        print(f"  Mean Latency: {avg_latency:.2f}s")
        print(f"  Mean Words: {avg_words:.1f}")
        print()
    
    # Find fastest
    fastest = min(results.keys(), key=lambda k: sum(r['total_latency'] for r in results[k]) / len(results[k]))
    fastest_latency = sum(r['total_latency'] for r in results[fastest]) / len(results[fastest])
    
    print(f"🏆 FASTEST: {fastest} ({fastest_latency:.2f}s mean)")
    
    # Save results
    report_path = project_root / 'results' / 'approach_2_5_optimized' / 'analysis' / 'prompt_optimization_test.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Prompt Optimization Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Images Tested: {len(images)}\n\n")
        
        for name in results.keys():
            f.write(f"{name.upper()}:\n")
            latencies = [r['total_latency'] for r in results[name]]
            word_counts = [r['word_count'] for r in results[name]]
            f.write(f"  Mean Latency: {sum(latencies)/len(latencies):.2f}s\n")
            f.write(f"  Mean Words: {sum(word_counts)/len(word_counts):.1f}\n\n")
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()

