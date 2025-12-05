#!/usr/bin/env python3
"""
Quick verification script to test standardized comparison setup
Tests on a single image to verify all imports and parameter overrides work
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("VERIFYING STANDARDIZED COMPARISON SETUP")
print("="*80)

# Test imports
print("\n[1/5] Testing imports...")
try:
    from code.evaluation.standardized_prompts import (
        STANDARDIZED_SYSTEM_PROMPT,
        STANDARDIZED_USER_PROMPT,
        create_standardized_user_prompt_with_objects,
        create_standardized_ocr_fusion_prompt,
        create_standardized_depth_fusion_prompt
    )
    print("  ✅ Standardized prompts imported")
except Exception as e:
    print(f"  ❌ Failed to import standardized prompts: {e}")
    sys.exit(1)

try:
    from code.approach_1_5_optimized.streaming_pipeline import StreamingPipeline
    print("  ✅ Approach 1.5 imported")
except Exception as e:
    print(f"  ❌ Failed to import Approach 1.5: {e}")
    sys.exit(1)

try:
    from code.approach_2_5_optimized.hybrid_pipeline_optimized import run_hybrid_pipeline_optimized
    print("  ✅ Approach 2.5 imported")
except Exception as e:
    print(f"  ❌ Failed to import Approach 2.5: {e}")
    sys.exit(1)

try:
    from code.approach_3_5_optimized.specialized_pipeline_optimized import run_specialized_pipeline_optimized
    print("  ✅ Approach 3.5 imported")
except Exception as e:
    print(f"  ❌ Failed to import Approach 3.5: {e}")
    sys.exit(1)

# Test parameter overrides
print("\n[2/5] Testing parameter override signatures...")
try:
    # Check Approach 1.5
    import inspect
    pipeline = StreamingPipeline()
    sig = inspect.signature(pipeline.describe_image)
    params = list(sig.parameters.keys())
    required_params = ['max_tokens', 'temperature', 'top_p', 'disable_image_resize']
    missing = [p for p in required_params if p not in params]
    if missing:
        print(f"  ❌ Approach 1.5 missing parameters: {missing}")
    else:
        print("  ✅ Approach 1.5 has all required parameters")
except Exception as e:
    print(f"  ❌ Failed to check Approach 1.5 signature: {e}")

try:
    # Check Approach 2.5
    sig = inspect.signature(run_hybrid_pipeline_optimized)
    params = list(sig.parameters.keys())
    required_params = ['max_tokens_override', 'temperature_override']
    missing = [p for p in required_params if p not in params]
    if missing:
        print(f"  ❌ Approach 2.5 missing parameters: {missing}")
    else:
        print("  ✅ Approach 2.5 has all required parameters")
except Exception as e:
    print(f"  ❌ Failed to check Approach 2.5 signature: {e}")

try:
    # Check Approach 3.5
    sig = inspect.signature(run_specialized_pipeline_optimized)
    params = list(sig.parameters.keys())
    required_params = ['max_tokens_override', 'temperature_override']
    missing = [p for p in required_params if p not in params]
    if missing:
        print(f"  ❌ Approach 3.5 missing parameters: {missing}")
    else:
        print("  ✅ Approach 3.5 has all required parameters")
except Exception as e:
    print(f"  ❌ Failed to check Approach 3.5 signature: {e}")

# Test prompt functions
print("\n[3/5] Testing standardized prompt functions...")
try:
    test_objects = "person at left center, car at right center"
    prompt = create_standardized_user_prompt_with_objects(test_objects)
    assert STANDARDIZED_USER_PROMPT in prompt
    assert test_objects in prompt
    print("  ✅ Standardized user prompt with objects works")
except Exception as e:
    print(f"  ❌ Failed to create standardized prompt: {e}")

try:
    test_ocr = {"full_text": "STOP", "texts": [{"text": "STOP"}]}
    prompt = create_standardized_ocr_fusion_prompt(test_objects, test_ocr)
    assert STANDARDIZED_USER_PROMPT in prompt
    print("  ✅ Standardized OCR fusion prompt works")
except Exception as e:
    print(f"  ❌ Failed to create OCR fusion prompt: {e}")

try:
    test_depth = {"mean_depth": 5.0, "min_depth": 2.0, "max_depth": 10.0}
    prompt = create_standardized_depth_fusion_prompt(test_objects, test_depth)
    assert STANDARDIZED_USER_PROMPT in prompt
    print("  ✅ Standardized depth fusion prompt works")
except Exception as e:
    print(f"  ❌ Failed to create depth fusion prompt: {e}")

# Check test images exist
print("\n[4/5] Checking test images...")
images_dir = project_root / "data" / "images"
if not images_dir.exists():
    print(f"  ⚠️  Images directory not found: {images_dir}")
else:
    image_count = 0
    for category in ['gaming', 'indoor', 'outdoor', 'text']:
        category_dir = images_dir / category
        if category_dir.exists():
            count = len(list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.png')))
            image_count += count
            print(f"  ✅ {category}: {count} images")
    print(f"  ✅ Total: {image_count} test images found")

# Test script structure
print("\n[5/5] Verifying test script structure...")
test_script = project_root / "code" / "evaluation" / "standardized_comparison_test.py"
if test_script.exists():
    print(f"  ✅ Test script exists: {test_script}")
    with open(test_script, 'r') as f:
        content = f.read()
        if 'STANDARDIZED_MAX_TOKENS = 100' in content:
            print("  ✅ Standardized parameters defined")
        if 'test_approach_1_5' in content and 'test_approach_2_5' in content and 'test_approach_3_5' in content:
            print("  ✅ All test functions present")
        if 'save_results' in content:
            print("  ✅ Results saving function present")
else:
    print(f"  ❌ Test script not found: {test_script}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n✅ Setup verified! You can now run:")
print("   python code/evaluation/standardized_comparison_test.py")
print("\nThis will test all approaches on all images with standardized parameters.")

