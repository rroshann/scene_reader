#!/usr/bin/env python3
"""
Quick setup test for Approach 5 streaming pipeline
Verifies all dependencies and model availability
"""
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "approach_4_local"))

print("=" * 60)
print("APPROACH 5 SETUP VERIFICATION")
print("=" * 60)
print()

# Test imports
print("1. Testing imports...")
try:
    import asyncio
    print("   ✅ asyncio")
except ImportError as e:
    print(f"   ❌ asyncio: {e}")
    sys.exit(1)

try:
    import openai
    from openai import AsyncOpenAI
    print(f"   ✅ openai (version {openai.__version__})")
    print("   ✅ AsyncOpenAI")
except ImportError as e:
    print(f"   ❌ openai: {e}")
    print("   Install with: pip install openai")
    sys.exit(1)

try:
    import transformers
    import torch
    print(f"   ✅ transformers (version {transformers.__version__})")
    print(f"   ✅ torch (version {torch.__version__})")
    if hasattr(torch.backends, 'mps'):
        print(f"   ✅ MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"   ❌ transformers/torch: {e}")
    print("   Install with: pip install transformers torch")
    sys.exit(1)

print()

# Test BLIP-2 import
print("2. Testing BLIP-2 model import...")
try:
    from blip2_model import BLIP2Model
    print("   ✅ BLIP2Model import successful")
except ImportError as e:
    print(f"   ❌ BLIP-2 import failed: {e}")
    sys.exit(1)

print()

# Test streaming pipeline imports
print("3. Testing streaming pipeline imports...")
try:
    # Import from current directory (approach_5_streaming)
    import importlib.util
    streaming_prompts_path = Path(__file__).parent / "prompts.py"
    spec = importlib.util.spec_from_file_location("streaming_prompts", streaming_prompts_path)
    streaming_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streaming_prompts)
    
    from streaming_pipeline import StreamingPipeline
    from model_wrappers import get_blip2_model, call_gpt4v_async
    TIER1_PROMPT = streaming_prompts.TIER1_PROMPT
    TIER2_SYSTEM_PROMPT = streaming_prompts.TIER2_SYSTEM_PROMPT
    print("   ✅ StreamingPipeline")
    print("   ✅ model_wrappers")
    print("   ✅ prompts")
except ImportError as e:
    print(f"   ❌ Streaming pipeline import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test BLIP-2 model initialization (without loading full model)
print("4. Testing BLIP-2 model availability...")
try:
    model = get_blip2_model()
    if model:
        print("   ✅ BLIP-2 model available")
        print(f"   ✅ Device: {model.device}")
    else:
        print("   ⚠️  BLIP-2 model not available (will use Tier2 only)")
except Exception as e:
    print(f"   ⚠️  BLIP-2 initialization warning: {e}")
    print("   Tier1 may not work, but Tier2 should still function")

print()

# Test API key
print("5. Testing API configuration...")
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print("   ✅ OPENAI_API_KEY configured")
    # Test API connection (quick test)
    try:
        client = AsyncOpenAI(api_key=api_key)
        print("   ✅ AsyncOpenAI client created")
    except Exception as e:
        print(f"   ⚠️  API client creation warning: {e}")
else:
    print("   ❌ OPENAI_API_KEY not found in .env")
    print("   Tier2 (GPT-4V) will not work without API key")

print()

# Test results directory
print("6. Testing directory structure...")
results_dir = project_root / 'results' / 'approach_5_streaming'
for subdir in ['raw', 'analysis', 'figures']:
    dir_path = results_dir / subdir
    dir_path.mkdir(parents=True, exist_ok=True)
    if dir_path.exists():
        print(f"   ✅ {subdir}/ directory exists")
    else:
        print(f"   ❌ {subdir}/ directory missing")

print()

# Test image availability
print("7. Testing image dataset...")
images_dir = project_root / 'data' / 'images'
image_count = 0
for category in ['gaming', 'indoor', 'outdoor', 'text']:
    cat_dir = images_dir / category
    if cat_dir.exists():
        count = len(list(cat_dir.glob('*.png')) + list(cat_dir.glob('*.jpg')))
        image_count += count
        print(f"   ✅ {category}: {count} images")
    else:
        print(f"   ⚠️  {category}/ directory not found")

print(f"\n   Total images: {image_count}")
if image_count < 40:
    print("   ⚠️  Expected ~42 images, found fewer")

print()
print("=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print()
print("Next steps:")
print("1. If BLIP-2 model needs to be loaded, it will download automatically")
print("2. Run: python batch_test_streaming.py")
print()

