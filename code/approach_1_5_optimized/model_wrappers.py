"""
Model wrappers for Approach 5: Streaming/Progressive Models
Async wrappers for BLIP-2 (local) and GPT-4V (cloud)
"""
import os
import sys
import asyncio
import base64
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
approach_4_path = Path(__file__).parent.parent / "approach_4_local"
sys.path.insert(0, str(approach_4_path))

load_dotenv()

# Import BLIP-2 model
BLIP2Model = None
BLIP2_AVAILABLE = False
try:
    # Try importing from approach_4_local
    sys.path.insert(0, str(approach_4_path))
    from blip2_model import BLIP2Model
    from local_vlm import get_device, get_model_cache_dir, format_device_name
    BLIP2_AVAILABLE = True
except ImportError as e:
    BLIP2_AVAILABLE = False
    print(f"Warning: BLIP-2 not available: {e}")
    print("Install dependencies from approach_4_local.")

# Import prompts from approach_1_5_optimized (not approach_4_local)
# Use absolute import to avoid conflicts
import importlib.util
prompts_path = Path(__file__).parent / "prompts.py"
spec = importlib.util.spec_from_file_location("optimized_prompts", prompts_path)
optimized_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized_prompts)
TIER1_PROMPT = optimized_prompts.TIER1_PROMPT
TIER2_SYSTEM_PROMPT = optimized_prompts.TIER2_SYSTEM_PROMPT
TIER2_USER_PROMPT = optimized_prompts.TIER2_USER_PROMPT


# Global BLIP-2 model instance (loaded once, reused)
_blip2_model = None
_blip2_executor = None


def get_blip2_model():
    """
    Get or initialize BLIP-2 model instance (singleton pattern)
    
    Returns:
        BLIP2Model instance or None if unavailable
    """
    global _blip2_model
    
    if not BLIP2_AVAILABLE:
        return None
    
    if _blip2_model is None:
        try:
            print("Loading BLIP-2 model for streaming pipeline...")
            _blip2_model = BLIP2Model()
            print("✅ BLIP-2 model loaded")
        except Exception as e:
            print(f"❌ Failed to load BLIP-2 model: {e}")
            return None
    
    return _blip2_model


def get_blip2_executor() -> ThreadPoolExecutor:
    """
    Get thread pool executor for BLIP-2 (synchronous model in async context)
    
    Returns:
        ThreadPoolExecutor instance
    """
    global _blip2_executor
    
    if _blip2_executor is None:
        _blip2_executor = ThreadPoolExecutor(max_workers=1)
    
    return _blip2_executor


async def run_blip2_async(image_path: Path, prompt: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Run BLIP-2 model asynchronously (wrapped in thread pool)
    
    Args:
        image_path: Path to image file
        prompt: Optional prompt (defaults to TIER1_PROMPT)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'device', 'success'
    """
    if prompt is None:
        prompt = TIER1_PROMPT
    
    model = get_blip2_model()
    if model is None:
        return None, "BLIP-2 model not available"
    
    executor = get_blip2_executor()
    loop = asyncio.get_event_loop()
    
    def _run_sync():
        """Synchronous BLIP-2 inference"""
        try:
            result, error = model.describe_image(
                image_path,
                prompt=prompt,
                max_new_tokens=50,  # Shorter for quick overview
                num_beams=1  # Fastest (greedy decoding)
            )
            if error:
                return None, error
            return {
                'description': result['description'],
                'latency': result['latency'],
                'device': result.get('device', 'unknown'),
                'success': True
            }, None
        except Exception as e:
            return None, str(e)
    
    # Run in thread pool
    try:
        result, error = await loop.run_in_executor(executor, _run_sync)
        return result, error
    except Exception as e:
        return None, f"BLIP-2 async execution error: {str(e)}"


async def call_gpt4v_async(
    image_path: Path, 
    system_prompt: Optional[str] = None, 
    user_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    disable_image_resize: bool = False
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call GPT-4V API asynchronously
    
    Args:
        image_path: Path to image file
        system_prompt: Optional system prompt (defaults to TIER2_SYSTEM_PROMPT)
        user_prompt: Optional user prompt (defaults to TIER2_USER_PROMPT)
        max_tokens: Optional max tokens override (defaults to 60)
        temperature: Optional temperature override (defaults to 0)
        top_p: Optional top_p override (defaults to 1.0)
        disable_image_resize: If True, skip image resizing/compression (for standardized comparison)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'tokens', 'success'
    """
    try:
        from openai import AsyncOpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found"
        
        client = AsyncOpenAI(api_key=api_key)
        
        sys_prompt = system_prompt if system_prompt is not None else TIER2_SYSTEM_PROMPT
        usr_prompt = user_prompt if user_prompt is not None else TIER2_USER_PROMPT
        
        # Read and optimize image (resize for faster processing)
        from PIL import Image
        import io
        
        img = Image.open(image_path)
        
        # Resize if image is too large (max 1024px on longest side for speed)
        # GPT-4V can handle high-res but smaller images are faster
        # Skip if disable_image_resize is True (for standardized comparison)
        if not disable_image_resize:
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  📐 Resized image: {img.size[0]}x{img.size[1]} (for faster processing)")
        
        # Convert to JPEG for smaller file size (faster upload)
        # Skip compression if disable_image_resize is True
        if not disable_image_resize:
            if img.format != 'JPEG':
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode()
            mime_type = 'image/jpeg'
        else:
            # Use original image format without compression
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or 'PNG')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode()
            mime_type = f'image/{img.format.lower() if img.format else "png"}'
        
        # Prepare messages
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                    {"type": "text", "text": usr_prompt if usr_prompt else "Describe this image."}
                ]
            }
        ]
        
        # Make async API call
        start_time = time.time()
        
        # Use overrides if provided, otherwise use defaults
        api_max_tokens = max_tokens if max_tokens is not None else 60
        api_temperature = temperature if temperature is not None else 0
        api_top_p = top_p if top_p is not None else 1.0
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=api_max_tokens,
            temperature=api_temperature,
            top_p=api_top_p
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
        
        # Calculate cost (GPT-4o pricing: $0.01/1K input tokens, $0.03/1K output tokens)
        cost = 0.0
        if tokens and hasattr(response, 'usage'):
            input_tokens = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
            cost = (input_tokens / 1000 * 0.01) + (output_tokens / 1000 * 0.03)
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens,
            'cost': cost,
            'success': True
        }, None
        
    except Exception as e:
        return None, f"GPT-4V API error: {str(e)}"

