"""
Specialized Pipeline for Approach 3
Combines OCR/depth estimation with object detection and LLM generation
"""
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add approach_2_yolo_llm to path for imports
project_root = Path(__file__).parent.parent.parent
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

# Import Approach 2 components (code reuse)
from yolo_detector import detect_objects, format_objects_for_prompt, get_detection_summary
from llm_generator import (
    generate_description_gpt4o_mini,
    generate_description_claude_haiku,
    generate_description_gpt35_turbo
)

# Import specialized components (from local module)
# Import from local prompts module explicitly
import importlib.util
specialized_dir = Path(__file__).parent
prompts_path = specialized_dir / 'prompts.py'
spec = importlib.util.spec_from_file_location("approach3_prompts", prompts_path)
approach3_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(approach3_prompts)

from ocr_processor import OCRProcessor
from depth_estimator import DepthEstimator

# Use imported module
OCR_FUSION_SYSTEM_PROMPT = approach3_prompts.OCR_FUSION_SYSTEM_PROMPT
DEPTH_FUSION_SYSTEM_PROMPT = approach3_prompts.DEPTH_FUSION_SYSTEM_PROMPT
BASE_SYSTEM_PROMPT = approach3_prompts.BASE_SYSTEM_PROMPT
create_ocr_fusion_prompt = approach3_prompts.create_ocr_fusion_prompt
create_depth_fusion_prompt = approach3_prompts.create_depth_fusion_prompt


def _generate_description_with_prompt(
    user_prompt: str,
    llm_model: str = 'gpt-4o-mini',
    system_prompt: Optional[str] = None
):
    """
    Generate description with custom prompt (bypasses create_user_prompt wrapper)
    
    Args:
        user_prompt: Full user prompt (already formatted)
        llm_model: LLM model name
        system_prompt: System prompt
    
    Returns:
        Tuple of (result_dict, error_string)
    """
    model_lower = llm_model.lower()
    
    if model_lower in ['gpt-4o-mini', 'gpt4o-mini', 'openai']:
        # Call GPT-4o-mini directly with custom prompt
        return _call_gpt4o_mini_direct(user_prompt, system_prompt)
    elif model_lower in ['gpt-3.5-turbo', 'gpt35-turbo', 'gpt-3.5', 'gpt35']:
        return _call_gpt35_turbo_direct(user_prompt, system_prompt)
    elif model_lower in ['claude-haiku', 'claude-haiku-20241022', 'claude', 'anthropic']:
        return _call_claude_haiku_direct(user_prompt, system_prompt)
    else:
        # Fallback to GPT-4o-mini
        return _call_gpt4o_mini_direct(user_prompt, system_prompt)


def _call_gpt4o_mini_direct(user_prompt: str, system_prompt: Optional[str] = None):
    """Call GPT-4o-mini directly with custom prompt"""
    try:
        import os
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        sys_prompt = system_prompt if system_prompt else BASE_SYSTEM_PROMPT
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
    except Exception as e:
        return None, str(e)


def _call_gpt35_turbo_direct(user_prompt: str, system_prompt: Optional[str] = None):
    """Call GPT-3.5-turbo directly with custom prompt"""
    try:
        import os
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        sys_prompt = system_prompt if system_prompt else BASE_SYSTEM_PROMPT
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
    except Exception as e:
        return None, str(e)


def _call_claude_haiku_direct(user_prompt: str, system_prompt: Optional[str] = None):
    """Call Claude Haiku directly with custom prompt"""
    try:
        import os
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None, "ANTHROPIC_API_KEY not found in environment"
        
        client = anthropic.Anthropic(api_key=api_key)
        sys_prompt = system_prompt if system_prompt else BASE_SYSTEM_PROMPT
        
        start_time = time.time()
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            system=sys_prompt,
            messages=[{
                "role": "user",
                "content": user_prompt
            }]
        )
        
        latency = time.time() - start_time
        description = message.content[0].text
        
        return {
            'description': description,
            'latency': latency,
            'tokens': None
        }, None
    except Exception as e:
        return None, str(e)


def run_specialized_pipeline(
    image_path: Path,
    category: Optional[str] = None,
    mode: Literal['auto', 'ocr', 'depth'] = 'auto',
    yolo_size: str = 'n',
    llm_model: str = 'gpt-4o-mini',
    confidence_threshold: float = 0.25,
    system_prompt: Optional[str] = None,
    ocr_processor: Optional[OCRProcessor] = None,
    depth_estimator: Optional[DepthEstimator] = None
) -> Dict:
    """
    Run specialized pipeline: OCR/Depth + YOLO + LLM
    
    Args:
        image_path: Path to image file
        category: Image category ('text', 'indoor', 'outdoor', 'gaming')
        mode: 'auto' (select based on category), 'ocr' (3A), or 'depth' (3B)
        yolo_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        llm_model: LLM model name (default: 'gpt-4o-mini')
        confidence_threshold: Minimum confidence for YOLO detections
        system_prompt: Optional custom system prompt
        ocr_processor: Optional pre-initialized OCR processor (for reuse)
        depth_estimator: Optional pre-initialized depth estimator (for reuse)
    
    Returns:
        Dict with complete results:
        {
            'success': bool,
            'description': str,
            'mode': str ('ocr' or 'depth'),
            'detection_latency': float,
            'ocr_latency': float (if OCR mode),
            'depth_latency': float (if depth mode),
            'generation_latency': float,
            'total_latency': float,
            'objects_detected': list,
            'ocr_results': dict (if OCR mode),
            'depth_results': dict (if depth mode),
            'num_objects': int,
            'tokens_used': int (if available),
            'error': str (if failed)
        }
    """
    start_time = time.time()
    result = {
        'success': False,
        'description': None,
        'mode': None,
        'detection_latency': None,
        'ocr_latency': None,
        'depth_latency': None,
        'generation_latency': None,
        'total_latency': None,
        'objects_detected': [],
        'ocr_results': None,
        'depth_results': None,
        'spatial_info': None,
        'num_objects': 0,
        'tokens_used': None,
        'yolo_model': f'yolov8{yolo_size}',
        'llm_model': llm_model,
        'error': None
    }
    
    try:
        # Auto-select mode based on category
        if mode == 'auto':
            if category == 'text':
                mode = 'ocr'
            elif category in ['indoor', 'outdoor']:
                mode = 'depth'
            else:
                # Default to depth for other categories
                mode = 'depth'
        
        result['mode'] = mode
        
        # Stage 1: Parallel Processing (YOLO + Specialized component)
        print(f"  🔍 Running specialized pipeline (mode: {mode})...")
        
        # Initialize processors if not provided
        if mode == 'ocr' and ocr_processor is None:
            ocr_processor = OCRProcessor(languages=['en'], gpu=True)
        if mode == 'depth' and depth_estimator is None:
            depth_estimator = DepthEstimator()
        
        # Run YOLO and specialized component in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            yolo_future = executor.submit(
                detect_objects,
                image_path,
                model_size=yolo_size,
                confidence_threshold=confidence_threshold
            )
            
            if mode == 'ocr':
                specialized_future = executor.submit(
                    ocr_processor.extract_text,
                    image_path
                )
            else:  # depth mode
                specialized_future = executor.submit(
                    depth_estimator.estimate_depth,
                    image_path
                )
            
            # Wait for both to complete
            detections, detection_latency, num_objects = yolo_future.result()
            specialized_result = specialized_future.result()
        
        result['detection_latency'] = detection_latency
        result['objects_detected'] = detections
        result['num_objects'] = num_objects
        
        if mode == 'ocr':
            result['ocr_latency'] = specialized_result.get('ocr_latency', 0.0)
            result['ocr_results'] = specialized_result
            print(f"  ✅ YOLO: {num_objects} objects in {detection_latency:.3f}s")
            print(f"  ✅ OCR: {specialized_result.get('num_texts', 0)} texts in {result['ocr_latency']:.3f}s")
        else:  # depth mode
            result['depth_latency'] = specialized_result.get('depth_latency', 0.0)
            result['depth_results'] = specialized_result
            print(f"  ✅ YOLO: {num_objects} objects in {detection_latency:.3f}s")
            print(f"  ✅ Depth: estimated in {result['depth_latency']:.3f}s")
        
        # Format objects for prompt
        objects_text = format_objects_for_prompt(detections, include_confidence=True)
        
        # Stage 2: Fusion - Create specialized prompt
        if mode == 'ocr':
            fusion_prompt = create_ocr_fusion_prompt(objects_text, specialized_result)
            fusion_system_prompt = system_prompt if system_prompt else OCR_FUSION_SYSTEM_PROMPT
        else:  # depth mode
            # Analyze spatial relationships
            if depth_estimator and specialized_result.get('depth_map') is not None:
                spatial_info = depth_estimator.analyze_spatial_relationships(
                    detections,
                    specialized_result['depth_map']
                )
                result['spatial_info'] = spatial_info
            else:
                spatial_info = None
            
            fusion_prompt = create_depth_fusion_prompt(
                objects_text,
                specialized_result,
                spatial_info
            )
            fusion_system_prompt = system_prompt if system_prompt else DEPTH_FUSION_SYSTEM_PROMPT
        
        # Stage 3: LLM Generation
        print(f"  🤖 Generating description with {llm_model}...")
        
        # Call LLM directly with fusion prompt (bypass create_user_prompt wrapper)
        # We need to call the underlying function directly since we have a custom fusion prompt
        llm_result, error = _generate_description_with_prompt(
            fusion_prompt,
            llm_model=llm_model,
            system_prompt=fusion_system_prompt
        )
        
        if error:
            result['error'] = f"LLM generation failed: {error}"
            result['total_latency'] = time.time() - start_time
            return result
        
        if not llm_result:
            result['error'] = "LLM generation returned no result"
            result['total_latency'] = time.time() - start_time
            return result
        
        result['generation_latency'] = llm_result['latency']
        result['description'] = llm_result['description']
        result['tokens_used'] = llm_result.get('tokens')
        result['success'] = True
        result['total_latency'] = time.time() - start_time
        
        print(f"  ✅ Generated description in {llm_result['latency']:.3f}s")
        print(f"  📊 Total latency: {result['total_latency']:.3f}s")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['total_latency'] = time.time() - start_time
        print(f"  ❌ Pipeline failed: {e}")
        return result

