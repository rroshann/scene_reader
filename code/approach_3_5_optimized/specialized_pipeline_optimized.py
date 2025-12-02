"""
Optimized Specialized Pipeline for Approach 3.5
Combines OCR/depth estimation with object detection and LLM generation
Optimizations: GPT-3.5-turbo, caching, adaptive max_tokens, optimized prompts, PaddleOCR
"""
import sys
import time
import os
from pathlib import Path
from typing import Dict, Optional, Literal
from concurrent.futures import ThreadPoolExecutor

# Add approach_2_yolo_llm to path for imports
project_root = Path(__file__).parent.parent.parent
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

# Import Approach 2 components (code reuse)
from yolo_detector import detect_objects, format_objects_for_prompt

# Import Approach 3 components (code reuse)
approach3_dir = project_root / "code" / "approach_3_specialized"
sys.path.insert(0, str(approach3_dir))
import importlib.util
depth_estimator_path = approach3_dir / 'depth_estimator.py'
spec = importlib.util.spec_from_file_location("depth_estimator", depth_estimator_path)
depth_estimator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_estimator_module)
DepthEstimator = depth_estimator_module.DepthEstimator

# Import Approach 3.5 optimized components
from cache_manager import get_cache_manager
from complexity_detector import detect_complexity, get_adaptive_max_tokens
from prompts_optimized import (
    OCR_FUSION_SYSTEM_PROMPT,
    DEPTH_FUSION_SYSTEM_PROMPT,
    BASE_SYSTEM_PROMPT,
    create_ocr_fusion_prompt,
    create_depth_fusion_prompt
)
from ocr_processor_optimized import OCRProcessorOptimized

# Global model instances for warmup/reuse
_global_ocr_processor: Optional[OCRProcessorOptimized] = None
_global_depth_estimator: Optional[DepthEstimator] = None


def warmup_models(mode: Literal['ocr', 'depth', 'both'] = 'both'):
    """
    Warmup models by initializing them upfront
    
    Args:
        mode: 'ocr' to initialize OCR, 'depth' for depth, 'both' for both
    """
    global _global_ocr_processor, _global_depth_estimator
    
    if mode in ['ocr', 'both']:
        if _global_ocr_processor is None:
            print("  🔥 Warming up OCR processor...")
            try:
                _global_ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
                print("  ✅ OCR processor warmed up")
            except Exception as e:
                print(f"  ⚠️  OCR warmup failed: {e}")
    
    if mode in ['depth', 'both']:
        if _global_depth_estimator is None:
            print("  🔥 Warming up depth estimator...")
            try:
                _global_depth_estimator = DepthEstimator()
                print("  ✅ Depth estimator warmed up")
            except Exception as e:
                print(f"  ⚠️  Depth warmup failed: {e}")


def _generate_description_optimized(
    user_prompt: str,
    llm_model: str = 'gpt-3.5-turbo',
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.4
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Generate description with optimized LLM (GPT-3.5-turbo by default)
    
    Args:
        user_prompt: Full user prompt (already formatted)
        llm_model: LLM model name (default: 'gpt-3.5-turbo')
        system_prompt: System prompt
        max_tokens: Optional max tokens (if None, uses default based on model)
    
    Returns:
        Tuple of (result_dict, error_string)
    """
    model_lower = llm_model.lower()
    
    # Default max_tokens if not provided
    if max_tokens is None:
        if 'gpt-3.5-turbo' in model_lower or 'gpt35' in model_lower:
            max_tokens = 200
        elif 'gpt-4o-mini' in model_lower or 'gpt4o-mini' in model_lower:
            max_tokens = 300
        else:
            max_tokens = 200
    
    if model_lower in ['gpt-3.5-turbo', 'gpt35-turbo', 'gpt-3.5', 'gpt35']:
        return _call_gpt35_turbo_direct(user_prompt, system_prompt, max_tokens, temperature)
    elif model_lower in ['gpt-4o-mini', 'gpt4o-mini', 'openai']:
        return _call_gpt4o_mini_direct(user_prompt, system_prompt, max_tokens, temperature)
    elif model_lower in ['claude-haiku', 'claude-haiku-20241022', 'claude', 'anthropic']:
        return _call_claude_haiku_direct(user_prompt, system_prompt, max_tokens, temperature)
    else:
        # Fallback to GPT-3.5-turbo
        return _call_gpt35_turbo_direct(user_prompt, system_prompt, max_tokens, temperature)


def _call_gpt35_turbo_direct(user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.4):
    """
    Call GPT-3.5-turbo directly with custom prompt (optimized for speed)
    
    Args:
        user_prompt: User prompt
        system_prompt: System prompt
        max_tokens: Maximum tokens (aggressively reduced)
        temperature: Temperature (lower = faster, more deterministic, default: 0.4 for speed)
    """
    try:
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
            max_tokens=max_tokens,
            temperature=temperature  # Lower temperature for faster generation
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        # Early stopping: limit response to first 2 sentences if too long
        # But preserve safety-critical information (don't truncate if contains warnings)
        if description:
            safety_keywords = ['warning', 'danger', 'hazard', 'caution', 'stop', 'avoid', 'unsafe']
            has_safety_info = any(keyword in description.lower() for keyword in safety_keywords)
            
            if not has_safety_info:  # Only truncate if no safety info
                sentences = description.split('. ')
                if len(sentences) > 2 and len(description) > 200:
                    description = '. '.join(sentences[:2]) + '.'
        
        tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
    except Exception as e:
        return None, str(e)


def _call_gpt4o_mini_direct(user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 300, temperature: float = 0.4):
    """Call GPT-4o-mini directly with custom prompt (fallback, optimized for speed)"""
    try:
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
            max_tokens=max_tokens,
            temperature=temperature  # Lower temperature for faster generation
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


def _call_claude_haiku_direct(user_prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 200, temperature: float = 0.4):
    """Call Claude Haiku directly with custom prompt (optimized for speed)"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None, "ANTHROPIC_API_KEY not found in environment"
        
        client = anthropic.Anthropic(api_key=api_key)
        sys_prompt = system_prompt if system_prompt else BASE_SYSTEM_PROMPT
        
        start_time = time.time()
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,  # Lower temperature for faster generation
            max_tokens=max_tokens,
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


def run_specialized_pipeline_optimized(
    image_path: Path,
    category: Optional[str] = None,
    mode: Literal['auto', 'ocr', 'depth'] = 'auto',
    yolo_size: str = 'n',
    llm_model: str = 'gpt-3.5-turbo',
    confidence_threshold: float = 0.25,
    system_prompt: Optional[str] = None,
    use_cache: bool = True,
    use_adaptive: bool = True,
    quality_mode: Literal['fast', 'balanced', 'quality'] = 'balanced',
    ocr_processor: Optional[OCRProcessorOptimized] = None,
    depth_estimator: Optional[DepthEstimator] = None
) -> Dict:
    """
    Run optimized specialized pipeline: OCR/Depth + YOLO + LLM (with optimizations)
    
    Args:
        image_path: Path to image file
        category: Image category ('text', 'indoor', 'outdoor', 'gaming')
        mode: 'auto' (select based on category), 'ocr' (3A), or 'depth' (3B)
        yolo_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        llm_model: LLM model name (default: 'gpt-3.5-turbo' - optimized)
        confidence_threshold: Minimum confidence for YOLO detections
        system_prompt: Optional custom system prompt
        use_cache: Whether to use caching (default: True)
        use_adaptive: Whether to use adaptive max_tokens (default: True)
        quality_mode: Quality mode ('fast', 'balanced', 'quality') - affects max_tokens and temperature
        ocr_processor: Optional pre-initialized OCR processor (for reuse)
        depth_estimator: Optional pre-initialized depth estimator (for reuse)
    
    Returns:
        Dict with complete results (same format as Approach 3, plus new fields):
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
            'cache_hit': bool,
            'complexity': str (simple/medium/complex),
            'max_tokens_used': int,
            'quality_mode': str (fast/balanced/quality),
            'temperature': float (temperature used for LLM),
            'approach_version': '3.5',
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
        'cache_hit': False,
        'complexity': None,
        'max_tokens_used': None,
        'approach_version': '3.5',
        'error': None
    }
    
    try:
        # Validate image path
        if not image_path.exists():
            result['error'] = f"Image path does not exist: {image_path}"
            result['total_latency'] = time.time() - start_time
            return result
        
        # Auto-select mode based on category
        if mode == 'auto':
            if category == 'text':
                mode = 'ocr'
            elif category in ['indoor', 'outdoor']:
                mode = 'depth'
            else:
                mode = 'depth'  # Default to depth
        
        result['mode'] = mode
        
        # Stage 1: Parallel Processing (YOLO + Specialized component)
        print(f"  🔍 Running optimized specialized pipeline (mode: {mode})...")
        
        # Use global instances if available, otherwise use provided or create new
        global _global_ocr_processor, _global_depth_estimator
        
        if mode == 'ocr':
            if ocr_processor is None:
                ocr_processor = _global_ocr_processor
            if ocr_processor is None:
                ocr_processor = OCRProcessorOptimized(languages=['en'], use_paddleocr=True)
                _global_ocr_processor = ocr_processor
        
        if mode == 'depth':
            if depth_estimator is None:
                depth_estimator = _global_depth_estimator
            if depth_estimator is None:
                depth_estimator = DepthEstimator()
                _global_depth_estimator = depth_estimator
        
        # Run YOLO and specialized component
        # For OCR: run in parallel (both always needed)
        # For Depth: run YOLO first, then conditionally run depth (skip if no objects)
        if mode == 'ocr':
            # OCR mode: run YOLO and OCR in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(
                    detect_objects,
                    image_path,
                    model_size=yolo_size,
                    confidence_threshold=confidence_threshold
                )
                ocr_future = executor.submit(
                    ocr_processor.extract_text,
                    image_path
                )
                detections, detection_latency, num_objects = yolo_future.result()
                specialized_result = ocr_future.result()
        else:  # depth mode
            # Run YOLO and depth estimation in parallel for better performance
            # Even if num_objects == 0, running in parallel saves time vs sequential
            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future = executor.submit(
                    detect_objects,
                    image_path,
                    model_size=yolo_size,
                    confidence_threshold=confidence_threshold
                )
                depth_future = executor.submit(
                    depth_estimator.estimate_depth,
                    image_path
                )
                
                # Get results (both run in parallel)
                detections, detection_latency, num_objects = yolo_future.result()
                specialized_result = depth_future.result()
            
            # Conditional depth processing: if no objects, mark depth as skipped
            # (Depth was computed in parallel, but we mark it as unused for consistency)
            if num_objects == 0:
                specialized_result['error'] = 'Skipped: no objects detected (computed in parallel but unused)'
        
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
        
        # Detect complexity for adaptive max_tokens
        complexity = detect_complexity(detections, category) if use_adaptive else None
        result['complexity'] = complexity
        
        # Stage 2: Check cache before LLM generation
        cache_hit = False
        cache_key = None
        cache_manager = None
        
        if use_cache:
            cache_manager = get_cache_manager()
            cache_key = cache_manager.get_cache_key(
                yolo_model=f'yolov8{yolo_size}',
                objects=detections,
                mode=mode,
                ocr_results=specialized_result if mode == 'ocr' else None,
                depth_results=specialized_result if mode == 'depth' else None,
                prompt_template='optimized'
            )
            
            cached_result = cache_manager.get_cached_result(cache_key)
            if cached_result:
                cache_hit = True
                print(f"  💾 Cache HIT! Using cached description")
                result['generation_latency'] = cached_result.get('generation_latency', 0.0)
                result['description'] = cached_result.get('description')
                result['tokens_used'] = cached_result.get('tokens_used')
                result['cache_hit'] = True
                result['total_latency'] = time.time() - start_time
                result['success'] = True
                print(f"  📊 Total latency: {result['total_latency']:.3f}s (cached)")
                return result
        
        # Stage 3: Fusion - Create optimized prompt (with smart truncation)
        if mode == 'ocr':
            fusion_prompt = create_ocr_fusion_prompt(
                objects_text, 
                specialized_result,
                detections=detections  # Pass detections for smart truncation
            )
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
                spatial_info,
                detections=detections  # Pass detections for smart truncation
            )
            fusion_system_prompt = system_prompt if system_prompt else DEPTH_FUSION_SYSTEM_PROMPT
        
        # Stage 4: LLM Generation with adaptive max_tokens
        print(f"  🤖 Generating description with {llm_model}...")
        
        # Determine max_tokens and temperature adaptively (with quality mode adjustment)
        temperature = 0.4  # Default optimized temperature
        if use_adaptive and complexity:
            base_max_tokens = get_adaptive_max_tokens(complexity)
            # Apply quality mode multiplier and get temperature
            try:
                from performance_optimizer import QualityMode, get_quality_mode_settings
                quality_enum = QualityMode(quality_mode)
                settings = get_quality_mode_settings(quality_enum)
                max_tokens = int(base_max_tokens * settings['max_tokens_multiplier'])
                temperature = settings.get('temperature', 0.4)  # Use quality mode temperature
            except Exception as e:
                # Fallback if performance_optimizer not available
                max_tokens = base_max_tokens
                print(f"  ⚠️  Quality mode settings failed: {e}, using defaults")
        else:
            # Default based on model
            if 'gpt-3.5-turbo' in llm_model.lower():
                max_tokens = 200
            else:
                max_tokens = 300
        
        result['max_tokens_used'] = max_tokens
        result['quality_mode'] = quality_mode
        result['temperature'] = temperature
        
        llm_result, error = _generate_description_optimized(
            fusion_prompt,
            llm_model=llm_model,
            system_prompt=fusion_system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
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
        result['cache_hit'] = False
        
        # Store in cache
        if use_cache and cache_manager and cache_key:
            cache_manager.store_result(cache_key, {
                'description': llm_result['description'],
                'generation_latency': llm_result['latency'],
                'tokens_used': llm_result.get('tokens'),
                'yolo_model': f'yolov8{yolo_size}',
                'llm_model': llm_model
            })
        
        print(f"  ✅ Generated description in {llm_result['latency']:.3f}s")
        print(f"  📊 Total latency: {result['total_latency']:.3f}s")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['total_latency'] = time.time() - start_time
        print(f"  ❌ Pipeline failed: {e}")
        return result

