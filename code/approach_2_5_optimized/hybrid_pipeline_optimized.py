"""
Optimized Hybrid Pipeline: YOLO Detection + LLM Generation (Approach 2.5)
Extends Approach 2 with GPT-3.5-turbo and optimizations
"""
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path first
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import Approach 2 components (code reuse) - use absolute imports
from code.approach_2_yolo_llm.yolo_detector import detect_objects, format_objects_for_prompt, get_detection_summary
from code.approach_2_yolo_llm.llm_generator import generate_description

# Keep approach2_dir for later use
approach2_dir = project_root / "code" / "approach_2_yolo_llm"

# Import cache manager (use absolute import)
from code.approach_2_5_optimized.cache_manager import get_cache_manager

# Import prompt mode selectors
from code.approach_2_5_optimized.prompts_optimized import get_system_prompt, get_user_prompt_function

# Import adaptive generator (optional)
try:
    from code.approach_2_5_optimized.llm_generator_optimized import generate_description_adaptive
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


def run_hybrid_pipeline_optimized(
    image_path: Path,
    yolo_size: str = 'n',
    llm_model: str = 'gpt-3.5-turbo',
    confidence_threshold: float = 0.25,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        use_adaptive: bool = False,
        prompt_mode: str = 'real_world',  # 'gaming' or 'real_world'
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        user_question: Optional[str] = None  # User's specific question to answer
    ) -> Dict:
    """
    Run optimized hybrid pipeline: YOLO detection + LLM generation (GPT-3.5-turbo)
    
    This is a wrapper around Approach 2's pipeline, optimized for speed with GPT-3.5-turbo.
    
    Args:
        image_path: Path to image file
        yolo_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        llm_model: 'gpt-3.5-turbo' (default, optimized) or other models
        confidence_threshold: Minimum confidence for YOLO detections
        system_prompt: Optional custom system prompt for LLM (overrides prompt_mode)
        use_cache: Whether to use caching (default: True)
        use_adaptive: Whether to use adaptive max_tokens (default: False)
        prompt_mode: 'gaming' or 'real_world' (default: 'real_world')
    
    Returns:
        Dict with complete results (same format as Approach 2):
        {
            'success': bool,
            'description': str,
            'detection_latency': float,
            'generation_latency': float,
            'total_latency': float,
            'objects_detected': list,
            'num_objects': int,
            'detection_summary': dict,
            'tokens_used': int (if available),
            'yolo_model': str,
            'llm_model': str,
            'error': str (if failed),
            'approach_version': '2.5',  # Added identifier
            'cache_hit': bool  # Whether result came from cache
        }
    """
    import time
    
    start_time = time.time()
    result = {
        'success': False,
        'description': None,
        'detection_latency': None,
        'generation_latency': None,
        'total_latency': None,
        'objects_detected': [],
        'num_objects': 0,
        'detection_summary': {},
        'tokens_used': None,
        'yolo_model': f'yolov8{yolo_size}',
        'llm_model': llm_model,
        'approach_version': '2.5',  # Identifier for Approach 2.5
        'cache_hit': False,  # Initialize cache_hit
        'error': None
    }
    
    try:
        # Stage 1: YOLO Detection (reuse from Approach 2)
        print(f"  🔍 Running YOLOv8{yolo_size.upper()} detection...")
        detections, detection_latency, num_objects = detect_objects(
            image_path,
            model_size=yolo_size,
            confidence_threshold=confidence_threshold
        )
        
        result['detection_latency'] = detection_latency
        result['objects_detected'] = detections
        result['num_objects'] = num_objects
        result['detection_summary'] = get_detection_summary(detections)
        
        print(f"  ✅ Detected {num_objects} object(s) in {detection_latency:.3f}s")
        
        # Format objects for LLM prompt (reuse from Approach 2)
        objects_text = format_objects_for_prompt(detections, include_confidence=True)
        
        # Use mode-specific prompts if system_prompt not provided
        if system_prompt is None:
            system_prompt = get_system_prompt(prompt_mode)
        
        # Get mode-specific user prompt function
        create_user_prompt_fn = get_user_prompt_function(prompt_mode)
        
        # Check cache before LLM generation
        cache_hit = False
        cache_key = None
        cache_manager = None
        
        if use_cache:
            cache_manager = get_cache_manager()
            cache_key = cache_manager.get_cache_key(
                yolo_model=f'yolov8{yolo_size}',
                objects=detections,
                prompt_template='default'
            )
            
            cached_result = cache_manager.get_cached_result(cache_key)
            if cached_result:
                cache_hit = True
                print(f"  💾 Cache HIT! Using cached description")
                result['generation_latency'] = cached_result.get('generation_latency', 0.0)
                result['description'] = cached_result.get('description')
                result['tokens_used'] = cached_result.get('tokens_used')
                result['total_latency'] = time.time() - start_time
                result['cache_hit'] = True
                result['success'] = True
                print(f"  📊 Total latency: {result['total_latency']:.3f}s (cached)")
                return result
        
        # Stage 2: LLM Generation (reuse from Approach 2, with GPT-3.5-turbo)
        if not cache_hit:
            print(f"  🤖 Generating description with {llm_model}...")
            
            # Use adaptive generation if enabled and available
            if use_adaptive and ADAPTIVE_AVAILABLE:
                # Extract category from image path if possible
                category = None
                if 'gaming' in str(image_path):
                    category = 'gaming'
                elif 'indoor' in str(image_path):
                    category = 'indoor'
                elif 'outdoor' in str(image_path):
                    category = 'outdoor'
                elif 'text' in str(image_path):
                    category = 'text'
                
                llm_result, error = generate_description_adaptive(
                    objects_text,
                    detections,
                    category=category,
                    llm_model=llm_model,
                    system_prompt=system_prompt,
                    use_adaptive=True
                )
            else:
                # Create user prompt using mode-specific function and temporarily override
                # the create_user_prompt function in the prompts module that generate_description uses
                import sys
                import importlib.util
                # Find the prompts module that was imported by llm_generator
                # Use the same approach as llm_generator to avoid conflicts
                prompts_mod_name = 'approach_2_prompts'
                prompts_mod = sys.modules.get(prompts_mod_name)
                if prompts_mod is None:
                    # Import using the same method as llm_generator
                    prompts_path = approach2_dir / "prompts.py"
                    spec = importlib.util.spec_from_file_location(prompts_mod_name, prompts_path)
                    prompts_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(prompts_mod)
                
                original_create_user_prompt = prompts_mod.create_user_prompt
                
                # Create wrapper that includes user question if provided
                def create_user_prompt_with_question(objects_text):
                    base_prompt = create_user_prompt_fn(objects_text)
                    if user_question:
                        # Detect if question is about text/reading
                        text_keywords = ['street', 'sign says', 'what does', 'read', 'label', 'name', 'says']
                        is_text_question = any(keyword in user_question.lower() for keyword in text_keywords)
                        
                        if is_text_question:
                            return f"{base_prompt}\n\nUser's question: {user_question}. IMPORTANT: This question is about READING TEXT. You can only see OBJECTS (like 'sign', 'car'), but you CANNOT READ TEXT. If the text is not visible in the detected objects, you MUST say 'I cannot read the text' or 'I cannot see the street name' - DO NOT guess. Answer directly based on what you can actually see."
                        else:
                            return f"{base_prompt}\n\nUser's question: {user_question}. Please answer this question directly based on what you see in the image. Focus on answering the question, not just describing what's in front."
                    return base_prompt
                
                prompts_mod.create_user_prompt = create_user_prompt_with_question
                
                try:
                    llm_result, error = generate_description(
                        objects_text,
                        llm_model=llm_model,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens_override,
                        temperature=temperature_override
                    )
                finally:
                    # Restore original function
                    prompts_mod.create_user_prompt = original_create_user_prompt
            
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
            
            # Store in cache (only if cache is enabled and we have a cache_key)
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

