"""
Optimized Hybrid Pipeline: YOLO Detection + LLM Generation (Approach 2.5)
Extends Approach 2 with GPT-3.5-turbo and optimizations
"""
import sys
from pathlib import Path
from typing import Dict, Optional

# Add approach_2_yolo_llm to path for imports
project_root = Path(__file__).parent.parent.parent
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

# Import Approach 2 components (code reuse)
from yolo_detector import detect_objects, format_objects_for_prompt, get_detection_summary
from llm_generator import generate_description

# Import cache manager
from cache_manager import get_cache_manager

# Import adaptive generator (optional)
try:
    from llm_generator_optimized import generate_description_adaptive
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
        use_adaptive: bool = False
    ) -> Dict:
    """
    Run optimized hybrid pipeline: YOLO detection + LLM generation (GPT-3.5-turbo)
    
    This is a wrapper around Approach 2's pipeline, optimized for speed with GPT-3.5-turbo.
    
    Args:
        image_path: Path to image file
        yolo_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        llm_model: 'gpt-3.5-turbo' (default, optimized) or other models
        confidence_threshold: Minimum confidence for YOLO detections
        system_prompt: Optional custom system prompt for LLM
        use_cache: Whether to use caching (default: True)
        use_adaptive: Whether to use adaptive max_tokens (default: False)
    
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
                llm_result, error = generate_description(
                    objects_text,
                    llm_model=llm_model,
                    system_prompt=system_prompt
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

