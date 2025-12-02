"""
Hybrid Pipeline: YOLO Detection + LLM Generation
Main orchestrator for Approach 2
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from yolo_detector import detect_objects, format_objects_for_prompt, get_detection_summary
from llm_generator import generate_description


def run_hybrid_pipeline(
    image_path: Path,
    yolo_size: str = 'n',
    llm_model: str = 'gpt-4o-mini',
    confidence_threshold: float = 0.25,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Run complete hybrid pipeline: YOLO detection + LLM generation
    
    Args:
        image_path: Path to image file
        yolo_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        llm_model: 'gpt-4o-mini' or 'claude-haiku'
        confidence_threshold: Minimum confidence for YOLO detections
        system_prompt: Optional custom system prompt for LLM
    
    Returns:
        Dict with complete results:
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
            'error': str (if failed)
        }
    """
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
        'error': None
    }
    
    try:
        # Stage 1: YOLO Detection
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
        
        # Format objects for LLM prompt
        objects_text = format_objects_for_prompt(detections, include_confidence=True)
        
        # Stage 2: LLM Generation
        print(f"  🤖 Generating description with {llm_model}...")
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
        
        print(f"  ✅ Generated description in {llm_result['latency']:.3f}s")
        print(f"  📊 Total latency: {result['total_latency']:.3f}s")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['total_latency'] = time.time() - start_time
        print(f"  ❌ Pipeline failed: {e}")
        return result


def test_pipeline(image_path: Path, yolo_size: str = 'n', llm_model: str = 'gpt-4o-mini'):
    """
    Test the hybrid pipeline on a single image
    
    Args:
        image_path: Path to test image
        yolo_size: YOLO model size
        llm_model: LLM model name
    """
    print("=" * 60)
    print(f"Testing Hybrid Pipeline")
    print(f"Image: {image_path.name}")
    print(f"YOLO: yolov8{yolo_size}")
    print(f"LLM: {llm_model}")
    print("=" * 60)
    
    result = run_hybrid_pipeline(image_path, yolo_size, llm_model)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Detection latency: {result['detection_latency']:.3f}s")
        print(f"Generation latency: {result['generation_latency']:.3f}s")
        print(f"Total latency: {result['total_latency']:.3f}s")
        print(f"Objects detected: {result['num_objects']}")
        print(f"\nDescription:\n{result['description']}")
    else:
        print(f"Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Test with a sample image
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
    else:
        # Try to find a test image
        test_image = Path("data/images/gaming/tic_tac_toe-opp_move_1.png")
        if not test_image.exists():
            test_image = Path("test_image.png")
    
    if not test_image.exists():
        print("Please provide an image path:")
        print("python hybrid_pipeline.py <path_to_image>")
        sys.exit(1)
    
    # Test with default configuration
    test_pipeline(test_image, yolo_size='n', llm_model='gpt-4o-mini')

