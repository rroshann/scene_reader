"""
Optimized LLM Generator for Approach 2.5
Wraps Approach 2 LLM generator with adaptive max_tokens based on scene complexity
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add approach_2_yolo_llm to path for imports
project_root = Path(__file__).parent.parent.parent
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

# Import Approach 2 LLM generator (code reuse)
from llm_generator import (
    generate_description_gpt35_turbo as base_generate_gpt35_turbo,
    generate_description_gpt4o_mini as base_generate_gpt4o_mini,
    generate_description_claude_haiku as base_generate_claude_haiku,
    generate_description_gemini_flash as base_generate_gemini_flash
)

# Import complexity detector
from complexity_detector import detect_complexity, get_adaptive_max_tokens


def generate_description_gpt35_turbo_adaptive(
    objects_text: str,
    detections: list,
    category: str = None,
    system_prompt: Optional[str] = None,
    use_adaptive: bool = True
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description with adaptive max_tokens based on scene complexity
    
    Args:
        objects_text: Formatted text describing detected objects
        detections: List of detected objects (for complexity detection)
        category: Scene category (gaming, indoor, outdoor, text)
        system_prompt: Optional custom system prompt
        use_adaptive: Whether to use adaptive max_tokens (default: True)
    
    Returns:
        Tuple of (result_dict, error_string)
    """
    try:
        from openai import OpenAI
        import os
        import time
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        
        # Determine max_tokens adaptively
        if use_adaptive:
            complexity = detect_complexity(detections, category)
            max_tokens = get_adaptive_max_tokens(complexity)
        else:
            max_tokens = 200  # Default fixed value
        
        # Use base function but with adaptive max_tokens
        # We need to call the API directly since base function uses fixed max_tokens
        from prompts import SYSTEM_PROMPT, create_user_prompt
        
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        user_prompt = create_user_prompt(objects_text)
        
        print(f"  📤 Sending request to GPT-3.5-turbo (max_tokens={max_tokens})...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        tokens_used = None
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used,
            'max_tokens_used': max_tokens,
            'complexity': detect_complexity(detections, category) if use_adaptive else None
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_description_adaptive(
    objects_text: str,
    detections: list,
    category: str = None,
    llm_model: str = 'gpt-3.5-turbo',
    system_prompt: Optional[str] = None,
    use_adaptive: bool = True
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description with adaptive parameters
    
    Currently only supports GPT-3.5-turbo with adaptive max_tokens.
    Other models fall back to base implementation.
    
    Args:
        objects_text: Formatted text describing detected objects
        detections: List of detected objects
        category: Scene category
        llm_model: LLM model name
        system_prompt: Optional custom system prompt
        use_adaptive: Whether to use adaptive parameters
    
    Returns:
        Tuple of (result_dict, error_string)
    """
    model_lower = llm_model.lower()
    
    if model_lower in ['gpt-3.5-turbo', 'gpt35-turbo', 'gpt-3.5', 'gpt35'] and use_adaptive:
        return generate_description_gpt35_turbo_adaptive(
            objects_text,
            detections,
            category,
            system_prompt,
            use_adaptive=True
        )
    else:
        # Fall back to base implementation
        from llm_generator import generate_description
        return generate_description(objects_text, llm_model, system_prompt)

