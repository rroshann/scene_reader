"""
Optimized Pipeline for Approach 1.5: Optimized Pure VLM
Fast GPT-4V with concise prompts - NO local models
"""
import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from model_wrappers import call_gpt4v_async


class StreamingPipeline:
    """
    Optimized GPT-4V pipeline - fast, cloud-only, no local models
    """
    
    def __init__(self):
        """Initialize the optimized pipeline"""
        self.model = "GPT-4V"
    
    async def describe_image(
        self,
        image_path: Path,
        tier1_prompt: Optional[str] = None,
        tier2_system_prompt: Optional[str] = None,
        tier2_user_prompt: Optional[str] = None,
        prompt_mode: str = 'gaming',  # 'gaming' or 'real_world'
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        disable_image_resize: bool = False
    ) -> Dict:
        """
        Describe an image using optimized GPT-4V
        
        Args:
            image_path: Path to image file
            tier1_prompt: Ignored (kept for compatibility)
            tier2_system_prompt: Optional system prompt (overrides prompt_mode)
            tier2_user_prompt: Optional user prompt (overrides prompt_mode)
            prompt_mode: 'gaming' or 'real_world' (default: 'gaming')
        
        Returns:
            Dict with results:
            {
                'success': bool,
                'tier1': {
                    'description': None,
                    'latency': None,
                    'success': False,
                    'error': 'BLIP-2 removed - GPT-4V only'
                },
                'tier2': {
                    'description': str,
                    'latency': float,
                    'tokens': int,
                    'cost': float,
                    'success': bool,
                    'error': Optional[str]
                },
                'time_to_first_output': float,  # Same as tier2 latency
                'total_latency': float,  # Same as tier2 latency
                'perceived_latency_improvement': None
            }
        """
        start_time = time.time()
        
        result = {
            'success': False,
            'tier1': {
                'description': None,
                'latency': None,
                'success': False,
                'error': 'BLIP-2 removed - GPT-4V only'
            },
            'tier2': {
                'description': None,
                'latency': None,
                'tokens': None,
                'cost': None,
                'success': False,
                'error': None
            },
            'time_to_first_output': None,
            'total_latency': None,
            'perceived_latency_improvement': None
        }
        
        # Use mode-specific prompts if not provided
        import importlib.util
        prompts_path = current_dir / 'prompts.py'
        spec = importlib.util.spec_from_file_location("prompts", prompts_path)
        prompts_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts_module)
        get_tier2_system_prompt = prompts_module.get_tier2_system_prompt
        get_tier2_user_prompt = prompts_module.get_tier2_user_prompt
        
        if tier2_system_prompt is None:
            tier2_system_prompt = get_tier2_system_prompt(prompt_mode)
        if tier2_user_prompt is None:
            tier2_user_prompt = get_tier2_user_prompt(prompt_mode)
        
        # Call GPT-4V directly (no BLIP-2, no delays)
        print("  🚀 Using GPT-4V (optimized, no local models)")
        tier2_result, tier2_error = await call_gpt4v_async(
            image_path, 
            tier2_system_prompt, 
            tier2_user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            disable_image_resize=disable_image_resize
        )
        
        total_latency = time.time() - start_time
        
        if tier2_result:
            result['tier2'] = {
                'description': tier2_result['description'],
                'latency': tier2_result['latency'],
                'tokens': tier2_result.get('tokens'),
                'cost': tier2_result.get('cost', 0.0),
                'success': True,
                'error': None
            }
            result['success'] = True
            result['time_to_first_output'] = total_latency
            result['total_latency'] = total_latency
        else:
            result['tier2'] = {
                'description': None,
                'latency': total_latency,
                'tokens': None,
                'cost': None,
                'success': False,
                'error': tier2_error
            }
            result['success'] = False
        
        return result


async def describe_image_async(
    image_path: Path,
    tier1_prompt: Optional[str] = None,
    tier2_system_prompt: Optional[str] = None,
    tier2_user_prompt: Optional[str] = None
) -> Dict:
    """
    Convenience function to describe an image using optimized pipeline
    
    Args:
        image_path: Path to image file
        tier1_prompt: Ignored (kept for compatibility)
        tier2_system_prompt: Optional system prompt
        tier2_user_prompt: Optional user prompt
    
    Returns:
        Dict with results
    """
    pipeline = StreamingPipeline()
    return await pipeline.describe_image(
        image_path,
        tier1_prompt,
        tier2_system_prompt,
        tier2_user_prompt
    )
