"""
Streaming Pipeline for Approach 5: Streaming/Progressive Models
Two-tier architecture: Fast BLIP-2 + Detailed GPT-4V
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from model_wrappers import run_blip2_async, call_gpt4v_async


class StreamingPipeline:
    """
    Two-tier streaming pipeline for progressive disclosure
    Tier 1: Fast local model (BLIP-2) for quick overview
    Tier 2: Detailed cloud model (GPT-4V) for comprehensive description
    """
    
    def __init__(self):
        """Initialize the streaming pipeline"""
        self.tier1_model = "BLIP-2"
        self.tier2_model = "GPT-4V"
    
    async def describe_image(
        self,
        image_path: Path,
        tier1_prompt: Optional[str] = None,
        tier2_system_prompt: Optional[str] = None,
        tier2_user_prompt: Optional[str] = None
    ) -> Dict:
        """
        Describe an image using two-tier streaming approach
        
        Args:
            image_path: Path to image file
            tier1_prompt: Optional prompt for tier1 (fast model)
            tier2_system_prompt: Optional system prompt for tier2 (detailed model)
            tier2_user_prompt: Optional user prompt for tier2
        
        Returns:
            Dict with complete results:
            {
                'success': bool,
                'tier1': {
                    'description': str,
                    'latency': float,
                    'success': bool,
                    'error': Optional[str]
                },
                'tier2': {
                    'description': str,
                    'latency': float,
                    'tokens': int,
                    'cost': float,
                    'success': bool,
                    'error': Optional[str]
                },
                'time_to_first_output': float,  # Tier1 latency
                'total_latency': float,  # Max of tier1 and tier2
                'perceived_latency_improvement': Optional[float]  # vs single tier2
            }
        """
        start_time = time.time()
        
        result = {
            'success': False,
            'tier1': {
                'description': None,
                'latency': None,
                'success': False,
                'error': None
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
        
        # Start both models simultaneously
        tier1_task = asyncio.create_task(
            run_blip2_async(image_path, tier1_prompt)
        )
        tier2_task = asyncio.create_task(
            call_gpt4v_async(image_path, tier2_system_prompt, tier2_user_prompt)
        )
        
        # Wait for tier1 (quick description) - this is the perceived latency
        tier1_result, tier1_error = await tier1_task
        
        tier1_latency = time.time() - start_time
        
        if tier1_result:
            result['tier1'] = {
                'description': tier1_result['description'],
                'latency': tier1_result['latency'],
                'success': True,
                'error': None
            }
            result['time_to_first_output'] = tier1_latency
        else:
            result['tier1'] = {
                'description': None,
                'latency': tier1_latency,
                'success': False,
                'error': tier1_error
            }
            result['time_to_first_output'] = tier1_latency  # Still record time even if failed
        
        # Wait for tier2 (detailed description)
        tier2_result, tier2_error = await tier2_task
        
        tier2_latency = time.time() - start_time
        
        if tier2_result:
            result['tier2'] = {
                'description': tier2_result['description'],
                'latency': tier2_result['latency'],
                'tokens': tier2_result.get('tokens'),
                'cost': tier2_result.get('cost', 0.0),
                'success': True,
                'error': None
            }
        else:
            result['tier2'] = {
                'description': None,
                'latency': tier2_latency,
                'tokens': None,
                'cost': None,
                'success': False,
                'error': tier2_error
            }
        
        # Calculate total latency (max of both tiers since they run in parallel)
        result['total_latency'] = max(
            tier1_latency if tier1_latency else 0,
            tier2_latency if tier2_latency else 0
        )
        
        # Success if at least one tier succeeded
        result['success'] = result['tier1']['success'] or result['tier2']['success']
        
        # Calculate perceived latency improvement (vs single tier2)
        # This is the key metric: how much faster does the user perceive the response?
        if result['tier2']['success'] and result['tier1']['success']:
            tier2_only_latency = result['tier2']['latency']
            perceived_latency = result['time_to_first_output']
            if tier2_only_latency > 0:
                improvement = ((tier2_only_latency - perceived_latency) / tier2_only_latency) * 100
                result['perceived_latency_improvement'] = improvement
        
        return result


async def describe_image_async(
    image_path: Path,
    tier1_prompt: Optional[str] = None,
    tier2_system_prompt: Optional[str] = None,
    tier2_user_prompt: Optional[str] = None
) -> Dict:
    """
    Convenience function to describe an image using streaming pipeline
    
    Args:
        image_path: Path to image file
        tier1_prompt: Optional prompt for tier1
        tier2_system_prompt: Optional system prompt for tier2
        tier2_user_prompt: Optional user prompt for tier2
    
    Returns:
        Dict with streaming results
    """
    pipeline = StreamingPipeline()
    return await pipeline.describe_image(
        image_path,
        tier1_prompt,
        tier2_system_prompt,
        tier2_user_prompt
    )

