"""
Performance Optimizer for Approach 3.5
Advanced optimizations: streaming, adaptive quality modes, enhanced caching
"""
from typing import Dict, Literal, Optional, Callable
from enum import Enum


class QualityMode(Enum):
    """Quality modes for adaptive performance"""
    FAST = "fast"  # Target: <1s, lower max_tokens, simpler prompts
    BALANCED = "balanced"  # Target: <1.5s, current optimized settings
    QUALITY = "quality"  # Target: <2.5s, higher max_tokens for complex scenes


def get_quality_mode_settings(mode: QualityMode) -> Dict:
    """
    Get settings for a quality mode
    
    Args:
        mode: Quality mode (FAST, BALANCED, QUALITY)
    
    Returns:
        Dict with max_tokens_multiplier, temperature, prompt_complexity
    """
    if mode == QualityMode.FAST:
        return {
            'max_tokens_multiplier': 0.7,  # 30% reduction
            'temperature': 0.3,  # Lower = faster
            'prompt_complexity': 'minimal',  # Simplest prompts
            'target_latency': 1.0
        }
    elif mode == QualityMode.BALANCED:
        return {
            'max_tokens_multiplier': 1.0,  # Current optimized settings
            'temperature': 0.4,  # Current optimized
            'prompt_complexity': 'optimized',  # Current optimized prompts
            'target_latency': 1.5
        }
    else:  # QUALITY
        return {
            'max_tokens_multiplier': 1.3,  # 30% increase for quality
            'temperature': 0.5,  # Slightly higher for quality
            'prompt_complexity': 'standard',  # More detailed prompts
            'target_latency': 2.5
        }


def adapt_quality_mode_based_on_latency(
    recent_latencies: list,
    current_mode: QualityMode = QualityMode.BALANCED
) -> QualityMode:
    """
    Adaptively adjust quality mode based on recent latency performance
    
    Args:
        recent_latencies: List of recent latencies (last 5-10 runs)
        current_mode: Current quality mode
    
    Returns:
        Recommended quality mode
    """
    if not recent_latencies:
        return current_mode
    
    avg_latency = sum(recent_latencies) / len(recent_latencies)
    
    # If consistently fast, can use QUALITY mode
    if avg_latency < 1.0 and current_mode != QualityMode.QUALITY:
        return QualityMode.QUALITY
    
    # If consistently slow, use FAST mode
    if avg_latency > 2.0 and current_mode != QualityMode.FAST:
        return QualityMode.FAST
    
    # Otherwise maintain BALANCED
    return QualityMode.BALANCED


def should_use_streaming(llm_model: str, target_latency: float = 1.5) -> bool:
    """
    Determine if streaming should be used based on model and target
    
    Args:
        llm_model: LLM model name
        target_latency: Target latency in seconds
    
    Returns:
        True if streaming should be used
    """
    # Streaming beneficial for slower models or strict latency targets
    slow_models = ['gpt-4o-mini', 'claude-haiku', 'gemini']
    return any(model in llm_model.lower() for model in slow_models) or target_latency < 1.0

