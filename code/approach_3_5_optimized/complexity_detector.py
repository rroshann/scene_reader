"""
Scene Complexity Detector for Approach 3.5
Determines scene complexity to enable adaptive max_tokens optimization
Reused from Approach 2.5
"""
from typing import Dict, List, Literal


def detect_complexity(
    detections: List[Dict],
    category: str = None
) -> Literal['simple', 'medium', 'complex']:
    """
    Detect scene complexity based on detected objects
    
    Args:
        detections: List of detected objects (dicts with class, bbox, confidence)
        category: Optional scene category (gaming, indoor, outdoor, text)
    
    Returns:
        Complexity level: 'simple', 'medium', or 'complex'
    """
    num_objects = len(detections)
    
    # Count unique object classes
    unique_classes = set(obj.get('class', '') for obj in detections)
    num_unique_classes = len(unique_classes)
    
    # Calculate average confidence
    confidences = [obj.get('confidence', 0.0) for obj in detections]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Complexity scoring
    complexity_score = 0
    
    # Object count contributes to complexity
    if num_objects == 0:
        complexity_score = 0  # Simple (no objects)
    elif num_objects <= 2:
        complexity_score = 1  # Simple
    elif num_objects <= 5:
        complexity_score = 2  # Medium
    else:
        complexity_score = 3  # Complex
    
    # Object diversity contributes
    if num_unique_classes > 3:
        complexity_score += 1
    
    # Category-based adjustments
    if category == 'gaming':
        # Gaming scenes often have UI elements, more complex
        complexity_score += 1
    elif category == 'text':
        # Text scenes are usually simpler (just text)
        complexity_score -= 1
    
    # Normalize to 3 levels
    if complexity_score <= 1:
        return 'simple'
    elif complexity_score <= 3:
        return 'medium'
    else:
        return 'complex'


def get_adaptive_max_tokens(complexity: Literal['simple', 'medium', 'complex']) -> int:
    """
    Get adaptive max_tokens based on complexity (optimized for speed)
    
    Args:
        complexity: Scene complexity level
    
    Returns:
        Recommended max_tokens value (aggressively reduced for 20-30% faster generation)
    """
    if complexity == 'simple':
        return 75  # Reduced from 100 for faster generation
    elif complexity == 'medium':
        return 100  # Reduced from 150 for faster generation
    else:  # complex
        return 150  # Reduced from 200 for faster generation


def get_complexity_stats(detections: List[Dict]) -> Dict:
    """
    Get complexity statistics for analysis
    
    Args:
        detections: List of detected objects
    
    Returns:
        Dict with complexity metrics
    """
    num_objects = len(detections)
    unique_classes = set(obj.get('class', '') for obj in detections)
    confidences = [obj.get('confidence', 0.0) for obj in detections]
    
    return {
        'num_objects': num_objects,
        'num_unique_classes': len(unique_classes),
        'unique_classes': list(unique_classes),
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'min_confidence': min(confidences) if confidences else 0.0,
        'max_confidence': max(confidences) if confidences else 0.0
    }

