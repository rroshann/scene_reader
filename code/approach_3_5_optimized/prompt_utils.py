"""
Prompt Utilities for Approach 3.5
Smart truncation and prioritization functions for preserving important information
"""
from typing import List, Dict, Optional
import re


# Safety-critical object classes (always preserve)
SAFETY_CRITICAL_CLASSES = {
    'person', 'people', 'human', 'pedestrian',
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
    'fire', 'smoke', 'hazard', 'danger', 'warning'
}


def score_object_importance(obj: Dict, mode: str = 'ocr', depth_info: Optional[Dict] = None) -> float:
    """
    Score object by importance for smart truncation
    
    Args:
        obj: Object dict with 'class', 'confidence', 'bbox'
        mode: 'ocr' or 'depth'
        depth_info: Optional dict with depth information for this object
    
    Returns:
        Importance score (higher = more important)
    """
    score = 0.0
    
    # Base score from confidence
    confidence = obj.get('confidence', 0.0)
    score += confidence * 10.0
    
    # Safety-critical objects get high priority
    obj_class = obj.get('class', '').lower()
    if obj_class in SAFETY_CRITICAL_CLASSES:
        score += 50.0  # High priority for safety
    
    # For depth mode, closer objects are more important
    if mode == 'depth' and depth_info:
        depth_val = depth_info.get('mean_depth', 0)
        if depth_val > 0:
            # Closer = lower depth value typically, so invert
            # Normalize depth (assuming 0-255 range)
            normalized_depth = depth_val / 255.0
            score += (1.0 - normalized_depth) * 20.0  # Closer objects get higher score
    
    return score


def smart_truncate_objects_text(
    objects_text: str,
    detections: List[Dict],
    max_chars: int = 200,
    mode: str = 'ocr',
    depth_info_dict: Optional[Dict] = None
) -> str:
    """
    Smart truncation of objects text, preserving most important objects
    
    Args:
        objects_text: Formatted text describing detected objects
        detections: List of detected objects (for prioritization)
        max_chars: Maximum characters to preserve
        mode: 'ocr' or 'depth'
        depth_info_dict: Optional dict mapping object indices to depth info
    
    Returns:
        Truncated objects text with important objects preserved
    """
    if not objects_text or not detections:
        return objects_text[:max_chars] if objects_text else "No objects detected"
    
    # If text is short enough, return as-is
    if len(objects_text) <= max_chars:
        return objects_text
    
    # Score and sort objects by importance
    scored_objects = []
    for i, obj in enumerate(detections):
        depth_info = depth_info_dict.get(i) if depth_info_dict else None
        score = score_object_importance(obj, mode, depth_info)
        scored_objects.append((score, i, obj))
    
    # Sort by score (highest first)
    scored_objects.sort(reverse=True, key=lambda x: x[0])
    
    # Try to preserve high-priority objects
    # Build truncated text by including objects in priority order
    preserved_indices = set()
    truncated_parts = []
    current_length = 0
    
    # First pass: preserve all safety-critical and high-confidence objects
    for score, idx, obj in scored_objects:
        obj_class = obj.get('class', '').lower()
        confidence = obj.get('confidence', 0.0)
        
        # Always preserve safety-critical or high-confidence objects
        if obj_class in SAFETY_CRITICAL_CLASSES or confidence >= 0.7:
            preserved_indices.add(idx)
            # Extract this object's text from objects_text
            # (Simplified: we'll use a regex to find object descriptions)
            obj_text = f"{obj.get('class', 'object')} (conf: {confidence:.2f})"
            if current_length + len(obj_text) + 10 <= max_chars:  # +10 for formatting
                truncated_parts.append(obj_text)
                current_length += len(obj_text) + 2
    
    # Second pass: add remaining objects until limit
    for score, idx, obj in scored_objects:
        if idx not in preserved_indices:
            obj_text = f"{obj.get('class', 'object')} (conf: {obj.get('confidence', 0.0):.2f})"
            if current_length + len(obj_text) + 10 <= max_chars:
                truncated_parts.append(obj_text)
                current_length += len(obj_text) + 2
                preserved_indices.add(idx)
    
    # If we have preserved objects, reconstruct text
    if truncated_parts:
        truncated = ', '.join(truncated_parts)
        if len(truncated) > max_chars:
            # Fallback to simple truncation at word boundary
            truncated = truncated[:max_chars]
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.7:
                truncated = truncated[:last_space] + "..."
        return truncated
    
    # Fallback: simple word-boundary truncation
    truncated = objects_text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space] + "..."
    return truncated


def smart_truncate_text(text: str, max_chars: int = 150, preserve_safety: bool = True) -> str:
    """
    Smart truncation of OCR text, preserving safety keywords
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters
        preserve_safety: Whether to preserve safety-related keywords
    
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text
    
    if preserve_safety:
        # Check for safety keywords
        safety_keywords = ['warning', 'danger', 'hazard', 'caution', 'stop', 'no entry', 'keep out']
        text_lower = text.lower()
        
        # If contains safety keywords, try to preserve them
        for keyword in safety_keywords:
            if keyword in text_lower:
                # Find keyword position
                idx = text_lower.find(keyword)
                # Try to include context around keyword
                start = max(0, idx - 20)
                end = min(len(text), idx + len(keyword) + 50)
                if end - start <= max_chars:
                    return text[start:end]
                # Otherwise, prioritize keyword area
                return text[max(0, idx - 10):min(len(text), idx + max_chars - 10)]
    
    # Standard truncation at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space] + "..."
    return truncated


def extract_object_descriptions(objects_text: str, detections: List[Dict]) -> List[tuple]:
    """
    Extract individual object descriptions from formatted objects text
    
    Args:
        objects_text: Formatted text (e.g., from format_objects_for_prompt)
        detections: List of detection dicts
    
    Returns:
        List of (index, description_text) tuples
    """
    # Parse objects_text to extract individual object descriptions
    # Format is typically: "class (conf: X.XX) at position"
    descriptions = []
    
    # Simple regex to find object patterns
    pattern = r'(\w+)\s*\(conf:\s*([\d.]+)\)'
    matches = re.finditer(pattern, objects_text)
    
    for i, match in enumerate(matches):
        if i < len(detections):
            descriptions.append((i, match.group(0)))
    
    return descriptions

