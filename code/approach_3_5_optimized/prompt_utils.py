"""
Utility functions for prompt optimization and text truncation
"""
from typing import List, Dict, Optional, Any


def smart_truncate_objects_text(
    objects_text: str,
    detections: Optional[List[Any]] = None,
    max_chars: int = 300, 
    mode: str = 'real_world',
    depth_info_dict: Optional[Dict] = None
) -> str:
    """
    Smart truncation of objects text, preserving safety-critical information
    
    Args:
        objects_text: Formatted objects text
        detections: List of detection objects (optional, for prioritization)
        max_chars: Maximum characters to keep
        mode: 'ocr', 'depth', or 'real_world' (affects prioritization)
        depth_info_dict: Optional depth information dictionary
    
    Returns:
        Truncated text
    """
    if not objects_text or len(objects_text) <= max_chars:
        return objects_text
    
    # Safety-critical keywords (prioritize these)
    safety_keywords = ['car', 'vehicle', 'person', 'bicycle', 'motorcycle', 'truck', 'bus', 
                       'traffic', 'crosswalk', 'stairs', 'door', 'obstacle']
    
    # If we have detections, prioritize high-confidence and safety-critical objects
    if detections:
        # Split into lines and prioritize safety-critical ones
        lines = objects_text.split('\n')
        safety_lines = []
        other_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in safety_keywords):
                safety_lines.append(line)
            else:
                other_lines.append(line)
        
        # Combine: safety first, then others
        prioritized_text = '\n'.join(safety_lines + other_lines)
        
        # Truncate from end, but keep safety info
        if len(prioritized_text) > max_chars:
            # Try to keep safety lines + partial other lines
            safety_text = '\n'.join(safety_lines)
            remaining_chars = max_chars - len(safety_text) - 10  # 10 for separator
            
            if remaining_chars > 0:
                other_text = '\n'.join(other_lines)
                truncated_other = other_text[:remaining_chars] + '...'
                return safety_text + '\n' + truncated_other
            else:
                return safety_text[:max_chars] + '...'
        
        return prioritized_text[:max_chars]
    else:
        # Simple truncation if no detections
        return objects_text[:max_chars] + '...'


def smart_truncate_text(text: str, max_chars: int = 200, preserve_safety: bool = False) -> str:
    """
    Smart truncation of general text
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters to keep
        preserve_safety: If True, try to preserve safety-related content
    
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text
    
    if preserve_safety:
        # Try to preserve sentences with safety keywords
        sentences = text.split('. ')
        safety_sentences = []
        other_sentences = []
        
        safety_keywords = ['car', 'vehicle', 'person', 'danger', 'safe', 'obstacle', 'watch']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in safety_keywords):
                safety_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # Combine and truncate
        prioritized = '. '.join(safety_sentences + other_sentences)
        if len(prioritized) > max_chars:
            return prioritized[:max_chars] + '...'
        return prioritized
    
    # Simple truncation
    return text[:max_chars] + '...'

