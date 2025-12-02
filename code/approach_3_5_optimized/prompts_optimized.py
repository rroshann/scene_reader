"""
Optimized prompts for Approach 3.5
Concise prompts with 30-40% token reduction while maintaining quality
Uses smart truncation to preserve important information
"""
from typing import Dict, List, Optional

# Import prompt_utils (handle both relative and absolute imports)
try:
    from .prompt_utils import smart_truncate_objects_text, smart_truncate_text
except ImportError:
    from prompt_utils import smart_truncate_objects_text, smart_truncate_text


# Ultra-optimized system prompts (40% further reduction, target: 60-70 tokens)
OCR_FUSION_SYSTEM_PROMPT = """Text reading assistant. Integrate objects with text. Prioritize accurate reading. Be concise."""

DEPTH_FUSION_SYSTEM_PROMPT = """Navigation assistant. Use objects and depth. Provide distances (e.g., "2m ahead"). Prioritize safety."""

BASE_SYSTEM_PROMPT = """Accessibility assistant. Concise descriptions: spatial layout, critical status, immediate concerns."""


def create_ocr_fusion_prompt(
    objects_text: str, 
    ocr_results: Dict,
    detections: Optional[List[Dict]] = None
) -> str:
    """
    Create optimized fusion prompt combining object detections with OCR results
    Uses smart truncation to preserve important information
    
    Args:
        objects_text: Formatted text describing detected objects (from YOLO)
        ocr_results: Dict with OCR results (texts, full_text, bboxes, confidences)
        detections: Optional list of detection dicts (for smart truncation prioritization)
    
    Returns:
        Formatted prompt string (optimized, ~30-40% shorter, preserves important info)
    """
    prompt_parts = []
    
    # Smart truncation of objects text (preserves high-confidence and safety-critical objects)
    if detections:
        truncated_objects = smart_truncate_objects_text(
            objects_text, 
            detections, 
            max_chars=200,
            mode='ocr'
        )
    else:
        # Fallback to simple truncation if detections not available
        truncated_objects = smart_truncate_text(objects_text, max_chars=200, preserve_safety=False) if objects_text else "No objects detected"
    
    prompt_parts.append("Objs:" + truncated_objects)
    
    # Smart truncation of OCR text (preserves safety keywords)
    if ocr_results.get('full_text'):
        truncated_text = smart_truncate_text(
            ocr_results['full_text'],
            max_chars=150,
            preserve_safety=True
        )
        prompt_parts.append(f"Text:\"{truncated_text}\"")
    
    prompt_parts.append("Describe: objects+text, explain meaning.")
    
    return '\n'.join(prompt_parts)


def create_depth_fusion_prompt(
    objects_text: str, 
    depth_info: Dict, 
    spatial_info: Dict = None,
    detections: Optional[List[Dict]] = None
) -> str:
    """
    Create optimized fusion prompt combining object detections with depth information
    Uses smart truncation to preserve important information (closer objects, safety-critical)
    
    Args:
        objects_text: Formatted text describing detected objects (from YOLO)
        depth_info: Dict with depth map statistics (mean_depth, min_depth, max_depth)
        spatial_info: Dict with spatial relationships (from analyze_spatial_relationships)
        detections: Optional list of detection dicts (for smart truncation prioritization)
    
    Returns:
        Formatted prompt string (optimized, ~30-40% shorter, preserves important info)
    """
    prompt_parts = []
    
    # Build depth info dict for smart truncation (maps object index to depth)
    depth_info_dict = {}
    if spatial_info and spatial_info.get('spatial_info') and detections:
        spatial_dict = spatial_info['spatial_info']
        if isinstance(spatial_dict, dict):
            # Map object class to depth info
            for i, obj in enumerate(detections):
                obj_class = obj.get('class', '')
                if obj_class in spatial_dict:
                    depth_info_dict[i] = spatial_dict[obj_class]
    
    # Smart truncation of objects text (preserves closer objects and safety-critical)
    if detections:
        truncated_objects = smart_truncate_objects_text(
            objects_text,
            detections,
            max_chars=200,
            mode='depth',
            depth_info_dict=depth_info_dict if depth_info_dict else None
        )
    else:
        # Fallback to simple truncation
        truncated_objects = smart_truncate_text(objects_text, max_chars=200, preserve_safety=False) if objects_text else "No objects detected"
    
    prompt_parts.append("Objs:" + truncated_objects)
    
    # Add depth information (ultra-condensed)
    if depth_info.get('mean_depth') is not None:
        mean_depth = depth_info['mean_depth']
        prompt_parts.append(f"Depth:{mean_depth:.0f}")
    
    # Add spatial relationships (ultra-condensed, prioritize closer objects)
    if spatial_info and spatial_info.get('spatial_info'):
        spatial_dict = spatial_info['spatial_info']
        if isinstance(spatial_dict, dict):
            # Sort by depth (closer first) and take top 3
            spatial_items_with_depth = []
            for obj_class, info in spatial_dict.items():
                if isinstance(info, dict):
                    depth_val = info.get('depth', 0)
                    if depth_val:
                        spatial_items_with_depth.append((depth_val, obj_class))
            
            # Sort by depth (lower = closer = more important)
            spatial_items_with_depth.sort(key=lambda x: x[0])
            
            # Take top 3 closest objects
            spatial_items = [
                f"{obj_class}:{depth_val:.0f}"
                for depth_val, obj_class in spatial_items_with_depth[:3]
            ]
            
            if spatial_items:
                prompt_parts.append(f"Dist:{','.join(spatial_items)}")
    
    prompt_parts.append("Describe with distances. Prioritize safety.")
    
    return '\n'.join(prompt_parts)

