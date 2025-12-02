"""
Specialized prompts for Approach 3
Fusion prompts that combine OCR/depth data with object detections
"""
from typing import List, Dict


# System prompt for OCR-enhanced descriptions (3A)
OCR_FUSION_SYSTEM_PROMPT = """You are a visual accessibility assistant specializing in text reading. You receive both object detections and extracted text from images. Your task is to create clear, actionable descriptions that integrate both visual objects and text content. When text is present, prioritize reading it accurately and explaining its context relative to objects in the scene. Be concise but comprehensive."""

# System prompt for depth-enhanced descriptions (3B)
DEPTH_FUSION_SYSTEM_PROMPT = """You are a visual accessibility assistant specializing in spatial navigation. You receive object detections and depth/distance information. Your task is to create clear, actionable descriptions that include actual distances and spatial relationships. Use the depth information to provide accurate distance estimates (e.g., "2 meters ahead", "5 meters to the left"). Prioritize safety-critical spatial information for navigation."""

# Base system prompt (fallback)
BASE_SYSTEM_PROMPT = """You are a visual accessibility assistant. Provide concise, actionable descriptions focusing on spatial layout, critical status, and immediate concerns. Prioritize user actionability."""


def create_ocr_fusion_prompt(objects_text: str, ocr_results: Dict) -> str:
    """
    Create fusion prompt combining object detections with OCR results
    
    Args:
        objects_text: Formatted text describing detected objects (from YOLO)
        ocr_results: Dict with OCR results (texts, full_text, bboxes, confidences)
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add object detections
    prompt_parts.append("Objects detected in the image:")
    prompt_parts.append(objects_text)
    
    # Add OCR text if available
    if ocr_results.get('full_text'):
        prompt_parts.append("\nText extracted from the image:")
        prompt_parts.append(f'"{ocr_results["full_text"]}"')
        
        # Add individual text items if multiple
        if len(ocr_results.get('texts', [])) > 1:
            prompt_parts.append("\nIndividual text items:")
            for i, text in enumerate(ocr_results['texts'], 1):
                confidence = ocr_results.get('confidences', [])[i-1] if i-1 < len(ocr_results.get('confidences', [])) else 0.0
                prompt_parts.append(f"  {i}. \"{text}\" (confidence: {confidence:.2f})")
    
    prompt_parts.append("\nDescribe this scene for a blind person, integrating both the objects and text content. Explain what the text says and how it relates to the objects in the scene.")
    
    return '\n'.join(prompt_parts)


def create_depth_fusion_prompt(objects_text: str, depth_info: Dict, spatial_info: Dict = None) -> str:
    """
    Create fusion prompt combining object detections with depth information
    
    Args:
        objects_text: Formatted text describing detected objects (from YOLO)
        depth_info: Dict with depth map statistics (mean_depth, min_depth, max_depth)
        spatial_info: Dict with spatial relationships (from analyze_spatial_relationships)
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add object detections
    prompt_parts.append("Objects detected in the image:")
    prompt_parts.append(objects_text)
    
    # Add depth information
    if depth_info.get('mean_depth') is not None:
        prompt_parts.append("\nDepth information:")
        prompt_parts.append(f"  Mean depth: {depth_info['mean_depth']:.2f}")
        prompt_parts.append(f"  Depth range: {depth_info['min_depth']:.2f} - {depth_info['max_depth']:.2f}")
    
    # Add spatial relationships if available
    if spatial_info and spatial_info.get('spatial_info'):
        prompt_parts.append("\nSpatial relationships (from depth analysis):")
        for obj_name, info in spatial_info['spatial_info'].items():
            depth_val = info.get('depth', 0)
            prompt_parts.append(f"  {obj_name}: depth value {depth_val:.2f}")
    
    prompt_parts.append("\nDescribe this scene for a blind person, including actual distances and spatial relationships. Use the depth information to provide distance estimates (e.g., '2 meters ahead', '5 meters to the left'). Prioritize safety-critical spatial information for navigation.")
    
    return '\n'.join(prompt_parts)


def create_user_prompt(objects_text: str) -> str:
    """
    Create standard user prompt (fallback, same as Approach 2)
    
    Args:
        objects_text: Formatted text describing detected objects
    
    Returns:
        Formatted prompt string
    """
    return f"""Analyze the image based on these detected objects:
{objects_text}

Describe the scene for a blind person. Include:
- Key objects and their positions (e.g., "person at center", "door on right")
- Any safety hazards or obstacles
- Actionable advice or important context

Keep it concise and actionable."""

