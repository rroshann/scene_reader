"""
Standardized Prompts for Fair Comparison Testing
Neutral, consistent prompts used across all approaches for standardized comparison
"""

# Standardized System Prompt (neutral, same for all approaches)
STANDARDIZED_SYSTEM_PROMPT = """You are a visual accessibility assistant helping a blind person understand their surroundings. 
Describe what you see in the image clearly and concisely. Focus on:
- Main objects and their locations
- Spatial relationships (left, right, center, ahead)
- Important details that affect navigation or safety
- Text content if present

Keep descriptions clear, factual, and helpful. Use everyday language."""

# Standardized User Prompt (simple, consistent format)
STANDARDIZED_USER_PROMPT = "Describe this image for a blind person. Focus on what is visible and where things are located."

# For Approach 2.5 (YOLO+LLM) - format objects into standardized prompt
def create_standardized_user_prompt_with_objects(objects_text: str) -> str:
    """
    Create standardized user prompt with detected objects
    
    Args:
        objects_text: Formatted text of detected objects from YOLO
    
    Returns:
        Standardized user prompt with objects
    """
    return f"{STANDARDIZED_USER_PROMPT}\n\nDetected objects:\n{objects_text}"

# For Approach 3.5 (Specialized) - format with OCR/depth data
# These functions match the signatures expected by Approach 3.5's fusion prompt functions
def create_standardized_ocr_fusion_prompt(
    objects_text: str,
    ocr_results: dict,
    detections: list = None,
    mode: str = 'real_world'
) -> str:
    """
    Create standardized OCR fusion prompt matching Approach 3.5 signature
    
    Args:
        objects_text: Formatted text of detected objects
        ocr_results: Dict with OCR results (texts, full_text, etc.)
        detections: Optional list of detection dicts (ignored for standardized)
        mode: Mode (ignored for standardized)
    
    Returns:
        Standardized user prompt with OCR and objects
    """
    ocr_text = ocr_results.get('full_text', '')
    if not ocr_text and ocr_results.get('texts'):
        ocr_text = ' '.join([t.get('text', '') for t in ocr_results['texts']])
    
    return f"{STANDARDIZED_USER_PROMPT}\n\nDetected objects:\n{objects_text}\n\nText found:\n{ocr_text}"

def create_standardized_depth_fusion_prompt(
    objects_text: str,
    depth_info: dict,
    spatial_info: dict = None,
    detections: list = None,
    mode: str = 'real_world'
) -> str:
    """
    Create standardized depth fusion prompt matching Approach 3.5 signature
    
    Args:
        objects_text: Formatted text of detected objects
        depth_info: Dict with depth estimation results
        spatial_info: Optional spatial information (ignored for standardized)
        detections: Optional list of detection dicts (ignored for standardized)
        mode: Mode (ignored for standardized)
    
    Returns:
        Standardized user prompt with depth and objects
    """
    depth_summary = f"Mean depth: {depth_info.get('mean_depth', 'N/A')}"
    if depth_info.get('min_depth') is not None:
        depth_summary += f", Range: {depth_info.get('min_depth', 0):.0f}-{depth_info.get('max_depth', 0):.0f}"
    
    return f"{STANDARDIZED_USER_PROMPT}\n\nDetected objects:\n{objects_text}\n\nDepth information:\n{depth_summary}"

