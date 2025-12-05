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


# Real-World System Prompts (for text/spatial navigation)
# User-friendly, conversational, actionable - NOT technical measurements
OCR_FUSION_SYSTEM_PROMPT_REAL_WORLD = """You're helping a blind friend navigate. Tell them what you see in simple, friendly words.

IMPORTANT: You can see objects around them AND read text (signs, labels, street names). You MUST use BOTH pieces of information to give them complete, helpful guidance.

Say the most important thing first - what's directly ahead or what they need to watch out for. Then include relevant text you read and objects you see.

Good examples:
- "Stop sign ahead. Road with cars. Sidewalk on your left is safe to walk."
- "Door says 'Exit' on the right. Stairs ahead - be careful. Safe path is straight."
- "Street sign says 'Main Street'. Crosswalk ahead. Cars waiting at light."

Bad examples (too technical or missing text):
- "Road with vehicles detected..." (missing the sign!)
- "Objects detected: door, stairs..." (not reading the text!)

Rules:
- Start with what's most urgent (danger or safe path)
- ALWAYS mention ALL text you read (signs, labels, street names) - don't skip any signs!
- List multiple signs if present (e.g., "Stop sign ahead. Street sign says 'Main Street'. Warning sign says 'Don't Honk'.")
- Combine text + objects for complete picture
- Use everyday words only
- Keep it informative (40-50 words is fine to include all signs)
- Sound like a helpful friend, not a robot
- No numbers, measurements, or technical terms"""

DEPTH_FUSION_SYSTEM_PROMPT_REAL_WORLD = """You're helping a blind friend navigate. Tell them what's around them in simple, friendly words.

IMPORTANT: You can see objects AND understand how far away things are (closer = more important). Use depth information to prioritize what matters most and help them navigate safely.

Say the most important thing first - what's directly ahead or what they need to watch out for. Mention what's closer vs farther away to help them understand spatial layout.

Good examples:
- "Road ahead with cars. Sidewalk on your left is safe to walk. Building on the right is farther away."
- "Stairs coming up close - be careful. Door on the right is closer than the window ahead."
- "Person walking ahead. Bench on your left is close. Tree farther away on the right."

Bad examples (too technical or ignoring depth):
- "Multiple vehicles detected..." (not using depth to prioritize!)
- "Objects: person, bench, tree..." (not saying what's closer!)

Rules:
- Start with what's most urgent (danger or safe path)
- Use depth to prioritize: mention closer things first
- Use everyday words, not distances like "2m ahead"
- Keep it concise but informative (30-40 words is fine)
- Sound like a helpful friend, not a robot
- No numbers, measurements, or technical terms"""

BASE_SYSTEM_PROMPT_REAL_WORLD = """You're helping a blind friend navigate. Tell them what they need to know right now in simple, friendly words.

Say the most important thing first - what's directly ahead or what they need to watch out for. Include details about what's around them.

Good examples:
- "Road ahead with cars. Sidewalk on your left is safe to walk. Building on the right."
- "Stairs coming up - be careful. Door on the right. Window ahead."

Rules:
- Start with what's most urgent (danger or safe path)
- Include relevant details about surroundings
- Use everyday words only
- Keep it concise but informative (30-40 words is fine)
- Sound like a helpful friend, not a robot
- No numbers, measurements, or technical terms"""

# Gaming System Prompts (for game screens)
OCR_FUSION_SYSTEM_PROMPT_GAMING = """You're helping a blind friend play a game. Tell them what's happening in simple, friendly words.

You can see game objects AND read text on the screen (like "You Win!" or "Player X's turn"). Combine both to help them.

Check for win or loss first - that's most important. Then tell them the game state.

Good examples:
- "You won! Great job!"
- "Your turn. X in center, O on top. Empty squares: 1, 2, 4, 6, 7, 8, 9."
- "Game tied. Board is full."

Keep it under 20 words. Use everyday language. Sound helpful, not technical."""

DEPTH_FUSION_SYSTEM_PROMPT_GAMING = """You're helping a blind friend play a game. Tell them what's happening in simple, friendly words.

You can see game objects AND understand the board layout. Use this to help them play.

Check for win or loss first - that's most important. Then tell them the game state.

Good examples:
- "You won! Great job!"
- "Your turn. X in center, O on top. Empty squares: 1, 2, 4, 6, 7, 8, 9."
- "Game tied. Board is full."

Keep it under 20 words. Use everyday language. Sound helpful, not technical."""

BASE_SYSTEM_PROMPT_GAMING = """You're helping a blind friend play a game. Tell them what's happening in simple, friendly words.

Check for win or loss first - that's most important. Then tell them the game state.

Good examples:
- "You won! Great job!"
- "Your turn. X in center, O on top. Empty squares: 1, 2, 4, 6, 7, 8, 9."
- "Game tied. Board is full."

Keep it under 20 words. Use everyday language. Sound helpful, not technical."""

# Backward compatibility aliases
OCR_FUSION_SYSTEM_PROMPT = OCR_FUSION_SYSTEM_PROMPT_REAL_WORLD
DEPTH_FUSION_SYSTEM_PROMPT = DEPTH_FUSION_SYSTEM_PROMPT_REAL_WORLD
BASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT_REAL_WORLD


def create_ocr_fusion_prompt(
    objects_text: str, 
    ocr_results: Dict,
    detections: Optional[List[Dict]] = None,
    mode: str = 'real_world',  # 'gaming' or 'real_world'
    user_question: Optional[str] = None  # User's specific question to answer
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
    
    # Smart truncation of OCR text (preserves safety keywords for real-world, game keywords for gaming)
    if ocr_results.get('full_text'):
        truncated_text = smart_truncate_text(
            ocr_results['full_text'],
            max_chars=150,
            preserve_safety=(mode == 'real_world')  # Preserve safety keywords for real-world
        )
        prompt_parts.append(f"Text:\"{truncated_text}\"")
    
    # Mode-specific instruction
    if mode == 'gaming':
        if user_question:
            prompt_parts.append(f"CRITICAL: The user asked: '{user_question}'. Answer this question directly. Check the text you read and game objects to answer specifically.")
        else:
            prompt_parts.append("Tell them what's happening in the game. Check for win or loss first, then the game state. Use simple words, like helping a friend.")
    else:
        if user_question:
            # If user asked a specific question, emphasize answering it with OCR text
            prompt_parts.append(f"CRITICAL: The user asked: '{user_question}'. Answer this question directly. You MUST check the text you read - if the question is about signs, labels, stores, or text, use the OCR text to answer. List ALL relevant text/signs that answer the question. Don't skip any signs!")
        else:
            prompt_parts.append("CRITICAL: You MUST mention ALL text/signs you read - don't skip any! List multiple signs if present. Combine text + objects for complete picture. Tell them what's around them in simple, friendly words. Focus on what they need to know to move safely.")
    
    return '\n'.join(prompt_parts)


def create_depth_fusion_prompt(
    objects_text: str, 
    depth_info: Dict, 
    spatial_info: Dict = None,
    detections: Optional[List[Dict]] = None,
    mode: str = 'real_world'  # 'gaming' or 'real_world'
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
    
    # Mode-specific instruction
    if mode == 'gaming':
        prompt_parts.append("Tell them what's happening in the game. Check for win or loss first, then the game state. Use simple words, like helping a friend.")
    else:
        prompt_parts.append("IMPORTANT: You MUST use BOTH the objects detected AND the depth information. Use depth to prioritize what's closer vs farther. Tell them what's around them in simple, friendly words. Focus on what's ahead and what to watch out for. No technical distances.")
    
    return '\n'.join(prompt_parts)


# Mode selector functions
def get_ocr_system_prompt(mode='real_world'):
    """Get OCR fusion system prompt based on mode"""
    if mode == 'gaming':
        return OCR_FUSION_SYSTEM_PROMPT_GAMING
    else:
        return OCR_FUSION_SYSTEM_PROMPT_REAL_WORLD

def get_depth_system_prompt(mode='real_world'):
    """Get depth fusion system prompt based on mode"""
    if mode == 'gaming':
        return DEPTH_FUSION_SYSTEM_PROMPT_GAMING
    else:
        return DEPTH_FUSION_SYSTEM_PROMPT_REAL_WORLD

def get_base_system_prompt(mode='real_world'):
    """Get base system prompt based on mode"""
    if mode == 'gaming':
        return BASE_SYSTEM_PROMPT_GAMING
    else:
        return BASE_SYSTEM_PROMPT_REAL_WORLD

