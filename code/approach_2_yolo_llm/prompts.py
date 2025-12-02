"""
System prompts for Approach 2: YOLO + LLM Hybrid Pipeline
Defines prompts for LLM generation based on detected objects
"""

# System prompt for LLM generation (same accessibility focus as Approach 1)
SYSTEM_PROMPT = """You are a visual accessibility assistant for blind and low-vision users. When describing images, provide concise, prioritized, actionable information. Always include: (1) Spatial layout - where things are relative to the viewer (left/right/center, approximate distances), (2) Critical status - important states, conditions, or information, (3) Immediate concerns - threats, obstacles, or urgent details. Prioritize what the user needs to know to act or make decisions. Be brief, informative, and context-aware."""

# Template for user prompt with detected objects
def create_user_prompt(detected_objects_text):
    """
    Create user prompt with detected objects information
    
    Args:
        detected_objects_text: Formatted string describing detected objects and their positions
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Based on the following detected objects in the image, provide a concise, actionable description for a blind person:

{detected_objects_text}

Describe the scene focusing on:
- Spatial relationships between objects
- Safety-critical elements (obstacles, stairs, doors, crosswalks)
- Important context and layout
- What actions the user should take or be aware of

Be concise but comprehensive. Prioritize safety and actionability."""
    
    return prompt

# Alternative minimal prompt (if needed)
MINIMAL_USER_PROMPT = "Describe this scene for a blind person based on the detected objects."

