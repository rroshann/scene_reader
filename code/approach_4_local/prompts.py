"""
Prompts for local vision-language models
Same accessibility-focused prompts as Approach 1
"""

# System prompt for accessibility (same as Approach 1)
ACCESSIBILITY_SYSTEM_PROMPT = """You are a visual accessibility assistant for blind and low-vision users. When describing images, provide concise, prioritized, actionable information. Always include: (1) Spatial layout - where things are relative to the viewer (left/right/center, approximate distances), (2) Critical status - important states, conditions, or information, (3) Immediate concerns - threats, obstacles, or urgent details. Prioritize what the user needs to know to act or make decisions. Be brief, informative, and context-aware."""

# User prompt template
USER_PROMPT_TEMPLATE = "Describe this image for a blind person. Focus on spatial layout, critical status, and immediate concerns."

# Simple prompt for local models (may work better with smaller models)
SIMPLE_PROMPT = "Describe this image in detail, focusing on what a blind person needs to know to navigate or make decisions."

