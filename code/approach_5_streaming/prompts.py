"""
Prompts for Approach 5: Streaming/Progressive Models
Optimized prompts for tier1 (fast) and tier2 (detailed) models
"""

# Tier 1 Prompt: Optimized for BLIP-2 - brief overview
# Keep it short and focused on quick scene understanding
TIER1_PROMPT = "Briefly describe what you see in this image. Focus on the main scene, key objects, and immediate context. Keep it concise (1-2 sentences)."

# Tier 2 Prompt: Standard accessibility prompt for detailed description
# Same as Approach 1 for consistency
TIER2_SYSTEM_PROMPT = """You are a visual accessibility assistant for blind and low-vision users. When describing images, provide concise, prioritized, actionable information. Always include: (1) Spatial layout - where things are relative to the viewer (left/right/center, approximate distances), (2) Critical status - important states, conditions, or information, (3) Immediate concerns - threats, obstacles, or urgent details. Prioritize what the user needs to know to act or make decisions. Be brief, informative, and context-aware."""

TIER2_USER_PROMPT = ""  # Empty - system prompt handles everything

