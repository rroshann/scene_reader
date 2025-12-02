"""
Optimized Prompts for Approach 2.5
Reduced token count while maintaining quality
"""
import sys
from pathlib import Path

# Import base prompts from Approach 2
project_root = Path(__file__).parent.parent.parent
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

from prompts import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT, create_user_prompt as BASE_CREATE_USER_PROMPT

# Optimized System Prompt (reduced verbosity, ~40% fewer tokens)
SYSTEM_PROMPT_MINIMAL = """Visual accessibility assistant for blind users. Provide concise, actionable descriptions. Include: (1) Spatial layout - object positions relative to viewer, (2) Critical status - important states/conditions, (3) Immediate concerns - threats/obstacles. Prioritize actionable information. Be brief and context-aware."""

# Structured System Prompt (bullet format, ~30% fewer tokens)
SYSTEM_PROMPT_STRUCTURED = """Visual accessibility assistant. Describe scenes concisely:
• Spatial layout: object positions (left/right/center, distances)
• Critical status: important states/conditions
• Immediate concerns: threats/obstacles
Prioritize actionable information. Be brief."""

# Template-based User Prompt (reduced verbosity)
def create_user_prompt_minimal(detected_objects_text):
    """Minimal user prompt - reduced verbosity"""
    return f"""Describe this scene for a blind person based on detected objects:

{detected_objects_text}

Focus on: spatial relationships, safety-critical elements, actionable context. Be concise."""

# Structured User Prompt (bullet format)
def create_user_prompt_structured(detected_objects_text):
    """Structured user prompt - bullet format"""
    return f"""Describe scene for blind person:

Detected objects:
{detected_objects_text}

Include:
• Spatial relationships
• Safety-critical elements (obstacles, stairs, doors)
• Actionable context
Be concise."""

# Template-based User Prompt (pre-formatted)
def create_user_prompt_template(detected_objects_text):
    """Template-based prompt - minimal structure"""
    return f"""Objects: {detected_objects_text}

Describe scene (spatial layout, safety, actions). Brief."""

# Default: Use minimal prompt (best balance)
SYSTEM_PROMPT = SYSTEM_PROMPT_MINIMAL
create_user_prompt = create_user_prompt_minimal

# Token count estimates (approximate):
# BASE_SYSTEM_PROMPT: ~80 tokens
# SYSTEM_PROMPT_MINIMAL: ~48 tokens (40% reduction)
# SYSTEM_PROMPT_STRUCTURED: ~56 tokens (30% reduction)
#
# BASE_CREATE_USER_PROMPT: ~150-300 tokens (depends on objects)
# create_user_prompt_minimal: ~90-180 tokens (40% reduction)
# create_user_prompt_structured: ~100-200 tokens (33% reduction)
# create_user_prompt_template: ~70-150 tokens (50% reduction)

