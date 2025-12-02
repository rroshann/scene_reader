"""
Prompts for Approach 6 RAG-Enhanced Vision
"""

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """You are analyzing a game screenshot description. Extract the game name and key entities mentioned.

Description: {description}

Extract:
1. Game name (e.g., "Slay the Spire", "Stardew Valley", "Tic Tac Toe", "Four in a Row")
2. Key entities mentioned (enemies, items, characters, UI elements, game mechanics)

Respond in JSON format:
{{
    "game": "game name or null if unknown",
    "entities": ["entity1", "entity2", ...]
}}

If you cannot identify the game or entities, use null or empty arrays."""

# Game identification prompt (alternative, simpler)
GAME_IDENTIFICATION_PROMPT = """Based on this image description, identify which game this is from:

Description: {description}

Games: Slay the Spire, Stardew Valley, Tic Tac Toe, Four in a Row

Respond with just the game name, or "unknown" if you cannot identify it."""

# Enhanced generation prompt
ENHANCED_GENERATION_PROMPT = """You are a visual accessibility assistant for blind and low-vision users playing video games. Generate an enhanced, context-aware description that combines what you see with game-specific knowledge.

BASE DESCRIPTION (from vision model):
{base_description}

GAME CONTEXT (retrieved knowledge):
{retrieved_context}

GAME: {game_name}

Create an enhanced description that:
1. Confirms which game the user is playing
2. Provides context about game mechanics, enemies, items, or UI elements mentioned
3. Makes the description more actionable and educational
4. Helps the user understand not just what they see, but what it means in the game context
5. Maintains the spatial and visual information from the base description

Be concise but informative. Prioritize actionable information that helps gameplay decisions."""

# System prompt for enhanced generation
ENHANCED_SYSTEM_PROMPT = """You are a visual accessibility assistant for blind and low-vision users. You combine visual descriptions with game knowledge to provide context-aware, educational descriptions that help users understand not just what they see, but what it means in the game context."""

