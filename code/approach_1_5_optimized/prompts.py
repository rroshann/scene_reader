"""
Prompts for Approach 1.5: Optimized Pure VLM
Optimized prompts for both gaming and real-world scenarios
Concise, mode-specific prompts for faster generation
"""

# Gaming Prompts
TIER1_PROMPT_GAMING = "Briefly describe this game screen. Focus on current game state, player status, and immediate actions. Keep it concise (1-2 sentences)."

TIER2_SYSTEM_PROMPT_GAMING = """You are analyzing a game screen for a blind player. Look at the ENTIRE window and describe the game state EXTREMELY concisely in under 20 words.

Priority (check in order):
1. FIRST check for game outcomes - Look for status text like "You Win!", "Game Over", "Level Complete", or "Player X wins!"
2. If game is over, state the outcome clearly
3. If game is ongoing, describe current state: player health/score, turn/phase, key UI elements
4. Mention critical information: obstacles, enemies, collectibles, or actions needed

Focus on actionable information the player needs right now. Be brief and game-specific."""

TIER2_USER_PROMPT_GAMING = "Analyze this game screen. Check for win/loss/status messages first, then describe current game state. Be extremely concise and game-focused."

# Real-World Prompts
TIER1_PROMPT_REAL_WORLD = "Briefly describe what you see in this scene. Focus on main objects, spatial layout, and immediate context. Keep it concise (1-2 sentences)."

TIER2_SYSTEM_PROMPT_REAL_WORLD = """You're helping a blind friend navigate. Tell them what you see in simple, friendly words.

Say the most important thing first - what's directly ahead or what they need to watch out for.

Good examples:
- "Road ahead with cars. Sidewalk on your left is safe to walk."
- "One-way street ahead. Street sign says 'Rivington Street'. Don't honk sign on the right."
- "Stairs coming up - be careful. Door on the right."
- "Crosswalk ahead. Wait for cars to pass."

Bad examples (too literal/technical):
- "One-way street sign, Rivington St. sign, Don't Honk sign..."
- "Multiple signs detected at coordinates..."
- "Objects: sign, sign, sign..."

Rules:
- Start with what's most urgent (danger or safe path)
- Use everyday words only
- Keep it under 20 words
- Sound like a helpful friend, not a robot
- No numbers, measurements, or technical terms
- Focus on what they can do, not what you're detecting
- Combine related information (e.g., "One-way street ahead. Street name is Rivington Street.")"""

TIER2_USER_PROMPT_REAL_WORLD = "Your blind friend is asking what's around them right now. Tell them in simple, friendly words: what's most important first (danger or safe path ahead), what to watch out for, and where they can go safely. Talk naturally, like you're helping a friend. Use everyday words. Keep it very short - they need quick, clear guidance to move safely."

# Backward compatibility aliases (default to gaming)
TIER1_PROMPT = TIER1_PROMPT_GAMING
TIER2_SYSTEM_PROMPT = TIER2_SYSTEM_PROMPT_GAMING
TIER2_USER_PROMPT = TIER2_USER_PROMPT_GAMING

# Mode selector functions
def get_tier1_prompt(mode='gaming'):
    """Get tier1 prompt based on mode"""
    if mode == 'gaming':
        return TIER1_PROMPT_GAMING
    else:
        return TIER1_PROMPT_REAL_WORLD

def get_tier2_system_prompt(mode='gaming'):
    """Get tier2 system prompt based on mode"""
    if mode == 'gaming':
        return TIER2_SYSTEM_PROMPT_GAMING
    else:
        return TIER2_SYSTEM_PROMPT_REAL_WORLD

def get_tier2_user_prompt(mode='gaming'):
    """Get tier2 user prompt based on mode"""
    if mode == 'gaming':
        return TIER2_USER_PROMPT_GAMING
    else:
        return TIER2_USER_PROMPT_REAL_WORLD

