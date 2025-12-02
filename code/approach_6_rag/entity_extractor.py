"""
Entity Extractor for Approach 6
Extracts game entities and identifies games from descriptions
"""
import os
import json
import time
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()


def extract_entities_llm(description: str, model: str = 'gpt-4o-mini') -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extract game entities using LLM
    
    Args:
        description: Base description from VLM
        model: LLM model to use ('gpt-4o-mini' or 'claude-haiku')
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'game', 'entities'
    """
    from prompts import ENTITY_EXTRACTION_PROMPT
    
    try:
        if model == 'gpt-4o-mini' or 'gpt' in model.lower():
            from openai import OpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None, "OPENAI_API_KEY not found"
            
            client = OpenAI(api_key=api_key)
            
            prompt = ENTITY_EXTRACTION_PROMPT.format(description=description)
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from text. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            latency = time.time() - start_time
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                result = json.loads(content)
                return {
                    'game': result.get('game'),
                    'entities': result.get('entities', []),
                    'latency': latency
                }, None
            except json.JSONDecodeError:
                # Fallback: try to extract game name manually
                game = None
                if 'slay the spire' in description.lower() or 'slaythespire' in description.lower():
                    game = 'slay_the_spire'
                elif 'stardew' in description.lower() or 'stardewvalley' in description.lower():
                    game = 'stardew_valley'
                elif 'tic tac toe' in description.lower() or 'tic_tac_toe' in description.lower():
                    game = 'tic_tac_toe'
                elif 'four in a row' in description.lower() or 'four_in_a_row' in description.lower():
                    game = 'four_in_a_row'
                
                return {
                    'game': game,
                    'entities': [],
                    'latency': latency
                }, None
        
        elif 'claude' in model.lower():
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return None, "ANTHROPIC_API_KEY not found"
            
            client = anthropic.Anthropic(api_key=api_key)
            
            prompt = ENTITY_EXTRACTION_PROMPT.format(description=description)
            
            start_time = time.time()
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                system="You are a helpful assistant that extracts structured information from text. Always respond with valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            latency = time.time() - start_time
            
            content = message.content[0].text.strip()
            
            # Try to parse JSON
            try:
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                result = json.loads(content)
                return {
                    'game': result.get('game'),
                    'entities': result.get('entities', []),
                    'latency': latency
                }, None
            except json.JSONDecodeError:
                # Fallback extraction
                game = None
                if 'slay the spire' in description.lower():
                    game = 'slay_the_spire'
                elif 'stardew' in description.lower():
                    game = 'stardew_valley'
                elif 'tic tac toe' in description.lower():
                    game = 'tic_tac_toe'
                elif 'four in a row' in description.lower():
                    game = 'four_in_a_row'
                
                return {
                    'game': game,
                    'entities': [],
                    'latency': latency
                }, None
        
        else:
            return None, f"Unknown model: {model}"
    
    except Exception as e:
        return None, str(e)


def identify_game_simple(description: str, image_path: str = None) -> Optional[str]:
    """
    Simple game identification from filename or description
    
    Args:
        description: Description text
        image_path: Optional image path (filename may contain game name)
    
    Returns:
        Game name or None
    """
    # Check filename first
    if image_path:
        path_lower = str(image_path).lower()
        if 'slaythespire' in path_lower or 'slay_the_spire' in path_lower:
            return 'slay_the_spire'
        elif 'stardewvalley' in path_lower or 'stardew_valley' in path_lower:
            return 'stardew_valley'
        elif 'tic_tac_toe' in path_lower or 'tictactoe' in path_lower:
            return 'tic_tac_toe'
        elif 'four_in_a_row' in path_lower or 'fourinarow' in path_lower:
            return 'four_in_a_row'
    
    # Check description
    desc_lower = description.lower()
    if 'slay the spire' in desc_lower or 'slaythespire' in desc_lower:
        return 'slay_the_spire'
    elif 'stardew' in desc_lower:
        return 'stardew_valley'
    elif 'tic tac toe' in desc_lower or 'tictactoe' in desc_lower:
        return 'tic_tac_toe'
    elif 'four in a row' in desc_lower:
        return 'four_in_a_row'
    
    return None


def extract_entities(description: str, image_path: str = None, use_llm: bool = True, model: str = 'gpt-4o-mini') -> Dict:
    """
    Extract entities and identify game (with fallback to simple method)
    
    Args:
        description: Base description from VLM
        image_path: Optional image path for filename-based identification
        use_llm: Whether to use LLM for extraction (default True)
        model: LLM model to use if use_llm is True
    
    Returns:
        Dict with 'game', 'entities', 'latency', 'method'
    """
    result = {
        'game': None,
        'entities': [],
        'latency': 0.0,
        'method': 'simple'
    }
    
    # Try simple identification first (fast, free)
    game = identify_game_simple(description, image_path)
    if game:
        result['game'] = game
        result['method'] = 'filename'
    
    # Use LLM if requested and game not found
    if use_llm and not result['game']:
        llm_result, error = extract_entities_llm(description, model)
        if llm_result and not error:
            result['game'] = llm_result.get('game')
            result['entities'] = llm_result.get('entities', [])
            result['latency'] = llm_result.get('latency', 0.0)
            result['method'] = 'llm'
        elif error:
            print(f"  ⚠️  LLM entity extraction failed: {error}, using simple method")
    
    return result


if __name__ == "__main__":
    # Test entity extraction
    test_descriptions = [
        "A screenshot from Slay the Spire showing a combat scene with the Defect character facing a Sentry enemy. Health bar shows 45/70.",
        "Stardew Valley farm scene with crops and a barn visible. Character is in the center with tools equipped.",
        "Tic tac toe board with X and O marks. It's player X's turn."
    ]
    
    for desc in test_descriptions:
        print(f"\nDescription: {desc[:80]}...")
        result = extract_entities(desc, use_llm=False)  # Test simple method first
        print(f"Game: {result['game']}")
        print(f"Method: {result['method']}")

