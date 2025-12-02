"""
Knowledge Processor for Approach 6
Processes game knowledge files and chunks them for vector storage
"""
import re
from pathlib import Path
from typing import List, Dict, Tuple


def load_game_wiki(game_name: str, knowledge_base_dir: Path = None) -> str:
    """
    Load game wiki/knowledge content from file
    
    Args:
        game_name: Name of the game (e.g., 'slay_the_spire', 'stardew_valley')
        knowledge_base_dir: Base directory for knowledge base (defaults to data/knowledge_base/games)
    
    Returns:
        Full text content of the game knowledge file
    """
    if knowledge_base_dir is None:
        # Go from code/approach_6_rag/ to project root, then to data/knowledge_base/games
        knowledge_base_dir = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "games"
    
    game_dir = knowledge_base_dir / game_name
    info_file = game_dir / "game_info.txt"
    
    if not info_file.exists():
        raise FileNotFoundError(f"Game knowledge file not found: {info_file}")
    
    with open(info_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content


def chunk_text(content: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Split text into semantic chunks for vector storage
    
    Args:
        content: Full text content to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dictionaries with 'text' and 'chunk_id' keys
    """
    # Split by sections (lines starting with uppercase or numbers)
    sections = re.split(r'\n(?=[A-Z0-9])', content)
    
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # If adding this section would exceed chunk size, save current chunk
        if current_chunk and len(current_chunk) + len(section) > chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                # Take last 'overlap' characters as overlap
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + "\n" + section
            else:
                current_chunk = section
        else:
            if current_chunk:
                current_chunk += "\n\n" + section
            else:
                current_chunk = section
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'text': current_chunk.strip(),
            'chunk_id': chunk_id
        })
    
    return chunks


def extract_game_entities(content: str) -> List[str]:
    """
    Extract key entities (enemies, items, abilities) from game content
    
    Args:
        content: Game knowledge content
    
    Returns:
        List of entity names found in the content
    """
    entities = []
    
    # Look for common patterns: "Entity Name:", "Entity Name -", capitalized words after colons
    patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):',  # "Entity Name:"
        r'-\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "- Entity Name"
        r'([A-Z][A-Z\s]+):',  # "ACRONYM:" or "MULTI WORD:"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        entities.extend(matches)
    
    # Remove duplicates and clean
    entities = list(set(entities))
    entities = [e.strip() for e in entities if len(e.strip()) > 2]
    
    return entities


def process_game_knowledge(game_name: str, knowledge_base_dir: Path = None) -> List[Dict[str, any]]:
    """
    Complete processing pipeline: load, chunk, and extract entities
    
    Args:
        game_name: Name of the game
        knowledge_base_dir: Base directory for knowledge base
    
    Returns:
        List of processed chunks with metadata
    """
    # Load content
    content = load_game_wiki(game_name, knowledge_base_dir)
    
    # Chunk content
    chunks = chunk_text(content, chunk_size=500, overlap=50)
    
    # Extract entities
    entities = extract_game_entities(content)
    
    # Add metadata to chunks
    processed_chunks = []
    for chunk in chunks:
        processed_chunks.append({
            'text': chunk['text'],
            'chunk_id': chunk['chunk_id'],
            'game': game_name,
            'entities': entities[:5]  # Include top entities for context
        })
    
    return processed_chunks


if __name__ == "__main__":
    # Test the processor
    games = ['slay_the_spire', 'stardew_valley', 'tic_tac_toe']
    
    for game in games:
        print(f"\n{'='*60}")
        print(f"Processing: {game}")
        print(f"{'='*60}")
        
        try:
            chunks = process_game_knowledge(game)
            print(f"Generated {len(chunks)} chunks")
            print(f"First chunk preview:")
            print(chunks[0]['text'][:200] + "...")
        except Exception as e:
            print(f"Error processing {game}: {e}")

