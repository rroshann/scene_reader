"""
Vector Store for Approach 6
Manages ChromaDB vector database for game knowledge retrieval
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


class VectorStore:
    """Manages vector database for game knowledge"""
    
    def __init__(self, db_path: Path = None, collection_name: str = "game_knowledge"):
        """
        Initialize vector store
        
        Args:
            db_path: Path to ChromaDB database (defaults to data/knowledge_base/vector_db)
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
        if db_path is None:
            # Go from code/approach_6_rag/ to project root, then to data/knowledge_base/vector_db
            db_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "vector_db"
        
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Game knowledge base for RAG"}
            )
            print(f"Created new collection: {collection_name}")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
    
    def add_game_knowledge(self, chunks: List[Dict], game_name: str) -> int:
        """
        Add game knowledge chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'chunk_id', 'game', etc.
            game_name: Name of the game
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Prepare documents, ids, and metadata
        documents = []
        ids = []
        metadatas = []
        
        for chunk in chunks:
            doc_id = f"{game_name}_chunk_{chunk['chunk_id']}"
            documents.append(chunk['text'])
            ids.append(doc_id)
            metadatas.append({
                'game': game_name,
                'chunk_id': chunk['chunk_id'],
                'entities': ','.join(chunk.get('entities', []))
            })
        
        # Add to collection
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks for {game_name}")
        return len(chunks)
    
    def search_knowledge(self, query: str, top_k: int = 3, game_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant knowledge chunks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            game_filter: Optional game name to filter results
        
        Returns:
            List of result dictionaries with 'text', 'score', 'metadata'
        """
        # Build where clause for filtering
        where = None
        if game_filter:
            where = {"game": game_filter}
        
        # Search
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'score': 1.0 - results['distances'][0][i] if 'distances' in results else None,  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    'id': results['ids'][0][i] if 'ids' in results else None
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }


def initialize_vector_store(db_path: Path = None) -> VectorStore:
    """
    Initialize and return a VectorStore instance
    
    Args:
        db_path: Optional path to database
    
    Returns:
        VectorStore instance
    """
    return VectorStore(db_path=db_path)


if __name__ == "__main__":
    # Test vector store
    from knowledge_processor import process_game_knowledge
    
    print("Initializing vector store...")
    store = initialize_vector_store()
    
    # Process and add game knowledge
    games = ['slay_the_spire', 'stardew_valley', 'tic_tac_toe']
    
    for game in games:
        print(f"\nProcessing {game}...")
        chunks = process_game_knowledge(game)
        store.add_game_knowledge(chunks, game)
    
    # Test search
    print("\n" + "="*60)
    print("Testing search...")
    print("="*60)
    
    test_queries = [
        "What enemies are in Slay the Spire?",
        "How does farming work in Stardew Valley?",
        "How do you win Tic Tac Toe?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = store.search_knowledge(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Game: {result['metadata'].get('game', 'unknown')}")
            print(f"     Text: {result['text'][:100]}...")
    
    # Stats
    stats = store.get_collection_stats()
    print(f"\n{'='*60}")
    print(f"Collection stats: {stats}")

