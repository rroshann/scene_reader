"""
Cache Manager for Approach 2.5
LRU cache for storing generated descriptions to avoid redundant API calls
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict


class CacheManager:
    """
    LRU cache manager for Approach 2.5 pipeline results
    
    Cache Key: Hash of (YOLO model, detected objects list, prompt template)
    Cache Value: Generated description, latency, tokens, timestamp
    """
    
    def __init__(self, max_size: int = 1000, persist_path: Optional[Path] = None):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of cache entries (LRU eviction)
            persist_path: Optional path to persist cache to disk
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'loads': 0,
            'saves': 0
        }
        
        # Load from disk if exists
        if persist_path and persist_path.exists():
            self.load_from_disk()
    
    def get_cache_key(self, yolo_model: str, objects: List[Dict], prompt_template: str = 'default') -> str:
        """
        Generate cache key from inputs
        
        Args:
            yolo_model: YOLO model identifier (e.g., 'yolov8n')
            objects: List of detected objects (dicts with class, bbox, confidence)
            prompt_template: Prompt template identifier
        
        Returns:
            Cache key (hash string)
        """
        # Create a stable representation of objects
        # Sort objects by class name for consistent hashing
        sorted_objects = sorted(objects, key=lambda x: x.get('class', ''))
        objects_str = json.dumps([
            {
                'class': obj.get('class', ''),
                'bbox': [round(coord, 1) for coord in obj.get('bbox', [])],  # Round bbox coords
                'confidence': round(obj.get('confidence', 0.0), 2)  # Round to avoid float precision issues
            }
            for obj in sorted_objects
        ], sort_keys=True)
        
        # Create hash
        key_string = f"{yolo_model}:{prompt_template}:{objects_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """
        Get cached result if exists
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached result dict or None if not found
        """
        if cache_key in self.cache:
            # Move to end (most recently used)
            result = self.cache.pop(cache_key)
            self.cache[cache_key] = result
            self.stats['hits'] += 1
            return result.copy()  # Return copy to prevent modification
        
        self.stats['misses'] += 1
        return None
    
    def store_result(self, cache_key: str, result: Dict) -> None:
        """
        Store result in cache
        
        Args:
            cache_key: Cache key
            result: Result dict with description, latency, tokens, etc.
        """
        # Add timestamp
        result['cached_at'] = time.time()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first item)
            self.stats['evictions'] += 1
        
        # Store new result (most recently used)
        self.cache[cache_key] = result.copy()
        
        # Persist to disk if configured
        if self.persist_path:
            self.save_to_disk()
    
    def clear_cache(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dict with hit rate, size, etc.
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'loads': self.stats['loads'],
            'saves': self.stats['saves']
        }
    
    def save_to_disk(self) -> None:
        """Save cache to disk"""
        if not self.persist_path:
            return
        
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'cache': dict(self.cache),
                    'stats': self.stats
                }, f, indent=2)
            self.stats['saves'] += 1
        except Exception as e:
            print(f"Warning: Failed to save cache to disk: {e}")
    
    def load_from_disk(self) -> None:
        """Load cache from disk"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cache = OrderedDict(data.get('cache', {}))
                # Restore stats if available
                if 'stats' in data:
                    self.stats.update(data['stats'])
            self.stats['loads'] += 1
        except Exception as e:
            print(f"Warning: Failed to load cache from disk: {e}")


# Global cache instance (singleton pattern)
_global_cache: Optional[CacheManager] = None


def get_cache_manager(max_size: int = 1000, persist_path: Optional[Path] = None) -> CacheManager:
    """
    Get or create global cache manager instance
    
    Args:
        max_size: Maximum cache size
        persist_path: Optional persistence path
    
    Returns:
        CacheManager instance
    """
    global _global_cache
    
    if _global_cache is None:
        if persist_path is None:
            # Default persistence path
            project_root = Path(__file__).parent.parent.parent
            persist_path = project_root / 'results' / 'approach_2_5_optimized' / 'cache.json'
        
        _global_cache = CacheManager(max_size=max_size, persist_path=persist_path)
    
    return _global_cache

