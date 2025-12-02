"""
Cache Manager for Approach 3.5
LRU cache for storing generated descriptions to avoid redundant API calls
Adapted from Approach 2.5 with support for OCR/depth data in cache keys
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from collections import OrderedDict


class CacheManager:
    """
    LRU cache manager for Approach 3.5 pipeline results
    
    Cache Key: Hash of (YOLO model, detected objects, OCR/depth data, prompt template, mode)
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
    
    def get_cache_key(
        self,
        yolo_model: str,
        objects: List[Dict],
        mode: str,
        ocr_results: Optional[Dict] = None,
        depth_results: Optional[Dict] = None,
        prompt_template: str = 'default'
    ) -> str:
        """
        Generate cache key from inputs (includes OCR/depth data for Approach 3.5)
        
        Args:
            yolo_model: YOLO model identifier (e.g., 'yolov8n')
            objects: List of detected objects (dicts with class, bbox, confidence)
            mode: Pipeline mode ('ocr' or 'depth')
            ocr_results: Optional OCR results dict (for OCR mode)
            depth_results: Optional depth results dict (for depth mode)
            prompt_template: Prompt template identifier
        
        Returns:
            Cache key (hash string)
        """
        # Create a stable representation of objects
        sorted_objects = sorted(objects, key=lambda x: x.get('class', ''))
        objects_str = json.dumps([
            {
                'class': obj.get('class', ''),
                'bbox': [round(coord, 1) for coord in obj.get('bbox', [])],
                'confidence': round(obj.get('confidence', 0.0), 2)
            }
            for obj in sorted_objects
        ], sort_keys=True)
        
        # Include OCR/depth data in cache key
        specialized_data = {}
        if mode == 'ocr' and ocr_results:
            # Hash OCR text for cache key
            ocr_text = ocr_results.get('full_text', '')
            specialized_data['ocr_text'] = ocr_text
            specialized_data['ocr_num_texts'] = ocr_results.get('num_texts', 0)
        elif mode == 'depth' and depth_results:
            # Enhanced depth cache key to prevent collisions
            specialized_data['mean_depth'] = round(depth_results.get('mean_depth', 0.0), 2)
            specialized_data['min_depth'] = round(depth_results.get('min_depth', 0.0), 2)
            specialized_data['max_depth'] = round(depth_results.get('max_depth', 0.0), 2)
            
            # Include depth map shape for uniqueness
            depth_map = depth_results.get('depth_map')
            if depth_map is not None:
                import numpy as np
                specialized_data['depth_shape'] = list(depth_map.shape) if hasattr(depth_map, 'shape') else None
                
                # Compute depth map hash (sample every 10th pixel for efficiency)
                # This prevents collisions: two scenes with same mean_depth but different distributions
                try:
                    # Sample depth map (every 10th pixel in both dimensions)
                    sampled = depth_map[::10, ::10]
                    # Flatten and convert to list for hashing
                    sampled_flat = sampled.flatten().tolist()
                    # Round to 2 decimal places to reduce noise
                    sampled_rounded = [round(x, 2) for x in sampled_flat[:100]]  # Limit to 100 samples
                    # Create hash of sampled values
                    sampled_str = ','.join(map(str, sampled_rounded))
                    depth_hash = hashlib.sha256(sampled_str.encode()).hexdigest()[:16]  # Use first 16 chars
                    specialized_data['depth_hash'] = depth_hash
                    
                    # Add depth distribution (histogram bins) for additional uniqueness
                    # Compute histogram with 10 bins
                    hist, _ = np.histogram(depth_map.flatten(), bins=10)
                    specialized_data['depth_hist'] = [int(x) for x in hist.tolist()]
                except Exception as e:
                    # Fallback if hash computation fails
                    specialized_data['depth_hash'] = None
                    specialized_data['depth_hist'] = None
        
        specialized_str = json.dumps(specialized_data, sort_keys=True)
        
        # Create hash
        key_string = f"{yolo_model}:{mode}:{prompt_template}:{objects_str}:{specialized_str}"
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
            return result.copy()
        
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
            self.cache.popitem(last=False)
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
            # Default persistence path for Approach 3.5
            project_root = Path(__file__).parent.parent.parent
            persist_path = project_root / 'results' / 'approach_3_5_optimized' / 'cache.json'
        
        _global_cache = CacheManager(max_size=max_size, persist_path=persist_path)
    
    return _global_cache

