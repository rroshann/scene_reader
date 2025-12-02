"""
Depth Estimator for Approach 3B
Uses Depth-Anything for monocular depth estimation
"""
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")


def get_device():
    """Get best available device (MPS/CUDA/CPU)"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class DepthEstimator:
    """
    Depth estimator using Depth-Anything model
    """
    
    def __init__(self, model_name: str = 'depth-anything/Depth-Anything-V2-Small-hf'):
        """
        Initialize depth estimator
        
        Args:
            model_name: HuggingFace model name (default: Depth-Anything-V2-Small)
                       Options: 'depth-anything/Depth-Anything-V2-Small-hf' (fastest)
                                'depth-anything/Depth-Anything-V2-Base-hf' (more accurate)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = get_device()
        
        print(f"  Initializing depth estimator: {model_name}")
        print(f"  Device: {self.device}")
        print("  Note: First run will download model (~200-500MB)")
        
        try:
            # Initialize depth estimation pipeline
            self.pipe = pipeline(
                "depth-estimation",
                model=model_name,
                device=0 if self.device.type in ['cuda', 'mps'] else -1  # -1 for CPU
            )
            print(f"  ✅ Depth estimator initialized successfully")
        except Exception as e:
            print(f"  ⚠️  Depth estimator initialization error: {e}")
            raise
    
    def estimate_depth(self, image_path: Path) -> Dict:
        """
        Estimate depth map from image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dict with:
            - 'depth_map': Depth map array
            - 'mean_depth': Mean depth value
            - 'min_depth': Minimum depth value
            'max_depth': Maximum depth value
            - 'depth_latency': Processing time in seconds
            - 'depth_shape': Shape of depth map
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                'depth_map': None,
                'mean_depth': 0.0,
                'min_depth': 0.0,
                'max_depth': 0.0,
                'depth_latency': 0.0,
                'depth_shape': None,
                'error': 'Transformers not installed'
            }
        
        start_time = time.time()
        
        try:
            # Estimate depth
            depth_result = self.pipe(str(image_path))
            
            # Extract depth map
            if isinstance(depth_result, dict) and 'depth' in depth_result:
                depth_map = np.array(depth_result['depth'])
            elif isinstance(depth_result, dict) and 'predicted_depth' in depth_result:
                depth_map = np.array(depth_result['predicted_depth'])
            else:
                # Handle PIL Image result
                from PIL import Image
                if isinstance(depth_result, Image.Image):
                    depth_map = np.array(depth_result)
                else:
                    depth_map = np.array(depth_result)
            
            depth_latency = time.time() - start_time
            
            # Calculate statistics
            mean_depth = float(np.mean(depth_map))
            min_depth = float(np.min(depth_map))
            max_depth = float(np.max(depth_map))
            
            return {
                'depth_map': depth_map,
                'mean_depth': mean_depth,
                'min_depth': min_depth,
                'max_depth': max_depth,
                'depth_latency': depth_latency,
                'depth_shape': depth_map.shape,
                'error': None
            }
            
        except Exception as e:
            depth_latency = time.time() - start_time
            return {
                'depth_map': None,
                'mean_depth': 0.0,
                'min_depth': 0.0,
                'max_depth': 0.0,
                'depth_latency': depth_latency,
                'depth_shape': None,
                'error': str(e)
            }
    
    def analyze_spatial_relationships(self, objects: List[Dict], depth_map: np.ndarray) -> Dict:
        """
        Analyze spatial relationships between objects using depth map
        
        Args:
            objects: List of detected objects (from YOLO) with bboxes
            depth_map: Depth map array
        
        Returns:
            Dict with spatial information:
            - 'object_depths': List of depth values for each object
            - 'relative_distances': Relative distance ordering
            - 'spatial_info': Detailed spatial relationships
        """
        if depth_map is None or len(objects) == 0:
            return {
                'object_depths': [],
                'relative_distances': [],
                'spatial_info': {}
            }
        
        try:
            object_depths = []
            spatial_info = {}
            
            for i, obj in enumerate(objects):
                bbox = obj.get('bbox', [])
                if len(bbox) >= 4:
                    # Get center point of bounding box
                    x_center = int((bbox[0] + bbox[2]) / 2)
                    y_center = int((bbox[1] + bbox[3]) / 2)
                    
                    # Clamp to depth map bounds
                    y_center = min(max(0, y_center), depth_map.shape[0] - 1)
                    x_center = min(max(0, x_center), depth_map.shape[1] - 1)
                    
                    # Get depth at center point
                    depth_value = float(depth_map[y_center, x_center])
                    object_depths.append(depth_value)
                    
                    # Estimate distance (depth values are relative, need calibration)
                    # For now, use relative ordering
                    spatial_info[obj.get('class', f'object_{i}')] = {
                        'depth': depth_value,
                        'bbox_center': (x_center, y_center)
                    }
            
            # Sort by depth (closer = smaller depth value typically)
            relative_distances = sorted(
                enumerate(object_depths),
                key=lambda x: x[1]
            )
            
            return {
                'object_depths': object_depths,
                'relative_distances': relative_distances,
                'spatial_info': spatial_info
            }
            
        except Exception as e:
            return {
                'object_depths': [],
                'relative_distances': [],
                'spatial_info': {},
                'error': str(e)
            }

