"""
YOLOv8 Object Detection Module
Handles object detection and spatial formatting for Approach 2
"""
import time
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


def load_yolo_model(model_size: str = 'n'):
    """
    Load YOLOv8 model
    
    Args:
        model_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
    
    Returns:
        YOLO model object
    """
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
    
    # Try project-local model first, then fallback to auto-download
    project_root = Path(__file__).parent.parent.parent
    local_model_path = project_root / 'data' / 'models' / 'yolo' / f'yolov8{model_size}.pt'
    
    if local_model_path.exists():
        model_name = str(local_model_path)
        print(f"  Loading YOLOv8{model_size.upper()} model from {local_model_path}...")
    else:
        # Fallback: YOLO will auto-download if not found
        model_name = f'yolov8{model_size}.pt'
        print(f"  Loading YOLOv8{model_size.upper()} model (will auto-download if needed)...")
    
    model = YOLO(model_name)
    return model


def detect_objects(image_path: Path, model_size: str = 'n', confidence_threshold: float = 0.25):
    """
    Detect objects in image using YOLOv8
    
    Args:
        image_path: Path to image file
        model_size: 'n' (nano), 'm' (medium), or 'x' (xlarge)
        confidence_threshold: Minimum confidence for detections (default: 0.25)
    
    Returns:
        Tuple of (detections_list, detection_latency, num_objects)
        detections_list: List of dicts with keys: 'class', 'bbox', 'confidence', 'position'
    """
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics package not installed")
    
    # Load model
    model = load_yolo_model(model_size)
    
    # Run detection
    start_time = time.time()
    results = model(str(image_path), conf=confidence_threshold, verbose=False)
    detection_latency = time.time() - start_time
    
    # Extract detections
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        
        # Get image dimensions for spatial calculations
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls[0])
            class_name = names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            
            # Calculate position (left/center/right, top/center/bottom)
            bbox_center_x = (bbox[0] + bbox[2]) / 2
            bbox_center_y = (bbox[1] + bbox[3]) / 2
            
            # Horizontal position
            if bbox_center_x < img_width / 3:
                h_position = "left"
            elif bbox_center_x > 2 * img_width / 3:
                h_position = "right"
            else:
                h_position = "center"
            
            # Vertical position
            if bbox_center_y < img_height / 3:
                v_position = "top"
            elif bbox_center_y > 2 * img_height / 3:
                v_position = "bottom"
            else:
                v_position = "center"
            
            # Calculate relative size (small/medium/large)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = img_width * img_height
            relative_size = bbox_area / img_area
            
            if relative_size < 0.01:
                size_label = "small"
            elif relative_size < 0.05:
                size_label = "medium"
            else:
                size_label = "large"
            
            detections.append({
                'class': class_name,
                'bbox': bbox.tolist(),
                'confidence': confidence,
                'h_position': h_position,
                'v_position': v_position,
                'position': f"{h_position} {v_position}",
                'size': size_label,
                'area_ratio': relative_size
            })
    
    return detections, detection_latency, len(detections)


def format_objects_for_prompt(detections: List[Dict], include_confidence: bool = True) -> str:
    """
    Format detected objects into text for LLM prompt
    
    Args:
        detections: List of detection dicts from detect_objects()
        include_confidence: Whether to include confidence scores
    
    Returns:
        Formatted string describing objects and positions
    """
    if not detections:
        return "No objects detected in the image."
    
    # Group objects by class for cleaner output
    objects_by_class = {}
    for det in detections:
        class_name = det['class']
        if class_name not in objects_by_class:
            objects_by_class[class_name] = []
        objects_by_class[class_name].append(det)
    
    # Format output
    lines = []
    lines.append(f"Detected {len(detections)} object(s):")
    
    for class_name, class_detections in sorted(objects_by_class.items()):
        if len(class_detections) == 1:
            det = class_detections[0]
            conf_text = f" (confidence: {det['confidence']:.2f})" if include_confidence else ""
            lines.append(f"- {class_name} at {det['position']}{conf_text}")
        else:
            # Multiple instances of same class
            positions = [f"{d['position']}" for d in class_detections]
            if include_confidence:
                confidences = [f"{d['confidence']:.2f}" for d in class_detections]
                conf_text = f" (confidences: {', '.join(confidences)})"
            else:
                conf_text = ""
            lines.append(f"- {len(class_detections)} {class_name}(s) at {', '.join(positions)}{conf_text}")
    
    # Add safety-critical objects emphasis
    safety_keywords = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic', 
                      'door', 'stair', 'step', 'obstacle', 'barrier', 'crosswalk', 'sign']
    safety_objects = [det for det in detections if any(keyword in det['class'].lower() for keyword in safety_keywords)]
    
    if safety_objects:
        lines.append("\nSafety-critical elements detected:")
        for det in safety_objects:
            lines.append(f"- {det['class']} at {det['position']} (confidence: {det['confidence']:.2f})")
    
    return "\n".join(lines)


def get_detection_summary(detections: List[Dict]) -> Dict:
    """
    Get summary statistics about detections
    
    Args:
        detections: List of detection dicts
    
    Returns:
        Dict with summary stats
    """
    if not detections:
        return {
            'num_objects': 0,
            'num_classes': 0,
            'avg_confidence': 0.0,
            'classes': []
        }
    
    classes = [det['class'] for det in detections]
    confidences = [det['confidence'] for det in detections]
    
    return {
        'num_objects': len(detections),
        'num_classes': len(set(classes)),
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
        'min_confidence': min(confidences) if confidences else 0.0,
        'max_confidence': max(confidences) if confidences else 0.0,
        'classes': sorted(set(classes))
    }

