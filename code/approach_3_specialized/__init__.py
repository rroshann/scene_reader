"""
Approach 3: Specialized Multi-Model System
Combines OCR, depth estimation, and object detection for task-specific accuracy
"""
import sys
from pathlib import Path

# Add parent directory to path to allow importing from sibling directories
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core components from Approach 2 for reuse
# This allows Approach 3 to extend functionality without duplicating code
from code.approach_2_yolo_llm import yolo_detector
from code.approach_2_yolo_llm import llm_generator
from code.approach_2_yolo_llm import prompts

