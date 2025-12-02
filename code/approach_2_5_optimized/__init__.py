"""
Approach 2.5: Optimized YOLO+LLM Hybrid Pipeline

This module extends Approach 2 with advanced optimizations:
- Faster LLM models (GPT-3.5-turbo)
- Prompt optimization
- Caching
- Adaptive parameters

Code Reuse Strategy:
- Imports and extends Approach 2 components (DRY principle)
- Only creates new code for optimizations
- Preserves Approach 2 functionality
"""

# Import Approach 2 components for reuse
import sys
from pathlib import Path

# Add code directory to path for imports
project_root = Path(__file__).parent.parent.parent
code_dir = project_root / "code"
sys.path.insert(0, str(code_dir))

# Import Approach 2 modules
from approach_2_yolo_llm import yolo_detector
from approach_2_yolo_llm import llm_generator
from approach_2_yolo_llm import prompts

# Re-export for convenience
__all__ = [
    'yolo_detector',
    'llm_generator', 
    'prompts'
]

