"""
Base interface for local Vision-Language Models
Handles device detection and common utilities
"""
import torch
from pathlib import Path
from typing import Optional


def get_device() -> torch.device:
    """
    Detect best available device for M1 Mac
    
    Priority: MPS (Metal) > CUDA > CPU
    
    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_model_cache_dir() -> Path:
    """
    Get the model cache directory (project-local)
    
    Returns:
        Path: Path to data/models/ directory
    """
    # Go from code/approach_4_local/ to project root, then to data/models
    # __file__ is code/approach_4_local/local_vlm.py
    # parent = code/approach_4_local/
    # parent.parent = code/
    # parent.parent.parent = project root (scene_reader/)
    current_file = Path(__file__).resolve()
    # Go up 3 levels: local_vlm.py -> approach_4_local -> code -> project_root
    project_root = current_file.parent.parent.parent
    cache_dir = project_root / "data" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def format_device_name(device: torch.device) -> str:
    """
    Format device name for logging
    
    Args:
        device: torch.device
        
    Returns:
        str: Human-readable device name
    """
    if device.type == "mps":
        return "MPS (Metal)"
    elif device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        return "CPU"


def get_memory_info() -> dict:
    """
    Get memory information for current device
    
    Returns:
        dict: Memory information
    """
    device = get_device()
    info = {
        'device': format_device_name(device),
        'device_type': device.type
    }
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_memory_gb'] = mem.total / (1024**3)
        info['available_memory_gb'] = mem.available / (1024**3)
        info['used_memory_gb'] = mem.used / (1024**3)
    except ImportError:
        # psutil not available, skip memory info
        info['total_memory_gb'] = None
        info['available_memory_gb'] = None
        info['used_memory_gb'] = None
    
    if device.type == "cuda":
        info['cuda_total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['cuda_allocated_memory_gb'] = torch.cuda.memory_allocated(0) / (1024**3)
        info['cuda_reserved_memory_gb'] = torch.cuda.memory_reserved(0) / (1024**3)
    
    return info

