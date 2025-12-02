"""
Device utilities for Approach 3
Handles device detection and model storage for M1 Mac optimization
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
    # Go from code/approach_3_specialized/ to project root, then to data/models
    current_file = Path(__file__).resolve()
    # Go up 3 levels: device_utils.py -> approach_3_specialized -> code -> project_root
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
        info['ram_total_gb'] = mem.total / (1024**3)
        info['ram_available_gb'] = mem.available / (1024**3)
        info['ram_used_percent'] = mem.percent
    except ImportError:
        pass
    
    return info

