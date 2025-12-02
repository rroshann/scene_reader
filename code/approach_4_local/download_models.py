#!/usr/bin/env python3
"""
Pre-download models for Approach 4
Downloads BLIP-2 model to data/models/ directory
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from local_vlm import get_model_cache_dir
from huggingface_hub import snapshot_download


def download_model(model_name: str, cache_dir: Path):
    """
    Download a model from HuggingFace Hub
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache the model
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}\n")
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True
        )
        print(f"✅ Successfully downloaded: {model_name}\n")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}\n")
        return False


def main():
    """Download all models"""
    print("="*60)
    print("Approach 4: Model Download Script")
    print("="*60)
    print()
    print("This script will download:")
    print("  1. BLIP-2 (Salesforce/blip2-opt-2.7b) - ~5GB")
    print()
    print("Total download size: ~5GB")
    print("Models will be stored in: data/models/")
    print()
    
    cache_dir = get_model_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Models to download
    models = [
        "Salesforce/blip2-opt-2.7b"
    ]
    
    results = []
    for model_name in models:
        success = download_model(model_name, cache_dir)
        results.append((model_name, success))
    
    # Summary
    print("="*60)
    print("Download Summary")
    print("="*60)
    for model_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {model_name}")
    
    all_success = all(success for _, success in results)
    if all_success:
        print("\n✅ All models downloaded successfully!")
        print("You can now run batch testing.")
    else:
        print("\n⚠️  Some models failed to download.")
        print("You can still try running batch testing - models will download automatically on first use.")


if __name__ == "__main__":
    main()

