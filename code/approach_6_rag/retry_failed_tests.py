"""
Retry failed tests from batch testing
Only runs tests that failed or are missing
"""
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

from rag_pipeline import run_rag_pipeline
from vector_store import initialize_vector_store
from batch_test_rag import test_configuration_on_image, save_results, get_gaming_images


def get_failed_tests():
    """Get list of tests that need to be retried"""
    csv_path = Path('results/approach_6_rag/raw/batch_results.csv')
    
    if not csv_path.exists():
        return []
    
    # Read existing results
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    # Get successful pairs
    successful_pairs = set(
        (r.get('filename'), r.get('configuration')) 
        for r in results 
        if r.get('success', '').lower() == 'true'
    )
    
    # Get all expected pairs
    images = get_gaming_images()
    vlm_models = ['gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku']
    use_rag_options = [False, True]
    
    all_expected = []
    for img_info in images:
        for vlm in vlm_models:
            for use_rag in use_rag_options:
                config_name = f"{vlm}{'+RAG' if use_rag else ''}"
                pair = (img_info['filename'], config_name)
                if pair not in successful_pairs:
                    all_expected.append({
                        'image_path': img_info['path'],
                        'category': img_info['category'],
                        'vlm_model': vlm,
                        'use_rag': use_rag,
                        'configuration': config_name
                    })
    
    return all_expected


def main():
    """Retry failed tests"""
    print("=" * 60)
    print("RETRY FAILED TESTS - Approach 6")
    print("=" * 60)
    print()
    
    failed_tests = get_failed_tests()
    
    if not failed_tests:
        print("✅ All tests already successful! Nothing to retry.")
        return
    
    print(f"Found {len(failed_tests)} tests to retry")
    print()
    
    # Initialize vector store
    print("Initializing vector store...")
    try:
        vector_store = initialize_vector_store()
        print("✅ Vector store initialized")
    except Exception as e:
        print(f"⚠️  Vector store init failed: {e}")
        vector_store = None
    
    # Output directory
    output_dir = Path('results/approach_6_rag/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Retry failed tests
    start_time = time.time()
    successful_retries = 0
    
    for idx, test_info in enumerate(failed_tests, 1):
        print(f"\n[{idx}/{len(failed_tests)}] Retrying: {test_info['configuration']} on {test_info['image_path'].name}")
        
        result = test_configuration_on_image(
            test_info['image_path'],
            test_info['category'],
            test_info['vlm_model'],
            test_info['use_rag'],
            output_dir,
            vector_store
        )
        
        if result['success']:
            successful_retries += 1
        
        # Save incrementally
        save_results([result], output_dir)
        
        # Rate limiting
        if idx < len(failed_tests):
            time.sleep(1)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("RETRY COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successfully retried: {successful_retries}/{len(failed_tests)}")
    
    # Final status
    csv_path = Path('results/approach_6_rag/raw/batch_results.csv')
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        successful = sum(1 for r in results if r.get('success', '').lower() == 'true')
        print(f"Total successful: {successful}/72")


if __name__ == "__main__":
    main()

