"""
Batch testing for Approach 6 RAG-Enhanced Vision
Tests all configurations on gaming images only
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


def get_gaming_images():
    """Get all gaming images from data/images/gaming folder"""
    gaming_dir = Path('data/images/gaming')
    images = []
    
    if gaming_dir.exists():
        for img_file in sorted(gaming_dir.glob('*.png')) + sorted(gaming_dir.glob('*.jpg')) + sorted(gaming_dir.glob('*.jpeg')):
            images.append({
                'path': img_file,
                'filename': img_file.name,
                'category': 'gaming'
            })
    
    return images


def test_configuration_on_image(image_path, category, vlm_model, use_rag, output_dir, vector_store=None):
    """
    Test a single configuration on a single image
    
    Args:
        image_path: Path to image
        category: Image category (should be 'gaming')
        vlm_model: VLM model ('gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku')
        use_rag: Whether to use RAG enhancement (True) or just base VLM (False)
        output_dir: Directory to save results
        vector_store: Optional VectorStore instance
    
    Returns:
        Result dict for CSV
    """
    config_name = f"{vlm_model}{'+RAG' if use_rag else ''}"
    print(f"\n  Testing: {config_name}")
    
    try:
        if use_rag:
            # Run RAG pipeline
            result = run_rag_pipeline(
                image_path,
                vlm_model=vlm_model,
                enhancement_model='gpt-4o-mini',  # Use lightweight model for enhancement
                use_llm_entity_extraction=False,  # Use filename-based identification (faster, free)
                top_k=3,
                vector_store=vector_store
            )
            
            if result['success']:
                csv_result = {
                    'filename': image_path.name,
                    'category': category,
                    'vlm_model': vlm_model,
                    'use_rag': True,
                    'configuration': config_name,
                    'base_description': result['base_description'],
                    'enhanced_description': result['enhanced_description'],
                    'game_name': result['game_name'],
                    'entities': str(result['entities']),
                    'num_retrieved_chunks': len(result['retrieved_chunks']),
                    'retrieved_chunks': str([c['text'][:100] for c in result['retrieved_chunks']]),  # Truncate for CSV
                    'base_latency': result['base_latency'],
                    'entity_extraction_latency': result['entity_extraction_latency'],
                    'retrieval_latency': result['retrieval_latency'],
                    'enhancement_latency': result['enhancement_latency'],
                    'total_latency': result['total_latency'],
                    'base_tokens': result['base_tokens'],
                    'enhancement_tokens': result['enhancement_tokens'],
                    'success': True,
                    'error': None,
                    'timestamp': datetime.now().isoformat()
                }
                print(f"    ✅ Success! Total latency: {result['total_latency']:.2f}s")
                return csv_result
            else:
                csv_result = {
                    'filename': image_path.name,
                    'category': category,
                    'vlm_model': vlm_model,
                    'use_rag': True,
                    'configuration': config_name,
                    'base_description': result.get('base_description'),
                    'enhanced_description': result.get('enhanced_description'),
                    'game_name': result.get('game_name'),
                    'entities': None,
                    'num_retrieved_chunks': 0,
                    'retrieved_chunks': None,
                    'base_latency': result.get('base_latency'),
                    'entity_extraction_latency': result.get('entity_extraction_latency'),
                    'retrieval_latency': result.get('retrieval_latency'),
                    'enhancement_latency': result.get('enhancement_latency'),
                    'total_latency': result.get('total_latency'),
                    'base_tokens': result.get('base_tokens'),
                    'enhancement_tokens': result.get('enhancement_tokens'),
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
                return csv_result
        
        else:
            # Run base VLM only (for comparison)
            from rag_pipeline import generate_base_description
            
            base_result, error = generate_base_description(image_path, vlm_model)
            
            if error or not base_result:
                csv_result = {
                    'filename': image_path.name,
                    'category': category,
                    'vlm_model': vlm_model,
                    'use_rag': False,
                    'configuration': config_name,
                    'base_description': None,
                    'enhanced_description': None,
                    'game_name': None,
                    'entities': None,
                    'num_retrieved_chunks': 0,
                    'retrieved_chunks': None,
                    'base_latency': base_result.get('latency') if base_result else None,
                    'entity_extraction_latency': None,
                    'retrieval_latency': None,
                    'enhancement_latency': None,
                    'total_latency': base_result.get('latency') if base_result else None,
                    'base_tokens': base_result.get('tokens') if base_result else None,
                    'enhancement_tokens': None,
                    'success': False,
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                }
                print(f"    ❌ Failed: {error}")
                return csv_result
            
            csv_result = {
                'filename': image_path.name,
                'category': category,
                'vlm_model': vlm_model,
                'use_rag': False,
                'configuration': config_name,
                'base_description': base_result['description'],
                'enhanced_description': base_result['description'],  # Same as base for non-RAG
                'game_name': None,
                'entities': None,
                'num_retrieved_chunks': 0,
                'retrieved_chunks': None,
                'base_latency': base_result['latency'],
                'entity_extraction_latency': None,
                'retrieval_latency': None,
                'enhancement_latency': None,
                'total_latency': base_result['latency'],
                'base_tokens': base_result.get('tokens'),
                'enhancement_tokens': None,
                'success': True,
                'error': None,
                'timestamp': datetime.now().isoformat()
            }
            print(f"    ✅ Success! Latency: {base_result['latency']:.2f}s")
            return csv_result
            
    except Exception as e:
        csv_result = {
            'filename': image_path.name,
            'category': category,
            'vlm_model': vlm_model,
            'use_rag': use_rag,
            'configuration': config_name,
            'base_description': None,
            'enhanced_description': None,
            'game_name': None,
            'entities': None,
            'num_retrieved_chunks': 0,
            'retrieved_chunks': None,
            'base_latency': None,
            'entity_extraction_latency': None,
            'retrieval_latency': None,
            'enhancement_latency': None,
            'total_latency': None,
            'base_tokens': None,
            'enhancement_tokens': None,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(f"    ❌ Exception: {e}")
        return csv_result


def save_results(results, output_dir):
    """Save results to CSV file (append mode)"""
    output_file = output_dir / 'batch_results.csv'
    
    # Define CSV columns
    fieldnames = [
        'filename', 'category', 'vlm_model', 'use_rag', 'configuration',
        'base_description', 'enhanced_description', 'game_name', 'entities',
        'num_retrieved_chunks', 'retrieved_chunks',
        'base_latency', 'entity_extraction_latency', 'retrieval_latency',
        'enhancement_latency', 'total_latency',
        'base_tokens', 'enhancement_tokens',
        'success', 'error', 'timestamp'
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = output_file.exists()
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)


def main():
    """Main batch testing function"""
    print("=" * 60)
    print("BATCH TESTING - Approach 6: RAG-Enhanced Vision")
    print("=" * 60)
    print()
    
    # Get gaming images
    images = get_gaming_images()
    print(f"Found {len(images)} gaming images")
    
    if not images:
        print("❌ No gaming images found in data/images/gaming/")
        return
    
    # Initialize vector store (shared across tests)
    print("\nInitializing vector store...")
    try:
        vector_store = initialize_vector_store()
        print("✅ Vector store initialized")
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        print("Continuing without vector store (RAG will fail but base VLM will work)")
        vector_store = None
    
    # Test configurations
    vlm_models = ['gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku']
    use_rag_options = [False, True]  # Test both base and RAG-enhanced
    
    configurations = []
    for vlm in vlm_models:
        for use_rag in use_rag_options:
            configurations.append((vlm, use_rag))
    
    print(f"\nTesting {len(configurations)} configurations:")
    for vlm, use_rag in configurations:
        config_name = f"{vlm}{'+RAG' if use_rag else ''}"
        print(f"  - {config_name}")
    
    total_tests = len(images) * len(configurations)
    print(f"\nTotal tests: {total_tests} ({len(images)} images × {len(configurations)} configurations)")
    print()
    
    # Output directory
    output_dir = Path('results/approach_6_rag/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    all_results = []
    start_time = time.time()
    test_count = 0
    
    for img_idx, img_info in enumerate(images, 1):
        print(f"\n{'='*60}")
        print(f"Image {img_idx}/{len(images)}: {img_info['filename']}")
        print(f"{'='*60}")
        
        image_results = []
        
        for config_idx, (vlm_model, use_rag) in enumerate(configurations, 1):
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] Configuration {config_idx}/{len(configurations)}")
            
            result = test_configuration_on_image(
                img_info['path'],
                img_info['category'],
                vlm_model,
                use_rag,
                output_dir,
                vector_store
            )
            image_results.append(result)
            all_results.append(result)
            
            # Save incrementally
            save_results([result], output_dir)
            
            # Rate limiting - wait between API calls
            if test_count < total_tests:
                time.sleep(1)  # 1 second between API calls
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / test_count
        remaining = avg_time * (total_tests - test_count)
        print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min | Est. remaining: {remaining/60:.1f} min")
        
        # Wait between images
        if img_idx < len(images):
            print("   Waiting 2 seconds before next image...")
            time.sleep(2)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("BATCH TESTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to: {output_dir / 'batch_results.csv'}")
    
    # Success rate
    successful = sum(1 for r in all_results if r['success'])
    print(f"Success rate: {successful}/{len(all_results)} ({successful*100/len(all_results):.1f}%)")
    
    # By configuration
    print("\nBy configuration:")
    for vlm_model in vlm_models:
        for use_rag in use_rag_options:
            config_results = [r for r in all_results if r['vlm_model'] == vlm_model and r['use_rag'] == use_rag]
            if config_results:
                config_success = sum(1 for r in config_results if r['success'])
                avg_latency = sum(float(r['total_latency']) for r in config_results if r['success'] and r['total_latency']) / config_success if config_success > 0 else 0
                config_name = f"{vlm_model}{'+RAG' if use_rag else ''}"
                print(f"  {config_name}: {config_success}/{len(config_results)} successful, avg latency: {avg_latency:.2f}s")
    
    # By RAG vs Base
    print("\nRAG vs Base:")
    base_results = [r for r in all_results if not r['use_rag']]
    rag_results = [r for r in all_results if r['use_rag']]
    if base_results:
        base_success = sum(1 for r in base_results if r['success'])
        base_avg_latency = sum(float(r['total_latency']) for r in base_results if r['success'] and r['total_latency']) / base_success if base_success > 0 else 0
        print(f"  Base VLM: {base_success}/{len(base_results)} successful, avg latency: {base_avg_latency:.2f}s")
    if rag_results:
        rag_success = sum(1 for r in rag_results if r['success'])
        rag_avg_latency = sum(float(r['total_latency']) for r in rag_results if r['success'] and r['total_latency']) / rag_success if rag_success > 0 else 0
        print(f"  RAG-Enhanced: {rag_success}/{len(rag_results)} successful, avg latency: {rag_avg_latency:.2f}s")


if __name__ == "__main__":
    main()

