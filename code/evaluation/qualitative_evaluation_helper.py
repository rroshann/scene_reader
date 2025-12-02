#!/usr/bin/env python3
"""
Helper tool for manual qualitative evaluation
Displays descriptions side-by-side for easy comparison and scoring
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

def load_results(csv_path):
    """Load results from CSV"""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '2025-11-22T20:' in row.get('timestamp', ''):
                results.append(row)
    return results

def organize_by_image(results):
    """Organize results by image filename"""
    by_image = defaultdict(dict)
    for r in results:
        filename = r['filename']
        model = r['model']
        if r['success'] == 'True':
            by_image[filename][model] = {
                'description': r.get('description', ''),
                'latency': r.get('latency_seconds', ''),
                'category': r.get('category', '')
            }
    return by_image

def create_evaluation_template():
    """Create a template CSV for manual scoring"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    results = load_results(csv_path)
    by_image = organize_by_image(results)
    
    output_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create scoring template
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'category', 'model',
            'completeness', 'clarity', 'conciseness', 'actionability', 'safety_focus',
            'overall_score', 'notes'
        ])
        
        for filename, models in sorted(by_image.items()):
            category = models.get('GPT-4V', models.get('Gemini', models.get('Claude', {}))).get('category', 'unknown')
            for model in ['GPT-4V', 'Gemini', 'Claude']:
                if model in models:
                    writer.writerow([
                        filename, category, model,
                        '', '', '', '', '', '', ''
                    ])
    
    print(f"✅ Created evaluation template: {output_path}")
    print(f"   Total rows: {len(results)}")
    print("\n📝 Instructions:")
    print("   1. Open the CSV file")
    print("   2. Score each description on a 1-5 scale:")
    print("      - Completeness: Does it cover all important elements?")
    print("      - Clarity: Is it easy to understand?")
    print("      - Conciseness: Is it brief and to the point?")
    print("      - Actionability: Can the user act on this information?")
    print("      - Safety Focus: Does it prioritize safety-critical info?")
    print("   3. Add overall_score (1-5) and any notes")
    print("   4. Save the file")

def display_comparison(filename, by_image):
    """Display all model descriptions for an image side-by-side"""
    if filename not in by_image:
        print(f"❌ Image not found: {filename}")
        return
    
    models = by_image[filename]
    category = models.get('GPT-4V', models.get('Gemini', models.get('Claude', {}))).get('category', 'unknown')
    
    print("=" * 80)
    print(f"Image: {filename}")
    print(f"Category: {category}")
    print("=" * 80)
    print()
    
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        if model in models:
            data = models[model]
            print(f"--- {model} ---")
            print(f"Latency: {data.get('latency', 'N/A')}s")
            print(f"Description:")
            print(data.get('description', 'N/A'))
            print()
        else:
            print(f"--- {model} ---")
            print("❌ No result available")
            print()
    
    print("=" * 80)

def interactive_evaluation():
    """Interactive mode to view descriptions and score them"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    results = load_results(csv_path)
    by_image = organize_by_image(results)
    
    print("=" * 80)
    print("QUALITATIVE EVALUATION HELPER")
    print("=" * 80)
    print()
    print("Available images:")
    print()
    
    images = sorted(by_image.keys())
    for i, filename in enumerate(images, 1):
        category = by_image[filename].get('GPT-4V', by_image[filename].get('Gemini', by_image[filename].get('Claude', {}))).get('category', 'unknown')
        print(f"  {i:2d}. {filename} ({category})")
    
    print()
    print("Commands:")
    print("  - Enter image number to view descriptions")
    print("  - Type 'all' to create comparison file for all images")
    print("  - Type 'template' to create scoring CSV template")
    print("  - Type 'quit' to exit")
    print()
    
    while True:
        try:
            cmd = input("> ").strip()
            
            if cmd.lower() == 'quit':
                break
            elif cmd.lower() == 'template':
                create_evaluation_template()
            elif cmd.lower() == 'all':
                create_all_comparisons(by_image)
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(images):
                    display_comparison(images[idx], by_image)
                else:
                    print("❌ Invalid image number")
            else:
                print("❌ Invalid command")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_all_comparisons(by_image):
    """Create a text file with all comparisons"""
    output_path = Path('results/approach_1_vlm/evaluation/all_descriptions_comparison.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for filename in sorted(by_image.keys()):
            models = by_image[filename]
            category = models.get('GPT-4V', models.get('Gemini', models.get('Claude', {}))).get('category', 'unknown')
            
            f.write("=" * 80 + "\n")
            f.write(f"Image: {filename}\n")
            f.write(f"Category: {category}\n")
            f.write("=" * 80 + "\n\n")
            
            for model in ['GPT-4V', 'Gemini', 'Claude']:
                if model in models:
                    data = models[model]
                    f.write(f"--- {model} ---\n")
                    f.write(f"Latency: {data.get('latency', 'N/A')}s\n")
                    f.write(f"Description:\n{data.get('description', 'N/A')}\n\n")
                else:
                    f.write(f"--- {model} ---\n")
                    f.write("❌ No result available\n\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"✅ Created comparison file: {output_path}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'template':
        create_evaluation_template()
    elif len(sys.argv) > 1 and sys.argv[1] == 'all':
        csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
        results = load_results(csv_path)
        by_image = organize_by_image(results)
        create_all_comparisons(by_image)
    else:
        interactive_evaluation()

