#!/usr/bin/env python3
"""
Helper tool for safety-critical error analysis
Focuses on navigation images (indoor + outdoor) to check for hazard detection
"""
import csv
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

def get_navigation_images(results):
    """Get all indoor and outdoor navigation images"""
    nav_results = []
    for r in results:
        if r.get('category') in ['indoor', 'outdoor'] and r['success'] == 'True':
            nav_results.append(r)
    return nav_results

def create_safety_checklist():
    """Create a checklist for safety-critical elements"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    results = load_results(csv_path)
    nav_results = get_navigation_images(results)
    
    output_path = Path('results/approach_1_vlm/evaluation/safety_analysis.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Organize by image
    by_image = defaultdict(dict)
    for r in nav_results:
        filename = r['filename']
        model = r['model']
        by_image[filename][model] = {
            'description': r.get('description', ''),
            'category': r.get('category', '')
        }
    
    # Define safety-critical elements by category
    indoor_hazards = ['stairs', 'steps', 'obstacle', 'door', 'wall', 'furniture', 'barrier', 'opening', 'exit', 'entrance']
    outdoor_hazards = ['crosswalk', 'traffic', 'vehicle', 'car', 'pedestrian', 'road', 'curb', 'sidewalk', 'obstacle', 'sign', 'light']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'category', 'model', 'description',
            'hazards_mentioned', 'hazards_detected_count', 'false_negative',
            'safety_score', 'notes'
        ])
        
        for filename in sorted(by_image.keys()):
            category = by_image[filename].get('GPT-4V', by_image[filename].get('Gemini', by_image[filename].get('Claude', {}))).get('category', 'unknown')
            hazards_list = indoor_hazards if category == 'indoor' else outdoor_hazards
            
            for model in ['GPT-4V', 'Gemini', 'Claude']:
                if model in by_image[filename]:
                    desc = by_image[filename][model]['description'].lower()
                    detected = [h for h in hazards_list if h in desc]
                    
                    writer.writerow([
                        filename, category, model, by_image[filename][model]['description'],
                        ', '.join(detected), len(detected), '', '', ''
                    ])
    
    print(f"✅ Created safety analysis template: {output_path}")
    print(f"   Navigation images analyzed: {len(by_image)}")
    print("\n📝 Instructions:")
    print("   1. Review each description")
    print("   2. Check if hazards were correctly identified")
    print("   3. Mark false_negative (Y/N) if critical hazard was missed")
    print("   4. Score safety (1-5): 5=excellent, 1=missed critical hazards")
    print("   5. Add notes about any safety concerns")

def analyze_safety_keywords():
    """Quick analysis of safety keywords in descriptions"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    results = load_results(csv_path)
    nav_results = get_navigation_images(results)
    
    indoor_hazards = ['stairs', 'steps', 'obstacle', 'door', 'wall', 'furniture', 'barrier', 'opening', 'exit', 'entrance']
    outdoor_hazards = ['crosswalk', 'traffic', 'vehicle', 'car', 'pedestrian', 'road', 'curb', 'sidewalk', 'obstacle', 'sign', 'light']
    
    print("=" * 80)
    print("SAFETY KEYWORD ANALYSIS (Quick Check)")
    print("=" * 80)
    print()
    
    by_model = defaultdict(lambda: {'indoor': [], 'outdoor': []})
    
    for r in nav_results:
        model = r['model']
        category = r.get('category', '')
        desc = r.get('description', '').lower()
        
        hazards = indoor_hazards if category == 'indoor' else outdoor_hazards
        detected = [h for h in hazards if h in desc]
        by_model[model][category].append(len(detected))
    
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        print(f"{model}:")
        for category in ['indoor', 'outdoor']:
            if by_model[model][category]:
                avg = sum(by_model[model][category]) / len(by_model[model][category])
                print(f"  {category.capitalize()}: {avg:.1f} hazards mentioned on average (n={len(by_model[model][category])})")
        print()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'keywords':
        analyze_safety_keywords()
    else:
        create_safety_checklist()

