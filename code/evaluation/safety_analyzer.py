#!/usr/bin/env python3
"""
Automated safety-critical error analysis for navigation images
Checks for hazard detection, false negatives, and safety scores
"""
import csv
from pathlib import Path
from collections import defaultdict

def analyze_safety(description, category, filename):
    """Analyze safety-critical elements in description"""
    desc_lower = description.lower()
    
    # Define hazards by category
    indoor_hazards = ['stairs', 'steps', 'obstacle', 'door', 'wall', 'furniture', 'barrier', 'opening', 'exit', 'entrance', 'person', 'cabinet', 'table']
    outdoor_hazards = ['crosswalk', 'traffic', 'vehicle', 'car', 'pedestrian', 'road', 'curb', 'sidewalk', 'obstacle', 'sign', 'light', 'person', 'bike', 'post']
    
    hazards_list = indoor_hazards if category == 'indoor' else outdoor_hazards
    
    # Find mentioned hazards
    detected_hazards = [h for h in hazards_list if h in desc_lower]
    
    # Check for spatial information (important for safety)
    has_spatial = any(word in desc_lower for word in ['left', 'right', 'center', 'ahead', 'behind', 'forward', 'distance', 'meter', 'foot'])
    
    # Check if safety info is prioritized (mentioned early)
    first_50_words = ' '.join(description.split()[:50]).lower()
    safety_in_start = sum(1 for h in hazards_list if h in first_50_words)
    
    # Score safety (1-5)
    # 5: Multiple hazards detected + spatial info + prioritized
    # 4: Hazards detected + spatial info
    # 3: Hazards detected
    # 2: Some spatial info but missing hazards
    # 1: Missing critical safety info
    
    if len(detected_hazards) >= 3 and has_spatial and safety_in_start >= 1:
        safety_score = 5
    elif len(detected_hazards) >= 2 and has_spatial:
        safety_score = 4
    elif len(detected_hazards) >= 2:
        safety_score = 3
    elif len(detected_hazards) >= 1 or has_spatial:
        safety_score = 2
    else:
        safety_score = 1
    
    # Check for false negatives (common missed hazards)
    # For indoor: stairs are critical
    # For outdoor: crosswalks, vehicles, traffic are critical
    false_negative = False
    notes = []
    
    if category == 'indoor':
        if 'stair' in filename.lower() or 'step' in filename.lower():
            if 'stair' not in desc_lower and 'step' not in desc_lower:
                false_negative = True
                notes.append("CRITICAL: Missed stairs/steps")
        if 'door' in filename.lower():
            if 'door' not in desc_lower:
                false_negative = True
                notes.append("Missed door")
    
    elif category == 'outdoor':
        if 'crosswalk' in filename.lower():
            if 'crosswalk' not in desc_lower:
                false_negative = True
                notes.append("CRITICAL: Missed crosswalk")
        if 'traffic' in filename.lower() or 'vehicle' in filename.lower():
            if 'traffic' not in desc_lower and 'vehicle' not in desc_lower and 'car' not in desc_lower:
                false_negative = True
                notes.append("CRITICAL: Missed traffic/vehicles")
        if 'pedestrian' in filename.lower():
            if 'pedestrian' not in desc_lower and 'person' not in desc_lower:
                false_negative = True
                notes.append("Missed pedestrian")
    
    return {
        'hazards_mentioned': ', '.join(detected_hazards[:5]),  # Limit to 5
        'hazards_detected_count': len(detected_hazards),
        'false_negative': 'Y' if false_negative else 'N',
        'safety_score': safety_score,
        'notes': '; '.join(notes) if notes else ''
    }

def analyze_all_navigation():
    """Analyze all navigation images"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    output_path = Path('results/approach_1_vlm/evaluation/safety_analysis.csv')
    
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True' and row.get('description'):
                    if row['category'] in ['indoor', 'outdoor']:
                        results.append(row)
    
    print(f"Analyzing {len(results)} navigation descriptions...")
    
    safety_results = []
    for r in results:
        description = r['description']
        category = r['category']
        filename = r['filename']
        model = r['model']
        
        safety_data = analyze_safety(description, category, filename)
        
        safety_results.append({
            'filename': filename,
            'category': category,
            'model': model,
            'description': description[:200] + '...' if len(description) > 200 else description,
            'hazards_mentioned': safety_data['hazards_mentioned'],
            'hazards_detected_count': safety_data['hazards_detected_count'],
            'false_negative': safety_data['false_negative'],
            'safety_score': safety_data['safety_score'],
            'notes': safety_data['notes']
        })
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'category', 'model', 'description', 'hazards_mentioned', 
                     'hazards_detected_count', 'false_negative', 'safety_score', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safety_results)
    
    print(f"✅ Saved {len(safety_results)} safety analyses to {output_path}")
    
    # Print summary
    print("\n📊 Safety Summary:")
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        model_results = [r for r in safety_results if r['model'] == model]
        if model_results:
            avg_score = sum(r['safety_score'] for r in model_results) / len(model_results)
            false_negatives = sum(1 for r in model_results if r['false_negative'] == 'Y')
            avg_hazards = sum(r['hazards_detected_count'] for r in model_results) / len(model_results)
            print(f"  {model}:")
            print(f"    Avg Safety Score: {avg_score:.2f}")
            print(f"    False Negatives: {false_negatives}/{len(model_results)}")
            print(f"    Avg Hazards Detected: {avg_hazards:.1f}")
    
    return safety_results

if __name__ == '__main__':
    analyze_all_navigation()

