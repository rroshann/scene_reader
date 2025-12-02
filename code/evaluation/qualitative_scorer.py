#!/usr/bin/env python3
"""
Automated qualitative evaluation of VLM descriptions
Scores descriptions on 5 dimensions: Completeness, Clarity, Conciseness, Actionability, Safety Focus
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

def score_completeness(description, category, filename):
    """Score 1-5: Does it mention all important elements?"""
    desc_lower = description.lower()
    
    # Category-specific key elements
    if category == 'gaming':
        # Check for: character, enemies/NPCs, UI elements, status, objects
        elements = ['character', 'player', 'enemy', 'npc', 'health', 'hp', 'card', 'ui', 'status', 'item', 'door', 'path']
        found = sum(1 for elem in elements if elem in desc_lower)
        # Gaming scenes are complex, expect 3-5 elements
        if found >= 5:
            return 5
        elif found >= 3:
            return 4
        elif found >= 2:
            return 3
        elif found >= 1:
            return 2
        else:
            return 1
    
    elif category in ['indoor', 'outdoor']:
        # Check for: obstacles, spatial layout, doors/exits, hazards
        elements = ['door', 'exit', 'entrance', 'stairs', 'step', 'obstacle', 'wall', 'left', 'right', 'center', 'ahead', 'behind', 'person', 'furniture']
        found = sum(1 for elem in elements if elem in desc_lower)
        # Navigation needs spatial info + obstacles
        if found >= 6:
            return 5
        elif found >= 4:
            return 4
        elif found >= 3:
            return 3
        elif found >= 2:
            return 2
        else:
            return 1
    
    elif category == 'text':
        # Check for: text content, meaning, context
        # Text descriptions should mention what the text says
        if len(description) > 50 and ('read' in desc_lower or 'say' in desc_lower or 'show' in desc_lower or 'display' in desc_lower):
            return 5
        elif len(description) > 30:
            return 4
        elif len(description) > 15:
            return 3
        else:
            return 2
    
    # Default scoring based on length and detail
    if len(description) > 100:
        return 4
    elif len(description) > 50:
        return 3
    else:
        return 2

def score_clarity(description):
    """Score 1-5: Is it easy to understand?"""
    desc_lower = description.lower()
    
    # Check for clear structure and language
    has_spatial = any(word in desc_lower for word in ['left', 'right', 'center', 'ahead', 'behind', 'above', 'below'])
    has_clear_verbs = any(word in desc_lower for word in ['shows', 'displays', 'contains', 'features', 'includes'])
    sentence_count = len(re.split(r'[.!?]+', description))
    avg_sentence_length = len(description.split()) / max(sentence_count, 1)
    
    # Good clarity: spatial references, clear verbs, reasonable sentence length
    score = 3  # Base score
    
    if has_spatial:
        score += 0.5
    if has_clear_verbs:
        score += 0.5
    if 10 <= avg_sentence_length <= 25:  # Good sentence length
        score += 0.5
    if sentence_count >= 2:  # Multiple sentences (better structure)
        score += 0.5
    
    # Check for confusing elements
    if 'unclear' in desc_lower or 'unable to' in desc_lower or 'cannot' in desc_lower:
        score -= 1
    
    return min(5, max(1, int(round(score))))

def score_conciseness(description):
    """Score 1-5: Is it appropriately brief?"""
    word_count = len(description.split())
    
    # Ideal: 50-100 words for most scenarios
    # Too verbose: >150 words
    # Too terse: <30 words
    
    if 50 <= word_count <= 100:
        return 5
    elif 40 <= word_count <= 120:
        return 4
    elif 30 <= word_count <= 150:
        return 3
    elif 20 <= word_count <= 200:
        return 2
    else:
        return 1

def score_actionability(description, category):
    """Score 1-5: Can user make decisions based on it?"""
    desc_lower = description.lower()
    
    # Check for actionable information
    has_directions = any(word in desc_lower for word in ['left', 'right', 'forward', 'ahead', 'behind', 'toward'])
    has_distances = any(word in desc_lower for word in ['meter', 'foot', 'feet', 'close', 'near', 'far', 'distance'])
    has_status = any(word in desc_lower for word in ['safe', 'danger', 'hazard', 'obstacle', 'clear', 'open', 'closed'])
    has_priorities = any(word in desc_lower for word in ['important', 'critical', 'immediate', 'urgent', 'priority'])
    
    actionable_elements = sum([has_directions, has_distances, has_status, has_priorities])
    
    if actionable_elements >= 3:
        return 5
    elif actionable_elements >= 2:
        return 4
    elif actionable_elements >= 1:
        return 3
    elif len(description) > 50:  # Has some information
        return 2
    else:
        return 1

def score_safety_focus(description, category):
    """Score 1-5: Does it prioritize safety-critical info?"""
    if category not in ['indoor', 'outdoor']:
        # For non-navigation, check if it mentions important status/concerns
        desc_lower = description.lower()
        has_concerns = any(word in desc_lower for word in ['danger', 'threat', 'warning', 'important', 'critical', 'urgent', 'obstacle'])
        if has_concerns:
            return 4
        else:
            return 3
    
    # For navigation, prioritize safety elements
    desc_lower = description.lower()
    
    # Safety-critical keywords
    safety_keywords = ['stairs', 'step', 'obstacle', 'hazard', 'danger', 'vehicle', 'traffic', 'crosswalk', 'door', 'wall', 'barrier', 'person', 'pedestrian']
    found_safety = sum(1 for keyword in safety_keywords if keyword in desc_lower)
    
    # Check if safety info is mentioned early/emphasized
    first_50_words = ' '.join(description.split()[:50]).lower()
    safety_in_start = sum(1 for keyword in safety_keywords if keyword in first_50_words)
    
    if found_safety >= 4 and safety_in_start >= 2:
        return 5
    elif found_safety >= 3:
        return 4
    elif found_safety >= 2:
        return 3
    elif found_safety >= 1:
        return 2
    else:
        return 1

def calculate_overall_score(scores):
    """Calculate overall score from 5 dimensions"""
    # Weighted average: Safety and Actionability are more important
    weights = {
        'completeness': 0.2,
        'clarity': 0.2,
        'conciseness': 0.15,
        'actionability': 0.25,
        'safety_focus': 0.2
    }
    
    weighted_sum = sum(scores[dim] * weights[dim] for dim in scores)
    return round(weighted_sum, 1)

def evaluate_all_descriptions():
    """Evaluate all descriptions in batch_results.csv"""
    csv_path = Path('results/approach_1_vlm/raw/batch_results.csv')
    output_path = Path('results/approach_1_vlm/evaluation/qualitative_scores.csv')
    
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only evaluate system prompt test results
            if '2025-11-22T20:' in row.get('timestamp', ''):
                if row['success'] == 'True' and row.get('description'):
                    results.append(row)
    
    print(f"Evaluating {len(results)} descriptions...")
    
    scored_results = []
    for r in results:
        description = r['description']
        category = r['category']
        filename = r['filename']
        model = r['model']
        
        scores = {
            'completeness': score_completeness(description, category, filename),
            'clarity': score_clarity(description),
            'conciseness': score_conciseness(description),
            'actionability': score_actionability(description, category),
            'safety_focus': score_safety_focus(description, category)
        }
        
        overall = calculate_overall_score(scores)
        
        scored_results.append({
            'filename': filename,
            'category': category,
            'model': model,
            'completeness': scores['completeness'],
            'clarity': scores['clarity'],
            'conciseness': scores['conciseness'],
            'actionability': scores['actionability'],
            'safety_focus': scores['safety_focus'],
            'overall_score': overall,
            'notes': ''
        })
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'category', 'model', 'completeness', 'clarity', 
                     'conciseness', 'actionability', 'safety_focus', 'overall_score', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_results)
    
    print(f"✅ Saved {len(scored_results)} scored descriptions to {output_path}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    for model in ['GPT-4V', 'Gemini', 'Claude']:
        model_scores = [r['overall_score'] for r in scored_results if r['model'] == model]
        if model_scores:
            avg = sum(model_scores) / len(model_scores)
            print(f"  {model}: {avg:.2f} avg overall score (n={len(model_scores)})")
    
    return scored_results

if __name__ == '__main__':
    evaluate_all_descriptions()

