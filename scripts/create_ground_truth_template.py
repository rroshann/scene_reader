#!/usr/bin/env python3
"""
Create ground truth CSV template from all images
This creates a template that you can fill in with actual labels
"""
import csv
from pathlib import Path
import os

def get_category_from_path(image_path):
    """Extract category from file path"""
    parts = image_path.parts
    if 'gaming' in parts:
        return 'gaming'
    elif 'indoor' in parts:
        return 'indoor'
    elif 'outdoor' in parts:
        return 'outdoor'
    elif 'text' in parts:
        return 'text'
    return 'unknown'

def create_ground_truth_template():
    """Create CSV template with all images"""
    images_dir = Path('data/images')
    output_file = Path('data/ground_truth.csv')
    
    # Find all images
    images = []
    for category_dir in ['gaming', 'indoor', 'outdoor', 'text']:
        cat_path = images_dir / category_dir
        if cat_path.exists():
            for img_file in sorted(cat_path.glob('*.png')) + sorted(cat_path.glob('*.jpg')) + sorted(cat_path.glob('*.jpeg')):
                images.append({
                    'filename': img_file.name,
                    'category': category_dir,
                    'subcategory': '',  # To be filled
                    'key_objects': '',  # To be filled
                    'safety_critical': '',  # To be filled
                    'spatial_layout': '',  # To be filled
                    'text_content': '',  # To be filled (if text category)
                    'description_requirements': '',  # To be filled
                    'difficulty': ''  # easy, medium, hard
                })
    
    # Write CSV
    fieldnames = [
        'filename', 'category', 'subcategory', 'key_objects', 
        'safety_critical', 'spatial_layout', 'text_content', 
        'description_requirements', 'difficulty'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(images)
    
    print(f"✅ Created ground truth template: {output_file}")
    print(f"   Total images: {len(images)}")
    print(f"   Categories:")
    for cat in ['gaming', 'indoor', 'outdoor', 'text']:
        count = sum(1 for img in images if img['category'] == cat)
        print(f"     - {cat}: {count} images")
    print(f"\n   Next step: Fill in the template with actual labels")

if __name__ == "__main__":
    create_ground_truth_template()

