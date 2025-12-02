#!/usr/bin/env python3
"""
Evaluate if images are worthy/suitable for Scene Reader project
Uses GPT-4V to assess if images are good for testing (not technical quality)
"""
import os
import sys
from pathlib import Path
from PIL import Image
import time
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

def check_technical_quality(image_path):
    """Check technical image quality metrics"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        
        # Check format
        format_name = img.format
        
        # Check if image is readable
        img.verify()
        
        # Reopen for mode check (verify closes the image)
        img = Image.open(image_path)
        mode = img.mode
        
        return {
            'valid': True,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'file_size_mb': round(file_size, 2),
            'format': format_name,
            'mode': mode,
            'aspect_ratio': round(width / height, 2) if height > 0 else 0
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def analyze_content_with_vlm(image_path, category):
    """Use GPT-4V to analyze image content and suitability"""
    try:
        from openai import OpenAI
        import base64
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OpenAI API key not found"
        
        client = OpenAI(api_key=api_key)
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        # Category-specific evaluation criteria
        category_prompts = {
            "gaming": """Evaluate this gaming screenshot for a visual accessibility project:

1. Does it show a clear gaming scene (not just menu/loading screen)?
2. Are there multiple elements visible (character, enemies, UI, environment)?
3. Is it complex enough to test VLM capabilities?
4. Would a blind gamer benefit from understanding this scene?
5. Any issues: too cluttered, too simple, unclear what's happening?

Rate: Good for testing / Needs improvement / Not suitable
Explain why.""",
            
            "indoor": """Evaluate this indoor navigation image for accessibility testing:

1. Does it show a clear indoor scene (hallway, room, stairs, doorway)?
2. Are obstacles/navigation elements visible?
3. Is it realistic and representative of real-world scenarios?
4. Would a blind person need to understand this scene to navigate safely?
5. Any issues: too dark, too cluttered, not representative?

Rate: Good for testing / Needs improvement / Not suitable
Explain why.""",
            
            "outdoor": """Evaluate this outdoor navigation image for accessibility testing:

1. Does it show a clear outdoor scene (street, sidewalk, crosswalk, entrance)?
2. Are safety-critical elements visible (obstacles, traffic, hazards)?
3. Is it realistic and representative of real-world scenarios?
4. Would a blind person need to understand this scene to navigate safely?
5. Any issues: too dark, too cluttered, not representative?

Rate: Good for testing / Needs improvement / Not suitable
Explain why.""",
            
            "text": """Evaluate this text/sign image for accessibility testing:

1. Is there clear, readable text in the image?
2. Is the text important/useful (sign, menu, label, not just random text)?
3. Would a blind person benefit from reading this text?
4. Is the text clear enough for OCR testing?
5. Any issues: text too small, blurry, not useful?

Rate: Good for testing / Needs improvement / Not suitable
Explain why."""
        }
        
        prompt = category_prompts.get(category, category_prompts["indoor"])
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content
        return analysis, None
        
    except Exception as e:
        return None, str(e)

def analyze_image(image_path, category=None):
    """Analyze a single image"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path.name}")
    print(f"{'='*60}")
    
    # Technical quality check
    tech = check_technical_quality(image_path)
    
    if not tech['valid']:
        print(f"❌ Invalid image: {tech.get('error', 'Unknown error')}")
        return
    
    print("\n📊 Technical Quality:")
    print(f"  Resolution: {tech['resolution']}")
    print(f"  File size: {tech['file_size_mb']} MB")
    print(f"  Format: {tech['format']}")
    print(f"  Color mode: {tech['mode']}")
    print(f"  Aspect ratio: {tech['aspect_ratio']}")
    
    # Quality recommendations
    print("\n✅ Quality Check:")
    issues = []
    
    if tech['width'] < 640 or tech['height'] < 480:
        issues.append("⚠️  Resolution too low (minimum 640x480 recommended)")
    elif tech['width'] >= 1280 and tech['height'] >= 720:
        print("  ✅ Good resolution (HD or better)")
    else:
        print("  ⚠️  Acceptable resolution (could be higher)")
    
    if tech['file_size_mb'] > 10:
        issues.append(f"⚠️  File size large ({tech['file_size_mb']} MB)")
    else:
        print("  ✅ File size reasonable")
    
    if tech['mode'] != 'RGB':
        issues.append(f"⚠️  Color mode is {tech['mode']} (RGB preferred)")
    else:
        print("  ✅ Color mode is RGB")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    # VLM content analysis (main feature - evaluates if image is worthy)
    if category:
        print(f"\n🤖 Content Suitability Analysis (using GPT-4V):")
        print(f"  Category: {category}")
        print("  Analyzing if this image is good for testing...")
        
        analysis, error = analyze_content_with_vlm(image_path, category)
        
        if analysis:
            print(f"\n  📝 Assessment:\n  {analysis}")
        else:
            print(f"  ❌ Error: {error}")
    else:
        print("\n⚠️  No category specified. Specify category for content analysis:")
        print("   Categories: gaming, indoor, outdoor, text")

def analyze_directory(directory, category=None):
    """Analyze all images in a directory"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    images = list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))
    
    if not images:
        print(f"❌ No images found in {directory}")
        return
    
    print(f"\n📁 Found {len(images)} images in {directory}")
    
    for img_path in sorted(images):
        analyze_image(img_path, category)
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/analyze_image_quality.py <image_path> [category]")
        print("  python scripts/analyze_image_quality.py <directory> [category]")
        print("\nCategories: gaming, indoor, outdoor, text")
        print("\nExample:")
        print("  python scripts/analyze_image_quality.py data/images/gaming/ gaming")
        print("  python scripts/analyze_image_quality.py data/images/gaming/game_01.png gaming")
        return
    
    path = Path(sys.argv[1])
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    if path.is_file():
        analyze_image(path, category)
    elif path.is_dir():
        analyze_directory(path, category)
    else:
        print(f"❌ Path not found: {path}")

if __name__ == "__main__":
    main()

