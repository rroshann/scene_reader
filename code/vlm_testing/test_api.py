"""
Real API test - Makes actual API calls to test all three VLMs
Requires a test image to work with
"""
import os
import sys
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

def test_openai_real(image_path):
    """Test OpenAI GPT-4V with actual API call"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        client = OpenAI(api_key=api_key)
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        print("  📤 Sending request to GPT-4V...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image for a blind person in 2-3 sentences."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }],
            max_tokens=200
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        return {
            'success': True,
            'description': description,
            'latency': latency,
            'tokens': response.usage.total_tokens if hasattr(response, 'usage') else None
        }, None
        
    except Exception as e:
        return None, str(e)

def test_google_real(image_path):
    """Test Google Gemini with actual API call"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        genai.configure(api_key=api_key)
        
        # Use fastest Flash models first (Flash is faster than Pro)
        # Available models: gemini-2.5-flash (fastest), gemini-2.0-flash, gemini-2.5-pro
        model_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-2.5-pro']
        model = None
        last_error = None
        used_model_name = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                used_model_name = model_name
                break
            except Exception as e:
                last_error = e
                continue
        
        if model is None:
            raise Exception(f"Could not find available model. Tried: {model_names}. Last error: {last_error}")
        
        print(f"  Using model: {used_model_name} (fastest)")
        
        # Load image
        img = Image.open(image_path)
        
        print("  📤 Sending request to Gemini...")
        start_time = time.time()
        
        response = model.generate_content([
            "Describe this image for a blind person in 2-3 sentences.",
            img
        ])
        
        latency = time.time() - start_time
        description = response.text
        
        return {
            'success': True,
            'description': description,
            'latency': latency
        }, None
        
    except Exception as e:
        return None, str(e)

def test_anthropic_real(image_path):
    """Test Anthropic Claude with actual API call"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        print("  📤 Sending request to Claude...")
        start_time = time.time()
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "Describe this image for a blind person in 2-3 sentences."}
                ],
            }],
        )
        
        latency = time.time() - start_time
        description = message.content[0].text
        
        return {
            'success': True,
            'description': description,
            'latency': latency
        }, None
        
    except Exception as e:
        return None, str(e)

def find_test_image():
    """Try to find a test image in common locations"""
    possible_paths = [
        Path("test_image.png"),
        Path("test_image.jpg"),
        Path("data/images/gaming") / "test.png",
    ]
    
    # Try to find any image in data folder
    data_dir = Path("data/images")
    if data_dir.exists():
        for category in ["gaming", "indoor", "outdoor", "text"]:
            cat_dir = data_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
                if images:
                    return images[0]
    
    # Check if any of the possible paths exist
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def main():
    print("=" * 60)
    print("REAL API TEST - Testing with actual API calls")
    print("=" * 60)
    print()
    
    # Get test image from command line or find one
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
    else:
        test_image = find_test_image()
    
    if not test_image or not test_image.exists():
        print("⚠️  No test image found!")
        print("   Please provide an image path as argument:")
        print("   python code/vlm_testing/test_api.py <path_to_image>")
        print()
        print("   Or place a test image in data/images/ folder")
        print("   Example: python code/vlm_testing/test_api.py data/images/gaming/test.png")
        return
    
    print(f"📷 Using test image: {test_image}")
    print()
    
    results = {}
    
    # Test OpenAI
    print("1️⃣  Testing OpenAI GPT-4V...")
    result, error = test_openai_real(test_image)
    if result:
        results['OpenAI'] = result
        print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        if result.get('tokens'):
            print(f"   📊 Tokens used: {result['tokens']}")
        print(f"   📝 Description: {result['description'][:150]}...")
    else:
        results['OpenAI'] = {'success': False, 'error': error}
        print(f"   ❌ Failed: {error}")
    print()
    
    # Test Google
    print("2️⃣  Testing Google Gemini...")
    result, error = test_google_real(test_image)
    if result:
        results['Google'] = result
        print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        print(f"   📝 Description: {result['description'][:150]}...")
    else:
        results['Google'] = {'success': False, 'error': error}
        print(f"   ❌ Failed: {error}")
    print()
    
    # Test Anthropic
    print("3️⃣  Testing Anthropic Claude...")
    result, error = test_anthropic_real(test_image)
    if result:
        results['Anthropic'] = result
        print(f"   ✅ Success! Latency: {result['latency']:.2f}s")
        print(f"   📝 Description: {result['description'][:150]}...")
    else:
        results['Anthropic'] = {'success': False, 'error': error}
        print(f"   ❌ Failed: {error}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for service, result in results.items():
        if result.get('success'):
            print(f"{service:15} ✅ Working (Latency: {result['latency']:.2f}s)")
        else:
            print(f"{service:15} ❌ Failed: {result.get('error', 'Unknown error')}")
    
    all_working = all(r.get('success', False) for r in results.values())
    
    if all_working:
        print("\n🎉 All APIs are working correctly!")
        print("\nFull descriptions:")
        print("-" * 60)
        for service, result in results.items():
            if result.get('success'):
                print(f"\n{service}:")
                print(result['description'])
    else:
        print("\n⚠️  Some APIs failed. Check errors above.")

if __name__ == "__main__":
    main()

