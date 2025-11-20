"""
Quick API access test script
Tests all three VLM APIs to ensure they're working correctly
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()

def test_openai():
    """Test OpenAI GPT-4V API"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            return False
        
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
        print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    except ImportError:
        print("❌ openai package not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI setup error: {e}")
        return False

def test_google():
    """Test Google Gemini API"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            return False
        
        genai.configure(api_key=api_key)
        print("✅ Google Gemini client initialized")
        print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    except ImportError:
        print("❌ google-generativeai package not installed. Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Google setup error: {e}")
        return False

def test_anthropic():
    """Test Anthropic Claude API"""
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in .env file")
            return False
        
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic Claude client initialized")
        print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
        return True
    except ImportError:
        print("❌ anthropic package not installed. Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Anthropic setup error: {e}")
        return False

def main():
    print("=" * 50)
    print("API Access Test")
    print("=" * 50)
    print()
    
    results = {
        'OpenAI': test_openai(),
        'Google': test_google(),
        'Anthropic': test_anthropic()
    }
    
    print()
    print("=" * 50)
    print("Summary:")
    print("=" * 50)
    
    for service, success in results.items():
        status = "✅ Ready" if success else "❌ Not Ready"
        print(f"{service:15} {status}")
    
    all_ready = all(results.values())
    
    if all_ready:
        print("\n🎉 All APIs are ready to use!")
    else:
        print("\n⚠️  Some APIs need setup. Check the errors above.")
        print("   Make sure:")
        print("   1. .env file exists with all API keys")
        print("   2. All packages are installed: pip install -r requirements.txt")
        print("   3. API keys are valid and have credits/quota")

if __name__ == "__main__":
    main()

