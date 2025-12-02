"""
RAG Pipeline for Approach 6
Orchestrates VLM → Entity Extraction → Retrieval → Enhanced Generation
"""
import os
import time
import base64
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()

# Import components
from entity_extractor import extract_entities, identify_game_simple
from vector_store import initialize_vector_store
from prompts import ENHANCED_GENERATION_PROMPT, ENHANCED_SYSTEM_PROMPT

# VLM functions - directly implemented to avoid import conflicts
# Using the same logic as test_api.py

# System prompt for VLM (same as Approach 1)
VLM_SYSTEM_PROMPT = """You are a visual accessibility assistant for blind and low-vision users. When describing images, provide concise, prioritized, actionable information. Always include: (1) Spatial layout - where things are relative to the viewer (left/right/center, approximate distances), (2) Critical status - important states, conditions, or information, (3) Immediate concerns - threats, obstacles, or urgent details. Prioritize what the user needs to know to act or make decisions. Be brief, informative, and context-aware."""
VLM_USER_PROMPT = ""


def _test_openai_real(image_path, system_prompt=None, user_prompt=None):
    """Test OpenAI GPT-4V with actual API call (internal function)"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        client = OpenAI(api_key=api_key)
        
        sys_prompt = system_prompt if system_prompt is not None else VLM_SYSTEM_PROMPT
        usr_prompt = user_prompt if user_prompt is not None else (VLM_USER_PROMPT if VLM_USER_PROMPT else "Describe this image.")
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        from PIL import Image
        img = Image.open(image_path)
        format_map = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'WEBP': 'image/webp',
            'GIF': 'image/gif'
        }
        mime_type = format_map.get(img.format, 'image/png')
        
        print("  📤 Sending request to GPT-4V...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": usr_prompt}
                    ]
                }
            ],
            max_tokens=300
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


def _test_google_real(image_path):
    """Test Google Gemini with actual API call (internal function)"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        genai.configure(api_key=api_key)
        
        model_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-2.5-pro']
        model = None
        last_error = None
        used_model_name = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=VLM_SYSTEM_PROMPT
                )
                used_model_name = model_name
                break
            except Exception as e:
                last_error = e
                continue
        
        if model is None:
            raise Exception(f"Could not find available model. Tried: {model_names}. Last error: {last_error}")
        
        print(f"  Using model: {used_model_name} (fastest)")
        
        img = Image.open(image_path)
        
        print("  📤 Sending request to Gemini...")
        start_time = time.time()
        
        user_content = [img]
        if VLM_USER_PROMPT:
            user_content.insert(0, VLM_USER_PROMPT)
        
        response = model.generate_content(user_content)
        
        latency = time.time() - start_time
        description = response.text
        
        return {
            'success': True,
            'description': description,
            'latency': latency
        }, None
        
    except Exception as e:
        return None, str(e)


def _test_anthropic_real(image_path):
    """Test Anthropic Claude with actual API call (internal function)"""
    try:
        import anthropic
        from PIL import Image
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None, "API key not found"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        img = Image.open(image_path)
        format_map = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'WEBP': 'image/webp',
            'GIF': 'image/gif'
        }
        media_type = format_map.get(img.format, 'image/png')
        
        print("  📤 Sending request to Claude...")
        start_time = time.time()
        
        model_names = [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ]
        
        message = None
        last_error = None
        
        for model_name in model_names:
            try:
                user_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    }
                ]
                if VLM_USER_PROMPT:
                    user_content.append({"type": "text", "text": VLM_USER_PROMPT})
                else:
                    user_content.append({"type": "text", "text": "Describe this image."})
                
                message = client.messages.create(
                    model=model_name,
                    max_tokens=200,
                    system=VLM_SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": user_content
                    }],
                )
                print(f"  ✅ Using model: {model_name}")
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if message is None:
            raise Exception(f"Could not find working Claude model. Tried: {model_names}. Last error: {last_error}")
        
        latency = time.time() - start_time
        description = message.content[0].text
        
        return {
            'success': True,
            'description': description,
            'latency': latency
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_base_description(image_path: Path, vlm_model: str = 'gpt-4o') -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate base description using VLM (Step 1)
    
    Args:
        image_path: Path to image
        vlm_model: VLM model to use ('gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku')
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'tokens'
    """
    if vlm_model == 'gpt-4o' or 'gpt' in vlm_model.lower():
        return _test_openai_real(image_path)
    elif 'gemini' in vlm_model.lower():
        return _test_google_real(image_path)
    elif 'claude' in vlm_model.lower():
        return _test_anthropic_real(image_path)
    else:
        return None, f"Unknown VLM model: {vlm_model}"


def retrieve_context(description: str, game_name: str, query: str = None, top_k: int = 3, vector_store=None) -> Tuple[List[Dict], float]:
    """
    Retrieve relevant context from knowledge base (Step 2-3)
    
    Args:
        description: Base description
        game_name: Identified game name
        query: Optional custom query (defaults to description)
        top_k: Number of chunks to retrieve
        vector_store: VectorStore instance (creates new if None)
    
    Returns:
        Tuple of (retrieved_chunks, retrieval_latency)
    """
    if vector_store is None:
        vector_store = initialize_vector_store()
    
    start_time = time.time()
    
    # Use description as query if no custom query provided
    search_query = query if query else description
    
    # Search knowledge base
    results = vector_store.search_knowledge(
        query=search_query,
        top_k=top_k,
        game_filter=game_name
    )
    
    latency = time.time() - start_time
    
    return results, latency


def generate_enhanced_description(
    base_description: str,
    retrieved_context: List[Dict],
    game_name: str,
    model: str = 'gpt-4o-mini'
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate enhanced description combining base + context (Step 4)
    
    Args:
        base_description: Base description from VLM
        retrieved_context: Retrieved knowledge chunks
        game_name: Game name
        model: LLM model for enhancement ('gpt-4o-mini' or 'claude-haiku')
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'tokens'
    """
    try:
        # Format retrieved context
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_context)
        ])
        
        if not context_text:
            context_text = "No relevant context found."
        
        # Build prompt
        user_prompt = ENHANCED_GENERATION_PROMPT.format(
            base_description=base_description,
            retrieved_context=context_text,
            game_name=game_name or "Unknown"
        )
        
        if model == 'gpt-4o-mini' or 'gpt' in model.lower():
            from openai import OpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None, "OPENAI_API_KEY not found"
            
            client = OpenAI(api_key=api_key)
            
            start_time = time.time()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            latency = time.time() - start_time
            
            description = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                'description': description,
                'latency': latency,
                'tokens': tokens
            }, None
        
        elif 'claude' in model.lower():
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return None, "ANTHROPIC_API_KEY not found"
            
            client = anthropic.Anthropic(api_key=api_key)
            
            start_time = time.time()
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=400,
                system=ENHANCED_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            latency = time.time() - start_time
            
            description = message.content[0].text
            tokens = None  # Claude doesn't always return token usage in same format
            
            return {
                'description': description,
                'latency': latency,
                'tokens': tokens
            }, None
        
        else:
            return None, f"Unknown model: {model}"
    
    except Exception as e:
        return None, str(e)


def run_rag_pipeline(
    image_path: Path,
    vlm_model: str = 'gpt-4o',
    enhancement_model: str = 'gpt-4o-mini',
    use_llm_entity_extraction: bool = False,
    top_k: int = 3,
    vector_store=None
) -> Dict:
    """
    Run complete RAG pipeline: VLM → Entity Extraction → Retrieval → Enhancement
    
    Args:
        image_path: Path to image
        vlm_model: VLM model for base description ('gpt-4o', 'gemini-2.5-flash', 'claude-3-5-haiku')
        enhancement_model: LLM model for enhancement ('gpt-4o-mini' or 'claude-haiku')
        use_llm_entity_extraction: Whether to use LLM for entity extraction (default False, uses filename)
        top_k: Number of knowledge chunks to retrieve
        vector_store: Optional VectorStore instance (creates new if None)
    
    Returns:
        Dict with complete results:
        {
            'success': bool,
            'base_description': str,
            'enhanced_description': str,
            'game_name': str,
            'entities': list,
            'retrieved_chunks': list,
            'base_latency': float,
            'entity_extraction_latency': float,
            'retrieval_latency': float,
            'enhancement_latency': float,
            'total_latency': float,
            'base_tokens': int,
            'enhancement_tokens': int,
            'error': str (if failed)
        }
    """
    start_time = time.time()
    result = {
        'success': False,
        'base_description': None,
        'enhanced_description': None,
        'game_name': None,
        'entities': [],
        'retrieved_chunks': [],
        'base_latency': None,
        'entity_extraction_latency': None,
        'retrieval_latency': None,
        'enhancement_latency': None,
        'total_latency': None,
        'base_tokens': None,
        'enhancement_tokens': None,
        'error': None
    }
    
    try:
        # Step 1: Generate base description
        print(f"  🔍 Step 1: Generating base description with {vlm_model}...")
        base_result, error = generate_base_description(image_path, vlm_model)
        
        if error or not base_result:
            result['error'] = f"Base description failed: {error}"
            result['total_latency'] = time.time() - start_time
            return result
        
        result['base_description'] = base_result['description']
        result['base_latency'] = base_result['latency']
        result['base_tokens'] = base_result.get('tokens')
        print(f"  ✅ Base description generated in {result['base_latency']:.3f}s")
        
        # Step 2: Extract entities and identify game
        print(f"  🔍 Step 2: Extracting entities and identifying game...")
        entity_start = time.time()
        entity_result = extract_entities(
            result['base_description'],
            image_path=image_path,
            use_llm=use_llm_entity_extraction,
            model=enhancement_model
        )
        result['entity_extraction_latency'] = time.time() - entity_start
        result['game_name'] = entity_result.get('game')
        result['entities'] = entity_result.get('entities', [])
        print(f"  ✅ Identified game: {result['game_name']} (method: {entity_result.get('method', 'unknown')})")
        
        # If no game identified, skip RAG enhancement
        if not result['game_name']:
            print(f"  ⚠️  No game identified, skipping RAG enhancement")
            result['enhanced_description'] = result['base_description']
            result['success'] = True
            result['total_latency'] = time.time() - start_time
            return result
        
        # Step 3: Retrieve relevant context
        print(f"  🔍 Step 3: Retrieving relevant context (top_k={top_k})...")
        retrieved_chunks, retrieval_latency = retrieve_context(
            result['base_description'],
            result['game_name'],
            top_k=top_k,
            vector_store=vector_store
        )
        result['retrieval_latency'] = retrieval_latency
        result['retrieved_chunks'] = retrieved_chunks
        print(f"  ✅ Retrieved {len(retrieved_chunks)} chunks in {retrieval_latency:.3f}s")
        
        # Step 4: Generate enhanced description
        print(f"  🔍 Step 4: Generating enhanced description with {enhancement_model}...")
        enhanced_result, error = generate_enhanced_description(
            result['base_description'],
            retrieved_chunks,
            result['game_name'],
            model=enhancement_model
        )
        
        if error or not enhanced_result:
            result['error'] = f"Enhancement failed: {error}"
            result['enhanced_description'] = result['base_description']  # Fallback to base
            result['total_latency'] = time.time() - start_time
            return result
        
        result['enhanced_description'] = enhanced_result['description']
        result['enhancement_latency'] = enhanced_result['latency']
        result['enhancement_tokens'] = enhanced_result.get('tokens')
        print(f"  ✅ Enhanced description generated in {result['enhancement_latency']:.3f}s")
        
        result['success'] = True
        result['total_latency'] = time.time() - start_time
        print(f"  📊 Total latency: {result['total_latency']:.3f}s")
        
        return result
    
    except Exception as e:
        result['error'] = str(e)
        result['total_latency'] = time.time() - start_time
        print(f"  ❌ Pipeline failed: {e}")
        return result


def test_rag_pipeline(image_path: Path, vlm_model: str = 'gpt-4o', enhancement_model: str = 'gpt-4o-mini'):
    """
    Test the RAG pipeline on a single image
    
    Args:
        image_path: Path to test image
        vlm_model: VLM model for base description
        enhancement_model: LLM model for enhancement
    """
    print("=" * 60)
    print("Testing RAG Pipeline")
    print(f"Image: {image_path.name}")
    print(f"VLM: {vlm_model}")
    print(f"Enhancement Model: {enhancement_model}")
    print("=" * 60)
    
    result = run_rag_pipeline(image_path, vlm_model, enhancement_model)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Game: {result['game_name']}")
        print(f"Base latency: {result['base_latency']:.3f}s")
        print(f"Entity extraction latency: {result['entity_extraction_latency']:.3f}s")
        print(f"Retrieval latency: {result['retrieval_latency']:.3f}s")
        print(f"Enhancement latency: {result['enhancement_latency']:.3f}s")
        print(f"Total latency: {result['total_latency']:.3f}s")
        print(f"\nBase Description:\n{result['base_description']}")
        print(f"\nEnhanced Description:\n{result['enhanced_description']}")
    else:
        print(f"Error: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Test with a sample image
    import sys
    
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])
    else:
        # Try to find a gaming image
        test_image = Path("data/images/gaming/SlayTheSpire_Defect_vs_Sentry_ZapPlus.png")
        if not test_image.exists():
            test_image = Path("data/images/gaming/tic_tac_toe-opp_move_1.png")
    
    if not test_image.exists():
        print("Please provide an image path:")
        print("python rag_pipeline.py <path_to_image>")
        sys.exit(1)
    
    # Test with default configuration
    test_rag_pipeline(test_image, vlm_model='gpt-4o', enhancement_model='gpt-4o-mini')

