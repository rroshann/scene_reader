"""
LLM Generation Module for Approach 2
Handles description generation using various LLM models:
- GPT-4o-mini (baseline)
- GPT-3.5-turbo (faster)
- Claude Haiku (baseline)
- Gemini Flash (text generation, fastest)
"""
import os
import time
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import prompts (use absolute import to avoid conflicts with other prompts.py files)
import sys
from pathlib import Path
import importlib.util

# Get the prompts.py file in the same directory as this file
current_dir = Path(__file__).parent
prompts_path = current_dir / "prompts.py"

# Use a unique module name to avoid conflicts
module_name = f"approach_2_yolo_llm_prompts_{id(prompts_path)}"

# Check if already loaded
if module_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(module_name, prompts_path)
    approach_2_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(approach_2_prompts)
    sys.modules[module_name] = approach_2_prompts
else:
    approach_2_prompts = sys.modules[module_name]

SYSTEM_PROMPT = approach_2_prompts.SYSTEM_PROMPT
create_user_prompt = approach_2_prompts.create_user_prompt


def generate_description_gpt4o_mini(objects_text: str, system_prompt: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description using OpenAI GPT-4o-mini
    
    Args:
        objects_text: Formatted text describing detected objects
        system_prompt: Optional custom system prompt (defaults to prompts.SYSTEM_PROMPT)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'tokens'
    """
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        
        # Use custom prompt if provided, otherwise use default
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        user_prompt = create_user_prompt(objects_text)
        
        print("  📤 Sending request to GPT-4o-mini...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        # Extract token usage if available
        tokens_used = None
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_description_claude_haiku(objects_text: str, system_prompt: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description using Anthropic Claude 3.5 Haiku
    
    Args:
        objects_text: Formatted text describing detected objects
        system_prompt: Optional custom system prompt (defaults to prompts.SYSTEM_PROMPT)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency'
    """
    try:
        import anthropic
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return None, "ANTHROPIC_API_KEY not found in environment"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Use custom prompt if provided, otherwise use default
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        user_prompt = create_user_prompt(objects_text)
        
        print("  📤 Sending request to Claude 3.5 Haiku...")
        start_time = time.time()
        
        # Try Claude 3.5 Haiku first, fallback to older models if needed
        model_names = [
            "claude-3-5-haiku-20241022",  # Latest Haiku
            "claude-3-haiku-20240307",    # Older Haiku
            "claude-3-5-sonnet-20241022"  # Fallback to Sonnet if Haiku unavailable
        ]
        
        message = None
        last_error = None
        
        for model_name in model_names:
            try:
                message = client.messages.create(
                    model=model_name,
                    max_tokens=300,
                    system=sys_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
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
        
        # Extract token usage if available
        tokens_used = None
        if hasattr(message, 'usage') and message.usage:
            tokens_used = message.usage.input_tokens + message.usage.output_tokens
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_description_gpt35_turbo(
    objects_text: str, 
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description using OpenAI GPT-3.5-turbo (faster than GPT-4o-mini)
    
    Args:
        objects_text: Formatted text describing detected objects
        system_prompt: Optional custom system prompt (defaults to prompts.SYSTEM_PROMPT)
        max_tokens: Optional max tokens override (defaults to 100)
        temperature: Optional temperature override (defaults to 0.9)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency', 'tokens'
    """
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        client = OpenAI(api_key=api_key)
        
        # Use custom prompt if provided, otherwise use default
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        user_prompt = create_user_prompt(objects_text)
        
        print("  📤 Sending request to GPT-3.5-turbo...")
        start_time = time.time()
        
        # Use overrides if provided, otherwise use defaults
        api_max_tokens = max_tokens if max_tokens is not None else 100
        api_temperature = temperature if temperature is not None else 0.9
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=api_max_tokens,
            temperature=api_temperature
        )
        
        latency = time.time() - start_time
        description = response.choices[0].message.content
        
        # Extract token usage if available
        tokens_used = None
        if hasattr(response, 'usage') and response.usage:
            tokens_used = response.usage.total_tokens
        
        return {
            'description': description,
            'latency': latency,
            'tokens': tokens_used
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_description_gemini_flash(objects_text: str, system_prompt: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description using Google Gemini Flash (text generation, fastest)
    
    Args:
        objects_text: Formatted text describing detected objects
        system_prompt: Optional custom system prompt (defaults to prompts.SYSTEM_PROMPT)
    
    Returns:
        Tuple of (result_dict, error_string)
        result_dict contains: 'description', 'latency'
    """
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return None, "GOOGLE_API_KEY not found in environment"
        
        genai.configure(api_key=api_key)
        
        # Use custom prompt if provided, otherwise use default
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        user_prompt = create_user_prompt(objects_text)
        
        # Try Gemini Flash models (fastest for text generation)
        model_names = [
            'gemini-2.5-flash',  # Latest Flash
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-2.5-pro'  # Fallback
        ]
        
        model = None
        used_model_name = None
        last_error = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=sys_prompt
                )
                used_model_name = model_name
                break
            except Exception as e:
                last_error = str(e)
                continue
        
        if model is None:
            raise Exception(f"Could not find working Gemini model. Tried: {model_names}. Last error: {last_error}")
        
        print(f"  📤 Sending request to {used_model_name}...")
        start_time = time.time()
        
        response = model.generate_content(
            user_prompt,
            generation_config={
                'max_output_tokens': 200,  # Reduced for speed
                'temperature': 0.7
            }
        )
        
        latency = time.time() - start_time
        
        # Check for safety filtering or blocked content (finish_reason 2)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                # finish_reason 2 = SAFETY (content blocked by safety filters)
                # finish_reason 3 = RECITATION (content blocked due to recitation)
                # finish_reason 4 = OTHER (other reasons)
                if finish_reason in [2, 3, 4]:
                    reason_map = {2: "SAFETY", 3: "RECITATION", 4: "OTHER"}
                    raise Exception(f"Gemini blocked content (finish_reason: {finish_reason} = {reason_map.get(finish_reason, 'UNKNOWN')}). This may be due to safety filters or content policy.")
        
        # Handle response - Gemini may return different response types
        if hasattr(response, 'text') and response.text:
            description = response.text
        elif hasattr(response, 'parts') and response.parts:
            # Some Gemini responses use parts
            description = ''.join([part.text for part in response.parts if hasattr(part, 'text')])
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle candidate-based responses
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                description = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
            else:
                raise Exception("Could not extract text from Gemini response")
        else:
            raise Exception(f"Unexpected Gemini response format: {type(response)}")
        
        if not description:
            raise Exception("Gemini returned empty description")
        
        return {
            'description': description,
            'latency': latency,
            'tokens': None  # Gemini doesn't always provide token counts
        }, None
        
    except Exception as e:
        return None, str(e)


def generate_description(
    objects_text: str, 
    llm_model: str = 'gpt-4o-mini', 
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate description using specified LLM model
    
    Args:
        objects_text: Formatted text describing detected objects
        llm_model: 'gpt-4o-mini', 'gpt-3.5-turbo', 'claude-haiku', or 'gemini-flash'
        system_prompt: Optional custom system prompt
        max_tokens: Optional max tokens override (only used for GPT-3.5-turbo currently)
        temperature: Optional temperature override (only used for GPT-3.5-turbo currently)
    
    Returns:
        Tuple of (result_dict, error_string)
    """
    model_lower = llm_model.lower()
    
    if model_lower in ['gpt-4o-mini', 'gpt4o-mini', 'openai']:
        return generate_description_gpt4o_mini(objects_text, system_prompt)
    elif model_lower in ['gpt-3.5-turbo', 'gpt35-turbo', 'gpt-3.5', 'gpt35']:
        return generate_description_gpt35_turbo(objects_text, system_prompt, max_tokens=max_tokens, temperature=temperature)
    elif model_lower in ['claude-haiku', 'claude-haiku-20241022', 'claude', 'anthropic']:
        return generate_description_claude_haiku(objects_text, system_prompt)
    elif model_lower in ['gemini-flash', 'gemini', 'google']:
        return generate_description_gemini_flash(objects_text, system_prompt)
    else:
        return None, f"Unknown LLM model: {llm_model}. Use 'gpt-4o-mini', 'gpt-3.5-turbo', 'claude-haiku', or 'gemini-flash'"

