"""
BLIP-2 Model Implementation
Salesforce BLIP-2 for local vision-language tasks
"""
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

import torch
from local_vlm import get_device, get_model_cache_dir, format_device_name
from prompts import SIMPLE_PROMPT


class BLIP2Model:
    """BLIP-2 model wrapper for local inference"""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", cache_dir: Optional[Path] = None):
        """
        Initialize BLIP-2 model
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Custom cache directory (defaults to data/models/)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        
        self.model_name = model_name
        self.device = get_device()
        self.cache_dir = cache_dir if cache_dir else get_model_cache_dir()
        
        print(f"Loading BLIP-2 model: {model_name}")
        print(f"Device: {format_device_name(self.device)}")
        print(f"Cache directory: {self.cache_dir}")
        
        # Load processor and model
        try:
            self.processor = Blip2Processor.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ BLIP-2 model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP-2 model: {e}")
    
    def describe_image(
        self,
        image_path: Path,
        prompt: Optional[str] = None,
        max_new_tokens: int = 100,
        num_beams: int = 3
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Generate description for an image
        
        Args:
            image_path: Path to image file
            prompt: Optional text prompt (defaults to SIMPLE_PROMPT)
            max_new_tokens: Maximum number of new tokens to generate (not including prompt)
            num_beams: Number of beams for beam search (1 = greedy, 3 = default, higher = slower but better quality)
            
        Returns:
            Tuple of (result_dict, error_string)
            result_dict contains: 'description', 'latency', 'device', 'num_beams'
        """
        if prompt is None:
            prompt = SIMPLE_PROMPT
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process inputs - BLIP-2 works better with question-answer format
            # If prompt doesn't have "Answer:" format, add it
            if "Answer:" not in prompt and "answer:" not in prompt.lower():
                formatted_prompt = f"{prompt} Answer:"
            else:
                formatted_prompt = prompt
            
            inputs = self.processor(
                images=image,
                text=formatted_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate description
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False
                )
            
            latency = time.time() - start_time
            
            # Decode output - BLIP-2 returns full sequence including prompt
            # We need to extract only the generated part (after the prompt)
            full_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in full_text:
                generated_text = full_text.split("Answer:")[-1].strip()
            elif "answer:" in full_text.lower():
                # Case-insensitive split
                parts = full_text.lower().split("answer:")
                if len(parts) > 1:
                    # Find the original case version
                    idx = full_text.lower().find("answer:")
                    generated_text = full_text[idx + len("answer:"):].strip()
                else:
                    generated_text = full_text
            else:
                # If no "Answer:" found, return the full text (might be just the answer)
                generated_text = full_text
            
            return {
                'description': generated_text,
                'latency': latency,
                'device': format_device_name(self.device),
                'model': self.model_name,
                'num_beams': num_beams
            }, None
            
        except Exception as e:
            return None, f"BLIP-2 inference error: {str(e)}"


def load_blip2_model(model_name: str = "Salesforce/blip2-opt-2.7b") -> BLIP2Model:
    """
    Convenience function to load BLIP-2 model
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        BLIP2Model instance
    """
    return BLIP2Model(model_name=model_name)

