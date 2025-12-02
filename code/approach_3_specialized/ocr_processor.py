"""
OCR Processor for Approach 3A
Uses EasyOCR for text extraction
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not installed. Install with: pip install easyocr")


class OCRProcessor:
    """
    OCR processor using EasyOCR
    """
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU (default: True, uses MPS on M1 Mac if available)
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed. Install with: pip install easyocr")
        
        self.languages = languages
        self.gpu = gpu
        
        # Initialize EasyOCR reader
        # First run will download models (~500MB)
        print(f"  Initializing EasyOCR reader (languages: {languages})...")
        print("  Note: First run will download models (~500MB)")
        
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print(f"  ✅ EasyOCR initialized successfully")
        except Exception as e:
            error_str = str(e)
            print(f"  ⚠️  EasyOCR initialization warning: {e}")
            
            # Handle SSL certificate errors (common on Mac)
            if 'SSL' in error_str or 'CERTIFICATE' in error_str or 'certificate' in error_str.lower():
                print("  ⚠️  SSL certificate issue detected. This is common on Mac.")
                print("  💡 Solution options:")
                print("     1. Run: /Applications/Python\\ 3.*/Install\\ Certificates.command")
                print("     2. Or manually download EasyOCR models")
                print("     3. Or use alternative: pip install paddleocr (more accurate for English)")
                raise ImportError(f"EasyOCR SSL certificate issue: {e}. Please fix SSL certificates or use alternative OCR.")
            
            # Try with GPU disabled if GPU fails
            if gpu:
                print("  Retrying with GPU disabled...")
                try:
                    self.reader = easyocr.Reader(languages, gpu=False)
                    self.gpu = False
                    print(f"  ✅ EasyOCR initialized successfully (CPU mode)")
                except Exception as e2:
                    raise ImportError(f"EasyOCR initialization failed: {e2}")
            else:
                raise
    
    def extract_text(self, image_path: Path, confidence_threshold: float = 0.5) -> Dict:
        """
        Extract text from image
        
        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for text detection (default: 0.5)
        
        Returns:
            Dict with:
            - 'texts': List of detected text strings
            - 'full_text': Combined text string
            - 'bboxes': List of bounding boxes [(x1,y1,x2,y2), ...]
            - 'confidences': List of confidence scores
            - 'ocr_latency': Processing time in seconds
            - 'num_texts': Number of text detections
        """
        if not EASYOCR_AVAILABLE:
            return {
                'texts': [],
                'full_text': '',
                'bboxes': [],
                'confidences': [],
                'ocr_latency': 0.0,
                'num_texts': 0,
                'error': 'EasyOCR not installed'
            }
        
        start_time = time.time()
        
        try:
            # Read text from image
            results = self.reader.readtext(str(image_path))
            
            # Filter by confidence and extract data
            texts = []
            bboxes = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    texts.append(text)
                    bboxes.append(bbox)
                    confidences.append(confidence)
            
            ocr_latency = time.time() - start_time
            full_text = ' '.join(texts)
            
            return {
                'texts': texts,
                'full_text': full_text,
                'bboxes': bboxes,
                'confidences': confidences,
                'ocr_latency': ocr_latency,
                'num_texts': len(texts),
                'error': None
            }
            
        except Exception as e:
            ocr_latency = time.time() - start_time
            return {
                'texts': [],
                'full_text': '',
                'bboxes': [],
                'confidences': [],
                'ocr_latency': ocr_latency,
                'num_texts': 0,
                'error': str(e)
            }

