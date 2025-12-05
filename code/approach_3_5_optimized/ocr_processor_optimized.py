"""
Optimized OCR Processor for Approach 3.5
Uses Google Cloud Vision API (fast cloud) as primary, PaddleOCR/EasyOCR as fallback
"""
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try Google Cloud Vision API first (fastest, cloud-based)
GOOGLE_VISION_AVAILABLE = False
GOOGLE_VISION_REST_AVAILABLE = False
try:
    from google.cloud import vision
    import base64
    import requests
    GOOGLE_VISION_AVAILABLE = True
    GOOGLE_VISION_REST_AVAILABLE = True
except ImportError:
    try:
        import requests
        import base64
        GOOGLE_VISION_REST_AVAILABLE = True
    except ImportError:
        pass

# Try PaddleOCR as fallback (local)
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# Try EasyOCR as fallback (local)
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class OCRProcessorOptimized:
    """
    Optimized OCR processor using Google Cloud Vision API (fast cloud) or local engines (PaddleOCR/EasyOCR)
    """
    
    def __init__(self, languages: List[str] = ['en'], use_cloud: bool = True, use_paddleocr: bool = True):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (default: ['en'])
            use_cloud: Whether to prefer Google Cloud Vision API (default: True, fastest)
            use_paddleocr: Whether to prefer PaddleOCR if cloud unavailable (default: True)
        """
        self.languages = languages
        self.use_cloud = use_cloud
        self.use_paddleocr = use_paddleocr
        self.ocr_engine = None
        self.engine_name = None
        
        # Try Google Cloud Vision API first (fastest, ~0.5-1s vs 60s local)
        if use_cloud:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                # Try REST API first (simpler, uses API key)
                if GOOGLE_VISION_REST_AVAILABLE:
                    try:
                        print(f"  Initializing Google Cloud Vision API (REST, fast cloud OCR)...")
                        self.ocr_engine = {'api_key': api_key, 'method': 'rest'}
                        self.engine_name = 'google_vision'
                        print(f"  ✅ Google Cloud Vision API (REST) initialized successfully")
                        return
                    except Exception as e:
                        print(f"  ⚠️  Google Cloud Vision REST API failed: {e}")
                
                # Try client library (requires service account or application default credentials)
                if GOOGLE_VISION_AVAILABLE:
                    try:
                        print(f"  Initializing Google Cloud Vision API (client library)...")
                        self.ocr_engine = vision.ImageAnnotatorClient()
                        self.engine_name = 'google_vision'
                        print(f"  ✅ Google Cloud Vision API (client) initialized successfully")
                        return
                    except Exception as e:
                        print(f"  ⚠️  Google Cloud Vision client failed: {e}")
                        print("  Falling back to local OCR...")
            else:
                print("  ⚠️  GOOGLE_API_KEY not found, using local OCR")
        
        # Try PaddleOCR as fallback (local)
        if use_paddleocr and PADDLEOCR_AVAILABLE:
            try:
                print(f"  Initializing PaddleOCR (languages: {languages})...")
                print("  Note: First run will download models (~100-200MB)")
                # Use minimal PaddleOCR initialization (only required parameters)
                # lang parameter: 'en' for English, or use languages[0] if provided
                lang_code = languages[0] if languages else 'en'
                # Note: use_angle_cls is deprecated, but still works for compatibility
                self.ocr_engine = PaddleOCR(lang=lang_code)
                self.engine_name = 'paddleocr'
                print(f"  ✅ PaddleOCR initialized successfully")
                return
            except Exception as e:
                print(f"  ⚠️  PaddleOCR initialization failed: {e}")
                print("  Falling back to EasyOCR...")
        
        # Fallback to EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                print(f"  Initializing EasyOCR (languages: {languages})...")
                print("  Note: First run will download models (~500MB)")
                self.ocr_engine = easyocr.Reader(languages, gpu=True)
                self.engine_name = 'easyocr'
                print(f"  ✅ EasyOCR initialized successfully")
                return
            except Exception as e:
                error_str = str(e)
                print(f"  ⚠️  EasyOCR initialization warning: {e}")
                
                # Handle SSL certificate errors
                if 'SSL' in error_str or 'CERTIFICATE' in error_str or 'certificate' in error_str.lower():
                    print("  ⚠️  SSL certificate issue detected.")
                    print("  💡 Solution: Run /Applications/Python\\ 3.*/Install\\ Certificates.command")
                
                # Try with GPU disabled
                try:
                    self.ocr_engine = easyocr.Reader(languages, gpu=False)
                    self.engine_name = 'easyocr'
                    print(f"  ✅ EasyOCR initialized successfully (CPU mode)")
                    return
                except Exception as e2:
                    print(f"  ❌ EasyOCR initialization failed: {e2}")
        
        # If all fail, raise error
        if not self.ocr_engine:
            error_msg = "No OCR engine available.\n"
            if not GOOGLE_VISION_AVAILABLE:
                error_msg += "Install Google Cloud Vision: pip install google-cloud-vision\n"
            if not PADDLEOCR_AVAILABLE:
                error_msg += "Or install PaddleOCR: pip install paddleocr\n"
            if not EASYOCR_AVAILABLE:
                error_msg += "Or install EasyOCR: pip install easyocr\n"
            raise ImportError(error_msg)
    
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
            - 'engine': Engine used ('google_vision', 'paddleocr', or 'easyocr')
            - 'error': Error message if failed
        """
        if not self.ocr_engine:
            return {
                'texts': [],
                'full_text': '',
                'bboxes': [],
                'confidences': [],
                'ocr_latency': 0.0,
                'num_texts': 0,
                'engine': None,
                'error': 'OCR engine not initialized'
            }
        
        start_time = time.time()
        
        try:
            texts = []
            bboxes = []
            confidences = []
            
            if self.engine_name == 'google_vision':
                # Google Cloud Vision API format (fastest, ~0.5-1s)
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                
                # Check if using REST API or client library
                if isinstance(self.ocr_engine, dict) and self.ocr_engine.get('method') == 'rest':
                    # REST API method (uses API key)
                    api_key = self.ocr_engine['api_key']
                    image_b64 = base64.b64encode(content).decode('utf-8')
                    
                    url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
                    payload = {
                        'requests': [{
                            'image': {'content': image_b64},
                            'features': [{'type': 'TEXT_DETECTION'}]
                        }]
                    }
                    
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if 'responses' in result and len(result['responses']) > 0:
                        annotations = result['responses'][0].get('textAnnotations', [])
                        # First annotation is full text, rest are words
                        for annotation in annotations[1:]:
                            text = annotation.get('description', '')
                            if text and text.strip():
                                texts.append(text.strip())
                                confidences.append(1.0)  # Google Vision doesn't provide per-word confidence
                                
                                # Get bounding box
                                vertices = annotation.get('boundingPoly', {}).get('vertices', [])
                                if vertices:
                                    x_coords = [v.get('x', 0) for v in vertices]
                                    y_coords = [v.get('y', 0) for v in vertices]
                                    if x_coords and y_coords:
                                        bboxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                                    else:
                                        bboxes.append([0, 0, 0, 0])
                                else:
                                    bboxes.append([0, 0, 0, 0])
                else:
                    # Client library method (requires service account)
                    image = vision.Image(content=content)
                    response = self.ocr_engine.text_detection(image=image)
                    
                    if response.text_annotations:
                        # First annotation is the full text, rest are individual words
                        for annotation in response.text_annotations[1:]:  # Skip first (full text)
                            text = annotation.description
                            confidence = 1.0  # Google Vision doesn't provide per-word confidence
                            
                            if text and text.strip():
                                texts.append(text.strip())
                                confidences.append(confidence)
                                
                                # Get bounding box
                                vertices = annotation.bounding_poly.vertices
                                if vertices:
                                    x_coords = [v.x for v in vertices if v.x is not None]
                                    y_coords = [v.y for v in vertices if v.y is not None]
                                    if x_coords and y_coords:
                                        bboxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                                    else:
                                        bboxes.append([0, 0, 0, 0])
                                else:
                                    bboxes.append([0, 0, 0, 0])
            
            elif self.engine_name == 'paddleocr':
                # PaddleOCR format (new API uses predict() method)
                # Returns list of OCRResult objects
                results = self.ocr_engine.predict(str(image_path))
                
                if results and len(results) > 0:
                    ocr_result = results[0]  # Get first (and usually only) result
                    rec_texts = ocr_result.get('rec_texts', [])
                    rec_scores = ocr_result.get('rec_scores', [])
                    rec_polys = ocr_result.get('rec_polys', [])
                    
                    # Ensure arrays have same length
                    min_length = min(len(rec_texts), len(rec_scores), len(rec_polys) if rec_polys else len(rec_texts))
                    
                    # Process each detected text
                    for i in range(min_length):
                        text = rec_texts[i] if i < len(rec_texts) else ""
                        confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                        
                        if confidence >= confidence_threshold and text:
                            texts.append(text)
                            confidences.append(float(confidence))
                            
                            # Convert polygon to bounding box [x1, y1, x2, y2]
                            if i < len(rec_polys) and rec_polys[i] is not None:
                                poly = rec_polys[i]
                                if len(poly) > 0:
                                    x_coords = [point[0] for point in poly]
                                    y_coords = [point[1] for point in poly]
                                    bboxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                                else:
                                    bboxes.append([0, 0, 0, 0])
                            else:
                                bboxes.append([0, 0, 0, 0])
            
            elif self.engine_name == 'easyocr':
                # EasyOCR format
                results = self.ocr_engine.readtext(str(image_path))
                
                for (bbox, text, confidence) in results:
                    if confidence >= confidence_threshold:
                        texts.append(text)
                        # Convert bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> [x1,y1,x2,y2]
                        if len(bbox) >= 2:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            bboxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                        else:
                            bboxes.append([0, 0, 0, 0])
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
                'engine': self.engine_name,
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
                'engine': self.engine_name,
                'error': str(e)
            }

