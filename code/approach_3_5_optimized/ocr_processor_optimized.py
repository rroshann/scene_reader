"""
Optimized OCR Processor for Approach 3.5
Uses PaddleOCR as primary (more accurate, avoids SSL issues), EasyOCR as fallback
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

# Try PaddleOCR first (primary)
PADDLEOCR_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# Try EasyOCR as fallback
EASYOCR_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class OCRProcessorOptimized:
    """
    Optimized OCR processor using PaddleOCR (primary) or EasyOCR (fallback)
    """
    
    def __init__(self, languages: List[str] = ['en'], use_paddleocr: bool = True):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (default: ['en'])
            use_paddleocr: Whether to prefer PaddleOCR (default: True)
        """
        self.languages = languages
        self.use_paddleocr = use_paddleocr
        self.ocr_engine = None
        self.engine_name = None
        
        # Try PaddleOCR first if requested
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
        
        # If both fail, raise error
        if not self.ocr_engine:
            raise ImportError(
                "No OCR engine available. Install PaddleOCR: pip install paddleocr\n"
                "Or install EasyOCR: pip install easyocr"
            )
    
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
            - 'engine': Engine used ('paddleocr' or 'easyocr')
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
            
            if self.engine_name == 'paddleocr':
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

