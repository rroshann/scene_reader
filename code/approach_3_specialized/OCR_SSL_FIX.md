# OCR SSL Certificate Fix Guide

## Problem

EasyOCR model download fails on Mac with SSL certificate error:
```
<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>
```

This is a common issue on macOS when Python's SSL certificates are not properly configured.

## Solutions

### Solution 1: Run Python Certificate Installer (Recommended)

1. **Find your Python installation:**
   ```bash
   which python3
   ```

2. **Run the certificate installer:**
   ```bash
   /Applications/Python\ 3.*/Install\ Certificates.command
   ```
   
   Or navigate to:
   - Applications → Python 3.x → Install Certificates.command
   - Double-click to run

3. **Verify fix:**
   ```bash
   python3 -c "import ssl; print(ssl.get_default_verify_paths())"
   ```

4. **Test EasyOCR:**
   ```python
   import easyocr
   reader = easyocr.Reader(['en'])
   ```

### Solution 2: Use PaddleOCR Alternative

PaddleOCR is more accurate for English and may avoid SSL issues:

1. **Install PaddleOCR:**
   ```bash
   pip install paddleocr
   ```

2. **Modify `ocr_processor.py`:**
   - Replace EasyOCR with PaddleOCR implementation
   - PaddleOCR API is similar but may require code changes

3. **Advantages:**
   - More accurate for English text
   - May avoid SSL certificate issues
   - Better performance on some images

### Solution 3: Manual Model Download

1. **Download EasyOCR models manually:**
   - Visit: https://github.com/JaidedAI/EasyOCR/releases
   - Download model files for English

2. **Place models in:**
   ```
   ~/.EasyOCR/model/
   ```

3. **Verify:**
   ```bash
   ls ~/.EasyOCR/model/
   ```

### Solution 4: Use certifi Package

1. **Install certifi:**
   ```bash
   pip install certifi
   ```

2. **Set SSL context in code:**
   ```python
   import ssl
   import certifi
   ssl_context = ssl.create_default_context(cafile=certifi.where())
   ```

3. **Note:** This may require modifying EasyOCR's internal code

## Testing After Fix

After applying any solution, test OCR functionality:

```python
from code.approach_3_specialized.ocr_processor import OCRProcessor
from pathlib import Path

# Initialize OCR processor
ocr = OCRProcessor(languages=['en'], gpu=True)

# Test on a text image
test_image = Path("data/images/text/TEXT_MenuBoard_RestaurantPricing_MultipleSections_MountedSign.jpg")
result = ocr.extract_text(test_image)

if result.get('num_texts', 0) > 0:
    print(f"✅ OCR working! Extracted {result['num_texts']} texts")
    print(f"Text: {result['full_text']}")
else:
    print("❌ OCR still not working")
```

## Running Batch Tests After Fix

Once OCR is working, you can run the full batch test:

```bash
python3 code/approach_3_specialized/batch_test_specialized.py
```

This will test both OCR mode (text images) and Depth mode (navigation images).

## Additional Resources

- EasyOCR GitHub: https://github.com/JaidedAI/EasyOCR
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
- Python SSL Certificate Issues: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error

---

**Last Updated:** November 24, 2025

