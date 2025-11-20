# SCENE READER PROJECT - DETAILED EXECUTION SCHEDULE
## Roshan Sivakumar & Dhesel Khando | DS-5690 Fall 2025

---

# WEEK 1: FOUNDATION & SETUP
**Goal:** Collect all test data, set up development environment, get proposal approved
**Total Time:** 12-15 hours (6-8 hours per person)

## Day 1: Project Kickoff (2-3 hours total)
### Joint Meeting (1 hour)
- [ ] Review approved proposal together
- [ ] Decide on communication method (Slack/Discord/WhatsApp)
- [ ] Set up shared Google Drive folder for:
  - Test images
  - Ground truth labels
  - Code repository
  - Meeting notes
- [ ] Divide Week 1 tasks
- [ ] Schedule daily 15-min check-ins for Week 1

### Individual Setup (1-2 hours each)
**Both:**
- [ ] Create accounts:
  - OpenAI (GPT-4V access)
  - Google AI Studio (Gemini API)
  - Anthropic (Claude API)
- [ ] Set up Python environment (3.10+)
- [ ] Install base packages:
  ```
  pip install openai google-generativeai anthropic pillow requests python-dotenv
  ```
- [ ] Create project folder structure:
  ```
  scene-reader/
  ├── data/
  │   ├── images/
  │   │   ├── gaming/
  │   │   ├── indoor/
  │   │   ├── outdoor/
  │   │   └── text/
  │   └── ground_truth.csv
  ├── code/
  │   ├── vlm_testing/
  │   ├── object_detection/
  │   ├── specialized/
  │   └── local_models/
  └── results/
  ```

---

## Day 2-3: Image Collection (4-6 hours total)

### Roshan: Gaming + Indoor Images (2-3 hours)
**Gaming Screenshots (15 images):**
- [ ] Hollow Knight: 8 screenshots
  - 3 navigation scenes (caves, platforms)
  - 3 combat scenes (with enemies)
  - 2 NPC interaction scenes
- [ ] Stardew Valley: 7 screenshots
  - 3 farm scenes (with objects/crops)
  - 2 town scenes (buildings, NPCs)
  - 2 indoor scenes (house, store)
- [ ] Save all as PNG, 1920x1080 or native resolution
- [ ] Name files: `game_hollowknight_nav_01.png`, etc.

**Indoor Navigation (8 images):**
- [ ] Take/find photos of:
  - 2 hallways (one empty, one with obstacles)
  - 2 doorways (open and closed)
  - 2 staircases (going up/down)
  - 2 rooms (cluttered and clean)
- [ ] Take from "eye level" perspective
- [ ] Ensure good lighting
- [ ] Name files: `indoor_hallway_01.png`, etc.

### Dhesel: Outdoor + Text Images (2-3 hours)
**Outdoor Navigation (10 images):**
- [ ] Take/find photos of:
  - 3 crosswalks/streets (different times of day)
  - 2 sidewalks (with/without obstacles)
  - 2 building entrances
  - 2 outdoor stairs/ramps
  - 1 parking lot
- [ ] Include varied conditions (sunny, cloudy)
- [ ] Name files: `outdoor_crosswalk_01.png`, etc.

**Text/Sign Reading (10 images):**
- [ ] Take/find photos of:
  - 3 street signs
  - 2 store signs
  - 2 product labels
  - 2 menus (restaurant/cafe)
  - 1 informational sign
- [ ] Ensure text is readable
- [ ] Varied fonts and sizes
- [ ] Name files: `text_sign_01.png`, etc.

**Upload all images to shared Google Drive**

---

## Day 4-5: Ground Truth Creation (3-4 hours total)

### Joint Task: Create Ground Truth Labels
**Set up spreadsheet** (30 min - one person)
Create `ground_truth.csv` with columns:
- filename
- category (gaming/indoor/outdoor/text)
- key_objects (list of important objects)
- safety_critical_elements (obstacles, hazards)
- text_content (if applicable)
- spatial_relationships (e.g., "door on left, person on right")
- description_goal (what a good description should include)

**Label images** (2-3 hours - divide work)
**Roshan labels:** Gaming + Indoor (23 images)
**Dhesel labels:** Outdoor + Text (20 images)

**For each image, document:**
1. All important objects visible
2. Any safety-critical elements (stairs, obstacles, doors)
3. Spatial layout (left/right/center/distance if applicable)
4. Text content (for text images)
5. What a "perfect" description should mention

**Example entry:**
```
filename: game_hollowknight_nav_01.png
category: gaming
key_objects: player character, platform, enemy (bug), health bar, soul meter
safety_critical_elements: pit below platform, enemy ahead
spatial_relationships: player at center-left, platform extends right, enemy 3m ahead, pit directly below
description_goal: Must mention platform, enemy location, pit danger, player health status
```

### Joint Meeting (30 min)
- [ ] Review all ground truth labels together
- [ ] Ensure consistency in labeling approach
- [ ] Finalize dataset
- [ ] Commit everything to shared drive

---

## Day 6-7: API Testing & Baseline (2-3 hours total)

### Individual: Test API Access (1 hour each)
**Both test all three APIs:**

**GPT-4V Test:**
```python
from openai import OpenAI
import base64

client = OpenAI(api_key="your-key")

# Test with one image
with open("test_image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image for a blind person."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ]
    }]
)
print(response.choices[0].message.content)
```

**Gemini Test:**
```python
import google.generativeai as genai

genai.configure(api_key="your-key")
model = genai.GenerativeModel('gemini-1.5-pro')

# Test with one image
import PIL.Image
img = PIL.Image.open("test_image.png")
response = model.generate_content(["Describe this image for a blind person.", img])
print(response.text)
```

**Claude Test:**
```python
import anthropic
import base64

client = anthropic.Anthropic(api_key="your-key")

with open("test_image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
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
            {"type": "text", "text": "Describe this image for a blind person."}
        ],
    }],
)
print(message.content[0].text)
```

- [ ] Verify all APIs work
- [ ] Document any issues
- [ ] Check API rate limits and costs

### Joint Meeting (1 hour)
- [ ] Share API test results
- [ ] Troubleshoot any issues together
- [ ] Review Week 1 completion
- [ ] Plan Week 2 division of labor
- [ ] **SUBMIT PROPOSAL** (if not done earlier)

---

# WEEK 2: BASELINE VLM TESTING & OBJECT DETECTION SETUP
**Goal:** Get all VLM baseline results + working object detection pipeline
**Total Time:** 15-18 hours (7-9 hours per person)

## ROSHAN'S TASKS: VLM Baseline Testing

### Day 1-2: GPT-4V Testing (3-4 hours)
**Create testing script** (1 hour)
```python
# vlm_tester.py
import openai
import pandas as pd
import time
import json
from pathlib import Path

def test_vlm_on_dataset(image_folder, ground_truth_csv, output_file, model="gpt-4o"):
    results = []
    df = pd.read_csv(ground_truth_csv)
    
    for idx, row in df.iterrows():
        image_path = Path(image_folder) / row['filename']
        
        # Measure latency
        start_time = time.time()
        
        # API call
        response = call_gpt4v(image_path)
        
        end_time = time.time()
        latency = end_time - start_time
        
        results.append({
            'filename': row['filename'],
            'model': model,
            'description': response,
            'latency_seconds': latency,
            'timestamp': time.time()
        })
        
        # Save incrementally
        pd.DataFrame(results).to_csv(output_file, index=False)
        
        # Rate limiting
        time.sleep(1)
    
    return pd.DataFrame(results)
```

**Run GPT-4V on all 43 images** (2-3 hours)
- [ ] Test all gaming images (15)
- [ ] Test all indoor images (8)
- [ ] Test all outdoor images (10)
- [ ] Test all text images (10)
- [ ] Save results to `results/gpt4v_baseline.csv`
- [ ] Document any errors or failures

### Day 3: Gemini Testing (2-3 hours)
- [ ] Adapt testing script for Gemini API
- [ ] Run on all 43 images
- [ ] Save results to `results/gemini_baseline.csv`
- [ ] Compare latency to GPT-4V

### Day 4: Claude Testing (2-3 hours)
- [ ] Adapt testing script for Claude API
- [ ] Run on all 43 images
- [ ] Save results to `results/claude_baseline.csv`
- [ ] Compare latency to previous models

### Day 5: Initial Analysis (1-2 hours)
- [ ] Calculate average latency for each model
- [ ] Do quick manual review of 10 random descriptions
- [ ] Note any obvious patterns or issues
- [ ] Prepare summary for mid-week check-in

**Deliverables:**
- `results/gpt4v_baseline.csv`
- `results/gemini_baseline.csv`
- `results/claude_baseline.csv`
- Summary document with initial findings

---

## DHESEL'S TASKS: Object Detection + LLM Pipeline

### Day 1-2: YOLO Setup (4-5 hours)
**Install and test YOLO** (2 hours)
```bash
pip install ultralytics opencv-python
```

```python
# test_yolo.py
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')  # nano model (fastest)

# Test on one image
results = model('test_image.png')

# Print detected objects
for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = r.names[class_id]
        confidence = float(box.conf[0])
        print(f"Detected: {class_name} (confidence: {confidence:.2f})")
```

- [ ] Download YOLOv8 models (nano, small, medium)
- [ ] Test on 5 sample images
- [ ] Verify detections make sense
- [ ] Measure inference time

**Create object detection pipeline** (2-3 hours)
```python
# object_detector.py
from ultralytics import YOLO
import time

class ObjectDetectionPipeline:
    def __init__(self, model_size='n'):
        self.model = YOLO(f'yolov8{model_size}.pt')
    
    def detect_objects(self, image_path, conf_threshold=0.25):
        start_time = time.time()
        
        results = self.model(image_path, conf=conf_threshold)
        
        detection_time = time.time() - start_time
        
        objects = []
        for r in results:
            for box in r.boxes:
                objects.append({
                    'class': r.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
        
        return {
            'objects': objects,
            'detection_time': detection_time
        }
```

- [ ] Test on all image categories
- [ ] Tune confidence threshold
- [ ] Save detection results

### Day 3-4: LLM Integration (3-4 hours)
**Create description generator** (2 hours)
```python
# description_generator.py
import openai

def generate_description_from_objects(object_list, image_category):
    prompt = f"""Based on these detected objects, create a clear description for a blind person:

Objects detected: {', '.join([obj['class'] for obj in object_list])}

Category: {image_category}

Provide a concise, helpful description focusing on:
1. Main objects and their spatial relationships
2. Any safety concerns (obstacles, hazards)
3. Actionable information

Keep it under 50 words."""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Cheaper model for this step
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

**Test full pipeline** (1-2 hours)
- [ ] Run detection → description on 10 test images
- [ ] Measure total latency (detection + generation)
- [ ] Evaluate description quality
- [ ] Debug any issues

### Day 5: Full Dataset Testing (2-3 hours)
**Run complete pipeline on all images**
- [ ] Process all 43 images
- [ ] Save results to `results/yolo_llm_baseline.csv`
- [ ] Include: detected objects, detection time, description, total latency
- [ ] Compare to VLM baseline

**Deliverables:**
- Working object detection + LLM pipeline
- `results/yolo_llm_baseline.csv`
- Performance comparison notes

---

## Mid-Week Joint Check-in (1 hour)
**Day 3 or 4 - both meet**
- [ ] Roshan shares VLM results so far
- [ ] Dhesel demos object detection pipeline
- [ ] Compare initial latency numbers
- [ ] Troubleshoot any issues
- [ ] Adjust timeline if needed

## End-of-Week Joint Meeting (1-2 hours)
**Day 6 or 7 - both meet**
- [ ] Review all baseline results
- [ ] Compare VLM vs Object Detection approaches
- [ ] Identify any data quality issues
- [ ] Plan Week 3 tasks
- [ ] Celebrate completing baseline! 🎉

---

# WEEK 3: SPECIALIZED MODELS & LOCAL MODELS
**Goal:** Implement remaining approaches, get all 4 methods tested
**Total Time:** 16-20 hours (8-10 hours per person)

## ROSHAN'S TASKS: Specialized Multi-Model Pipeline

### Day 1-2: OCR Setup (3-4 hours)
**Install EasyOCR** (30 min)
```bash
pip install easyocr
```

**Create OCR module** (2-3 hours)
```python
# ocr_module.py
import easyocr
import time

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def extract_text(self, image_path):
        start_time = time.time()
        
        results = self.reader.readtext(image_path)
        
        ocr_time = time.time() - start_time
        
        texts = [text for (bbox, text, confidence) in results if confidence > 0.5]
        
        return {
            'texts': texts,
            'full_text': ' '.join(texts),
            'ocr_time': ocr_time
        }
```

**Test OCR pipeline** (1 hour)
- [ ] Test on all 10 text images
- [ ] Evaluate accuracy vs ground truth
- [ ] Measure latency
- [ ] Note failure cases

### Day 3-4: Depth Estimation Setup (4-5 hours)
**Install depth estimation** (1 hour)
```bash
pip install torch torchvision
pip install transformers
```

**Create depth module** (2-3 hours)
```python
# depth_module.py
from transformers import pipeline
import numpy as np
import time

class DepthEstimator:
    def __init__(self):
        self.pipe = pipeline("depth-estimation", model="Intel/dpt-large")
    
    def estimate_depth(self, image_path):
        start_time = time.time()
        
        depth = self.pipe(image_path)
        
        depth_time = time.time() - start_time
        
        # Analyze depth map
        depth_array = np.array(depth['depth'])
        
        return {
            'depth_map': depth_array,
            'mean_depth': np.mean(depth_array),
            'min_depth': np.min(depth_array),
            'max_depth': np.max(depth_array),
            'depth_time': depth_time
        }
```

**Test and tune** (1-2 hours)
- [ ] Test on navigation images (indoor + outdoor)
- [ ] Verify depth estimates make sense
- [ ] Measure latency (this will be slow!)
- [ ] Consider using smaller/faster model if too slow

### Day 5-7: Integration & Testing (3-4 hours)
**Create integrated pipeline** (2 hours)
```python
# specialized_pipeline.py
from object_detector import ObjectDetectionPipeline
from ocr_module import OCRProcessor
from depth_module import DepthEstimator
import openai

class SpecializedPipeline:
    def __init__(self):
        self.detector = ObjectDetectionPipeline()
        self.ocr = OCRProcessor()
        self.depth = DepthEstimator()
    
    def analyze_image(self, image_path, category):
        # Run all models
        objects = self.detector.detect_objects(image_path)
        
        text = None
        if category == 'text':
            text = self.ocr.extract_text(image_path)
        
        depth = None
        if category in ['indoor', 'outdoor']:
            depth = self.depth.estimate_depth(image_path)
        
        # Generate comprehensive description
        description = self.generate_description(objects, text, depth, category)
        
        return {
            'objects': objects,
            'text': text,
            'depth': depth,
            'description': description
        }
```

**Test on full dataset** (1-2 hours)
- [ ] Run on all 43 images
- [ ] Save results to `results/specialized_baseline.csv`
- [ ] Document latency breakdown (detection + OCR + depth + generation)

**Deliverables:**
- Working specialized pipeline
- `results/specialized_baseline.csv`
- Notes on which components take longest

---

## DHESEL'S TASKS: Local Model Testing

### Day 1-3: LLaVA Setup (5-7 hours)
**Set up LLaVA** (3-4 hours)
```bash
pip install llava
# OR use Ollama for easier setup:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llava
```

**Test LLaVA** (2-3 hours)
```python
# local_vlm.py
import subprocess
import json
import time

class LocalVLM:
    def __init__(self, model="llava"):
        self.model = model
    
    def describe_image(self, image_path, prompt="Describe this image for a blind person."):
        start_time = time.time()
        
        # Using Ollama CLI (easier than direct LLaVA)
        cmd = [
            "ollama", "run", "llava",
            prompt,
            "--image", image_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        latency = time.time() - start_time
        
        return {
            'description': result.stdout,
            'latency': latency
        }
```

- [ ] Test on 10 sample images
- [ ] Compare quality to cloud VLMs
- [ ] Measure latency carefully
- [ ] Note hardware specs (CPU/GPU usage)

**Troubleshooting Note:** 
- If LLaVA is too complicated, pivot to testing smaller VLMs via HuggingFace
- Alternative: Use BLIP-2 (smaller, easier to set up)

### Day 4-5: Full Testing (2-3 hours)
**Run local model on all images**
- [ ] Process all 43 images
- [ ] Save to `results/local_vlm_baseline.csv`
- [ ] Compare to cloud models
- [ ] Document resource usage

### Day 6-7: Backup/Alternative Testing (2-3 hours)
**If time allows, test alternatives:**
- [ ] Try different local models (BLIP-2, MiniGPT)
- [ ] Test model size variations (7B vs 13B)
- [ ] Measure quality vs speed tradeoffs

**OR if behind schedule:**
- [ ] Help Roshan with specialized pipeline
- [ ] Start on Week 4 analysis tasks

**Deliverables:**
- Working local VLM setup
- `results/local_vlm_baseline.csv`
- Resource usage documentation

---

## Mid-Week Check-in (1 hour)
**Day 3-4**
- [ ] Share progress on specialized vs local approaches
- [ ] Troubleshoot any blocking issues
- [ ] Adjust timeline if one approach is taking too long

## End-of-Week Meeting (2 hours)
**Day 7**
- [ ] Review ALL four approaches
- [ ] Compare results side-by-side
- [ ] Identify clear winners and losers
- [ ] Start planning analysis
- [ ] HIGH FIVE - hardest week done! 🎉

---

# WEEK 4: COMPREHENSIVE ANALYSIS & MID-PROJECT CHECK-IN
**Goal:** Deep analysis, create visualizations, prepare for instructor meeting
**Total Time:** 15-18 hours (7-9 hours per person)

## Day 1-2: Quantitative Analysis (4-5 hours)

### Joint: Consolidate All Results (1 hour)
- [ ] Merge all CSV files into master dataset
- [ ] Ensure consistent formatting
- [ ] Handle any missing data

### Roshan: Latency Analysis (2-3 hours)
**Create analysis notebook:**
```python
# latency_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all results
gpt4v = pd.read_csv('results/gpt4v_baseline.csv')
gemini = pd.read_csv('results/gemini_baseline.csv')
claude = pd.read_csv('results/claude_baseline.csv')
yolo = pd.read_csv('results/yolo_llm_baseline.csv')
specialized = pd.read_csv('results/specialized_baseline.csv')
local = pd.read_csv('results/local_vlm_baseline.csv')

# Calculate statistics
stats = {
    'GPT-4V': {
        'mean': gpt4v['latency_seconds'].mean(),
        'p50': gpt4v['latency_seconds'].median(),
        'p95': gpt4v['latency_seconds'].quantile(0.95),
        'p99': gpt4v['latency_seconds'].quantile(0.99),
    },
    # ... repeat for all models
}

# Create visualizations
plt.figure(figsize=(12, 6))
plt.boxplot([gpt4v['latency_seconds'], gemini['latency_seconds'], ...])
plt.xlabel('Approach')
plt.ylabel('Latency (seconds)')
plt.title('Latency Distribution by Approach')
plt.savefig('figures/latency_boxplot.png')
```

**Deliverables:**
- [ ] Latency statistics table
- [ ] Box plot of latency distributions
- [ ] Bar chart of mean latency by approach
- [ ] Latency by category (gaming, indoor, outdoor, text)

### Dhesel: Cost Analysis (2-3 hours)
**Calculate costs:**
```python
# cost_analysis.py

API_COSTS = {
    'gpt-4o': 0.0025 / 1000,  # per token (input)
    'gemini-1.5-pro': 0.00125 / 1000,
    'claude-3-5-sonnet': 0.003 / 1000,
    'gpt-4o-mini': 0.00015 / 1000
}

def estimate_cost_per_query(model, avg_tokens_input, avg_tokens_output):
    # Calculate based on actual usage
    pass

# Calculate for 1000 queries
costs_1000 = {
    'GPT-4V': ...,
    'Gemini': ...,
    'Claude': ...,
    'YOLO+GPT': ...,  # Only LLM part
    'Specialized': ...,
    'Local': 0  # No API cost
}
```

**Deliverables:**
- [ ] Cost per query table
- [ ] Cost per 1000 queries
- [ ] Cost vs latency scatter plot
- [ ] Cost vs quality analysis

---

## Day 3-4: Qualitative Analysis (5-6 hours)

### Joint: Manual Evaluation Session (3-4 hours)
**Set up evaluation rubric:**
```
For each description, rate 1-5:
1. Completeness - Did it mention all key objects?
2. Spatial accuracy - Did it correctly describe positions?
3. Safety focus - Did it mention hazards/obstacles?
4. Clarity - Is it easy to understand?
5. Conciseness - Appropriate length?
```

**Divide and evaluate:**
- [ ] Each person evaluates 20-25 descriptions
- [ ] Score all approaches for same images
- [ ] Document specific strengths/weaknesses
- [ ] Note any hallucinations

**Create evaluation spreadsheet:**
```
image_id, approach, completeness, spatial_accuracy, safety_focus, clarity, conciseness, notes
```

### Roshan: Accuracy Analysis (1-2 hours)
**Object detection accuracy:**
- [ ] Compare detected objects to ground truth
- [ ] Calculate precision and recall
- [ ] Identify commonly missed objects
- [ ] Analyze false positives

**For text images:**
- [ ] Calculate OCR accuracy (character/word level)
- [ ] Compare across different text types

### Dhesel: Failure Mode Analysis (1-2 hours)
**Categorize failures:**
```
Failure Types:
1. Missed important objects
2. Incorrect spatial relationships
3. Hallucinated objects
4. Wrong safety assessment
5. Missed text content
6. Technical errors (API timeout, etc.)
```

- [ ] Categorize every error
- [ ] Count failures by type and approach
- [ ] Identify patterns (which approach fails how?)
- [ ] Document safety-critical failures

---

## Day 5: Visualization Creation (3-4 hours)

### Roshan: Create Comparison Charts (2 hours)
- [ ] Latency comparison (bar chart)
- [ ] Accuracy by category (grouped bar chart)
- [ ] Cost vs accuracy scatter plot
- [ ] Approach performance radar chart

### Dhesel: Create Analysis Tables (2 hours)
- [ ] Use case suitability matrix:
  ```
  |              | Gaming | Indoor Nav | Outdoor Nav | Text Reading |
  |--------------|--------|------------|-------------|--------------|
  | GPT-4V       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐    | ⭐⭐⭐⭐     | ⭐⭐⭐⭐⭐     |
  | YOLO+LLM     | ⭐⭐⭐   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐    | ⭐⭐          |
  | Specialized  | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐     |
  | Local        | ⭐⭐    | ⭐⭐       | ⭐⭐        | ⭐⭐⭐        |
  ```
- [ ] Tradeoff summary table
- [ ] Failure rate by approach

---

## Day 6: Prepare for Mid-Check-in (2-3 hours)

### Joint: Create Presentation (2 hours)
**Slide outline:**
1. Quick project recap
2. Progress summary (4 approaches implemented ✓)
3. Preliminary findings:
   - Latency results
   - Accuracy highlights
   - Cost comparison
4. Interesting discoveries
5. Challenges encountered
6. Plan for Weeks 5
7. Questions for instructor

### Individual: Prepare Talking Points (1 hour each)
- [ ] Roshan: VLM and specialized approaches
- [ ] Dhesel: Object detection and local models
- [ ] Both: Be ready to discuss tradeoffs

---

## Day 7: MID-PROJECT CHECK-IN WITH INSTRUCTOR/TA

### Meeting (30-60 min)
- [ ] Present findings so far
- [ ] Get feedback on methodology
- [ ] Ask questions about Week 5 deliverables
- [ ] Adjust timeline based on feedback

### Post-Meeting (1 hour)
- [ ] Document feedback
- [ ] Update Week 5 plan
- [ ] Celebrate being 80% done! 🎉

---

# WEEK 5: DEMO, RECOMMENDATIONS, & FINAL DELIVERABLES
**Goal:** Create demo, write deployment recommendations, finalize everything
**Total Time:** 16-20 hours (8-10 hours per person)

## Day 1-2: Build Interactive Demo (6-8 hours)

### Roshan: Create Gradio Interface (4-5 hours)
```python
# demo.py
import gradio as gr
import pandas as pd

def compare_approaches(image):
    results = {}
    
    # Run all 4 approaches
    results['GPT-4V'] = run_gpt4v(image)
    results['YOLO+LLM'] = run_yolo_llm(image)
    results['Specialized'] = run_specialized(image)
    results['Local'] = run_local(image)
    
    return results

demo = gr.Interface(
    fn=compare_approaches,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Textbox(label="GPT-4V"),
        gr.Textbox(label="YOLO+LLM"),
        gr.Textbox(label="Specialized"),
        gr.Textbox(label="Local")
    ],
    title="Scene Reader: Approach Comparison",
    description="Upload an image to compare all four approaches"
)

demo.launch()
```

**Features to implement:**
- [ ] Image upload
- [ ] Side-by-side comparison of all 4 approaches
- [ ] Show latency for each
- [ ] Show cost estimate
- [ ] Dropdown for test scenarios
- [ ] Metrics display

### Dhesel: Create Demo Assets (2-3 hours)
- [ ] Select 10 best example images
- [ ] Pre-run all approaches on examples
- [ ] Create demo script/walkthrough
- [ ] Record screen capture of demo (backup)
- [ ] Test demo thoroughly

### Joint: Polish & Test (1-2 hours)
- [ ] Test demo with fresh eyes
- [ ] Fix any bugs
- [ ] Ensure it runs smoothly
- [ ] Practice demo presentation

---

## Day 3-4: Write Deployment Recommendations (5-6 hours)

### Joint: Brainstorm Recommendations (1 hour)
**Decision tree structure:**
```
IF latency < 1s required:
    → Use YOLO+LLM or Local
ELSE IF highest accuracy needed:
    → Use GPT-4V or Specialized
ELSE IF cost-sensitive:
    → Use YOLO+LLM or Local
ELSE IF offline/privacy required:
    → Use Local
ELSE IF need OCR:
    → Use Specialized
```

### Roshan: Write Technical Recommendations (2-3 hours)
**Create document:** `deployment_guide.md`

**Sections:**
1. **Use Case: Gaming Accessibility**
   - Recommended: GPT-4V or Gemini
   - Reasoning: Best object recognition, good context understanding
   - Implementation notes
   
2. **Use Case: Real-time Navigation**
   - Recommended: YOLO+LLM (if cloud OK) or Local (if offline)
   - Reasoning: Balance of speed and accuracy
   - Safety considerations
   
3. **Use Case: Text Reading**
   - Recommended: Specialized (with OCR)
   - Reasoning: Dedicated OCR module performs best
   - Alternative: GPT-4V for complex layouts

4. **Use Case: Indoor Navigation**
   - Recommended: Specialized (with depth) or YOLO+LLM
   - Reasoning: Spatial awareness important
   - Trade-offs

5. **Use Case: Outdoor Navigation**
   - Recommended: Specialized or GPT-4V
   - Reasoning: Complex scenes need strong understanding
   - Safety-critical considerations

### Dhesel: Write Implementation Guidelines (2-3 hours)
**Create document:** `implementation_guide.md`

**Sections:**
1. **Setup Requirements**
   - Hardware requirements for each approach
   - API setup steps
   - Cost budgeting guide

2. **Integration Patterns**
   - How to integrate each approach into an app
   - Code examples
   - Error handling strategies

3. **Optimization Tips**
   - Caching strategies
   - Batch processing
   - Rate limiting

4. **Production Considerations**
   - Monitoring and logging
   - Fallback strategies
   - Cost management

5. **Future Improvements**
   - Hybrid approaches
   - Model fine-tuning
   - User feedback integration

---

## Day 5-6: Write Final Report (6-8 hours)

### Both: Outline Together (1 hour)
**Report structure:**
1. Executive Summary
2. Introduction
3. Methodology
4. Results
   - Quantitative findings
   - Qualitative findings
5. Analysis
   - Approach comparison
   - Tradeoff discussion
6. Deployment Recommendations
7. Limitations & Future Work
8. Conclusion
9. Appendices

### Divide Writing Tasks:
**Roshan writes:**
- [ ] Executive Summary (500 words)
- [ ] Introduction (800 words)
- [ ] Methodology - VLM approaches (1000 words)
- [ ] Results - Quantitative analysis (1500 words)
- [ ] Limitations (500 words)

**Dhesel writes:**
- [ ] Methodology - Detection approaches (1000 words)
- [ ] Results - Qualitative analysis (1500 words)
- [ ] Analysis section (1500 words)
- [ ] Deployment Recommendations (section 6 from above)
- [ ] Future Work (500 words)
- [ ] Conclusion (300 words)

### Joint: Edit & Integrate (2-3 hours)
- [ ] Merge all sections
- [ ] Ensure consistent style
- [ ] Add all figures and tables
- [ ] Proofread together
- [ ] Format professionally

---

## Day 7: Create Presentation (3-4 hours)

### Both: Build Slides Together (2-3 hours)
**Presentation outline (15-20 min talk):**

1. **Title Slide** (1 slide)
   - Project title, names

2. **Problem & Motivation** (2 slides)
   - 7M+ visually impaired people need help
   - Existing solutions limitations
   - Our approach

3. **Methodology** (3 slides)
   - Four approaches tested
   - Test dataset description
   - Evaluation metrics

4. **Results** (5-6 slides)
   - Latency comparison (chart)
   - Accuracy comparison (chart)
   - Cost comparison (table)
   - Use case suitability matrix
   - Failure mode analysis

5. **Key Findings** (2-3 slides)
   - Surprising discoveries
   - Clear winners/losers by scenario
   - Important tradeoffs

6. **Deployment Recommendations** (2 slides)
   - Decision tree graphic
   - Quick reference table

7. **Demo** (2-3 min live demo)
   - Show comparison interface
   - Run on 2 example images

8. **Limitations & Future Work** (1 slide)

9. **Conclusion** (1 slide)
   - Summary of contributions
   - Impact

10. **Q&A**

### Individual: Practice (1 hour each)
- [ ] Practice your sections
- [ ] Time yourself
- [ ] Prepare for questions

---

## FINAL DELIVERABLES CHECKLIST

### Code & Data:
- [ ] All source code organized and commented
- [ ] Test dataset with ground truth
- [ ] All results CSV files
- [ ] Demo application
- [ ] README with setup instructions

### Documentation:
- [ ] Final report (PDF)
- [ ] Deployment guide
- [ ] Implementation guide
- [ ] Code documentation

### Presentation:
- [ ] Presentation slides (PDF + PPT)
- [ ] Demo video (backup)
- [ ] Script/notes

### Submission:
- [ ] Upload everything to course system
- [ ] Submit to AI Showcase (optional but recommended!)
- [ ] Backup everything to personal storage

---

# WEEKLY TIME BREAKDOWN SUMMARY

| Week | Roshan Hours | Dhesel Hours | Joint Hours | Total |
|------|-------------|--------------|-------------|-------|
| 1    | 6-8         | 6-8          | 2-3         | 14-19 |
| 2    | 7-9         | 7-9          | 2-3         | 16-21 |
| 3    | 8-10        | 8-10         | 2-3         | 18-23 |
| 4    | 7-9         | 7-9          | 4-5         | 18-23 |
| 5    | 8-10        | 8-10         | 4-5         | 20-25 |
|**Total**| **36-46**   | **36-46**    | **14-19**   | **86-111** |

**Per person average:** 8-11 hours/week (totally doable!)

---

# COMMUNICATION PROTOCOL

## Daily During Active Development (Weeks 2-3):
- [ ] Quick Slack/text check-in each morning (5 min)
- [ ] Share blockers immediately
- [ ] End-of-day status update

## Regular Meetings:
- [ ] Week 1: 2 meetings (kickoff + wrap-up)
- [ ] Week 2: 2 meetings (mid-week + end)
- [ ] Week 3: 2 meetings (mid-week + end)
- [ ] Week 4: 1 formal meeting + instructor check-in
- [ ] Week 5: 3 meetings (demo planning + editing + practice)

## Emergency Protocol:
- [ ] If stuck >2 hours, message partner
- [ ] If critical blocker, schedule emergency call
- [ ] If way behind, adjust scope together (don't suffer alone!)

---

# BACKUP PLANS

## If Behind After Week 2:
**Drop:** Specialized pipeline (OCR + depth)
**Keep:** VLMs + YOLO+LLM + Local
**Still have:** 3 approaches = complete project

## If Behind After Week 3:
**Drop:** Local models
**Keep:** VLMs + YOLO+LLM
**Still have:** 2 contrasting approaches = valid comparison

## If Technical Blocker:
**YOLO won't install?** → Use pre-trained object detection API (Google Cloud Vision)
**LLaVA too complex?** → Use smaller local model via HuggingFace
**Depth estimation fails?** → Skip it, document why

## If One Person Falls Behind:
**Partner helps!** This is the advantage of teamwork
**Redistribute tasks** based on progress
**No judgment** - tech stuff breaks, that's normal

---

# SUCCESS TIPS

## For Roshan:
- Week 2 is your lighter week (just API calls) - use extra time to help Dhesel if needed
- Week 3 is heavier (specialized pipeline) - start early!
- Your analysis skills will shine in Week 4

## For Dhesel:
- Week 2 is your heavier week (YOLO setup) - start on Day 1!
- Week 3 might be tricky (local models) - don't hesitate to pivot
- Your implementation docs in Week 5 are crucial

## For Both:
- **Start every week strong** - Monday sets the tone
- **Communicate constantly** - Don't go dark
- **Celebrate small wins** - Each working approach is a victory
- **Be flexible** - Tech projects never go 100% to plan
- **Document as you go** - Don't save it all for Week 5
- **Take breaks** - Burnout helps no one

---

# YOU'VE GOT THIS! 🚀

This is ambitious but totally doable. You have:
- ✅ A partner to share the load
- ✅ Clear weekly goals
- ✅ Backup plans for everything
- ✅ A feasible timeline
- ✅ Built-in flex time

**Remember:** Even if you only complete 3 approaches, you have a complete, impressive project. The goal is rigorous comparison, not perfection on all 4 approaches.

**When in doubt, communicate!** You're a team.

Now go build something amazing! 💪
