# SCENE READER: Comparative Analysis of Vision AI for Accessibility

**Project Repository Documentation**

---

## 📋 PROJECT OVERVIEW

### Project Title
**Scene Reader: Comparative Analysis of Computer Vision Approaches for Real-Time Visual Accessibility**

### Team Members
- **Roshan Sivakumar** - roshan.sivakumar@vanderbilt.edu
- **Dhesel Khando** - dhesel.khando@vanderbilt.edu

### Course Information
- **Course:** DS-5690 - Generative AI Models in Theory and Practice
- **Semester:** Fall 2025
- **Instructor:** Jesse Spencer-Smith
- **Institution:** Vanderbilt University Data Science Institute

### Project Timeline
- **Start Date:** November 18, 2025
- **Submission Date:** December 4, 2025
- **Duration:** 16 days (compressed schedule)

---

## 🎯 PROJECT OBJECTIVES

### Primary Goal
Systematically evaluate and compare different computer vision and multimodal AI approaches to determine which provides the best balance of accuracy, speed, and usability for helping blind and low-vision users understand visual scenes in real-time.

### Research Questions
1. Which vision AI approach achieves the best latency-accuracy tradeoff for accessibility applications?
2. How do different architectures (pure VLMs vs. hybrid detection+LLM) perform across diverse scenarios?
3. What are the cost implications of each approach for practical deployment?
4. Which approach is most suitable for specific use cases (gaming, navigation, text reading)?
5. What are the failure modes and safety-critical limitations of each approach?

### Success Criteria
- ✅ Working implementations of 2-3 distinct approaches
- ✅ Comprehensive testing across 40+ diverse images
- ✅ Quantitative metrics: latency, accuracy, cost
- ✅ Qualitative analysis: failure modes, usability assessment
- ✅ Actionable deployment recommendations for developers

---

## 🔬 TECHNICAL APPROACH

### Approaches Under Comparison

#### **Approach 1: Vision-Language Models (VLMs) - BASELINE**
**Models Tested:**
- GPT-4V (OpenAI)
- Gemini 1.5 Pro (Google)
- Claude 3.5 Sonnet (Anthropic)

**Architecture:**
- End-to-end multimodal transformers
- Vision transformer (ViT) backbone + language decoder
- Cross-modal attention mechanisms
- Direct image → text generation

**Strengths:**
- Highest description quality
- Best contextual understanding
- Handles complex scenes well
- No separate components to integrate

**Weaknesses:**
- Slowest inference (2-5 seconds)
- Most expensive per query
- Requires internet/API access
- Potential hallucinations

**Use Cases:**
- Gaming accessibility (complex scenes)
- General scene understanding
- When accuracy > speed

---

#### **Approach 2: Object Detection + LLM (HYBRID) - OPTIONAL**
**Components:**
- YOLOv8 or DETR for object detection
- GPT-4o-mini for description generation

**Architecture:**
- Two-stage pipeline:
  1. Transformer-based object detector identifies objects + locations
  2. LLM synthesizes natural language description from structured data

**Strengths:**
- Faster than pure VLMs (1-2 seconds)
- More reliable object identification
- Cheaper (only LLM call costs API usage)
- Structured intermediate representation

**Weaknesses:**
- Two points of failure
- May miss contextual relationships
- Limited to pre-defined object classes
- More complex implementation

**Use Cases:**
- Indoor/outdoor navigation (obstacle detection)
- When speed matters
- Cost-sensitive applications

---

### Test Scenarios

#### **Scenario 1: Gaming Accessibility (10 images)**
**Games:** Hollow Knight, Stardew Valley
**Challenge:** Complex visual information, UI elements, character positions
**Key Elements to Identify:**
- Player character location
- Enemies/NPCs
- Interactive objects (doors, items)
- Health/status indicators
- Navigation paths

**Success Metric:** Can a blind gamer understand what's on screen and make informed decisions?

---

#### **Scenario 2: Indoor Navigation (10 images)**
**Environments:** Hallways, doorways, stairs, rooms
**Challenge:** Spatial relationships, obstacle detection, safety-critical information
**Key Elements to Identify:**
- Obstacles (furniture, people, objects)
- Doorways and exits
- Stairs (going up/down)
- Room layout and furniture placement
- Potential hazards

**Success Metric:** Can a blind person navigate safely without missing obstacles?

---

#### **Scenario 3: Outdoor Navigation (10 images)**
**Environments:** Crosswalks, sidewalks, building entrances, stairs
**Challenge:** Dynamic scenes, varying lighting, safety-critical elements
**Key Elements to Identify:**
- Crosswalk status (safe to cross?)
- Sidewalk obstacles (bikes, posts, people)
- Building entrances and steps
- Traffic and vehicles
- Directional information

**Success Metric:** Can a blind person navigate outdoor spaces safely?

---

#### **Scenario 4: Text/Sign Reading (10 images)**
**Content:** Street signs, store signs, product labels, menus
**Challenge:** Text extraction accuracy, handling varied fonts/sizes
**Key Elements to Identify:**
- Text content (accurate OCR)
- Sign purpose and meaning
- Important information (prices, directions, warnings)
- Text layout and organization

**Success Metric:** Can text be accurately read and understood?

---

## 📊 EVALUATION FRAMEWORK

### Quantitative Metrics

#### **Latency Measurements**
- **End-to-end latency:** Time from image capture to audio-ready description
- **Percentiles:** p50 (median), p95, p99
- **By scenario:** Gaming, indoor, outdoor, text
- **Target:** <2 seconds for real-time usability

**Measurement Method:**
```python
import time

start = time.time()
description = model.describe(image)
latency = time.time() - start
```

---

#### **Accuracy Metrics**
**Object Detection Accuracy:**
- Precision: % of identified objects that are correct
- Recall: % of actual objects that were identified
- F1 Score: Harmonic mean of precision and recall

**Spatial Relationship Accuracy:**
- Correct positioning (left/right/center): 0-100%
- Distance estimation (if applicable): qualitative assessment

**Text Reading Accuracy:**
- Character-level accuracy: Levenshtein distance
- Word-level accuracy: % correct words
- Semantic accuracy: Does it convey correct meaning?

**Safety-Critical Element Detection:**
- Hazard detection rate: % of obstacles/stairs identified
- False negative rate: % of hazards missed (CRITICAL metric)
- False positive rate: % of non-existent hazards reported

---

#### **Cost Analysis**
**Per-Query Cost:**
```
GPT-4V: ~$0.03-0.08 per image (depending on resolution)
Gemini 1.5 Pro: ~$0.02-0.05 per image
Claude 3.5 Sonnet: ~$0.03-0.06 per image
YOLO + GPT-4o-mini: ~$0.01 per query
```

**Cost per 1000 Queries:**
- Essential for deployment viability
- Compare against budget constraints
- Calculate break-even vs. human assistance

---

### Qualitative Metrics

#### **Description Quality (1-5 Scale)**
**Completeness:** Does it mention all important elements?
- 1 = Misses most key objects
- 3 = Mentions main objects
- 5 = Comprehensive coverage

**Clarity:** Is the description easy to understand?
- 1 = Confusing or ambiguous
- 3 = Understandable but could be clearer
- 5 = Crystal clear

**Conciseness:** Is it appropriately brief?
- 1 = Too verbose or too terse
- 3 = Acceptable length
- 5 = Perfect balance

**Actionability:** Can user make decisions based on it?
- 1 = Not useful for decision-making
- 3 = Somewhat helpful
- 5 = Enables confident action

**Safety Focus:** Does it prioritize hazards?
- 1 = Misses safety-critical info
- 3 = Mentions hazards
- 5 = Properly emphasizes dangers

---

#### **Failure Mode Analysis**
**Category 1: Missed Objects**
- Important objects not mentioned
- Frequency by approach and scenario
- Impact on usability

**Category 2: Hallucinations**
- Objects described that aren't present
- Severity: minor (extra detail) vs. critical (fake hazard)
- Frequency by model

**Category 3: Incorrect Spatial Relationships**
- "Left" vs "right" errors
- Distance errors ("close" vs "far")
- Impact on navigation safety

**Category 4: Text Misreading**
- OCR errors
- Misinterpreted signs
- Missing text entirely

**Category 5: Context Failures**
- Misunderstood scene purpose
- Missed important relationships
- Incorrect scene interpretation

---

## 🗂️ PROJECT STRUCTURE

```
scene-reader/
├── README.md                          # Project overview and setup
├── PROJECT.md                         # This file - comprehensive documentation
├── SCHEDULE.md                        # Detailed timeline and tasks
├── DATA_COLLECTION_GUIDE.md          # How to gather test images
│
├── proposal/
│   ├── Sivakumar_Khando_SceneReader_OnePage.docx
│   └── Sivakumar_Khando_SceneReader_OnePage.pdf
│
├── data/
│   ├── images/
│   │   ├── gaming/                   # 10 game screenshots
│   │   ├── indoor/                   # 10 indoor navigation scenes
│   │   ├── outdoor/                  # 10 outdoor navigation scenes
│   │   └── text/                     # 10 text/sign images
│   ├── ground_truth.csv              # Labeled test data
│   └── sources.txt                   # Image source documentation
│
├── code/
│   ├── vlm_testing/
│   │   ├── test_gpt4v.py            # GPT-4V testing script
│   │   ├── test_gemini.py           # Gemini testing script
│   │   ├── test_claude.py           # Claude testing script
│   │   └── vlm_baseline.py          # Unified VLM testing framework
│   │
│   ├── object_detection/             # OPTIONAL if time allows
│   │   ├── yolo_detector.py         # YOLO object detection
│   │   ├── llm_descriptor.py        # LLM description generation
│   │   └── hybrid_pipeline.py       # Full pipeline integration
│   │
│   ├── evaluation/
│   │   ├── calculate_metrics.py     # Quantitative analysis
│   │   ├── manual_evaluation.py     # Qualitative scoring interface
│   │   └── failure_analysis.py      # Categorize and analyze failures
│   │
│   └── utils/
│       ├── image_loader.py          # Image handling utilities
│       ├── timer.py                 # Latency measurement
│       └── cost_calculator.py       # API cost tracking
│
├── results/
│   ├── raw/
│   │   ├── gpt4v_results.csv        # Raw GPT-4V outputs
│   │   ├── gemini_results.csv       # Raw Gemini outputs
│   │   └── claude_results.csv       # Raw Claude outputs
│   │
│   ├── analysis/
│   │   ├── latency_analysis.ipynb   # Jupyter notebook for latency
│   │   ├── accuracy_analysis.ipynb  # Accuracy calculations
│   │   └── cost_analysis.ipynb      # Cost comparisons
│   │
│   └── figures/
│       ├── latency_comparison.png
│       ├── accuracy_by_scenario.png
│       ├── cost_vs_accuracy.png
│       └── failure_mode_breakdown.png
│
├── demo/                             # OPTIONAL - simple version OK
│   ├── app.py                       # Simple Gradio/Streamlit demo
│   ├── examples/                    # Pre-loaded example images
│   └── demo_video.mp4              # Screen recording backup
│
├── docs/
│   ├── deployment_guide.md          # Recommendations for developers
│   ├── implementation_guide.md      # How to integrate approaches
│   └── api_setup.md                 # API account setup instructions
│
├── report/
│   ├── final_report.pdf             # Main deliverable
│   ├── sections/                    # Individual report sections
│   └── appendices/                  # Additional materials
│
└── presentation/
    ├── slides.pptx                  # Presentation slides
    ├── slides.pdf                   # PDF version
    └── script.md                    # Presentation notes
```

---

## 🛠️ TECHNICAL STACK

### Required Tools & Libraries

#### **Python Environment**
```bash
# Python 3.10 or higher
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **Core Dependencies**
```bash
pip install openai                    # GPT-4V API
pip install google-generativeai       # Gemini API
pip install anthropic                 # Claude API
pip install pillow                    # Image processing
pip install pandas                    # Data analysis
pip install numpy                     # Numerical operations
pip install matplotlib seaborn        # Visualization
pip install jupyter                   # Analysis notebooks
pip install python-dotenv             # Environment variables
pip install requests                  # HTTP requests
```

#### **Optional Dependencies (if doing object detection)**
```bash
pip install ultralytics              # YOLOv8
pip install opencv-python            # Computer vision
pip install torch torchvision        # PyTorch (for YOLO)
```

#### **Demo Dependencies (if time allows)**
```bash
pip install gradio                   # Interactive demos
# OR
pip install streamlit                # Alternative demo framework
```

---

### API Setup

#### **OpenAI (GPT-4V)**
1. Create account: https://platform.openai.com/
2. Add payment method (required for GPT-4V)
3. Generate API key: https://platform.openai.com/api-keys
4. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

**Costs:**
- GPT-4V: $0.01 per 1K input tokens + $0.03 per 1K output tokens
- Image: ~$0.03-0.08 per image depending on resolution

---

#### **Google (Gemini)**
1. Go to: https://aistudio.google.com/
2. Click "Get API Key"
3. Create new API key
4. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="..."
   ```

**Costs:**
- Gemini 1.5 Pro: $0.00125 per 1K input tokens
- Generous free tier: 50 requests/day

---

#### **Anthropic (Claude)**
1. Create account: https://console.anthropic.com/
2. Add credits (minimum $5)
3. Generate API key
4. Set environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

**Costs:**
- Claude 3.5 Sonnet: $0.003 per 1K input tokens
- Images: ~$0.024 per image (1.15M pixels)

---

### Environment Variables Setup

Create `.env` file in project root:
```bash
# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...

# Project Configuration
PROJECT_NAME=scene-reader
DATA_DIR=./data
RESULTS_DIR=./results
```

**Load in Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
```

---

## 📝 DATA SPECIFICATION

### Image Requirements

#### **Technical Specifications**
- **Format:** PNG (preferred) or JPG (>90% quality)
- **Resolution:** Minimum 640x480, recommended 1280x720 or higher
- **Color space:** RGB (not grayscale)
- **File size:** <10MB per image
- **Lighting:** Well-lit, no extreme darkness or overexposure

#### **Content Requirements**
- **Clear subject:** Main elements in focus
- **Contextual:** Representative of real-world use cases
- **Diverse:** Variety within each category
- **Ethical:** No personal information, blurred faces if present

---

### Ground Truth Format

**File:** `data/ground_truth.csv`

**Columns:**
- `filename`: Image filename (e.g., "game_hollowknight_nav_01.png")
- `category`: gaming | indoor | outdoor | text
- `subcategory`: Specific type (e.g., "combat", "hallway", "crosswalk")
- `key_objects`: Comma-separated list of important objects
- `safety_critical`: Comma-separated list of hazards/obstacles
- `spatial_layout`: Text description of spatial relationships
- `text_content`: Actual text in image (if category=text)
- `description_requirements`: What a good description must include
- `difficulty`: easy | medium | hard

**Example Entry:**
```csv
filename,category,subcategory,key_objects,safety_critical,spatial_layout,text_content,description_requirements,difficulty
"indoor_hallway_01.png","indoor","hallway","hallway,door,person,trash_can","trash_can","Door at end of hallway (center), person walking toward camera (left side), trash can on right side","","Must mention door location, person position, trash can as obstacle",medium
```

---

## 🧪 TESTING PROTOCOL

### Phase 1: API Validation (Day 1)
**Goal:** Ensure all APIs work correctly

**Steps:**
1. Test each API with 1 sample image
2. Verify response format
3. Check latency is reasonable
4. Confirm API keys are working
5. Test error handling

**Success Criteria:**
- All 3 APIs return descriptions
- No authentication errors
- Latency <10 seconds

---

### Phase 2: Baseline Testing (Days 2-4)
**Goal:** Run all images through all VLMs

**Process:**
```python
for each image in dataset:
    for each model in [gpt4v, gemini, claude]:
        start_time = record_time()
        description = model.describe(image)
        latency = calculate_latency(start_time)
        
        save_result(
            image=image,
            model=model,
            description=description,
            latency=latency,
            timestamp=now()
        )
        
        wait(1 second)  # Rate limiting
```

**Success Criteria:**
- All 40 images × 3 models = 120 descriptions generated
- All latencies recorded
- All results saved to CSV

---

### Phase 3: Manual Evaluation (Days 5-6)
**Goal:** Qualitatively assess description quality

**Process:**
1. Load ground truth for image
2. Read VLM description
3. Score on 5 dimensions (1-5 scale each)
4. Note specific failures
5. Categorize failure type if applicable

**Division of Labor:**
- Each person evaluates 60 descriptions (20 images × 3 models)
- Cross-check 10 images for consistency

---

### Phase 4: Analysis (Days 7-10)
**Goal:** Calculate all metrics and create visualizations

**Quantitative:**
- Calculate mean, median, p95, p99 latency for each model
- Calculate accuracy metrics from manual scores
- Estimate costs based on usage
- Create comparison charts

**Qualitative:**
- Aggregate failure modes
- Identify patterns (which model fails how?)
- Document safety-critical failures
- Synthesize insights

---

## 📈 EXPECTED OUTCOMES

### Hypothesis: VLM Performance

**GPT-4V:**
- Highest accuracy (90-95% object detection)
- Slowest latency (3-5 seconds)
- Most expensive (~$0.05 per query)
- Best at complex scenes and context

**Gemini 1.5 Pro:**
- Good accuracy (85-90%)
- Medium latency (2-4 seconds)
- Medium cost (~$0.03 per query)
- Strong at diverse scenarios

**Claude 3.5 Sonnet:**
- Good accuracy (85-90%)
- Medium latency (2-4 seconds)
- Medium cost (~$0.04 per query)
- Excellent at safety-critical details

---

### Use Case Recommendations (Predicted)

**For Gaming Accessibility:**
→ **GPT-4V** (accuracy matters more than speed)

**For Real-Time Navigation:**
→ **Gemini or Claude** (balance speed and accuracy)

**For Text Reading:**
→ **Any VLM** (all perform well on text)

**For Cost-Sensitive Applications:**
→ **Gemini** (best price-performance ratio)

---

## 🚧 KNOWN LIMITATIONS

### Technical Limitations

**Latency Constraint:**
- All VLMs currently 2-5 seconds
- True real-time (<500ms) not achievable with current models
- "Near real-time" is the best we can offer

**Accuracy Challenges:**
- Hallucinations inevitable with generative models
- Spatial relationships sometimes incorrect
- Small objects may be missed

**Cost Barriers:**
- Commercial API costs add up for heavy usage
- Free tiers insufficient for production deployment

---

### Scope Limitations

**What We're NOT Doing:**
- Building production-ready system
- User testing with blind individuals
- Real-time video processing (only static images)
- Audio integration (text-to-speech)
- Mobile app development
- Custom model training

**What We ARE Doing:**
- Systematic comparison of existing approaches
- Feasibility analysis for deployment
- Recommendations for developers
- Documentation of capabilities and limitations

---

### Dataset Limitations

**Size:** 40 images (not thousands)
- Sufficient for comparison
- Not sufficient for training
- Focused on breadth over depth

**Diversity:** Limited scenarios
- 4 categories covered
- Missing: medical, workplace, transportation
- Generalizations must be cautious

**Ground Truth:** Manual labeling
- Subjective elements exist
- Inter-rater reliability not formally measured
- Assumes our labels are correct

---

## 📚 RELATED WORK & REFERENCES

### Key Papers

**Vision-Language Models:**
- "CLIP: Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
- "GPT-4V(ision) System Card" (OpenAI, 2023)

**Object Detection:**
- "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- "DETR: End-to-End Object Detection with Transformers" (Carion et al., 2020)
- "YOLOv8: Pushing the Boundaries of Real-Time Object Detection" (Ultralytics, 2023)

**Accessibility Applications:**
- "VizWiz: Nearly Real-Time Answers to Visual Questions" (Bigham et al., 2010)
- "Seeing AI: A Research Platform for Understanding Human-AI Interaction" (Microsoft, 2017)

---

### Existing Accessibility Tools

**Microsoft Seeing AI:**
- Mobile app for blind users
- Scene description, text reading, person recognition
- Not real-time, not gaming-focused

**Be My Eyes:**
- Human volunteer assistance platform
- Recently integrated GPT-4V
- High latency (requires human response)

**Envision AI:**
- Mobile app for text reading and scene understanding
- Limited gaming support
- Commercial product

**Our Contribution:**
- First systematic comparison of approaches
- Gaming accessibility focus (underexplored)
- Explicit tradeoff analysis for developers

---

## 🎓 LEARNING OBJECTIVES ALIGNMENT

### Course Goals Met

**From DS-5690 Syllabus:**

✅ **"Speak knowledgeably about generative AI, how it works"**
- Deep dive into vision-language transformers
- Understanding of multimodal architectures
- Practical experience with state-of-the-art models

✅ **"Describe succinctly architectures, algorithms and approaches"**
- Comparison of VLM architectures
- Analysis of vision transformers (ViT)
- Cross-modal attention mechanisms

✅ **"Quickly ascertain abilities and limitations of new models"**
- Systematic testing methodology
- Failure mode identification
- Performance boundary documentation

✅ **"Understand technical tradeoffs in deployed systems"**
- Latency vs. accuracy analysis
- Cost vs. performance tradeoffs
- Cloud vs. edge deployment considerations

✅ **"Demonstrate mastery through implementation and analysis"**
- Working implementations of multiple approaches
- Rigorous quantitative and qualitative evaluation
- Actionable recommendations for practitioners

---

## 🤝 TEAM RESPONSIBILITIES

### Roshan Sivakumar
**Primary Areas:**
- Gaming + Indoor image collection
- VLM testing implementation (GPT-4V, Gemini, Claude)
- Latency analysis and visualization
- Report: Introduction, Methodology (VLMs), Results (Quantitative)

**Backup Support:**
- Help with object detection if Dhesel stuck
- Review analysis code

---

### Dhesel Khando
**Primary Areas:**
- Outdoor + Text image collection
- Object detection pipeline (if time allows)
- Cost analysis and failure mode categorization
- Report: Methodology (Detection), Results (Qualitative), Analysis

**Backup Support:**
- Help with VLM testing if Roshan stuck
- Demo creation

---

### Joint Responsibilities
- Ground truth labeling (divide images)
- Manual evaluation sessions
- Final report editing and integration
- Presentation preparation and practice
- Code review and documentation

---

## 📞 COMMUNICATION PROTOCOL

### Regular Check-ins
**Daily (Nov 19-26):**
- Morning: 5-min Slack message (progress, blockers)
- Evening: Quick status update

**Meetings:**
- Nov 19 evening: Kickoff + image review
- Nov 22: Mid-week check-in (1 hour)
- Nov 25: Week 1 wrap-up (1 hour)
- Nov 27: Analysis planning (1 hour)
- Dec 1: Report review (2 hours)
- Dec 3: Presentation practice (1 hour)

### Escalation Protocol
**If Stuck (>2 hours):**
1. Google/ChatGPT for solutions (30 min)
2. Message partner
3. Schedule call if needed

**If Critical Blocker:**
- Immediately notify partner
- Assess scope adjustment
- No suffering in silence!

---

## ✅ SUBMISSION CHECKLIST

### Code Deliverables
- [ ] All source code committed to repository
- [ ] Code is documented with comments
- [ ] README.md with setup instructions
- [ ] requirements.txt with all dependencies
- [ ] .env.example showing required variables

### Data Deliverables
- [ ] 40 test images in organized folders
- [ ] ground_truth.csv with all labels
- [ ] sources.txt documenting image origins
- [ ] Sample images for demo

### Results Deliverables
- [ ] All raw results CSVs (one per model)
- [ ] Analysis notebooks (executed with outputs)
- [ ] All figures and visualizations (PNG/PDF)
- [ ] Summary statistics tables

### Documentation Deliverables
- [ ] Final report (PDF, 15-20 pages)
- [ ] Deployment recommendations guide
- [ ] Implementation guide for developers
- [ ] PROJECT.md (this file)

### Presentation Deliverables
- [ ] Presentation slides (PPTX + PDF)
- [ ] Presenter notes/script
- [ ] Demo video (backup if live demo fails)
- [ ] Example outputs to show

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Project (B Grade)
- ✅ 2 VLMs tested successfully
- ✅ 30+ images in dataset
- ✅ Basic latency and accuracy metrics
- ✅ Simple comparative analysis
- ✅ Clear presentation

### Target Project (A- to A Grade)
- ✅ 3 VLMs tested comprehensively
- ✅ 40 images across all scenarios
- ✅ Comprehensive quantitative metrics
- ✅ Detailed qualitative analysis
- ✅ Failure mode categorization
- ✅ Deployment recommendations
- ✅ Professional report and presentation

### Exceptional Project (A+ Grade)
- ✅ All of above +
- ✅ Working object detection comparison
- ✅ Interactive demo
- ✅ Novel insights or findings
- ✅ Publication-quality documentation
- ✅ Submission to AI Showcase

---

## 🚀 GETTING STARTED

### Day 1 Quick Start (Nov 19)

**Morning (Both):**
```bash
# 1. Set up project structure
mkdir scene-reader
cd scene-reader
mkdir -p data/images/{gaming,indoor,outdoor,text}
mkdir -p code results docs

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install openai google-generativeai anthropic pillow pandas

# 4. Set up .env file
touch .env
# Add your API keys to .env
```

**Afternoon (Image Collection):**
- Roshan: Gaming + Indoor
- Dhesel: Outdoor + Text
- Target: 40 images by end of day

**Evening (Meeting):**
- Upload images to shared Google Drive
- Create initial ground_truth.csv
- Test API access
- Plan tomorrow's tasks

---

## 📖 ADDITIONAL RESOURCES

### Tutorials & Documentation
- **OpenAI Vision API:** https://platform.openai.com/docs/guides/vision
- **Gemini API:** https://ai.google.dev/tutorials/python_quickstart
- **Claude Vision:** https://docs.anthropic.com/claude/docs/vision
- **YOLOv8:** https://docs.ultralytics.com/

### Datasets for Reference
- **VizWiz:** https://vizwiz.org/
- **COCO:** https://cocodataset.org/
- **ImageNet:** https://www.image-net.org/

### Tools
- **Gradio:** https://www.gradio.app/ (for demos)
- **Weights & Biases:** https://wandb.ai/ (experiment tracking)
- **Jupyter:** https://jupyter.org/ (analysis)

---

## 📄 LICENSE & ATTRIBUTION

### Project License
This project is for academic purposes only.
- Code: MIT License (if open-sourced later)
- Documentation: CC BY 4.0
- Dataset: Not for redistribution (contains third-party images)

### Image Attribution
All image sources documented in `data/sources.txt`
- Personal photos: Original work
- Stock photos: Licensed for academic use (Unsplash/Pexels)
- Game screenshots: Fair use for academic research
- Dataset images: Cited with original source

### Model Attribution
- GPT-4V: OpenAI (https://openai.com/)
- Gemini: Google DeepMind (https://deepmind.google/)
- Claude: Anthropic (https://www.anthropic.com/)

---

## 📮 CONTACT & SUPPORT

### Team Contact
- **Roshan Sivakumar:** roshan.sivakumar@vanderbilt.edu
- **Dhesel Khando:** dhesel.khando@vanderbilt.edu

### Course Staff
- **Instructor:** Jesse Spencer-Smith (jesse.spencer-smith@vanderbilt.edu)
  - Office Hours: Fridays 8am-9am
  - AI Friday Hours: 10am-12pm, Data Science Team Space

- **TA:** Shivam Tyagi (shivam.tyagi@vanderbilt.edu)
  - Office Hours: Tuesday/Thursday 3pm-4pm
  - Location: DSI Common Room

### Emergency Contacts
- DSI Help: dsi@vanderbilt.edu
- Technical Issues: Course Slack channel

---

## 🎉 ACKNOWLEDGMENTS

### Special Thanks
- Professor Jesse Spencer-Smith for project guidance
- Shivam Tyagi for technical support
- Vanderbilt Data Science Institute for resources
- Open-source community for tools and libraries

### Inspiration
This project was inspired by the need to make visual information accessible to the 7+ million blind and visually impaired individuals in the US, and the 250+ million worldwide.

---

## 📅 VERSION HISTORY

**v1.0 (Nov 18, 2025)** - Initial project setup
- Project scope defined
- Team assigned
- Documentation created

**v1.1 (Nov 19, 2025)** - Data collection phase
- Images collected
- Ground truth begun
- APIs tested

**v2.0 (Dec 4, 2025)** - Project completion
- All testing complete
- Final report submitted
- Presentation delivered

---

## 🔮 FUTURE WORK

### Potential Extensions
1. **User Study:** Test with actual blind users
2. **Real-time Video:** Extend to video streams
3. **Mobile App:** Develop smartphone application
4. **Custom Training:** Fine-tune models on accessibility data
5. **Multimodal Output:** Add audio descriptions and haptic feedback
6. **Cost Optimization:** Implement intelligent caching and batching
7. **Edge Deployment:** Port to mobile/embedded devices
8. **Additional Scenarios:** Medical, workplace, transportation contexts

---

## 📊 PROJECT STATUS

**Current Status:** 🟡 In Progress

**Phase:** Data Collection
**Progress:** 0% → 100%
**Next Milestone:** API Testing Complete
**Target Completion:** December 4, 2025

**Last Updated:** November 18, 2025, 9:00 PM EST

---

**End of PROJECT.md**

*This document is a living reference and will be updated throughout the project lifecycle.*
