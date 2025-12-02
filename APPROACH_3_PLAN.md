# Approach 3: Specialized Multi-Model System Implementation Plan

## Overview

**Approach 3** combines specialized models (OCR, depth estimation, object detection) with LLM generation for task-specific accuracy. Two sub-approaches:
- **3A: OCR-Enhanced System** - Text reading specialist (text images)
- **3B: Depth-Enhanced System** - Spatial specialist (navigation images)

**Architecture:** Local processing (OCR/depth/YOLO) + Cloud LLM (GPT-4o-mini)

**Target:** Maximum accuracy for specific tasks, latency 3-6 seconds expected

## Phase 0: Setup and Code Reuse Strategy

### 0.1 Establish Directory Structure
- **Directory:** `code/approach_3_specialized/` (new)
- **Files:**
  - `__init__.py` - Package initialization
  - `ocr_processor.py` - OCR module (EasyOCR)
  - `depth_estimator.py` - Depth estimation module (Depth-Anything)
  - `specialized_pipeline.py` - Main orchestrator
  - `prompts.py` - Specialized prompts for fusion
  - `batch_test_specialized.py` - Full batch testing script
  - `test_subset.py` - Subset testing script (validation)
  - `README.md` - Documentation

### 0.2 Code Reuse Strategy
- **Reuse from Approach 2:**
  - Import `yolo_detector` from `approach_2_yolo_llm` (YOLOv8 detection)
  - Import `llm_generator` from `approach_2_yolo_llm` (GPT-4o-mini generation)
  - Import `prompts` as reference (adapt for specialized fusion)
- **Rationale:** DRY principle, consistency, maintainability
- **New Components:** OCR and depth estimation (not in Approach 2)

### 0.3 Device Detection Setup
- **File:** `code/approach_3_specialized/device_utils.py` (new)
- **Reuse pattern from Approach 4:** MPS detection for M1 Mac
- **Functions:**
  - `get_device()` - Detect MPS/CUDA/CPU
  - `get_model_cache_dir()` - Project-local model storage
  - `format_device_name()` - Logging helper
- **Purpose:** Optimize for M1 Mac GPU acceleration

## Phase 1: OCR Module Implementation (3A)

### 1.1 Install Dependencies
- **Package:** `easyocr` (multi-language OCR)
- **Alternative:** `paddleocr` (more accurate for English, optional)
- **Command:** `pip install easyocr`
- **Note:** First run downloads model (~500MB), store in `data/models/easyocr/`

### 1.2 Create OCR Processor
- **File:** `code/approach_3_specialized/ocr_processor.py` (new)
- **Class:** `OCRProcessor`
- **Methods:**
  - `__init__(languages=['en'])` - Initialize EasyOCR reader
  - `extract_text(image_path)` - Extract text with bounding boxes
  - Returns: `{'texts': list, 'full_text': str, 'bboxes': list, 'confidences': list, 'ocr_latency': float}`
- **Features:**
  - Confidence threshold (default 0.5)
  - Bounding box coordinates
  - Text position tracking
  - Latency measurement
- **Error Handling:** Graceful degradation if OCR fails

### 1.3 Test OCR Module
- **Script:** `code/approach_3_specialized/test_ocr.py` (new, optional)
- **Scope:** Test on 2-3 text images from `data/images/text/`
- **Metrics:** Accuracy, latency, failure cases
- **Output:** Validation that OCR works correctly

## Phase 2: Depth Estimation Module Implementation (3B)

### 2.1 Install Dependencies
- **Package:** `transformers` (already installed)
- **Model:** Depth-Anything (via transformers pipeline)
- **Alternative models:** MiDaS, ZoeDepth (can test if Depth-Anything too slow)
- **Note:** Models auto-download on first use, store in `data/models/transformers/`

### 2.2 Create Depth Estimator
- **File:** `code/approach_3_specialized/depth_estimator.py` (new)
- **Class:** `DepthEstimator`
- **Methods:**
  - `__init__(model_name='depth-anything-v2-small-hf')` - Initialize depth pipeline
  - `estimate_depth(image_path)` - Generate depth map
  - `analyze_spatial_relationships(objects, depth_map)` - Calculate distances
  - Returns: `{'depth_map': array, 'mean_depth': float, 'spatial_info': dict, 'depth_latency': float}`
- **Features:**
  - MPS acceleration (M1 Mac)
  - Depth map analysis
  - Distance estimation for detected objects
  - Latency measurement
- **Error Handling:** Fallback if depth estimation fails

### 2.3 Test Depth Module
- **Script:** `code/approach_3_specialized/test_depth.py` (new, optional)
- **Scope:** Test on 2-3 navigation images (1 indoor + 1-2 outdoor)
- **Metrics:** Latency, depth map quality, spatial accuracy
- **Output:** Validation that depth estimation works

## Phase 3: Specialized Pipeline Integration

### 3.1 Create Fusion Prompts
- **File:** `code/approach_3_specialized/prompts.py` (new)
- **Prompts:**
  - `OCR_FUSION_SYSTEM_PROMPT` - System prompt for OCR-enhanced descriptions
  - `DEPTH_FUSION_SYSTEM_PROMPT` - System prompt for depth-enhanced descriptions
  - `create_ocr_fusion_prompt(objects, ocr_results)` - Format OCR + objects
  - `create_depth_fusion_prompt(objects, depth_info)` - Format depth + objects
- **Strategy:** Combine specialized data (OCR text, depth distances) with object detections

### 3.2 Create Main Pipeline
- **File:** `code/approach_3_specialized/specialized_pipeline.py` (new)
- **Function:** `run_specialized_pipeline(image_path, category, mode='auto')`
- **Modes:**
  - `'ocr'` - OCR-enhanced (3A) - for text images
  - `'depth'` - Depth-enhanced (3B) - for navigation images
  - `'auto'` - Auto-select based on category
- **Pipeline Flow:**
  1. Detect category (or use provided)
  2. Run YOLO detection (reuse from Approach 2)
  3. Run specialized processing (OCR or depth) in parallel with YOLO
  4. Fusion: Combine results
  5. LLM generation (reuse from Approach 2)
- **Returns:** Dict with all latencies, results, and breakdowns

### 3.3 Parallel Processing Implementation
- **Strategy:** Use `concurrent.futures.ThreadPoolExecutor` for parallel execution
- **Parallel Tasks:**
  - OCR + YOLO (for 3A)
  - Depth + YOLO (for 3B)
- **Rationale:** Reduce total latency by running independent operations simultaneously

### 3.4 Error Handling
- **Component-level:** Each component (OCR, depth, YOLO, LLM) has try-except
- **Graceful degradation:** If OCR fails, fall back to YOLO-only
- **If depth fails:** Fall back to YOLO-only
- **If LLM fails:** Return error with partial results

## Phase 4: Subset Testing (Validation)

### 4.1 Create Subset Test Script
- **File:** `code/approach_3_specialized/test_subset.py` (new)
- **Purpose:** Quick validation on small subset before full batch test
- **Strategy:**
  - Test 3A (OCR) on 3-4 text images (subset)
  - Test 3B (depth) on 4-6 navigation images (2-3 indoor + 2-3 outdoor)
  - Total: 7-10 images for quick validation
- **Configurations:**
  - 3A: YOLOv8N + EasyOCR + GPT-4o-mini
  - 3B: YOLOv8N + Depth-Anything + GPT-4o-mini
- **Output:** Console output + optional CSV for subset results
- **Purpose:** Validate pipeline works, check latencies, identify issues early

### 4.2 Run Subset Tests
- **3A Subset:** 3-4 text images
- **3B Subset:** 4-6 navigation images
- **Total:** 7-10 images
- **Expected Time:** ~2-3 minutes (quick validation)
- **Validation:**
  - Verify all components work (OCR, depth, YOLO, LLM)
  - Check latency ranges (should be 3-6s)
  - Identify any setup issues
  - Validate MPS acceleration working

### 4.3 Analyze Subset Results
- Review latency breakdowns
- Check for errors or failures
- Verify output quality
- Decide if ready for full batch test

## Phase 5: Full Batch Testing

### 5.1 Create Batch Test Script
- **File:** `code/approach_3_specialized/batch_test_specialized.py` (new)
- **Based on:** `code/approach_2_yolo_llm/batch_test_yolo_llm.py`
- **Strategy:**
  - Test 3A (OCR) on all text images (10 images)
  - Test 3B (depth) on all navigation images (20 images: 10 indoor + 10 outdoor)
  - Skip gaming images (not target for Approach 3)
- **Configurations:**
  - 3A: YOLOv8N + EasyOCR + GPT-4o-mini
  - 3B: YOLOv8N + Depth-Anything + GPT-4o-mini
- **Output:** `results/approach_3_specialized/raw/batch_results.csv`
- **Fields:** filename, category, mode (ocr/depth), all latencies (ocr/depth/detection/generation/total), results, success

### 5.2 Run Full Batch Tests
- **3A Tests:** 10 text images
- **3B Tests:** 20 navigation images
- **Total:** 30 images
- **Expected Time:** ~5-10 minutes (30 images × 3-6s each)
- **Incremental Saving:** Save after each test to prevent data loss

### 5.3 Validate Results
- Calculate latency statistics per component
- Compare OCR vs depth latencies
- Identify failure cases
- Verify category-specific performance

## Phase 6: Evaluation and Analysis

### 6.1 Create Analysis Script
- **File:** `code/evaluation/analyze_specialized_results.py` (new)
- **Metrics:**
  - Latency breakdown (OCR, depth, detection, generation, total)
  - Category-specific performance (text vs navigation)
  - Component contribution analysis
  - Failure rate per component
- **Output:** `results/approach_3_specialized/analysis/comprehensive_analysis.txt`

### 6.2 Create Comparison Script
- **File:** `code/evaluation/compare_specialized_vs_baseline.py` (new)
- **Compare:**
  - Approach 3A vs Approach 2 (text images)
  - Approach 3B vs Approach 2 (navigation images)
  - Quality improvement (OCR accuracy, depth accuracy)
  - Latency tradeoff
- **Output:** `results/approach_3_specialized/analysis/comparison.txt`

### 6.3 Create Visualizations
- **File:** `code/evaluation/create_specialized_visualizations.py` (new)
- **Plots:**
  - Latency breakdown by component (stacked bar)
  - OCR vs depth latency comparison
  - Category-specific performance
  - Comparison with Approach 2 baseline
- **Output:** `results/approach_3_specialized/figures/`

### 6.4 Statistical Tests
- **File:** `code/evaluation/statistical_tests_specialized.py` (new)
- **Tests:**
  - Paired t-test: Approach 3 vs Approach 2 (same images)
  - Component latency analysis
  - Quality improvement significance
- **Output:** `results/approach_3_specialized/analysis/statistical_tests.txt`

## Phase 7: Documentation

### 7.1 Create README
- **File:** `code/approach_3_specialized/README.md` (new)
- **Sections:**
  - Overview and architecture
  - Installation (dependencies)
  - Usage examples
  - M1 Mac optimization notes
  - Results summary
  - Troubleshooting

### 7.2 Update PROJECT.md
- **File:** `PROJECT.md`
- **Update:** Approach 3 section with status, results, findings

### 7.3 Update FINDINGS.md
- **File:** `FINDINGS.md`
- **Add:** Approach 3 results section with analysis

## Success Criteria

### Performance Targets
- **Latency:** 3-6 seconds (acceptable for specialized accuracy)
- **OCR Accuracy:** >85% text extraction accuracy
- **Depth Quality:** Reasonable depth estimates (qualitative)
- **Success Rate:** >90% (graceful degradation)

### Quality Targets
- **OCR Enhancement:** Better text reading than Approach 2
- **Depth Enhancement:** More spatial detail than Approach 2
- **Description Quality:** More accurate and detailed

## Implementation Order

1. **Phase 0:** Setup and code reuse structure
2. **Phase 1:** OCR module (3A foundation)
3. **Phase 2:** Depth module (3B foundation)
4. **Phase 3:** Pipeline integration (combine components)
5. **Phase 4:** Subset testing (validate on small set first)
6. **Phase 5:** Full batch testing (complete dataset)
7. **Phase 6:** Analysis and evaluation
8. **Phase 7:** Documentation

## Files to Create

- `code/approach_3_specialized/__init__.py`
- `code/approach_3_specialized/device_utils.py`
- `code/approach_3_specialized/ocr_processor.py`
- `code/approach_3_specialized/depth_estimator.py`
- `code/approach_3_specialized/specialized_pipeline.py`
- `code/approach_3_specialized/prompts.py`
- `code/approach_3_specialized/test_subset.py` (subset validation)
- `code/approach_3_specialized/batch_test_specialized.py` (full batch)
- `code/approach_3_specialized/test_ocr.py` (optional)
- `code/approach_3_specialized/test_depth.py` (optional)
- `code/approach_3_specialized/README.md`
- `code/evaluation/analyze_specialized_results.py`
- `code/evaluation/compare_specialized_vs_baseline.py`
- `code/evaluation/create_specialized_visualizations.py`
- `code/evaluation/statistical_tests_specialized.py`

## Files to Modify

- `PROJECT.md` - Update Approach 3 status
- `FINDINGS.md` - Add Approach 3 results
- `requirements.txt` - Add easyocr (if not already present)

## Dependencies to Add

- `easyocr>=1.7.0` - OCR processing
- `transformers>=4.35.0` - Already present, for depth estimation
- `torch>=2.0.0` - Already present, for MPS support

## Notes

- **M1 Mac Optimization:** Use MPS acceleration for depth estimation and OCR (if supported)
- **Model Storage:** Store models in `data/models/` (project-local)
- **Category-Specific Testing:** OCR for text, depth for navigation (not gaming)
- **Parallel Processing:** Run OCR/depth in parallel with YOLO to reduce latency
- **Graceful Degradation:** If specialized component fails, fall back to YOLO-only
- **Code Reuse:** Maximize reuse from Approach 2 (YOLO, LLM) to reduce duplication
- **Error Handling:** Robust error handling for each component (OCR, depth, YOLO, LLM)
- **Latency Tracking:** Track latency for each component separately for analysis
- **Subset Testing First:** Always test on subset (7-10 images) before full batch test to validate pipeline

