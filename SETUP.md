# Quick Setup Guide

## Step 1: Activate Virtual Environment

```bash
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template (you'll need to create this manually)
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
PROJECT_NAME=scene-reader
DATA_DIR=./data
RESULTS_DIR=./results
EOF
```

Or manually create `.env` and add:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
PROJECT_NAME=scene-reader
DATA_DIR=./data
RESULTS_DIR=./results
```

## Step 4: Get API Keys

1. **OpenAI (GPT-4V):**
   - Go to: https://platform.openai.com/api-keys
   - Create account and add payment method
   - Generate API key

2. **Google (Gemini):**
   - Go to: https://aistudio.google.com/
   - Click "Get API Key"
   - Create new API key

3. **Anthropic (Claude):**
   - Go to: https://console.anthropic.com/
   - Create account and add credits ($5 minimum)
   - Generate API key

## Step 5: Test API Access

Run a quick test to verify everything works:

```bash
python code/vlm_testing/test_api_access.py
```

## Next Steps

1. Collect test images (see DATA_COLLECTION_GUIDE.md)
2. Create ground truth labels
3. Start VLM testing (see SCHEDULE.md)

