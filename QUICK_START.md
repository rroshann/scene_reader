# ⚡ Quick Start - Do This Now!

## 🎯 Right Now: Set Up APIs (15-30 minutes)

### 1. OpenAI (GPT-4V) - ~$10-15 needed
- Go to: https://platform.openai.com/api-keys
- Sign up/login
- **Add payment method** (required!)
- Create API key
- **Set usage limit** ($20 recommended)

### 2. Google (Gemini) - FREE tier available!
- Go to: https://aistudio.google.com/
- Sign in with Google
- Click "Get API Key"
- **Free tier: 50 requests/day** (might be enough!)

### 3. Anthropic (Claude) - $5 minimum
- Go to: https://console.anthropic.com/
- Sign up/login
- **Add credits** ($5 minimum)
- Create API key

---

## 🔑 After Getting Keys: Create .env File

**Option 1: Use the helper script** (easiest)
```bash
source venv/bin/activate
python scripts/setup_env.py
```

**Option 2: Manual**
```bash
# Create .env file
touch .env

# Edit it (use any text editor)
# Add your keys:
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## ✅ Test It Works

```bash
# Test 1: Check keys are loaded (no API calls, free)
python code/vlm_testing/test_api_access.py

# Test 2: Real API call (costs money!)
python code/vlm_testing/test_api.py test_image.png
```

---

## 📋 What's Next?

After APIs work:
1. ✅ Collect images (40 total)
2. ✅ Create ground truth labels
3. ✅ Build batch testing script
4. ✅ Run full evaluation

---

## 💰 Budget Summary

| Service | Cost | Status |
|---------|------|--------|
| OpenAI | $10-15 | ⏳ Need to add |
| Google | $0-5 | ✅ Free tier! |
| Anthropic | $5 | ⏳ Need to add |
| **Total** | **$15-25** | |

---

**Start with Google (it's free!) to test everything works, then add credits to the others.**

