# API Setup Guide - Step by Step

## 🚨 IMPORTANT: You need to add credits/payment to use these APIs!

---

## Step 1: OpenAI (GPT-4V) Setup

### 1.1 Create Account & Add Payment
1. Go to: https://platform.openai.com/
2. Sign up or log in
3. **Add Payment Method** (required for GPT-4V):
   - Click on your profile → Billing
   - Add credit card or payment method
   - **Minimum:** $5 recommended to start
   - **Cost:** ~$0.03-0.08 per image (varies by resolution)

### 1.2 Get API Key
1. Go to: https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Name it: "Scene Reader Project"
4. **Copy the key immediately** (you won't see it again!)
5. Format: `sk-proj-...` (starts with `sk-`)

### 1.3 Set Usage Limits (Optional but Recommended)
1. Go to: https://platform.openai.com/account/billing/limits
2. Set a hard limit (e.g., $20) to avoid overspending
3. You can always increase it later

**Estimated Cost for Project:**
- 40 images × 3 models = 120 API calls
- At ~$0.05 per call = **~$6 total** (just for GPT-4V)
- Add buffer: **$10-15 recommended**

---

## Step 2: Google AI Studio (Gemini) Setup

### 2.1 Create Account
1. Go to: https://aistudio.google.com/
2. Sign in with your Google account
3. Accept terms of service

### 2.2 Get API Key
1. Click "Get API Key" button
2. Create new API key
3. Copy the key
4. Format: Usually a long string (no prefix)

### 2.3 Check Free Tier
- **Free tier:** 50 requests/day (generous!)
- **Paid tier:** $0.00125 per 1K input tokens (very cheap)
- For this project, free tier might be enough!

**Estimated Cost:**
- If using free tier: **$0**
- If exceeding: Very cheap, maybe $1-2 total

---

## Step 3: Anthropic (Claude) Setup

### 3.1 Create Account & Add Credits
1. Go to: https://console.anthropic.com/
2. Sign up or log in
3. **Add Credits** (required):
   - Click "Add Credits" or "Billing"
   - **Minimum:** $5 required
   - Add payment method
   - Purchase credits

### 3.2 Get API Key
1. Go to: https://console.anthropic.com/settings/keys
2. Click "Create Key"
3. Name it: "Scene Reader"
4. Copy the key
5. Format: `sk-ant-...` (starts with `sk-ant-`)

**Estimated Cost:**
- ~$0.003 per 1K tokens
- ~$0.024 per image
- 40 images × 3 = 120 calls ≈ **$3-5 total**

---

## Step 4: Create .env File

### 4.1 Copy Template
```bash
cp .env.example .env
```

### 4.2 Edit .env File
Open `.env` in a text editor and replace the placeholder values:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
GOOGLE_API_KEY=your-actual-google-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
```

**⚠️ IMPORTANT:**
- Never commit `.env` to git (it's in .gitignore)
- Keep your keys secret
- Don't share them publicly

---

## Step 5: Test Your Setup

### 5.1 Test API Access (No API Calls)
```bash
python code/vlm_testing/test_api_access.py
```

Expected output:
```
✅ OpenAI client initialized
✅ Google Gemini client initialized
✅ Anthropic Claude client initialized
```

### 5.2 Test Real API Calls (Uses Credits!)
```bash
python code/vlm_testing/test_api.py test_image.png
```

This will:
- Make actual API calls (costs money!)
- Test all 3 VLMs
- Show latency and descriptions
- Verify everything works

---

## 💰 Total Estimated Costs

| Service | Cost per Image | Total (120 calls) | Recommended Budget |
|---------|---------------|-------------------|---------------------|
| **OpenAI GPT-4V** | $0.03-0.08 | $3.60-9.60 | **$10-15** |
| **Google Gemini** | Free tier | $0 (or $1-2) | **$0-5** |
| **Anthropic Claude** | $0.024 | $2.88 | **$5** |
| **TOTAL** | | | **$15-25** |

**Note:** These are estimates. Actual costs depend on:
- Image resolution
- Response length
- API pricing changes
- Testing/retries

---

## 🚨 Troubleshooting

### "API key not found"
- Make sure `.env` file exists in project root
- Check that keys are correct (no extra spaces)
- Restart terminal/IDE after creating .env

### "Insufficient credits"
- Add more credits to your account
- Check billing page for balance

### "Rate limit exceeded"
- Wait a few minutes and try again
- Check your API usage limits
- Google has 50 requests/day free tier

### "Model not found" (Google)
- Some models may be unavailable
- The script tries multiple models automatically
- Check: https://aistudio.google.com/app/apikey

---

## ✅ Checklist

- [ ] OpenAI account created
- [ ] Payment method added to OpenAI ($10-15)
- [ ] OpenAI API key obtained
- [ ] Google AI Studio account created
- [ ] Google API key obtained
- [ ] Anthropic account created
- [ ] Credits added to Anthropic ($5)
- [ ] Anthropic API key obtained
- [ ] `.env` file created with all keys
- [ ] `test_api_access.py` runs successfully
- [ ] `test_api.py` runs successfully (makes real API calls)

---

## 🎯 Next Steps After API Setup

1. ✅ APIs are working → **DONE!**
2. Collect test images (40 total)
3. Create ground truth labels
4. Build batch testing script
5. Run full evaluation

---

**Questions?** Check the main README.md or PROJECT.md for more details.

