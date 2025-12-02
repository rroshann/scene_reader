# 🚀 Getting Started - Scene Reader Project

## Current Status ✅
- ✅ Project structure created
- ✅ Code files ready
- ✅ Dependencies installed
- ⏳ **NEXT: Set up API accounts and keys**

---

## Step 1: Set Up API Accounts & Add Credits 💰

**You're right - you need to add money/credits to the APIs!**

### Quick Summary:
1. **OpenAI (GPT-4V)**: ~$10-15 recommended
2. **Google (Gemini)**: Free tier available (50 requests/day) or ~$5
3. **Anthropic (Claude)**: $5 minimum

**Total Budget Needed: ~$15-25**

### Detailed Instructions:
👉 **See `API_SETUP_GUIDE.md` for complete step-by-step instructions**

**Quick Links:**
- OpenAI: https://platform.openai.com/api-keys
- Google: https://aistudio.google.com/
- Anthropic: https://console.anthropic.com/

---

## Step 2: Create .env File 🔑

After you get your API keys:

1. **Create `.env` file** in the project root:
   ```bash
   touch .env
   ```

2. **Add your keys** (use any text editor):
   ```bash
   OPENAI_API_KEY=sk-your-actual-key-here
   GOOGLE_API_KEY=your-actual-google-key-here
   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```

3. **Or use the helper script** (I'll create this for you):
   ```bash
   python scripts/setup_env.py
   ```

---

## Step 3: Test API Access 🧪

### Test 1: Check if keys are loaded (no API calls)
```bash
source venv/bin/activate
python code/vlm_testing/test_api_access.py
```

Expected: ✅ All APIs ready

### Test 2: Make real API calls (costs money!)
```bash
python code/vlm_testing/test_api.py test_image.png
```

This will test all 3 VLMs with a real image.

---

## Step 4: Collect Test Images 📸

You need **40 images total**:
- 10 gaming screenshots
- 10 indoor navigation scenes
- 10 outdoor navigation scenes
- 10 text/sign images

👉 **See `DATA_COLLECTION_GUIDE.md` for detailed instructions**

**Quick options:**
- **Gaming**: Steam Community screenshots (free, legal)
- **Indoor/Outdoor**: Your own photos or Unsplash (free stock photos)
- **Text**: Your own photos of signs/menus

---

## Step 5: Create Ground Truth Labels 📝

After collecting images, create `data/ground_truth.csv` with:
- Image filename
- Category
- Key objects
- Safety-critical elements
- Spatial relationships
- Text content (if applicable)

---

## Step 6: Build Batch Testing Script 🔄

I'll help you create a script that:
- Loads all images from `data/images/`
- Runs each VLM on each image
- Records latency, descriptions, costs
- Saves results to `results/raw/`

---

## Step 7: Run Full Evaluation 📊

- Process all 40 images × 3 models = 120 API calls
- Collect all results
- Calculate metrics

---

## Step 8: Analysis & Visualization 📈

- Calculate latency statistics
- Evaluate accuracy
- Compare costs
- Create visualizations
- Write report

---

## 🎯 Your Immediate Next Steps:

1. **Right Now**: Set up API accounts and add credits
   - Follow `API_SETUP_GUIDE.md`
   - Budget: $15-25 total

2. **After APIs**: Create `.env` file with keys

3. **Test**: Run `test_api_access.py` to verify

4. **Then**: Collect images (can do in parallel with API setup)

---

## 💡 Pro Tips:

- **Start with Google Gemini** - it has a free tier, so you can test without spending money
- **Set usage limits** on OpenAI to avoid overspending
- **Test with 1-2 images first** before running full batch
- **Collect images while waiting** for API approval/credits

---

## ❓ Need Help?

- Check `API_SETUP_GUIDE.md` for API setup
- Check `DATA_COLLECTION_GUIDE.md` for image collection
- Check `PROJECT.md` for full project details
- Check `SCHEDULE.md` for timeline

---

**Ready to start? Begin with Step 1: API Setup!** 🚀

