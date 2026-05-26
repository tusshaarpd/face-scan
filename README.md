# AI Facial Stress & Wellness Analyzer

A Streamlit Cloud-ready AI wellness assistant that analyzes a **live webcam capture** or uploaded face image using **both local ML and OpenAI Vision** for indicative stress, fatigue, and recovery signals.

> This system is an AI wellness assistant and not a medical diagnostic tool. Results are estimations based on visual indicators only.

## How It Works

1. **Live capture** — take a photo with your webcam (built-in countdown timer) or upload an image.
2. **Local ML analysis** — MediaPipe face mesh + OpenCV runs instantly with no API key, extracting eye openness, under-eye darkness, forehead tension, jaw tension, blur, and lighting quality.
3. **OpenAI Vision analysis** — when an API key is provided (via sidebar input or environment variable), GPT-4o also analyses the same photo and returns a structured wellness report.
4. **Dual-mode blending** — when both run, final scores are blended: **OpenAI 60% + local ML 40%**, giving a result informed by geometric CV signals and GPT-4o's visual reasoning. A side-by-side breakdown of both sets of scores is shown in the dashboard.

## Features

- Live webcam capture with 3-second readiness countdown
- JPG, JPEG, and PNG upload support
- Dual analysis: MediaPipe + OpenCV local ML **and** GPT-4o Vision (when key provided)
- Blended scoring with ML vs AI side-by-side comparison dashboard
- Eye openness, under-eye brightness, facial asymmetry, tension proxies, blur, and lighting observations
- CrewAI-style multi-agent orchestration with graceful fallback when AI services are unavailable
- Stress, fatigue, wellness, recovery, and eye-strain gauge dashboards
- Facial overlays, radar chart, heatmap, and historical trends
- SQLite trend storage for numeric scan metrics only

## Safety And Privacy

- This is not a medical, mental-health, or diagnostic product.
- The app does not permanently store facial images by default.
- Trend tracking stores only numeric metrics, timestamps, recovery need, confidence, and summary text.
- Users must provide consent before analysis.
- Secrets are read from Streamlit secrets or environment variables and are never hardcoded.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI API Key Setup

**Option 1 — In the running app (easiest):**
Open the sidebar → paste your key into the "OpenAI API Key" field (`sk-...`). The key is stored in your browser session only and never written to disk.

**Option 2 — Environment variable:**
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
streamlit run app.py
```

**Option 3 — Streamlit Cloud secrets or `.streamlit/secrets.toml`:**
```toml
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-4o-2024-08-06"   # optional, this is the default
```

Get your key at **platform.openai.com → API Keys**.

If no API key is configured, the app still runs full local ML scoring and clearly marks AI analysis as unavailable.

## Streamlit Cloud Deployment

1. Push this project to a GitHub repository.
2. Create a Streamlit Cloud app pointed at `app.py`.
3. Add `OPENAI_API_KEY` under app secrets.
4. Deploy.

`packages.txt` includes Linux packages commonly needed by OpenCV/MediaPipe.

## Optional Heavy Integrations

The app uses OpenCV and MediaPipe as the primary production path. DeepFace, FER, `face_recognition`, and dlib are intentionally optional because dlib-based packages can be fragile on Streamlit Cloud. The code imports optional analyzers defensively and continues without them when unavailable.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── agents/
├── services/
├── utils/
├── database/
├── assets/
└── .streamlit/
```

