# EcoScan ♻️

A lightweight, beginner-friendly demo app that helps you identify household waste items and shows how to dispose of them responsibly.  
Built with a **Vision Transformer (ViT)** image classifier, and simple **Tailwind CSS** templates for the UI.  

> This repository contains the full learning journey: notebooks for experimentation, a minimal demo interface, and example assets.

---

## Demo

- **Open the HTML interface locally** → upload or drag & drop a photo (e.g., a plastic bottle).  
- The **model notebook** runs inference and saves results.  
- The **results page** displays the predicted material + quick recycling guidance.  
-  A button can link to Google Maps to find the closest recycling point.  

---

## Project Overview

EcoScan aims to nudge travelers and residents toward better waste sorting by making it **fast** and **friendly**:
- Snap or upload a photo → get a **predicted category** (plastic, paper, glass, metal, cardboard, etc.).
- Show **plain-language guidance** (what goes / doesn’t go).
- Keep the stack **simple**: just notebooks, a lightweight model, and basic HTML templates (no backend needed).

---

## Repository Structure

```
ecoscan/
├── app/                   
│   ├── templates/          # Jinja2 HTML (Tailwind via CDN)
│   ├── static/             # app assets
├── data/                   # sample images / metadata for tests
├── models/                 # model weights / artifacts (excluded from git if large)
├── notebooks/              # experiments (EDA, training, inference tests)
├── server/                 # app.py
├── requirements.txt        # Python dependencies
└── README.md               # this file
```

---

## Local Setup

1. **Clone & enter the project**
   ```bash
   git clone https://github.com/<your-username>/ecoscan.git
   cd ecoscan
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**
   - Place model files under `models/` or let the notebook pull them from Hugging Face the first time.

---

## How to Run

There are **two ways** to try EcoScan locally:

### 1. From Jupyter Notebooks
- Open the notebook in `notebooks/`.  
- Run the inference cells with your own image.  
- The output will show the predicted label + confidence score.  

### 2. From the Demo HTML
- Open `app/templates/index.html` in your browser.  
- Upload a sample image from `data/`.  
- View the **results page** for the prediction and recycling guidance.  

---

## Example Output

```json
{
  "label": "plastic",
  "confidence": 0.92,
  "guidance": "Rinse the bottle and remove the cap. Check local rules for film plastics."
}
```

---

## Notes

- **Models and datasets** are excluded from the repo to keep it lightweight.  
- Add them manually in `models/` or `data/` if you want to retrain or test.  
- Tailwind is included via CDN, so no build step is required.  
