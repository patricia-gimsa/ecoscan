# app.py — EcoScan Local API (relative paths + energy/confidence reject)
# ---------------------------------------------------------------------
# Exposes:
#   GET /health   -> quick status & threshold info
#   POST /predict -> send image; returns {label, confidence, rejected, debug}
#
# Reject logic supported:
#   1) Plain confidence threshold:
#        - JSON is a number (e.g., 0.65), or {"threshold": 0.65}
#        -> reject if confidence < threshold
#   2) Energy-based threshold (your case):
#        - JSON like:
#            {
#              "method": "energy_T1.0_target_ood_reject",
#              "reject_threshold": -4.2220,
#              "rule": "accept if energy <= threshold; else reject",
#              "metrics_at_threshold": { "threshold": -4.2220, ... }
#            }
#        -> energy = -logsumexp(logits); reject if energy > threshold

import io, json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

# -------------------------------
# Resolve project paths (portable)
# -------------------------------
# BASE_DIR = .../ecoscan/server
BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = .../ecoscan
PROJECT_ROOT = BASE_DIR.parent
# MODEL_DIR = .../ecoscan/models/vit_ecoscan_v1   (contains config.json, model.safetensors, etc.)
MODEL_DIR = PROJECT_ROOT / "models" / "vit_ecoscan_v1"
# Optional reject threshold file
REJECT_FILE = MODEL_DIR / "eval" / "reject_threshold.json"

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
model.eval()  # no gradients

# Map class indices to labels (from config.json)
id2label = {int(k): v for k, v in model.config.id2label.items()}
labels = [id2label[i] for i in range(len(id2label))]

# -------------------------------
# Parse reject threshold config
# -------------------------------
reject_mode = None          # "confidence" | "energy" | None
reject_threshold = None     # float

if REJECT_FILE.exists():
    try:
        cfg = json.loads(REJECT_FILE.read_text(encoding="utf-8"))

        # Case A) plain number -> confidence threshold
        if isinstance(cfg, (int, float, str)):
            reject_mode = "confidence"
            reject_threshold = float(cfg)

        # Case B) {"threshold": X} -> confidence threshold
        elif isinstance(cfg, dict) and "threshold" in cfg and isinstance(cfg["threshold"], (int, float)):
            reject_mode = "confidence"
            reject_threshold = float(cfg["threshold"])

        # Case C) your JSON with energy method + reject_threshold
        elif isinstance(cfg, dict):
            # Prefer explicit energy method if present
            method = str(cfg.get("method", "")).lower()
            rule = str(cfg.get("rule", "")).lower()

            # 1) Try "reject_threshold" key
            if "reject_threshold" in cfg and isinstance(cfg["reject_threshold"], (int, float)):
                reject_threshold = float(cfg["reject_threshold"])
                reject_mode = "energy" if ("energy" in method or "energy" in rule) else "confidence"

            # 2) Or fall back to nested metrics_at_threshold.threshold
            elif "metrics_at_threshold" in cfg:
                mat = cfg["metrics_at_threshold"] or {}
                thr = mat.get("threshold", None)
                if isinstance(thr, (int, float)):
                    reject_threshold = float(thr)
                    reject_mode = "energy" if ("energy" in method or "energy" in rule) else "confidence"

        # Final sanity check
        if reject_mode not in ("confidence", "energy"):
            reject_mode = None
            reject_threshold = None

    except Exception as e:
        print("WARN: Could not parse reject_threshold.json:", e)
        reject_mode = None
        reject_threshold = None

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="EcoScan ViT Local API")

# Allow static HTML/JS front-end to call this API during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": str(MODEL_DIR),
        "labels": len(labels),
        "reject_mode": reject_mode,
        "reject_threshold": reject_threshold,
    }

# -------------------------------
# Prediction
# -------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Multipart upload:
      - field name: 'image'
    Returns:
      {
        "label": "Plastic" | null,    # null if rejected
        "confidence": 0.92,           # softmax top-1
        "rejected": true|false,
        "debug": { ... }              # useful while developing
      }
    """
    # 1) Read image
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # 2) Preprocess
    inputs = processor(images=img, return_tensors="pt")

    # 3) Forward pass
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits                        # [1, num_classes]

        # Softmax confidence (top-1)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
        conf = float(conf.item())
        idx = int(idx.item())
        label = id2label[idx]

        # Energy score (for energy-based reject): E = -logsumexp(logits)
        energy = float((-torch.logsumexp(logits, dim=1)).item())

    # 4) Apply reject rule (if configured)
    rejected = False
    if reject_threshold is not None:
        if reject_mode == "energy":
            # Your JSON rule: accept if energy <= threshold; else reject
            rejected = energy > reject_threshold
        else:
            # Confidence mode: accept if conf >= threshold; else reject
            rejected = conf < reject_threshold

    # 5) Build response
    return {
        "label": (None if rejected else label),
        "confidence": round(conf, 4),
        "rejected": rejected,
        "debug": {  # keep for MVP; remove later if you want
            "mode": reject_mode,
            "threshold": reject_threshold,
            "energy": round(energy, 6),
            "top_index": idx,
            "raw_conf": conf,
        },
    }

# Run with:
#   uvicorn app:app --reload --host 127.0.0.1 --port 8000
