import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from cellpose import models

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading Cellpose Models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Chromocenter model (required) ─────────────────────────────────────────────
MODEL_PATH_AUG = os.path.join(BASE_DIR, "models", "cp_chromo_aug")
MODEL_PATH_NO_AUG = os.path.join(BASE_DIR, "models", "cp_chromo_no_aug")
MODEL_SOURCE = None
MODEL_LOAD_ERROR = None
model = None

for candidate_path in [MODEL_PATH_AUG, MODEL_PATH_NO_AUG]:
    if not os.path.exists(candidate_path):
        continue
    try:
        print(f"Loading chromocenter model from {candidate_path}")
        model = models.CellposeModel(gpu=False, pretrained_model=candidate_path)
        MODEL_SOURCE = candidate_path
        break
    except Exception as e:
        MODEL_LOAD_ERROR = f"Failed to load model at {candidate_path}: {e}"
        print(MODEL_LOAD_ERROR)

if model is None:
    MODEL_LOAD_ERROR = MODEL_LOAD_ERROR or (
        "Chromocenter model did not load. Expected one of: "
        f"{MODEL_PATH_AUG} or {MODEL_PATH_NO_AUG}."
    )
    print(MODEL_LOAD_ERROR)

# ── Nucleus model (optional until trained) ────────────────────────────────────
NUCLEUS_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "cp_nucleus"),
    # Cellpose train() may append another "models" level depending on save_path.
    os.path.join(BASE_DIR, "models", "models", "cp_nucleus"),
]
nucleus_model = None
NUCLEUS_MODEL_SOURCE = None
nucleus_fallback_model = None

for nucleus_path in NUCLEUS_MODEL_CANDIDATES:
    if not os.path.exists(nucleus_path):
        continue

    try:
        print(f"Loading nucleus model from {nucleus_path}")
        nucleus_model = models.CellposeModel(gpu=False, pretrained_model=nucleus_path)
        NUCLEUS_MODEL_SOURCE = nucleus_path
        print("Nucleus model loaded successfully")
        break
    except Exception as e:
        print(f"Warning: Failed to load nucleus model at {nucleus_path}: {e}")

if nucleus_model is None:
    print(
        "Nucleus model not found. Checked: "
        + ", ".join(NUCLEUS_MODEL_CANDIDATES)
        + ". Run the notebook to train cp_nucleus. "
        "Nucleoplasm segmentation will be unavailable until then."
    )

try:
    # Always keep a stable fallback for nucleus/background separation.
    nucleus_fallback_model = models.CellposeModel(gpu=False, model_type='nuclei')
    print("Loaded pretrained nuclei fallback model for nucleus segmentation")
except Exception as e:
    print(f"Warning: Failed to load pretrained nuclei fallback model: {e}")


def prepare_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert input image to single-channel uint8 for stable Cellpose inference.

    TIFF images are often 16-bit, so this normalizes dynamic range to 8-bit.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        return image

    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (image.astype(np.float32) - min_val) / (max_val - min_val)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)

@app.get("/api/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)

    return {
        "status": "ok",
        "model_source": MODEL_SOURCE,
        "nucleus_model_available": nucleus_model is not None,
    }

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)

        # 1. Read the uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or unsupported TIFF encoding",
            )

        img = prepare_grayscale_uint8(img)

        # 2a. Run chromocenter model
        chromo_masks, _flows, _styles = model.eval(img, diameter=None, channels=[0, 0])
        chromo_instances = np.asarray(chromo_masks)

        # 2b. Run nucleus model if available
        nucleus_instances = None
        if nucleus_model is not None:
            nuc_masks, _nf, _ns = nucleus_model.eval(
                img,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=-2.0,
            )
            nucleus_instances = np.asarray(nuc_masks)

        # If custom nucleus model is unavailable or collapses to empty masks,
        # fall back to pretrained nuclei segmentation so background = outside nucleus.
        if (nucleus_instances is None or np.max(nucleus_instances) == 0) and nucleus_fallback_model is not None:
            fb_masks, _ff, _fs = nucleus_fallback_model.eval(
                img,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=-2.0,
            )
            nucleus_instances = np.asarray(fb_masks)

        # 3. Build 3-class semantic mask (grayscale PNG, values 0 / 128 / 255):
        #      0   = background (outside nucleus)
        #      128 = nucleoplasm (inside nucleus, not chromocenter)
        #      255 = chromocenter
        semantic_mask = np.zeros_like(chromo_instances, dtype=np.uint8)

        if nucleus_instances is not None:
            # Mark full nucleus region as nucleoplasm
            semantic_mask[nucleus_instances > 0] = 128

        # Chromocenter pixels override nucleoplasm (always on top)
        semantic_mask[chromo_instances > 0] = 255

        # 4. Encode browser-safe PNG of the original input
        _, original_buffer = cv2.imencode('.png', img)
        original_base64 = base64.b64encode(original_buffer.tobytes()).decode('utf-8')

        # 5. Encode the semantic mask
        _, mask_buffer = cv2.imencode('.png', semantic_mask)
        mask_base64 = base64.b64encode(mask_buffer.tobytes()).decode('utf-8')

        return {
            "filename": file.filename,
            # semantic_mask carries all three classes (0 / 128 / 255)
            "semantic_mask_base64": f"data:image/png;base64,{mask_base64}",
            # legacy alias so older clients keep working
            "mask_base64": f"data:image/png;base64,{mask_base64}",
            "original_base64": f"data:image/png;base64,{original_base64}",
            "model_source": MODEL_SOURCE,
            "nucleus_model_available": nucleus_model is not None,
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))