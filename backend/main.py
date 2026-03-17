import os
import cv2
import io
import zipfile
import base64
import numpy as np
import tifffile as tiff
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cellpose import models

from architecture_team_1.unetpp.infer_unetpp import load_unetpp_model, run_unetpp_inference

app = FastAPI()

# lets frontend talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loads models
print("Loading models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cellpose models
MODEL_PATH_AUG = os.path.join(BASE_DIR, "models", "cp_chromo_aug")
MODEL_PATH_NO_AUG = os.path.join(BASE_DIR, "models", "cp_chromo_no_aug")
MODEL_SOURCE = None
MODEL_LOAD_ERROR = None
model = None

for candidate_path in [MODEL_PATH_AUG, MODEL_PATH_NO_AUG]:
    if not os.path.exists(candidate_path):
        continue
    try:
        print(f"Loading Cellpose chromocenter model from {candidate_path}")
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

# nucleus model for Cellpose
NUCLEUS_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "cp_nucleus"),
    os.path.join(BASE_DIR, "models", "models", "cp_nucleus"),
]
nucleus_model = None
NUCLEUS_MODEL_SOURCE = None
nucleus_fallback_model = None

for nucleus_path in NUCLEUS_MODEL_CANDIDATES:
    if not os.path.exists(nucleus_path):
        continue

    try:
        print(f"Loading Cellpose nucleus model from {nucleus_path}")
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
        + ". Nucleoplasm segmentation will use fallback if possible."
    )

try:
    # fallback nucleus model
    nucleus_fallback_model = models.CellposeModel(gpu=False, model_type="nuclei")
    print("Loaded pretrained nuclei fallback model for Cellpose")
except Exception as e:
    print(f"Warning: Failed to load pretrained nuclei fallback model: {e}")



# U-Net++ model
unetpp_model = None
unetpp_device = None
unetpp_cfg = None
UNETPP_SOURCE = None
UNETPP_LOAD_ERROR = None

try:
    unetpp_model, unetpp_device, unetpp_cfg, UNETPP_SOURCE = load_unetpp_model()
    print(f"Loaded U-Net++ model from {UNETPP_SOURCE}")
except Exception as e:
    UNETPP_LOAD_ERROR = f"Failed to load U-Net++ model: {e}"
    print(UNETPP_LOAD_ERROR)


def prepare_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """
    make image single-channel uint8
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


class MaskDownloadRequest(BaseModel):
    file_name: str
    chromocenter_mask: str
    nuclei_mask: str
    background_mask: str


def _data_url_to_binary_mask(mask_data_url: str) -> np.ndarray:
    """
    turn a data url mask into a black binary mask
    """
    if "," in mask_data_url:
        encoded = mask_data_url.split(",", 1)[1]
    else:
        encoded = mask_data_url

    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode mask image")

    if img.ndim == 2:
        gray = img
        alpha = None
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        alpha = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = None

    if alpha is not None:
        binary = np.where(alpha > 0, 0, 255).astype(np.uint8)
    else:
        binary = np.where(gray > 0, 0, 255).astype(np.uint8)

    return binary


def _build_mask_zip(file_name: str, chromocenter_mask: str, nuclei_mask: str, background_mask: str) -> io.BytesIO:
    """
    build a zip with 3 tif masks
    """
    base_name = os.path.splitext(file_name)[0]

    chrom_arr = _data_url_to_binary_mask(chromocenter_mask)
    nuclei_arr = _data_url_to_binary_mask(nuclei_mask)
    bg_arr = _data_url_to_binary_mask(background_mask)

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        tif_buffer = io.BytesIO()
        tiff.imwrite(tif_buffer, chrom_arr)
        zip_file.writestr(f"{base_name}_chromocenter.tif", tif_buffer.getvalue())

        tif_buffer = io.BytesIO()
        tiff.imwrite(tif_buffer, nuclei_arr)
        zip_file.writestr(f"{base_name}_nuclei.tif", tif_buffer.getvalue())

        tif_buffer = io.BytesIO()
        tiff.imwrite(tif_buffer, bg_arr)
        zip_file.writestr(f"{base_name}_background.tif", tif_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "cellpose_available": model is not None,
        "cellpose_model_source": MODEL_SOURCE,
        "cellpose_nucleus_model_available": nucleus_model is not None,
        "unetpp_available": unetpp_model is not None,
        "unetpp_model_source": UNETPP_SOURCE,
        "cellpose_error": MODEL_LOAD_ERROR,
        "unetpp_error": UNETPP_LOAD_ERROR,
    }


@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(...),
    # model_name: str = Form("cellpose"),
    model_name: str = Form("unetpp"),
):
    try:
        # reads uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or unsupported TIFF encoding",
            )

        # Cellpose path
        if model_name == "cellpose":
            if model is None:
                raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR)

            img = prepare_grayscale_uint8(img)

            # runs chromocenter model
            chromo_masks, _flows, _styles = model.eval(img, diameter=None, channels=[0, 0])
            chromo_instances = np.asarray(chromo_masks)

            # runs nucleus model if we have it
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

            # fallbacks if custom nucleus model is missing or empty
            if (nucleus_instances is None or np.max(nucleus_instances) == 0) and nucleus_fallback_model is not None:
                fb_masks, _ff, _fs = nucleus_fallback_model.eval(
                    img,
                    diameter=None,
                    channels=[0, 0],
                    flow_threshold=0.4,
                    cellprob_threshold=-2.0,
                )
                nucleus_instances = np.asarray(fb_masks)

            # builds semantic mask
            semantic_mask = np.zeros_like(chromo_instances, dtype=np.uint8)

            if nucleus_instances is not None:
                semantic_mask[nucleus_instances > 0] = 128

            semantic_mask[chromo_instances > 0] = 255

            # encodes original image
            _, original_buffer = cv2.imencode(".png", img)
            original_base64 = base64.b64encode(original_buffer.tobytes()).decode("utf-8")

            # encodes mask
            _, mask_buffer = cv2.imencode(".png", semantic_mask)
            mask_base64 = base64.b64encode(mask_buffer.tobytes()).decode("utf-8")

            return {
                "filename": file.filename,
                "semantic_mask_base64": f"data:image/png;base64,{mask_base64}",
                "mask_base64": f"data:image/png;base64,{mask_base64}",
                "original_base64": f"data:image/png;base64,{original_base64}",
                "model_name": "cellpose",
                "model_source": MODEL_SOURCE,
                "nucleus_model_available": nucleus_model is not None,
            }

        
        # U-Net++ path
        elif model_name == "unetpp":
            if unetpp_model is None:
                raise HTTPException(status_code=503, detail=UNETPP_LOAD_ERROR)

            result = run_unetpp_inference(
                image=img,
                model=unetpp_model,
                device=unetpp_device,
                cfg=unetpp_cfg,
            )

            prepared_image = result["prepared_image"]
            semantic_mask = result["semantic_mask"]
            summary = result["summary"]
            cells_review = result["cells_review"]  

            # encodes original image
            _, original_buffer = cv2.imencode(".png", prepared_image)
            original_base64 = base64.b64encode(original_buffer.tobytes()).decode("utf-8")

            # encodes mask
            _, mask_buffer = cv2.imencode(".png", semantic_mask)
            mask_base64 = base64.b64encode(mask_buffer.tobytes()).decode("utf-8")

            return {
                "filename": file.filename,
                "semantic_mask_base64": f"data:image/png;base64,{mask_base64}",
                "mask_base64": f"data:image/png;base64,{mask_base64}",
                "original_base64": f"data:image/png;base64,{original_base64}",
                "model_name": "unetpp",
                "model_source": UNETPP_SOURCE,
                "nucleus_model_available": True,
                "summary": summary,
                "cells_review": cells_review
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model_name '{model_name}'. Use 'cellpose' or 'unetpp'.",
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download-masks")
async def download_masks(request: MaskDownloadRequest):
    try:
        zip_buffer = _build_mask_zip(
            file_name=request.file_name,
            chromocenter_mask=request.chromocenter_mask,
            nuclei_mask=request.nuclei_mask,
            background_mask=request.background_mask,
        )

        base_name = os.path.splitext(request.file_name)[0]
        zip_name = f"{base_name}_masks.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
        )
    except Exception as e:
        print(f"Error creating mask download: {e}")
        raise HTTPException(status_code=500, detail=str(e))