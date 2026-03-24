"""
infer_cellpose.py

this file handles Cellpose inference for one image at a time

it is doing these things
- loading trained Cellpose models from backend/models
- preparing an input image for Cellpose
- running chromocenter inference
- running nucleus inference if available
- falling back to pretrained nuclei model if needed
- building a 3-class semantic mask

output mask values
- 0   = background
- 128 = nucleoplasm
- 255 = chromocenter
"""

import os
import cv2
import numpy as np
from cellpose import models


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")

MODEL_PATH_AUG = os.path.join(MODEL_DIR, "cp_chromo_aug")
MODEL_PATH_NO_AUG = os.path.join(MODEL_DIR, "cp_chromo_no_aug")

NUCLEUS_MODEL_CANDIDATES = [
    os.path.join(MODEL_DIR, "cp_nucleus"),
    os.path.join(MODEL_DIR, "models", "cp_nucleus"),
]


def prepare_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """
    convert input image to single-channel uint8 for stable Cellpose inference
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


def load_cellpose_models(use_gpu=False):
    """
    load chromocenter and nucleus models

    returns
    - chromo_model
    - nucleus_model
    - nucleus_fallback_model
    - model_source
    - model_load_error
    """
    print("Loading Cellpose Models...")

    chromo_model = None
    model_source = None
    model_load_error = None

    for candidate_path in [MODEL_PATH_AUG, MODEL_PATH_NO_AUG]:
        if not os.path.exists(candidate_path):
            continue

        try:
            print(f"Loading chromocenter model from {candidate_path}")
            chromo_model = models.CellposeModel(
                gpu=use_gpu,
                pretrained_model=candidate_path,
            )
            model_source = candidate_path
            break
        except Exception as e:
            model_load_error = f"Failed to load model at {candidate_path}: {e}"
            print(model_load_error)

    if chromo_model is None:
        model_load_error = model_load_error or (
            "Chromocenter model did not load. Expected one of: "
            f"{MODEL_PATH_AUG} or {MODEL_PATH_NO_AUG}."
        )
        print(model_load_error)

    nucleus_model = None
    for nucleus_path in NUCLEUS_MODEL_CANDIDATES:
        if not os.path.exists(nucleus_path):
            continue

        try:
            print(f"Loading nucleus model from {nucleus_path}")
            nucleus_model = models.CellposeModel(
                gpu=use_gpu,
                pretrained_model=nucleus_path,
            )
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

    nucleus_fallback_model = None
    try:
        nucleus_fallback_model = models.CellposeModel(
            gpu=use_gpu,
            model_type="nuclei",
        )
        print("Loaded pretrained nuclei fallback model for nucleus segmentation")
    except Exception as e:
        print(f"Warning: Failed to load pretrained nuclei fallback model: {e}")

    return (
        chromo_model,
        nucleus_model,
        nucleus_fallback_model,
        model_source,
        model_load_error,
    )


def build_semantic_mask(chromo_instances: np.ndarray, nucleus_instances: np.ndarray | None) -> np.ndarray:
    """
    build the 3-class semantic mask

    output values
    - 0   background
    - 128 nucleoplasm
    - 255 chromocenter
    """
    semantic_mask = np.zeros_like(chromo_instances, dtype=np.uint8)

    if nucleus_instances is not None:
        semantic_mask[nucleus_instances > 0] = 128

    semantic_mask[chromo_instances > 0] = 255
    return semantic_mask


def run_cellpose_inference(
    image: np.ndarray,
    chromo_model,
    nucleus_model=None,
    nucleus_fallback_model=None,
):
    """
    run Cellpose inference on one image

    returns a dict with:
    - prepared image
    - chromocenter instances
    - nucleus instances
    - semantic mask
    """
    if chromo_model is None:
        raise ValueError("Chromocenter model is not loaded")

    img = prepare_grayscale_uint8(image)

    chromo_masks, _flows, _styles = chromo_model.eval(
        img,
        diameter=None,
        channels=[0, 0],
    )
    chromo_instances = np.asarray(chromo_masks)

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

    if (nucleus_instances is None or np.max(nucleus_instances) == 0) and nucleus_fallback_model is not None:
        fb_masks, _ff, _fs = nucleus_fallback_model.eval(
            img,
            diameter=None,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=-2.0,
        )
        nucleus_instances = np.asarray(fb_masks)

    semantic_mask = build_semantic_mask(chromo_instances, nucleus_instances)

    return {
        "prepared_image": img,
        "chromo_instances": chromo_instances,
        "nucleus_instances": nucleus_instances,
        "semantic_mask": semantic_mask,
    }