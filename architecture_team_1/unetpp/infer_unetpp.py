"""
infer_unetpp.py

this file handles U-Net++ inference for one image at a time

it is doing these things
- loading the trained U-Net++ checkpoint
- preparing an input image the same way as the shared dataset pipeline
- running U-Net++ inference
- building a 3-class semantic mask

output mask values
- 0   = background
- 128 = nucleoplasm
- 255 = chromocenter
"""

import os
import cv2
import yaml
import numpy as np
import torch
import segmentation_models_pytorch as smp


def load_config():
    """
    loads config.yaml from the project root and returns it as a dict
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """
    convert input image to single-channel uint8
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


def preprocess_image_for_unetpp(img_raw: np.ndarray, preprocess_mode="basic") -> np.ndarray:
    """
    preprocess microscopy image the same way as the shared dataset pipeline
    """
    img = img_raw.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    img = np.clip(img, lo, hi)
    img01 = (img - lo) / (hi - lo + 1e-8)

    if preprocess_mode == "basic":
        result = img01
    elif preprocess_mode == "full":
        img01 = cv2.bilateralFilter(img01, d=5, sigmaColor=0.08, sigmaSpace=3)
        k = 51
        if k % 2 == 0:
            k += 1
        k = max(k, 15)
        bg = cv2.GaussianBlur(img01, (k, k), 0)
        corr = img01 - bg
        lo2, hi2 = np.percentile(corr, 1), np.percentile(corr, 99.5)
        corr = np.clip(corr, lo2, hi2)
        result = (corr - lo2) / (hi2 - lo2 + 1e-8)
    else:
        raise ValueError("Unknown preprocess mode")

    return result.astype(np.float32)


def resize_for_unetpp(img01: np.ndarray, target_size) -> np.ndarray:
    """
    resize image to target size used by the model
    """
    resized = cv2.resize(
        img01,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32)


def build_semantic_mask(pred_mask: np.ndarray) -> np.ndarray:
    """
    convert predicted class mask {0,1,2} into visualization mask
    - 0   background
    - 128 nucleoplasm
    - 255 chromocenter
    """
    semantic_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    semantic_mask[pred_mask == 1] = 128
    semantic_mask[pred_mask == 2] = 255
    return semantic_mask


def load_unetpp_model():
    """
    load trained U-Net++ model and checkpoint info

    returns
    - model
    - device
    - cfg
    - checkpoint_path
    """
    cfg = load_config()

    checkpoint_path = os.path.join(
        "architecture_team_1",
        "unetpp",
        "runs_unetpp",
        "best_unetpp.pt",
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, device, cfg, checkpoint_path


def run_unetpp_inference(image: np.ndarray, model, device, cfg):
    """
    run U-Net++ inference on one image

    returns a dict with:
    - prepared_image_uint8
    - preprocessed_image
    - pred_mask
    - semantic_mask
    """
    img_uint8 = prepare_grayscale_uint8(image)

    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    img_pp = preprocess_image_for_unetpp(img_uint8, preprocess_mode=preprocess_mode)
    img_rs = resize_for_unetpp(img_pp, target_size=target_size)

    img_tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        original_h, original_w = img_uint8.shape[:2]

        pred_mask = cv2.resize(
            pred_mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

    semantic_mask = build_semantic_mask(pred_mask)

    return {
        "prepared_image": img_uint8,
        "preprocessed_image": img_rs,
        "pred_mask": pred_mask,
        "semantic_mask": semantic_mask,
    }