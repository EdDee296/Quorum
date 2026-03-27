"""
infer_deeplabv3.py

this file handles DeepLabV3 inference for one image at a time

it is doing these things
- loading the trained DeepLabV3 checkpoint
- preparing an input image the same way as the shared dataset pipeline
- running DeepLabV3 inference
- building a 3-class semantic mask
- collecting simple review metrics for the results page
- estimating prediction confidence using simple test-time augmentation (TTA)

output mask values
- 0   = background
- 128 = nucleoplasm
- 255 = chromocenter
"""

import os
import sys
import cv2
import yaml
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from architecture_team_4.deeplabv3_model import build_deeplabv3


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_grayscale_uint8(image: np.ndarray) -> np.ndarray:
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


def preprocess_image_for_deeplabv3(img_raw: np.ndarray, preprocess_mode="basic") -> np.ndarray:
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


def resize_for_deeplabv3(img01: np.ndarray, target_size) -> np.ndarray:
    resized = cv2.resize(
        img01,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32)


def build_semantic_mask(pred_mask: np.ndarray) -> np.ndarray:
    semantic_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    semantic_mask[pred_mask == 1] = 128
    semantic_mask[pred_mask == 2] = 255
    return semantic_mask


def build_review_metrics(pred_mask: np.ndarray):
    nucleus_mask = (pred_mask == 1) | (pred_mask == 2)
    chromocenter_mask = (pred_mask == 2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        nucleus_mask.astype(np.uint8),
        connectivity=8,
    )

    cells_review = []
    total_chromocenters = 0
    flagged_cells_count = 0

    image_h, image_w = pred_mask.shape[:2]
    image_area = image_h * image_w

    cell_id = 1
    while cell_id < num_labels:
        x = int(stats[cell_id, cv2.CC_STAT_LEFT])
        y = int(stats[cell_id, cv2.CC_STAT_TOP])
        w = int(stats[cell_id, cv2.CC_STAT_WIDTH])
        h = int(stats[cell_id, cv2.CC_STAT_HEIGHT])
        nucleus_area = int(stats[cell_id, cv2.CC_STAT_AREA])

        cell_region = labels == cell_id
        cell_chrom = chromocenter_mask & cell_region

        chrom_num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            cell_chrom.astype(np.uint8),
            connectivity=8,
        )

        chromocenter_count = max(0, chrom_num_labels - 1)
        chromocenter_area = int(cell_chrom.sum())
        total_chromocenters += chromocenter_count

        chrom_ratio = float(chromocenter_area / nucleus_area) if nucleus_area > 0 else 0.0
        touches_border = (x == 0) or (y == 0) or (x + w >= image_w) or (y + h >= image_h)

        review_reasons = []
        review_score = 0

        if nucleus_area < max(20, int(image_area * 0.0005)):
            review_reasons.append("small nucleus")
            review_score += 1

        if nucleus_area > int(image_area * 0.30):
            review_reasons.append("large nucleus")
            review_score += 1

        if chromocenter_count == 0:
            review_reasons.append("no chromocenter detected")
            review_score += 1

        if chromocenter_count > 12:
            review_reasons.append("many chromocenters")
            review_score += 1

        if chrom_ratio > 0.60:
            review_reasons.append("high chromocenter ratio")
            review_score += 1

        if touches_border:
            review_reasons.append("touches image border")
            review_score += 1

        is_flagged = review_score > 0
        if is_flagged:
            flagged_cells_count += 1

        cells_review.append({
            "cell_id": int(cell_id),
            "nucleus_area": nucleus_area,
            "chromocenter_area": chromocenter_area,
            "chromocenter_count": int(chromocenter_count),
            "chromocenter_ratio": round(chrom_ratio, 4),
            "touches_border": bool(touches_border),
            "review_score": int(review_score),
            "review_reasons": review_reasons,
            "is_flagged": bool(is_flagged),
            "bbox": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            },
        })

        cell_id += 1

    cells_review.sort(key=lambda row: row["review_score"], reverse=True)

    summary = {
        "nuclei_count": int(num_labels - 1),
        "chromocenter_count": int(total_chromocenters),
        "flagged_cells_count": int(flagged_cells_count),
    }

    return summary, cells_review


def _predict_mask_from_preprocessed(img_rs: np.ndarray, model, device) -> np.ndarray:
    img_tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).float().to(device)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    return pred_mask


def build_uncertainty_summary(img_pp: np.ndarray, model, device, original_size):
    original_h, original_w = original_size

    pred_original = _predict_mask_from_preprocessed(img_pp, model, device)

    img_hflip = np.ascontiguousarray(np.fliplr(img_pp))
    pred_hflip = _predict_mask_from_preprocessed(img_hflip, model, device)
    pred_hflip = np.fliplr(pred_hflip)

    img_vflip = np.ascontiguousarray(np.flipud(img_pp))
    pred_vflip = _predict_mask_from_preprocessed(img_vflip, model, device)
    pred_vflip = np.flipud(pred_vflip)

    stacked = np.stack([pred_original, pred_hflip, pred_vflip], axis=0)

    agreement_map = np.all(stacked == stacked[0:1], axis=0)
    mean_agreement = float(agreement_map.mean())

    chrom_presence = np.any(stacked == 2, axis=0)
    chrom_all_same = np.all(stacked == 2, axis=0)
    chrom_pixels = int(chrom_presence.sum())

    if chrom_pixels > 0:
        chromocenter_agreement = float(chrom_all_same.sum() / chrom_pixels)
    else:
        chromocenter_agreement = 1.0

    if mean_agreement >= 0.75 and chromocenter_agreement >= 0.70:
        confidence_label = "High"
        needs_review = False
    elif mean_agreement >= 0.60 and chromocenter_agreement >= 0.50:
        confidence_label = "Moderate"
        needs_review = False
    else:
        confidence_label = "Low"
        needs_review = True

    uncertainty_summary = {
        "mean_agreement": round(mean_agreement, 4),
        "chromocenter_agreement": round(chromocenter_agreement, 4),
        "confidence_label": confidence_label,
        "needs_review": bool(needs_review),
        "tta_views_used": 3,
        "resized_height": int(img_pp.shape[0]),
        "resized_width": int(img_pp.shape[1]),
        "original_height": int(original_h),
        "original_width": int(original_w),
    }

    return uncertainty_summary


def load_deeplabv3_model():
    cfg = load_config()

    checkpoint_path = os.path.join(
        THIS_DIR,
        "runs_deeplabv3",
        "best_deeplabv3.pt",
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")

    model = build_deeplabv3(
        backbone=backbone,
        pretrained_backbone=False,
        out_channels=3,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, device, cfg, checkpoint_path


def run_deeplabv3_inference(image: np.ndarray, model, device, cfg):
    img_uint8 = prepare_grayscale_uint8(image)

    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    img_pp = preprocess_image_for_deeplabv3(img_uint8, preprocess_mode=preprocess_mode)
    img_rs = resize_for_deeplabv3(img_pp, target_size=target_size)

    pred_mask_rs = _predict_mask_from_preprocessed(img_rs, model, device)

    original_h, original_w = img_uint8.shape[:2]

    pred_mask = cv2.resize(
        pred_mask_rs,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    semantic_mask = build_semantic_mask(pred_mask)
    summary, cells_review = build_review_metrics(pred_mask)
    uncertainty_summary = build_uncertainty_summary(
        img_pp=img_rs,
        model=model,
        device=device,
        original_size=(original_h, original_w),
    )

    return {
        "prepared_image": img_uint8,
        "preprocessed_image": img_rs,
        "pred_mask": pred_mask,
        "semantic_mask": semantic_mask,
        "summary": summary,
        "cells_review": cells_review,
        "uncertainty_summary": uncertainty_summary,
    }