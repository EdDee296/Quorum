"""
infer_unetpp.py

this file handles U-Net++ inference for one image at a time

it is doing these things
- loading the trained U-Net++ checkpoint
- preparing an input image the same way as the shared dataset pipeline
- running U-Net++ inference
- building a 3-class semantic mask
- collecting simple review metrics for the results page
- estimating prediction confidence using test-time augmentation (TTA)
  with entropy from averaged softmax probabilities

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


def build_review_metrics(pred_mask: np.ndarray):
    """
    build simple review metrics from the predicted mask

    this does not use ground truth.
    it just gives useful review flags for the results page.
    """
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

    for cell_id in range(1, num_labels):
        x = int(stats[cell_id, cv2.CC_STAT_LEFT])
        y = int(stats[cell_id, cv2.CC_STAT_TOP])
        w = int(stats[cell_id, cv2.CC_STAT_WIDTH])
        h = int(stats[cell_id, cv2.CC_STAT_HEIGHT])
        nucleus_area = int(stats[cell_id, cv2.CC_STAT_AREA])

        cell_region = labels == cell_id
        cell_chrom = chromocenter_mask & cell_region

        chrom_num_labels, _, _chrom_stats, _ = cv2.connectedComponentsWithStats(
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

    cells_review.sort(key=lambda row: row["review_score"], reverse=True)

    summary = {
        "nuclei_count": int(num_labels - 1),
        "chromocenter_count": int(total_chromocenters),
        "flagged_cells_count": int(flagged_cells_count),
    }

    return summary, cells_review


def _predict_probs_from_preprocessed(img_rs: np.ndarray, model, device) -> torch.Tensor:
    """
    run model on one already-preprocessed + resized image
    returns softmax probabilities at resized resolution as [C,H,W]
    """
    img_tensor = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    return probs


def _apply_tta_numpy(img_rs: np.ndarray, mode: str) -> np.ndarray:
    """
    apply deterministic TTA to resized image
    """
    if mode == "orig":
        return img_rs
    if mode == "hflip":
        return np.ascontiguousarray(np.fliplr(img_rs))
    if mode == "vflip":
        return np.ascontiguousarray(np.flipud(img_rs))
    if mode == "rot90":
        return np.ascontiguousarray(np.rot90(img_rs, k=1))

    raise ValueError(f"Unknown TTA mode: {mode}")


def _invert_tta_probs(probs_chw: torch.Tensor, mode: str) -> torch.Tensor:
    """
    invert deterministic TTA on probability tensor [C,H,W]
    """
    if mode == "orig":
        return probs_chw
    if mode == "hflip":
        return torch.flip(probs_chw, dims=[2])
    if mode == "vflip":
        return torch.flip(probs_chw, dims=[1])
    if mode == "rot90":
        return torch.rot90(probs_chw, k=3, dims=[1, 2])

    raise ValueError(f"Unknown TTA mode: {mode}")


def predict_with_tta(img_rs: np.ndarray, model, device):
    """
    run TTA inference and return:
    - mean_probs_rs: [C,H,W]
    - pred_mask_rs: [H,W]
    - entropy_rs: [H,W]
    """
    tta_modes = ["orig", "hflip", "vflip", "rot90"]
    probs_list = []

    for mode in tta_modes:
        aug_img = _apply_tta_numpy(img_rs, mode)
        probs = _predict_probs_from_preprocessed(aug_img, model, device)
        probs = _invert_tta_probs(probs, mode)
        probs_list.append(probs)

    prob_stack = torch.stack(probs_list, dim=0)    # [T,C,H,W]
    mean_probs_rs = prob_stack.mean(dim=0)         # [C,H,W]
    pred_mask_rs = torch.argmax(mean_probs_rs, dim=0).numpy().astype(np.uint8)

    eps = 1e-8
    entropy_rs = -torch.sum(mean_probs_rs * torch.log(mean_probs_rs + eps), dim=0).numpy().astype(np.float32)

    return mean_probs_rs, pred_mask_rs, entropy_rs


def build_uncertainty_summary(mean_probs_rs: torch.Tensor, entropy_rs: np.ndarray, original_size):
    """
    estimate uncertainty from averaged TTA softmax probabilities

    output
    - uncertainty_summary: dict
    """
    original_h, original_w = original_size

    mean_confidence = float(torch.max(mean_probs_rs, dim=0).values.mean().item())
    mean_entropy = float(entropy_rs.mean())
    max_entropy = float(entropy_rs.max())

    # normalized entropy for 3 classes
    max_possible_entropy = float(np.log(3.0))
    normalized_mean_entropy = mean_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0

    if normalized_mean_entropy <= 0.20 and mean_confidence >= 0.85:
        confidence_label = "High"
        needs_review = False
    elif normalized_mean_entropy <= 0.35 and mean_confidence >= 0.70:
        confidence_label = "Medium"
        needs_review = False
    else:
        confidence_label = "Low"
        needs_review = True

    uncertainty_summary = {
        "mean_confidence": round(mean_confidence, 4),
        "mean_entropy": round(mean_entropy, 4),
        "max_entropy": round(max_entropy, 4),
        "normalized_mean_entropy": round(normalized_mean_entropy, 4),
        "confidence_label": confidence_label,
        "needs_review": bool(needs_review),
        "tta_views_used": 4,
        "resized_height": int(entropy_rs.shape[0]),
        "resized_width": int(entropy_rs.shape[1]),
        "original_height": int(original_h),
        "original_width": int(original_w),
    }

    return uncertainty_summary


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
        "backend",
        "models",
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
    - summary
    - cells_review
    - uncertainty_summary
    """
    img_uint8 = prepare_grayscale_uint8(image)

    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    img_pp = preprocess_image_for_unetpp(img_uint8, preprocess_mode=preprocess_mode)
    img_rs = resize_for_unetpp(img_pp, target_size=target_size)

    mean_probs_rs, pred_mask_rs, entropy_rs = predict_with_tta(img_rs, model, device)

    original_h, original_w = img_uint8.shape[:2]

    pred_mask = cv2.resize(
        pred_mask_rs,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    semantic_mask = build_semantic_mask(pred_mask)
    summary, cells_review = build_review_metrics(pred_mask)
    uncertainty_summary = build_uncertainty_summary(
        mean_probs_rs=mean_probs_rs,
        entropy_rs=entropy_rs,
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