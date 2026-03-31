"""
eval_unetpp.py

this file is used for evaluating the trained U-Net++ model on the fixed validation split.

what this file does:

- loads config.yaml to get dataset path and preprocessing settings
- loads val_ids.txt to use the exact same validation set every time
- loads the trained U-Net++ checkpoint (best_unetpp.pt)
- runs TTA-based inference on every validation image
- predicts 3-class segmentation (0 background, 1 nucleus, 2 chromocenter)
- computes Dice and IoU for:
    class 1 (nucleus)
    class 2 (chromocenter)
- computes entropy-based uncertainty from TTA mean probabilities
- saves predicted masks (scaled for viewing: 0=black, 1=gray, 2=white)
- saves uncertainty maps
- saves per-image metrics into metrics.csv
- prints average metrics and worst 5 cells for chromocenter

important before running:

- make sure you already trained the model using train_unetpp.py
- make sure best_unetpp.pt exists inside:
    architecture_team_1/unetpp/runs_unetpp/
- make sure config.yaml has correct data_root path
- make sure val_ids.txt has not been modified

how to run:

from the project root folder:

python -m architecture_team_1.unetpp.eval_unetpp

outputs:

- metrics.csv saved to:
    architecture_team_1/unetpp/outputs_unetpp/
- predicted masks saved to:
    architecture_team_1/unetpp/outputs_unetpp/pred_masks/
- uncertainty maps saved to:
    architecture_team_1/unetpp/outputs_unetpp/uncertainty_maps/

notes:

- this evaluates both nucleus and chromocenter
- background is not evaluated because it is trivial and would inflate metrics
- chromocenter Dice is usually the main performance number
- TTA here uses deterministic transforms:
    original, horizontal flip, vertical flip, rotate 90
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import tifffile as tiff

from data_utils.dataset import CellDataset
import segmentation_models_pytorch as smp


def load_config():
    """
    Loading config.yaml file.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dice_iou_for_class(pred, gt, cls):
    """
    Computing Dice and IoU for a single class.

    Args:
        pred (ndarray): HxW predicted mask with values {0,1,2}
        gt (ndarray):   HxW ground truth mask with values {0,1,2}
        cls (int):      class id to evaluate (1 nucleus, 2 chromocenter)

    Returns:
        tuple:
            dice (float)
            iou (float)
    """
    p = (pred == cls)
    g = (gt == cls)

    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    p_sum = p.sum()
    g_sum = g.sum()

    denom = p_sum + g_sum
    if denom == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / float(denom)

    if union == 0:
        iou = 1.0
    else:
        iou = inter / float(union)

    return float(dice), float(iou)


def save_scaled_mask(mask_hw, out_path):
    """
    Saving predicted mask in scaled format for visualization.

    Args:
        mask_hw (ndarray): HxW mask with values {0,1,2}
        out_path (str): path to save .tif file

    Output:
        Writes scaled mask to disk where:
            0   -> background
            127 -> nucleus
            255 -> chromocenter
    """
    vis = np.zeros_like(mask_hw, dtype=np.uint8)
    vis[mask_hw == 1] = 127
    vis[mask_hw == 2] = 255
    tiff.imwrite(out_path, vis)


def save_uncertainty_map(uncertainty_hw, out_path):
    """
    Save uncertainty map as uint8 image for visualization.

    Higher value = higher uncertainty.
    """
    u = uncertainty_hw.astype(np.float32)
    u_min = float(u.min())
    u_max = float(u.max())

    if u_max <= u_min:
        vis = np.zeros_like(u, dtype=np.uint8)
    else:
        vis = ((u - u_min) / (u_max - u_min + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)

    tiff.imwrite(out_path, vis)


def apply_tta(img_bchw, mode):
    """
    Apply one deterministic test-time augmentation to image tensor [1,C,H,W].
    """
    if mode == "orig":
        return img_bchw
    if mode == "hflip":
        return torch.flip(img_bchw, dims=[3])
    if mode == "vflip":
        return torch.flip(img_bchw, dims=[2])
    if mode == "rot90":
        return torch.rot90(img_bchw, k=1, dims=[2, 3])

    raise ValueError(f"Unknown TTA mode: {mode}")


def invert_tta(prob_bchw, mode):
    """
    Invert deterministic test-time augmentation on probability tensor [1,C,H,W].
    """
    if mode == "orig":
        return prob_bchw
    if mode == "hflip":
        return torch.flip(prob_bchw, dims=[3])
    if mode == "vflip":
        return torch.flip(prob_bchw, dims=[2])
    if mode == "rot90":
        return torch.rot90(prob_bchw, k=3, dims=[2, 3])

    raise ValueError(f"Unknown TTA mode: {mode}")


def predict_with_tta(model, img_bchw):
    """
    Run TTA inference and return:
    - mean_probs: [C,H,W]
    - pred_hw: [H,W]
    - entropy_hw: [H,W]
    """
    tta_modes = ["orig", "hflip", "vflip", "rot90"]
    prob_list = []

    with torch.no_grad():
        for mode in tta_modes:
            aug_img = apply_tta(img_bchw, mode)
            logits = model(aug_img)
            probs = torch.softmax(logits, dim=1)
            probs = invert_tta(probs, mode)
            prob_list.append(probs[0].cpu())

    prob_stack = torch.stack(prob_list, dim=0)   # [T,C,H,W]
    mean_probs = prob_stack.mean(dim=0)          # [C,H,W]
    pred_hw = torch.argmax(mean_probs, dim=0).numpy().astype(np.uint8)

    eps = 1e-8
    entropy_hw = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=0).numpy().astype(np.float32)

    return mean_probs, pred_hw, entropy_hw


def main():
    """
    Running full evaluation pipeline.

    What this function does:
    - loads config and validation dataset
    - loads trained U-Net++ checkpoint
    - runs TTA inference on validation images
    - computes Dice and IoU for nucleus and chromocenter
    - computes entropy-based uncertainty maps from TTA mean probabilities
    - saves predicted masks, uncertainty maps, and metrics.csv
    - prints summary results
    """
    t0 = time.time()

    cfg = load_config()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    val_ids = CellDataset.load_split_ids("val_ids.txt")

    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )

    ckpt_path = os.path.join("architecture_team_1", "unetpp", "runs_unetpp", "best_unetpp.pt")
    out_dir = os.path.join("architecture_team_1", "unetpp", "outputs_unetpp")
    pred_dir = os.path.join(out_dir, "pred_masks")
    unc_dir = os.path.join(out_dir, "uncertainty_maps")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(unc_dir, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", ckpt_path)
    print("Out dir:", out_dir)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []

    for i in range(len(val_ds)):
        img, gt_mask = val_ds[i]
        cid = val_ds.samples[i]
        img = img.unsqueeze(0).to(device)

        _, pred, entropy_map = predict_with_tta(model, img)

        gt = gt_mask.cpu().numpy().astype(np.uint8)

        dice1, iou1 = dice_iou_for_class(pred, gt, cls=1)
        dice2, iou2 = dice_iou_for_class(pred, gt, cls=2)

        pred_out_path = os.path.join(pred_dir, f"Pred_mask_{cid}.tif")
        save_scaled_mask(pred, pred_out_path)

        unc_out_path = os.path.join(unc_dir, f"Uncertainty_{cid}.tif")
        save_uncertainty_map(entropy_map, unc_out_path)

        rows.append({
            "cell_id": cid,
            "dice_class1": dice1,
            "iou_class1": iou1,
            "dice_class2": dice2,
            "iou_class2": iou2,
            "mean_entropy": float(entropy_map.mean()),
            "max_entropy": float(entropy_map.max()),
            "pred_mask_file": os.path.basename(pred_out_path),
            "uncertainty_file": os.path.basename(unc_out_path),
        })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved metrics:", csv_path)
    print("Saved predictions:", pred_dir)
    print("Saved uncertainty maps:", unc_dir)
    print("Seconds:", int(time.time() - t0))

    print("\nAverages on val:")
    print("Mean Dice class1:", float(df["dice_class1"].mean()))
    print("Mean IoU  class1:", float(df["iou_class1"].mean()))
    print("Mean Dice class2:", float(df["dice_class2"].mean()))
    print("Mean IoU  class2:", float(df["iou_class2"].mean()))
    print("Mean entropy:", float(df["mean_entropy"].mean()))

    print("\nWorst 5 cells by Dice (class2):")
    worst = df.sort_values("dice_class2", ascending=True).head(5)
    print(worst[["cell_id", "dice_class2", "iou_class2", "mean_entropy"]].to_string(index=False))


if __name__ == "__main__":
    main()