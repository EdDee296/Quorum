"""
eval_unet.py

this file evaluates the trained U-Net model on the fixed validation split.
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
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dice_iou_for_class(pred, gt, cls):
    p = (pred == cls)
    g = (gt == cls)

    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()

    denom = p_sum + g_sum
    dice = 1.0 if denom == 0 else (2.0 * inter) / float(denom)
    iou = 1.0 if union == 0 else inter / float(union)
    return float(dice), float(iou)


def save_scaled_mask(mask_hw, out_path):
    vis = np.zeros_like(mask_hw, dtype=np.uint8)
    vis[mask_hw == 1] = 127
    vis[mask_hw == 2] = 255
    tiff.imwrite(out_path, vis)


def main():
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

    ckpt_path = os.path.join("architecture_team_2", "unet", "runs_unet", "best_unet.pt")
    out_dir = os.path.join("architecture_team_2", "unet", "outputs_unet")
    pred_dir = os.path.join(out_dir, "pred_masks")

    os.makedirs(pred_dir, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", ckpt_path)
    print("Out dir:", out_dir)

    model = smp.Unet(
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

        with torch.no_grad():
            logits = model(img)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        gt = gt_mask.cpu().numpy().astype(np.uint8)

        dice1, iou1 = dice_iou_for_class(pred, gt, cls=1)
        dice2, iou2 = dice_iou_for_class(pred, gt, cls=2)

        out_path = os.path.join(pred_dir, f"Pred_mask_{cid}.tif")
        save_scaled_mask(pred, out_path)

        rows.append({
            "cell_id": cid,
            "dice_class1": dice1,
            "iou_class1": iou1,
            "dice_class2": dice2,
            "iou_class2": iou2,
            "pred_mask_file": os.path.basename(out_path),
        })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved metrics:", csv_path)
    print("Saved predictions:", pred_dir)
    print("Seconds:", int(time.time() - t0))

    print("\nAverages on val:")
    print("Mean Dice class1:", float(df["dice_class1"].mean()))
    print("Mean IoU  class1:", float(df["iou_class1"].mean()))
    print("Mean Dice class2:", float(df["dice_class2"].mean()))
    print("Mean IoU  class2:", float(df["iou_class2"].mean()))

    print("\nWorst 5 cells by Dice (class2):")
    worst = df.sort_values("dice_class2", ascending=True).head(5)
    print(worst[["cell_id", "dice_class2", "iou_class2"]].to_string(index=False))


if __name__ == "__main__":
    main()
