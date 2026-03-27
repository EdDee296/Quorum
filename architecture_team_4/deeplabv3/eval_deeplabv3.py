"""
eval_deeplabv3.py

this file is used for evaluating the trained DeepLabV3 model on the fixed validation split.

what this file does:

- loads config.yaml to get dataset path and preprocessing settings
- loads val_ids.txt to use the exact same validation set every time
- loads the trained DeepLabV3 checkpoint (best_deeplabv3.pt)
- runs inference on every validation image
- predicts 3-class segmentation (0 background, 1 nucleus, 2 chromocenter)
- computes Dice and IoU for:
    class 1 (nucleus)
    class 2 (chromocenter)
- saves predicted masks (scaled for viewing: 0=black, 1=gray, 2=white)
- saves per-image metrics into metrics.csv
- prints average metrics and worst 5 cells for chromocenter

important before running:

- make sure you already trained the model using train_deeplabv3.py
- make sure best_deeplabv3.pt exists inside:
    architecture_team_4/deeplabv3/runs_deeplabv3/
- make sure config.yaml has correct data_root path
- make sure val_ids.txt has not been modified

how to run:

from project root folder:

python -m architecture_team_4.deeplabv3.eval_deeplabv3

outputs:

- metrics.csv saved to:
    architecture_team_4/deeplabv3/outputs_deeplabv3/
- predicted masks saved to:
    architecture_team_4/deeplabv3/outputs_deeplabv3/pred_masks/

notes:

- this evaluates both nucleus and chromocenter
- background is not evaluated because it is trivial and would inflate metrics
- chromocenter Dice is usually the main performance number
"""

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import torch
import tifffile as tiff

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils.dataset import CellDataset
from architecture_team_4.deeplabv3_model import build_deeplabv3


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repeat_to_3ch(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def dice_iou_for_class(pred, gt, cls):
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

    val_ids = CellDataset.load_split_ids(os.path.join(PROJECT_ROOT, "val_ids.txt"))

    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )

    ckpt_path = os.path.join(THIS_DIR, "runs_deeplabv3", "best_deeplabv3.pt")
    out_dir = os.path.join(THIS_DIR, "outputs_deeplabv3")
    pred_dir = os.path.join(out_dir, "pred_masks")

    os.makedirs(pred_dir, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", ckpt_path)
    print("Out dir:", out_dir)

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")

    model = build_deeplabv3(
        backbone=backbone,
        pretrained_backbone=False,
        out_channels=3,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []

    i = 0
    while i < len(val_ds):
        img, gt_mask = val_ds[i]
        cid = val_ds.samples[i]

        img = img.unsqueeze(0).to(device)
        img = repeat_to_3ch(img)

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

        i += 1

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