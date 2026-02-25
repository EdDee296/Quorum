"""
train_unet.py

this file trains a U-Net model for 3-class segmentation.
"""

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from data_utils.dataset import CellDataset


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_val_ids(root_dir, preprocess_mode, target_size, val_ids_path):
    val_ids = CellDataset.load_split_ids(val_ids_path)
    val_set = set(val_ids)

    ds_all = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=None,
    )

    all_ids = ds_all.samples

    train_ids = []
    i = 0
    while i < len(all_ids):
        cid = all_ids[i]
        if cid not in val_set:
            train_ids.append(cid)
        i += 1

    return train_ids, val_ids


def dice_iou_for_class(pred_hw, target_hw, class_id):
    pred_fg = (pred_hw == class_id)
    targ_fg = (target_hw == class_id)

    inter = np.logical_and(pred_fg, targ_fg).sum()
    union = np.logical_or(pred_fg, targ_fg).sum()

    pred_sum = pred_fg.sum()
    targ_sum = targ_fg.sum()

    denom = pred_sum + targ_sum
    dice = 1.0 if denom == 0 else (2.0 * inter) / float(denom)
    iou = 1.0 if union == 0 else inter / float(union)

    return float(dice), float(iou)


def eval_one_epoch(model, loader, device):
    model.eval()

    dice1_sum = 0.0
    iou1_sum = 0.0
    dice2_sum = 0.0
    iou2_sum = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            b = 0
            while b < preds.shape[0]:
                pred_np = preds[b].detach().cpu().numpy()
                targ_np = masks[b].detach().cpu().numpy()

                d1, j1 = dice_iou_for_class(pred_np, targ_np, class_id=1)
                d2, j2 = dice_iou_for_class(pred_np, targ_np, class_id=2)

                dice1_sum += d1
                iou1_sum += j1
                dice2_sum += d2
                iou2_sum += j2
                count += 1
                b += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        dice1_sum / float(count),
        iou1_sum / float(count),
        dice2_sum / float(count),
        iou2_sum / float(count),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    loss_total = 0.0
    count = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        count += 1

    if count == 0:
        return 0.0

    return loss_total / float(count)


def main():
    cfg = load_config()

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))
    aug_strength = str(cfg.get("augmentation", "standard")).lower()

    out_dir = os.path.join("architecture_team_2", "unet", "runs_unet")
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device:", device)
    print("data_root:", root_dir)
    print("preprocess_mode:", preprocess_mode)
    print("target_size:", target_size)
    print("augmentation:", aug_strength)
    print("out_dir:", out_dir)

    train_ids, val_ids = build_train_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path="val_ids.txt",
    )

    train_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength=aug_strength,
        target_size=target_size,
        split_ids=train_ids,
    )

    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_dice2 = -1.0
    best_path = os.path.join(out_dir, "best_unet.pt")

    log_path = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_dice_class1,val_iou_class1,val_dice_class2,val_iou_class2\n")

    epochs = int(cfg.get("epochs", 18))
    epoch = 1
    while epoch <= epochs:
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_dice1, val_iou1, val_dice2, val_iou2 = eval_one_epoch(model, val_loader, device)

        dt = time.time() - t0
        print(
            "Epoch",
            epoch,
            "| train_loss:",
            round(train_loss, 6),
            "| val_dice(nuc):",
            round(val_dice1, 6),
            "| val_iou(nuc):",
            round(val_iou1, 6),
            "| val_dice(chrom):",
            round(val_dice2, 6),
            "| val_iou(chrom):",
            round(val_iou2, 6),
            "| sec:",
            int(dt),
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                str(epoch) + "," + str(train_loss) + "," +
                str(val_dice1) + "," + str(val_iou1) + "," +
                str(val_dice2) + "," + str(val_iou2) + "\n"
            )

        if val_dice2 > best_val_dice2:
            best_val_dice2 = val_dice2
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice_class1": val_dice1,
                    "val_iou_class1": val_iou1,
                    "val_dice_class2": val_dice2,
                    "val_iou_class2": val_iou2,
                    "config": cfg,
                },
                best_path,
            )
            print("Saved best >", best_path)

        epoch += 1

    print("Done. Best val Dice (class2):", best_val_dice2)


if __name__ == "__main__":
    main()
