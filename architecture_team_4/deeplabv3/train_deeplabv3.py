"""
train_deeplabv3.py

this file is used for training DeepLabV3 on the fixed train/validation split.

what this file does:

- loads config.yaml to get dataset path and preprocessing settings
- loads val_ids.txt to build the exact same validation set every time
- creates train and validation datasets using CellDataset
- trains DeepLabV3 for 3-class segmentation
- predicts 3 classes:
    0 background
    1 nucleus
    2 chromocenter
- computes Dice and IoU on the validation split
- saves the best checkpoint based on mean foreground Dice

important before running:

- make sure config.yaml has correct data_root
- make sure val_ids.txt has not been modified
- make sure dataset folder structure is correct

how to run:

from project root:

python -m architecture_team_4.deeplabv3.train_deeplabv3

outputs:

- best checkpoint saved to:
    architecture_team_4/deeplabv3/runs_deeplabv3/best_deeplabv3.pt
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils.dataset import CellDataset
from architecture_team_4.deeplabv3_model import build_deeplabv3


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repeat_to_3ch(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def dice_iou_for_class_tensor(pred, gt, cls, eps=1e-6):
    p = (pred == cls).float()
    g = (gt == cls).float()

    inter = (p * g).sum(dim=(1, 2))
    p_sum = p.sum(dim=(1, 2))
    g_sum = g.sum(dim=(1, 2))
    union = p_sum + g_sum - inter

    dice = (2.0 * inter + eps) / (p_sum + g_sum + eps)
    iou = (inter + eps) / (union + eps)

    return float(dice.mean().item()), float(iou.mean().item())


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


def train_one_epoch(model, loader, optimizer, criterion, device, amp):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    total_loss = 0.0
    total_count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        images = repeat_to_3ch(images)

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    dice_class1_all = []
    iou_class1_all = []
    dice_class2_all = []
    iou_class2_all = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        images = repeat_to_3ch(images)

        logits = model(images)
        loss = criterion(logits, masks)

        pred = torch.argmax(logits, dim=1)

        dice1, iou1 = dice_iou_for_class_tensor(pred, masks, cls=1)
        dice2, iou2 = dice_iou_for_class_tensor(pred, masks, cls=2)

        dice_class1_all.append(dice1)
        iou_class1_all.append(iou1)
        dice_class2_all.append(dice2)
        iou_class2_all.append(iou2)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    val_loss = total_loss / max(1, total_count)

    dice_class1 = float(np.mean(dice_class1_all))
    iou_class1 = float(np.mean(iou_class1_all))
    dice_class2 = float(np.mean(dice_class2_all))
    iou_class2 = float(np.mean(iou_class2_all))

    mean_dice_fg = float((dice_class1 + dice_class2) / 2.0)
    mean_iou_fg = float((iou_class1 + iou_class2) / 2.0)

    return val_loss, dice_class1, iou_class1, dice_class2, iou_class2, mean_dice_fg, mean_iou_fg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class0_weight", type=float, default=1.0)
    parser.add_argument("--class1_weight", type=float, default=1.0)
    parser.add_argument("--class2_weight", type=float, default=5.0)

    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--pretrained_backbone", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config()

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))
    augmentation = str(cfg.get("augmentation", "standard")).lower()

    train_ids, val_ids = build_train_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path=os.path.join(PROJECT_ROOT, "val_ids.txt"),
    )

    train_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength=augmentation,
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

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    amp = (device == "cuda")
    pin_memory = (device == "cuda")

    print("device:", device)
    print("train dataset size:", len(train_ds))
    print("val dataset size:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_deeplabv3(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        out_channels=3,
    ).to(device)

    class_weights = torch.tensor(
        [args.class0_weight, args.class1_weight, args.class2_weight],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = os.path.join(THIS_DIR, "runs_deeplabv3")
    os.makedirs(run_dir, exist_ok=True)

    best_ckpt = os.path.join(run_dir, "best_deeplabv3.pt")
    best_mean_dice_fg = -1.0

    print("=========== Training DeepLabV3 ===========")

    epoch = 1
    while epoch <= args.epochs:
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, amp)
        val_loss, dice1, iou1, dice2, iou2, mean_dice_fg, mean_iou_fg = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"dice=['{dice1:.3f}', '{dice2:.3f}'] | "
            f"iou=['{iou1:.3f}', '{iou2:.3f}'] | "
            f"mean_dice_fg={mean_dice_fg:.4f} | "
            f"mean_iou_fg={mean_iou_fg:.4f} | "
            f"seconds={int(time.time() - t0)}"
        )

        if mean_dice_fg > best_mean_dice_fg:
            best_mean_dice_fg = mean_dice_fg
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_mean_dice_fg": best_mean_dice_fg,
                    "backbone": args.backbone,
                },
                best_ckpt,
            )

        epoch += 1

    print(f"[Info] Best val mean_dice_fg for deeplabv3: {best_mean_dice_fg:.4f}")
    print(f"[Info] Saved best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()