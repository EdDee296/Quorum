"""
train_unetpp.py

this file is training a U-Net++ model for 3-class segmentation

it is doing these things
- reading config.yaml for data path and settings
- reading val_ids.txt so validation split stays fixed for everyone
- building train ids by taking everything not in val_ids
- loading the dataset using the shared data_utils/dataset.py pipeline
- training a 3-class U-Net++ (background, nucleus, chromocenter)
- using CrossEntropyLoss because this is multi-class segmentation
- after every epoch it is calculating dice and iou for nucleus (class 1) and chromocenter (class 2)
- saving the best checkpoint based on chromocenter dice because that is the harder and more important class for us
- writing a train_log.csv so we can look at learning progress later

before running
- make sure config.yaml data_root is correct on your machine
- do not edit val_ids.txt
- make sure segmentation_models_pytorch is installed
- sanity check should already be passing so we know shapes and labels are correct

how to run
from the project root
python -m architecture_team_1.unetpp.train_unetpp

outputs
- best checkpoint
  architecture_team_1/unetpp/runs_unetpp/best_unetpp.pt
- backend copy of best checkpoint
  backend/models/best_unetpp.pt
- training log
  architecture_team_1/unetpp/runs_unetpp/train_log.csv
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
    """
    loads config.yaml from the project root and returns it as a dict
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_val_ids(root_dir, preprocess_mode, target_size, val_ids_path):
    """
    builds the train/val split in a reproducible way

    what it does
    - reads val_ids.txt to get the fixed validation ids
    - scans the dataset to find all valid cell ids
    - returns train_ids = all_ids minus val_ids

    output
    - train_ids: list[str]
    - val_ids: list[str]
    """
    # reading fixed validation ids
    val_ids = CellDataset.load_split_ids(val_ids_path)
    val_set = set(val_ids)

    # scanning dataset to find all valid cell ids
    ds_all = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=None,
    )

    all_ids = ds_all.samples

    # building train_ids by excluding validation ids
    train_ids = []
    i = 0
    while i < len(all_ids):
        cid = all_ids[i]
        if cid not in val_set:
            train_ids.append(cid)
        i += 1

    return train_ids, val_ids


def dice_iou_for_class(pred_hw, target_hw, class_id):
    """
    computes Dice + IoU for ONE class only

    inputs
    - pred_hw: [H,W] ints (0/1/2)
    - target_hw: [H,W] ints (0/1/2)
    - class_id: 1 (nucleus) or 2 (chromocenter)

    output
    - (dice, iou): floats
    """
    # selecting only pixels that belong to this class
    pred_fg = (pred_hw == class_id)
    targ_fg = (target_hw == class_id)

    # computing intersection and union
    inter = np.logical_and(pred_fg, targ_fg).sum()
    union = np.logical_or(pred_fg, targ_fg).sum()

    # counting foreground pixels
    pred_sum = pred_fg.sum()
    targ_sum = targ_fg.sum()

    denom = pred_sum + targ_sum
    if denom == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / float(denom)

    if union == 0:
        iou = 1.0
    else:
        iou = inter / float(union)

    return float(dice), float(iou)


def eval_one_epoch(model, loader, device):
    """
    runs 1 full pass over the validation loader (no gradients)

    what it does
    - model predicts logits [B,3,H,W]
    - we take argmax -> predicted class per pixel [B,H,W]
    - compute dice/iou for nucleus (class 1) and chromocenter (class 2)

    output
    - val_dice1, val_iou1, val_dice2, val_iou2 (averaged across all images)
    """
    model.eval()  # turning off dropout/batchnorm updates

    # accumulating metrics across validation set
    dice1_sum = 0.0
    iou1_sum = 0.0
    dice2_sum = 0.0
    iou2_sum = 0.0
    count = 0

    with torch.no_grad():  # disabling gradient computation
        for imgs, masks in loader:
            imgs = imgs.to(device)    
            masks = masks.to(device)  

            logits = model(imgs)                 
            preds = torch.argmax(logits, dim=1)  # predicted class per pixel

            # looping through batch images
            b = 0
            while b < preds.shape[0]:
                pred_np = preds[b].detach().cpu().numpy()
                targ_np = masks[b].detach().cpu().numpy()

                d1, j1 = dice_iou_for_class(pred_np, targ_np, class_id=1)  # nucleus
                d2, j2 = dice_iou_for_class(pred_np, targ_np, class_id=2)  # chromocenter

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
    """
    runs 1 full training epoch

    what it does
    - forward pass - logits [B,3,H,W]
    - CrossEntropyLoss compares logits vs masks [B,H,W]
    - backward + optimizer step

    output
    - average training loss for this epoch
    """
    model.train() # enabling training mode

    loss_total = 0.0
    count = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)          # forward pass
        loss = loss_fn(logits, masks) # multi-class cross entropy

        optimizer.zero_grad() # clearing old gradients
        loss.backward() # backpropagation
        optimizer.step()  # updating weights

        loss_total += float(loss.item())
        count += 1

    if count == 0:
        return 0.0

    return loss_total / float(count)


def main():
    """
    main training entry point.

    what this function does:
    - reading config.yaml for dataset + training settings
    - building train/validation split using val_ids.txt
    - creating datasets and dataloaders
    - building U-Net++ model (3 classes)
    - training for multiple epochs
    - computing validation Dice/IoU for nucleus and chromocenter
    - saving best checkpoint based on chromocenter Dice
    - logging training progress into train_log.csv

    outputs:
    - best_unetpp.pt saved in runs_unetpp/
    - best_unetpp.pt copied to backend/models/
    - train_log.csv saved in runs_unetpp/   
    """
    cfg = load_config()

    # reading config values
    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))
    aug_strength = str(cfg.get("augmentation", "standard")).lower()

    out_dir = os.path.join("architecture_team_1", "unetpp", "runs_unetpp")
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

    # building train/val split
    train_ids, val_ids = build_train_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path="val_ids.txt",
    )

    # creating training dataset (with augmentation)
    train_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength=aug_strength,
        target_size=target_size,
        split_ids=train_ids,
    )

    # creating validation dataset (no augmentation)
    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # we are saving best based on chromocenter dice (class 2)
    best_val_dice2 = -1.0
    best_path = os.path.join(out_dir, "best_unetpp.pt")

    backend_best_path = os.path.join("backend", "models", "best_unetpp.pt")
    os.makedirs(os.path.join("backend", "models"), exist_ok=True)

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

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice_class1": val_dice1,
                "val_iou_class1": val_iou1,
                "val_dice_class2": val_dice2,
                "val_iou_class2": val_iou2,
                "config": cfg,
            }

            torch.save(checkpoint_data, best_path)
            torch.save(checkpoint_data, backend_best_path)

            print("Saved best >", best_path)
            print("Saved backend copy >", backend_best_path)

        epoch += 1

    print("Done. Best val Dice (class2):", best_val_dice2)


if __name__ == "__main__":
    main()