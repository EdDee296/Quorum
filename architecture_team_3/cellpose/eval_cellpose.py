"""
eval_cellpose.py

this file evaluates trained Cellpose models using the same shared project setup style

it is doing these things
- reading config.yaml for data path and settings
- reading val_ids.txt so the validation split stays fixed
- loading the dataset using the shared data_utils/dataset.py pipeline
- converting semantic masks into instance masks for Cellpose evaluation
- loading saved Cellpose models from backend/models
- evaluating pretrained, no-aug, and aug models
- reporting mean Dice and IoU

before running
- make sure config.yaml data_root is correct on your machine
- make sure the trained Cellpose models already exist in backend/models
- do not edit val_ids.txt

how to run
from the project root
python -m architecture_team_3.cellpose.eval_cellpose
"""

import os
import yaml
import numpy as np
import torch
from cellpose import models

from data_utils.dataset import CellDataset
from architecture_team_3.cellpose.cellpose_utils import CellposeHelper


def load_config():
    """
    loads config.yaml from the project root and returns it as a dict
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_val_ids(root_dir, preprocess_mode, target_size, val_ids_path):
    """
    builds the validation split in the same way as unet++

    what it does
    - reads val_ids.txt to get the fixed validation ids
    - scans the dataset to find all valid cell ids
    - keeps only ids that exist in the dataset
    """
    val_ids = CellDataset.load_split_ids(val_ids_path)

    ds_all = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=None,
    )

    all_ids = set(ds_all.samples)

    filtered_val_ids = []
    i = 0
    while i < len(val_ids):
        cid = val_ids[i]
        if cid in all_ids:
            filtered_val_ids.append(cid)
        i += 1

    return filtered_val_ids


def build_val_dataset(root_dir, preprocess_mode, target_size, val_ids):
    """
    builds validation dataset
    """
    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )
    return val_ds


def evaluate_model(model, imgs, gt_masks, model_name):
    """
    evaluates one Cellpose model on the validation set

    output
    - mean_iou
    - mean_dice
    """
    print(f"\nEvaluating {model_name}...")

    iou_scores = []
    dice_scores = []

    i = 0
    while i < len(imgs):
        img = imgs[i]
        gt = gt_masks[i]

        pred_masks, _flows, _styles = model.eval(
            img,
            diameter=None,
            channels=[0, 0],
        )

        iou, dice = CellposeHelper.calc_metrics(pred_masks, gt)
        iou_scores.append(iou)
        dice_scores.append(dice)

        i += 1

    mean_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0
    mean_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0

    print(f"{model_name} - Mean IoU:  {mean_iou:.4f}")
    print(f"{model_name} - Mean Dice: {mean_dice:.4f}")

    return mean_iou, mean_dice


def main():
    """
    main evaluation entry point

    what this function does
    - reading config.yaml
    - building validation split using val_ids.txt
    - creating validation dataset using shared data_utils pipeline
    - converting masks into Cellpose instance format
    - loading pretrained and trained Cellpose models
    - evaluating all available models
    """
    cfg = load_config()

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    if os.path.exists("val_ids.txt"):
        val_ids_path = "val_ids.txt"
    else:
        val_ids_path = os.path.join("architecture_team_1", "unetpp", "val_ids.txt")

    model_dir = os.path.join("backend", "models")
    use_gpu = torch.cuda.is_available()

    print("data_root:", root_dir)
    print("preprocess_mode:", preprocess_mode)
    print("target_size:", target_size)
    print("val_ids_path:", val_ids_path)
    print("model_dir:", model_dir)
    print("use_gpu:", use_gpu)

    val_ids = build_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path=val_ids_path,
    )

    print("val images:", len(val_ids))

    val_ds = build_val_dataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids=val_ids,
    )

    print("Preparing validation data for Cellpose...")
    val_imgs, val_masks = CellposeHelper.extract_instances_for_cellpose(val_ds)

    print(f"Validation set ready: {len(val_imgs)} images")

    # pretrained baseline
    model_pretrained = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    # trained models
    path_no_aug = os.path.join(model_dir, "cp_chromo_no_aug")
    path_aug = os.path.join(model_dir, "cp_chromo_aug")

    model_no_aug = None
    model_aug = None

    if os.path.exists(path_no_aug):
        print(f"Loading trained no-aug model from {path_no_aug}")
        model_no_aug = models.CellposeModel(gpu=use_gpu, pretrained_model=path_no_aug)
    else:
        print(f"Could not find no-aug model at {path_no_aug}")

    if os.path.exists(path_aug):
        print(f"Loading trained aug model from {path_aug}")
        model_aug = models.CellposeModel(gpu=use_gpu, pretrained_model=path_aug)
    else:
        print(f"Could not find aug model at {path_aug}")

    results = []

    pretrained_iou, pretrained_dice = evaluate_model(
        model_pretrained,
        val_imgs,
        val_masks,
        "Pretrained nuclei baseline",
    )
    results.append(("Pretrained nuclei baseline", pretrained_iou, pretrained_dice))

    if model_no_aug is not None:
        no_aug_iou, no_aug_dice = evaluate_model(
            model_no_aug,
            val_imgs,
            val_masks,
            "Trained chromocenter model (no aug)",
        )
        results.append(("Trained chromocenter model (no aug)", no_aug_iou, no_aug_dice))

    if model_aug is not None:
        aug_iou, aug_dice = evaluate_model(
            model_aug,
            val_imgs,
            val_masks,
            "Trained chromocenter model (aug)",
        )
        results.append(("Trained chromocenter model (aug)", aug_iou, aug_dice))

    print("\nFinal summary:")
    i = 0
    while i < len(results):
        name, mean_iou, mean_dice = results[i]
        print(f"{name}: IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
        i += 1


if __name__ == "__main__":
    main()