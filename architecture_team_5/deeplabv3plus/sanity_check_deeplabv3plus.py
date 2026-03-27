"""
sanity_check_deeplabv3plus.py

this file is used for checking that the dataset pipeline is working correctly
before starting to train DeepLabV3+.

what this file is doing:

- reading config.yaml to get data_root and preprocessing settings
- reading val_ids.txt to build fixed train and validation splits
- creating train and val datasets using CellDataset
- checking dataset sizes
- checking image shape (should be 1,H,W)
- checking mask shape (should be H,W)
- checking that mask labels are only {0,1,2}
- loading a small batch with DataLoader to make sure batching works
- printing pixel counts to confirm class distribution looks reasonable

this file does NOT train anything.
it is only verifying shapes, labels, splits, and data loading.

before running:

- make sure config.yaml has correct data_root
- make sure val_ids.txt exists
- make sure dataset folder structure is correct

how to run:

from project root:

python -m architecture_team_5.deeplabv3plus.sanity_check_deeplabv3plus

if this runs without crashing and labels are {0,1,2},
then the dataset is safe and training can start.
"""

import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils.dataset import CellDataset


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
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


def main():
    cfg = load_config()

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    print("data_root:", root_dir)
    print("preprocess_mode:", preprocess_mode)
    print("target_size:", target_size)

    train_ids, val_ids = build_train_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path=os.path.join(PROJECT_ROOT, "val_ids.txt"),
    )

    print("train_ids count:", len(train_ids))
    print("val_ids count:", len(val_ids))

    train_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength=str(cfg.get("augmentation", "standard")).lower(),
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

    print("train dataset size:", len(train_ds))
    print("val dataset size:", len(val_ds))

    img, mask = val_ds[0]
    print("single image shape:", tuple(img.shape))
    print("single mask shape:", tuple(mask.shape))

    uniq = torch.unique(mask).cpu().numpy().tolist()
    print("mask unique labels:", uniq)

    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    images, masks = next(iter(loader))
    print("batch images shape:", tuple(images.shape))
    print("batch masks shape:", tuple(masks.shape))

    uniq2 = torch.unique(masks).cpu().numpy().tolist()
    print("batch mask unique labels:", uniq2)

    masks_np = masks.cpu().numpy()
    counts = {0: 0, 1: 0, 2: 0}

    b = 0
    while b < masks_np.shape[0]:
        m = masks_np[b]
        counts[0] += int(np.sum(m == 0))
        counts[1] += int(np.sum(m == 1))
        counts[2] += int(np.sum(m == 2))
        b += 1

    print("pixel counts in this batch:", counts)
    print("Sanity check done.")


if __name__ == "__main__":
    main()