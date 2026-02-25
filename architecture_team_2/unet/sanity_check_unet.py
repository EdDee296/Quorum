"""
sanity_check_unet.py

this file checks that the dataset pipeline is working correctly
before starting to train U-Net.
"""
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.dataset import CellDataset


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
        val_ids_path="val_ids.txt",
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
