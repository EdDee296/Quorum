"""
train_cellpose.py

this file trains Cellpose models using the same shared project setup style
we already use for unet++

it is doing these things
- reading config.yaml for data path and settings
- reading val_ids.txt so the validation split stays fixed
- building train ids by taking everything not in val_ids
- loading the dataset using the shared data_utils/dataset.py pipeline
- converting semantic masks into instance masks for Cellpose training
- training Cellpose models for chromocenters and nuclei
- saving trained models into backend/models

before running
- make sure config.yaml data_root is correct on your machine
- do not edit val_ids.txt
- make sure cellpose is installed

how to run
from the project root
python -m architecture_team_3.cellpose.train_cellpose
"""

import os
import yaml
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


def build_train_val_ids(root_dir, preprocess_mode, target_size, val_ids_path):
    """
    builds the train/val split in the same way as unet++

    what it does
    - reads val_ids.txt to get the fixed validation ids
    - scans the dataset to find all valid cell ids
    - returns train_ids = all_ids minus val_ids
    """
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


def build_datasets(root_dir, preprocess_mode, target_size, aug_strength, train_ids, val_ids):
    """
    builds train and validation datasets
    """
    train_ds_no_aug = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=train_ids,
    )

    train_ds_aug = CellDataset(
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

    return train_ds_no_aug, train_ds_aug, val_ds


def train_chromocenter_models(train_ds_no_aug, train_ds_aug, val_ds, model_save_dir, use_gpu=True):
    """
    trains two chromocenter Cellpose models
    - one with no augmentation
    - one with augmentation
    """
    print("Preparing chromocenter data for Cellpose...")

    train_imgs_na, train_masks_na = CellposeHelper.extract_instances_for_cellpose(train_ds_no_aug)
    train_imgs_a, train_masks_a = CellposeHelper.extract_instances_for_cellpose(train_ds_aug)
    val_imgs, val_masks = CellposeHelper.extract_instances_for_cellpose(val_ds)

    print("Extraction Complete:")
    print(f"Training set (no aug): {len(train_imgs_na)} images")
    print(f"Training set (aug): {len(train_imgs_a)} images")
    print(f"Validation set: {len(val_imgs)} images")

    model_no_aug = models.CellposeModel(gpu=use_gpu, model_type="nuclei")
    model_aug = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    print("\nTraining Model: Non-Augmented")
    path_no_aug = model_no_aug.train(
        train_imgs_na,
        train_masks_na,
        test_data=val_imgs,
        test_labels=val_masks,
        channels=[0, 0],
        save_path=model_save_dir,
        n_epochs=17,
        model_name="cp_chromo_no_aug",
    )
    print(f"Saved non-augmented model to: {path_no_aug}")

    print("\nTraining Model: Augmented")
    path_aug = model_aug.train(
        train_imgs_a,
        train_masks_a,
        test_data=val_imgs,
        test_labels=val_masks,
        channels=[0, 0],
        save_path=model_save_dir,
        n_epochs=17,
        model_name="cp_chromo_aug",
    )
    print(f"Saved augmented model to: {path_aug}")

    return {
        "model_no_aug": model_no_aug,
        "model_aug": model_aug,
        "path_no_aug": path_no_aug,
        "path_aug": path_aug,
    }


def train_nucleus_model(train_ds_no_aug, val_ds, model_save_dir, use_gpu=True):
    """
    trains a nucleus Cellpose model
    """
    print("Preparing nucleus data for Cellpose...")

    train_imgs, train_masks = CellposeHelper.extract_nucleus_instances_for_cellpose(train_ds_no_aug)
    val_imgs, val_masks = CellposeHelper.extract_nucleus_instances_for_cellpose(val_ds)

    print(f"Nucleus train set: {len(train_imgs)} images")
    print(f"Nucleus val set: {len(val_imgs)} images")

    model_nucleus = models.CellposeModel(gpu=use_gpu, model_type="nuclei")

    print("\nTraining nucleus model")
    path_nucleus = model_nucleus.train(
        train_imgs,
        train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        channels=[0, 0],
        save_path=model_save_dir,
        n_epochs=17,
        model_name="cp_nucleus",
        min_train_masks=1,
    )
    print(f"Saved nucleus model to: {path_nucleus}")

    return {
        "model_nucleus": model_nucleus,
        "path_nucleus": path_nucleus,
    }


def main():
    """
    main training entry point

    what this function does
    - reading config.yaml
    - building train/validation split using val_ids.txt
    - creating datasets using shared data_utils pipeline
    - training Cellpose models
    - saving trained models into backend/models
    """
    cfg = load_config()

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))
    aug_strength = str(cfg.get("augmentation", "standard")).lower()

    backend_model_dir = os.path.join("backend")
    os.makedirs(backend_model_dir, exist_ok=True)

    if os.path.exists("val_ids.txt"):
        val_ids_path = "val_ids.txt"
    else:
        val_ids_path = os.path.join("architecture_team_1", "unetpp", "val_ids.txt")

    use_gpu = torch.cuda.is_available()

    print("data_root:", root_dir)
    print("preprocess_mode:", preprocess_mode)
    print("target_size:", target_size)
    print("augmentation:", aug_strength)
    print("model_save_dir:", backend_model_dir)
    print("val_ids_path:", val_ids_path)
    print("use_gpu:", use_gpu)

    train_ids, val_ids = build_train_val_ids(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        val_ids_path=val_ids_path,
    )

    print("train images:", len(train_ids))
    print("val images:", len(val_ids))

    train_ds_no_aug, train_ds_aug, val_ds = build_datasets(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        target_size=target_size,
        aug_strength=aug_strength,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    train_chromocenter_models(
        train_ds_no_aug=train_ds_no_aug,
        train_ds_aug=train_ds_aug,
        val_ds=val_ds,
        model_save_dir=backend_model_dir,
        use_gpu=use_gpu,
    )

    train_nucleus_model(
        train_ds_no_aug=train_ds_no_aug,
        val_ds=val_ds,
        model_save_dir=backend_model_dir,
        use_gpu=use_gpu,
    )

    print("\nDone training Cellpose models.")


if __name__ == "__main__":
    main()