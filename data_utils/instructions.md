# Data Pipeline Technical Instructions

This document explains how to use the `data_utils/dataset.py` module. I have standardized this pipeline to ensure everyone is training and evaluating on the same processed images.

---


## Quick Start

To use the dataset in your notebook or script:

```python
from data_utils.dataset import CellDataset

# 1. Load the fixed validation IDs from the text file
val_ids = CellDataset.load_split_ids('val_ids.txt')

# 2. Read config.yaml for all parameters (recommended)
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 3. Define your 'train_ids' by excluding val_ids from the full list
# (Assume you have a helper or logic for this)

"""
Augmentation strength must be a string: 'none', 'light', or 'standard' (case-insensitive).
Use 'none' for no augmentation (only resizing), 'light' for flips/rotations, 'standard' for full augmentations.
Passing None or any other value will raise an error.
"""

# 4. Instantiate for Training (Apply augmentations, exclude val_ids)
train_ds = CellDataset(
    root_dir=config['data_root'],
    preprocess_mode=config['preprocess_mode'],
    aug_strength=str(config['augmentation']).lower(),  # must be 'none', 'light', or 'standard'
    target_size=tuple(config['target_size']),
    split_ids=train_ids
)

# 5. Instantiate for Evaluation (No augmentations, use ONLY val_ids)
val_ds = CellDataset(
    root_dir=config['data_root'],
    preprocess_mode=config['preprocess_mode'],
    aug_strength='none',  # must be string 'none' for no augmentation
    target_size=tuple(config['target_size']),
    split_ids=val_ids
)
```

**Best Practice:**
- Always use a string for `aug_strength`: 'none', 'light', or 'standard' (case-insensitive).
- Do not use `None` or other values.
- Read `data_root` and other parameters from `config.yaml` instead of hardcoding them. This ensures all team members use the same settings and makes your code portable between Colab and local environments.

---

## Preprocessing Modes

The `preprocess_mode` parameter is **strict**. If you pass anything other than these two, the code will raise a `ValueError`.

| Mode   | Description                                                                 | When to use                |
|--------|-----------------------------------------------------------------------------|----------------------------|
| basic  | 1%-99.5% percentile clipping + [0,1] normalization.                        | Quick baseline testing.     |
| full   | **Recommended.** Bilateral Filtering (noise reduction) and Gaussian-based Background Subtraction. | Final training and evaluation. |

---

## Output Format

Every item returned by `dataset[i]` follows this format:

- **Image Tensor:** `(1, H, W)` - Grayscale, `Float32`.
- **Mask Tensor:** `(H, W)` - Long (`Int64`).
- **Mask Label Key:**
    - `0`: Background (Black)
    - `1`: Nucleus (Grey)
    - `2`: Chromocenters (White)

**Warning:** Your model's final layer must output **3 channels** (one for each class) if you are using `CrossEntropyLoss`.

---

## Augmentation Strengths

The `aug_strength` parameter controls the albumentations pipeline:

- `None`: Only Resizing (**Use this for Evaluation.**)
- `'light'`: Flips and 90-degree rotations.
- `'standard'`: Flips, rotations, elastic transforms, and brightness/contrast shifts.

---

## Expected Data Structure

The `root_dir` you pass must contain these exact folder names:

```
Dataset_Root/
├── Microscopy_images/      # .tif files
└── Ground_truth_masks/     # Nucleus and Chromocenter .tif files
```

---

## Helper Methods

`CellDataset.load_split_ids(path)`: Use this to read `val_ids.txt`. It returns a list of strings that you can pass directly into the `split_ids` argument of the class constructor.