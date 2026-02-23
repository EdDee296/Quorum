# 🛠️ Data Pipeline Technical Instructions

This document explains how to use the `data_utils/dataset.py` module. As the Data Input Owner, I have standardized this pipeline to ensure everyone is training and evaluating on the same processed images.

---

## 🚀 Quick Start

To use the dataset in your notebook or script:

```python
from data_utils.dataset import CellDataset

# 1. Load the fixed validation IDs from the text file
val_ids = CellDataset.load_split_ids('val_ids.txt')

# 2. Instantiate for Training (Apply augmentations, exclude val_ids)
# Note: You'll need to define your 'train_ids' by excluding 'val_ids' from the full list
train_ds = CellDataset(
    root_dir='/content/drive/MyDrive/CMPUT469_Cell',
    preprocess_mode='full',
    aug_strength='standard',
    split_ids=train_ids
)

# 3. Instantiate for Evaluation (No augmentations, use ONLY val_ids)
val_ds = CellDataset(
    root_dir='/content/drive/MyDrive/CMPUT469_Cell',
    preprocess_mode='full',
    aug_strength=None,
    split_ids=val_ids
)
```

---

## 🖼️ Preprocessing Modes

The `preprocess_mode` parameter is **strict**. If you pass anything other than these two, the code will raise a `ValueError`.

| Mode   | Description                                                                 | When to use                |
|--------|-----------------------------------------------------------------------------|----------------------------|
| basic  | 1%-99.5% percentile clipping + [0,1] normalization.                        | Quick baseline testing.     |
| full   | **Recommended.** Bilateral Filtering (noise reduction) and Gaussian-based Background Subtraction. | Final training and evaluation. |

---

## 🧪 Output Format

Every item returned by `dataset[i]` follows this format:

- **Image Tensor:** `(1, H, W)` - Grayscale, `Float32`.
- **Mask Tensor:** `(H, W)` - Long (`Int64`).
- **Mask Label Key:**
    - `0`: Background (Black)
    - `1`: Nucleus (Grey)
    - `2`: Chromocenters (White)

**Warning:** Your model's final layer must output **3 channels** (one for each class) if you are using `CrossEntropyLoss`.

---

## 🔄 Augmentation Strengths

The `aug_strength` parameter controls the albumentations pipeline:

- `None`: Only Resizing (**Use this for Evaluation.**)
- `'light'`: Flips and 90-degree rotations.
- `'standard'`: Flips, rotations, elastic transforms, and brightness/contrast shifts.

---

## 📂 Expected Data Structure

The `root_dir` you pass must contain these exact folder names:

```
Dataset_Root/
├── Microscopy_images/      # .tif files
└── Ground_truth_masks/     # Nucleus and Chromocenter .tif files
```

---

## 📜 Helper Methods

`CellDataset.load_split_ids(path)`: Use this to read `val_ids.txt`. It returns a list of strings that you can pass directly into the `split_ids` argument of the class constructor.