# Quorum: Cell Segmentation Benchmark

This repository enables a **reproducible, fair comparison** of two independent model architectures for microscopy cell segmentation. All teams use a shared data pipeline and fixed validation split. Code is versioned on GitHub; data is stored on Google Drive.

---

## Project Structure

```
quorum/
├── data_utils/
│   └── dataset.py
├── architecture_team_1/
├── architecture_team_2/
├── evaluation/
├── config.yaml
├── val_ids.txt
├── requirements.txt
└── README.md
```

---

## Dataset Location

**Data is NOT stored in this repo.**

On Google Drive, expected structure:

```
CMPUT469_Cell/
├── Microscopy_images/
├── Ground_truth_masks/
```

**Never copy dataset files into the repository.**

---

## Quick Start (Google Colab)

1. **Mount Google Drive**
	```python
	from google.colab import drive
	drive.mount('/content/drive')
	```
2. **Clone this repository**
	```bash
	!git clone https://github.com/your-username/quorum.git
	```
3. **Import the shared dataset module**
	```python
	import sys
	sys.path.append('/content/quorum')
	from data_utils.dataset import CellDataset
	```
4. **Load the dataset**
	```python
	from torch.utils.data import DataLoader
	dataset = CellDataset()
	loader = DataLoader(dataset, batch_size=4, shuffle=True)
	images, masks = next(iter(loader))
	print(images.shape, masks.shape)
	```

---


## Configuration (`config.yaml`)

All paths and parameters are set in `config.yaml`. **Never hardcode paths** in notebooks or scripts—always use `config.yaml`.

Current fields:
```yaml
data_root: /content/drive/MyDrive/CMPUT469_Cell  # Path to dataset root (update if running locally)
preprocess_mode: basic         # 'basic' or 'full' (see instructions)
augmentation: standard         # 'none', 'light', or 'standard'
target_size: [256, 256]        # [height, width] for resizing images and masks
```

**If your data is stored elsewhere, update only the `data_root` field in your local config.yaml.**

---

## Team Responsibilities

### Data Pipeline Team
- Maintains `dataset.py` and `config.yaml`
- Defines preprocessing and validation split

### Architecture Teams
- Implement models in their own folders
- Import `CellDataset` from `data_utils`
- **Do not modify shared modules**
- Save predictions to Google Drive using original Cell IDs

### Evaluation Team
- Compares predictions from both teams
- Computes Dice and IoU metrics
- Uses fixed validation IDs from `val_ids.txt`

---

## Reproducibility Rules

- **Do not modify** `dataset.py`
- **Always use** validation IDs from `val_ids.txt`
- **Always use** `config.yaml` for all paths
- **Never store** dataset files in the repository
- **Save predictions** with original Cell ID filenames

---

## Validation Split (`val_ids.txt`)

- 200 microscopy images total
- 80/20 split: 160 train, 40 validation
- Validation IDs are listed in `val_ids.txt` (one Cell ID per line)
- File was generated once with a fixed random seed—**do not modify**

> Changing `val_ids.txt` breaks reproducibility and invalidates model comparison.

---

## How to Use the Validation Split

**You can use the helper in `data_utils/dataset.py` to obtain the split.**

The helper:
- Reads `val_ids.txt`
- Separates training and validation IDs
- Ensures consistency across all teams

**Please don't redefine validation IDs in notebooks or scripts.**

