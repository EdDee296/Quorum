# Architecture Team 4 – DeepLabV3

## Project Structure

```text
Quorum/
│
├── architecture_team_4/
│   ├── deeplabv3_model.py
│   │
│   ├── deeplabv3/
│   │   ├── sanity_check_deeplabv3.py
│   │   ├── train_deeplabv3.py
│   │   ├── eval_deeplabv3.py
│   │   ├── infer_deeplabv3.py
│   │   ├── runs_deeplabv3/
│   │   └── outputs_deeplabv3/
│   │
│   └── README.md
```

---

# Overview

This module implements **DeepLabV3 for 3-class cell segmentation** on microscopy images.

### Classes

* **0** → Background
* **1** → Nucleus
* **2** → Chromocenter

The model is designed to:

* support **single-channel microscopy images**
* support **batch_size = 1**
* provide **uncertainty estimation for real-world usage**

---

# Sanity Check

## Run

```bash
python -m architecture_team_4.deeplabv3.sanity_check_deeplabv3
```

## What it does

* verifies dataset loading
* checks image shape → `[1, H, W]`
* checks mask shape → `[H, W]`
* confirms label values → `{0,1,2}`
* validates train / validation split
* tests DataLoader batching
* prints pixel distribution

⚠️ This step must pass before training.

---

# Training

## Run

```bash
python -m architecture_team_4.deeplabv3.train_deeplabv3 \
    --batch_size 1 \
    --epochs 25 \
    --class2_weight 8 \
    --lr 0.0003
```

## What happens

* loads dataset configuration (`config.yaml`)
* builds train / validation split using `val_ids.txt`
* applies preprocessing + augmentation (train only)
* trains DeepLabV3 model
* evaluates each epoch:

  * Dice (per class)
  * IoU (per class)
* computes uncertainty metrics
* saves best model (based on foreground Dice)

---

## Output

```text
architecture_team_4/deeplabv3/runs_deeplabv3/
    └── best_deeplabv3.pt
```

---

# Uncertainty Estimation

Since ground truth is unavailable at inference time, Dice cannot be used.

Instead, reliability is estimated from model probabilities.

## Metrics

* **mean_entropy** → higher = more uncertain
* **mean_max_prob** → lower = less confident
* **boundary_entropy** → uncertainty near object edges
* **boundary_max_prob** → confidence at boundaries
* **class2_mean_prob** → chromocenter confidence
* **class2_area_ratio** → predicted chromocenter size

## Risk Score

```text
risk_score = entropy + (1 - confidence)
```

Higher risk → less reliable prediction

---

# Evaluation

## Run

```bash
python -m architecture_team_4.deeplabv3.eval_deeplabv3
```

## Requirements

* training must be completed
* `best_deeplabv3.pt` must exist

---

## What it does

* loads trained model
* runs inference on validation set
* computes:

  * Dice (class 1: nucleus)
  * Dice (class 2: chromocenter)
  * IoU (class 1 & 2)
* saves predicted masks
* exports per-image metrics
* reports worst-performing cells (chromocenter)

---

## Output

```text
architecture_team_4/deeplabv3/outputs_deeplabv3/
    ├── metrics.csv
    └── pred_masks/
```

---

## Mask Format

Each `.tif` file:

* **0** → Background
* **127** → Nucleus
* **255** → Chromocenter

---

## Console Output

```text
Mean Dice class1
Mean IoU  class1
Mean Dice class2
Mean IoU  class2

Worst 5 cells by Dice (class2)
```

👉 Chromocenter Dice is the primary evaluation metric.

---

# Inference (Single Image)

## Usage

```python
from architecture_team_4.deeplabv3.infer_deeplabv3 import run_deeplabv3_inference
```

## Output includes

* predicted segmentation mask
* visualization mask
* cell-level statistics
* uncertainty estimation

---

# Key Design Decisions

## 1. Class-weighted Loss

```text
background = 1
nucleus = 1
chromocenter = 5–10
```

→ improves learning of small chromocenter regions

---

## 2. GroupNorm instead of BatchNorm

* avoids instability with batch_size = 1
* applied to ASPP pooling branch

---

## 3. Single-channel to 3-channel conversion

```python
img = img.repeat(1, 3, 1, 1)
```

---

## 4. Spatial Consistency

* resizing preserves pixel alignment
* coordinates remain valid for downstream tracking

---

# Requirements

```bash
pip install torch torchvision pandas numpy tifffile opencv-python
```

---

# Running Notes

Always run from project root:

```bash
cd Quorum-master
```

Run scripts using module mode:

```bash
python -m architecture_team_4.deeplabv3.xxx
```

---

# Summary

This implementation provides:

* robust DeepLabV3 training pipeline
* accurate nucleus and chromocenter segmentation
* deployment-ready uncertainty estimation
* stable performance with small batch sizes

