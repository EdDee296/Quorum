# Architecture Team 5 – DeepLabV3+

## Project Structure

```text
Quorum/
│
├── architecture_team_5/
│   ├── deeplabv3plus_model.py
│   │
│   ├── deeplabv3plus/
│   │   ├── sanity_check_deeplabv3plus.py
│   │   ├── train_deeplabv3plus.py
│   │   ├── eval_deeplabv3plus.py
│   │   ├── infer_deeplabv3plus.py
│   │   ├── runs_deeplabv3plus/
│   │   └── outputs_deeplabv3plus/
│   │
│   └── README.md
```

---

# Overview

This module implements **DeepLabV3+ for 3-class cell segmentation** on microscopy images.

### Classes

* **0** → Background
* **1** → Nucleus
* **2** → Chromocenter

Compared to DeepLabV3, DeepLabV3+ improves:

* boundary accuracy
* small object segmentation (chromocenters)
* spatial detail recovery

---

# Model Design (DeepLabV3+)

DeepLabV3+ consists of:

### 1. Encoder (Backbone)

* ResNet50 / ResNet101
* extracts high-level semantic features

---

### 2. ASPP (Atrous Spatial Pyramid Pooling)

* multi-scale context aggregation
* captures different receptive fields

---

### 3. Decoder (Key Improvement)

* combines:

  * high-level features (ASPP output)
  * low-level features (early backbone layers)
* improves boundary precision

---

### Key Advantage

```text
DeepLabV3   → strong semantics
DeepLabV3+  → semantics + sharp boundaries
```

---

# Sanity Check

## Run

```bash
python -m architecture_team_5.deeplabv3plus.sanity_check_deeplabv3plus
```

## What it does

* verifies dataset loading
* checks image shape → `[1, H, W]`
* checks mask shape → `[H, W]`
* confirms labels → `{0,1,2}`
* validates train / validation split
* tests DataLoader batching
* prints pixel distribution

---

# Training

## Run

```bash
python -m architecture_team_5.deeplabv3plus.train_deeplabv3plus \
    --batch_size 4 \
    --epochs 25 \
    --class2_weight 8 \
    --lr 0.0001
```

## What happens

* loads dataset configuration (`config.yaml`)
* builds split using `val_ids.txt`
* applies preprocessing + augmentation
* trains DeepLabV3+ model
* evaluates each epoch:

  * Dice (per class)
  * IoU (per class)
* computes uncertainty metrics
* saves best checkpoint (foreground Dice)

---

## Output

```text
architecture_team_5/deeplabv3plus/runs_deeplabv3plus/
    └── best_deeplabv3plus.pt
```

---

# Uncertainty Estimation

Same principle as DeepLabV3:

Since ground truth is unavailable during deployment, we estimate reliability using model outputs.

## Metrics

* **mean_entropy** → uncertainty level
* **mean_max_prob** → prediction confidence
* **boundary_entropy** → boundary difficulty
* **boundary_max_prob** → boundary confidence
* **class2_mean_prob** → chromocenter confidence
* **class2_area_ratio** → chromocenter size

## Risk Score

```text
risk_score = entropy + (1 - confidence)
```

Higher risk → less reliable prediction

---

# Evaluation

## Run

```bash
python -m architecture_team_5.deeplabv3plus.eval_deeplabv3plus
```

## Requirements

* training must be completed
* `best_deeplabv3plus.pt` must exist

---

## What it does

* loads trained model
* runs inference on validation set
* computes:

  * Dice (class 1 & 2)
  * IoU (class 1 & 2)
* saves predicted masks
* exports per-image metrics
* identifies worst-performing cells (chromocenter)

---

## Output

```text
architecture_team_5/deeplabv3plus/outputs_deeplabv3plus/
    ├── metrics.csv
    └── pred_masks/
```

---

## Mask Format

Each `.tif` mask:

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

👉 Chromocenter Dice remains the key metric.

---

# Inference (Single Image)

## Usage

```python
from architecture_team_5.deeplabv3plus.infer_deeplabv3plus import run_deeplabv3plus_inference
```

## Output includes

* predicted segmentation mask
* visualization mask
* cell-level analysis
* uncertainty estimation

---

# Key Design Decisions

## 1. Class-weighted Loss

```text
background = 1
nucleus = 1
chromocenter = 5–10
```

→ improves performance on small chromocenter regions

---

## 2. Decoder for Boundary Refinement

* integrates low-level features
* improves edge sharpness
* reduces over-smoothing

---

## 3. Batch Size Stability

* GroupNorm used when necessary
* avoids BatchNorm instability

---

## 4. Single-channel Handling

```python
img = img.repeat(1, 3, 1, 1)
```

---

## 5. Spatial Consistency

* resizing preserves spatial alignment
* compatible with downstream tracking

---

# Requirements

```bash
pip install torch torchvision pandas numpy tifffile opencv-python
```

---

# Running Notes

Run from project root:

```bash
cd Quorum-master
```

Use module mode:

```bash
python -m architecture_team_5.deeplabv3plus.xxx
```

---

# Summary

DeepLabV3+ improves upon DeepLabV3 by:

* better boundary precision
* improved chromocenter segmentation
* stronger small-object detection

This makes it more suitable for **high-precision cell structure analysis** and real-world deployment scenarios.
