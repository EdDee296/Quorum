# Architecture Team 1

## Project Structure

```text
Quorum/
│
├── architecture_team_1/
│   ├── unetpp/
│   │   ├── sanity_check_unetpp.py
│   │   ├── train_unetpp.py
│   │   ├── eval_unetpp.py
│   │   ├── runs_unetpp/
│   │   └── outputs_unetpp/
│   │
│   └── README.md
│
└── architecture_team_2/
```

## U-Net++ Implementation

## Sanity Check

**Sanity check script:**
```bash
python -m architecture_team_1.unetpp.sanity_check_unetpp
```
**What it does:**
* Verifies dataset loading
* Checks image and mask shapes
* Confirms mask labels are {0, 1, 2}
* Confirms train/validation split sizes
* Tests DataLoader batching
* Prints basic pixel counts

**Note:**
This script does not train the model. It is only used to confirm the dataset pipeline is working correctly before training. It is recommended to run the sanity check once after setting `config.yaml`.

## Training

**Training script:**
```bash
python -m architecture_team_1.unetpp.train_unetpp
```
### What happens during training:

* Reads `config.yaml`
* Builds train/validation split using `val_ids.txt`
* Applies preprocessing + augmentation (for train only)
* Trains U-Net++ for multi-class segmentation
* Evaluates Dice/IoU for nucleus and chromocenter each epoch
* Saves best checkpoint based on chromocenter Dice
* Logs training metrics to CSV

**Outputs:**

```text
architecture_team_1/unetpp/runs_unetpp/
    ├── best_unetpp.pt
    └── train_log.csv
```

## Evaluation

**Evaluation script:**
> **Important:**
> * Training must be run first to generate `best_unetpp.pt`.
> * Evaluation will fail if no trained checkpoint exists.
```bash
python -m architecture_team_1.unetpp.eval_unetpp
```


**What it does:**

* Loads best checkpoint
* Runs inference on validation set
* Computes Dice and IoU for class 1 (nucleus) and class 2 (chromocenter)
* Saves per-cell metrics to CSV
* Saves predicted masks (scaled for visualization)

**Outputs:**

```text
architecture_team_1/unetpp/outputs_unetpp/
    ├── metrics.csv
    └── pred_masks/
```
The `pred_masks/` folder contains predicted segmentation masks for each validation cell.

Each file corresponds to one cell image and is saved as a `.tif` mask where:
* **0** → Background
* **127** → Nucleus (class 1, scaled for visualization)
* **255** → Chromocenter (class 2, scaled for visualization)

These masks are scaled for easier viewing in image software.
