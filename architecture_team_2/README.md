# Architecture Team 2

## Project Structure

```text
Quorum/
|
|- architecture_team_1/
|
|- architecture_team_2/
|  |- unet/
|  |  |- sanity_check_unet.py
|  |  |- train_unet.py
|  |  |- eval_unet.py
|  |  |- runs_unet/
|  |  `- outputs_unet/
|  |
|  `- README.md
|
`- data_utils/
```

## U-Net Implementation

## Sanity Check

Sanity check script:
```bash
python -m architecture_team_2.unet.sanity_check_unet
```

What it does:
- Verifies dataset loading
- Checks image and mask shapes
- Confirms mask labels are `{0, 1, 2}`
- Confirms train/validation split sizes
- Tests DataLoader batching
- Prints basic pixel counts

Note:
This script does not train the model. It only verifies the dataset pipeline before training.

## Training

Training script:
```bash
python -m architecture_team_2.unet.train_unet
```

What happens during training:
- Reads `config.yaml`
- Builds train/validation split using `val_ids.txt`
- Applies preprocessing + augmentation (train only)
- Trains U-Net for multi-class segmentation
- Evaluates Dice/IoU for nucleus and chromocenter every epoch
- Saves best checkpoint based on chromocenter Dice
- Logs metrics to CSV

Outputs:
```text
architecture_team_2/unet/runs_unet/
    |- best_unet.pt
    `- train_log.csv
```

## Evaluation

Important:
- Training must run first to create `best_unet.pt`
- Evaluation fails if checkpoint does not exist

Evaluation script:
```bash
python -m architecture_team_2.unet.eval_unet
```

What it does:
- Loads best checkpoint
- Runs inference on validation set
- Computes Dice/IoU for class 1 (nucleus) and class 2 (chromocenter)
- Saves per-cell metrics to CSV
- Saves predicted masks (scaled for visualization)

Outputs:
```text
architecture_team_2/unet/outputs_unet/
    |- metrics.csv
    `- pred_masks/
```
