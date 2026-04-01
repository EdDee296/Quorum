# report_eval/

Unified comparison suite for the Quorum project report.
One script generates every quantitative result and figure you need.

---

## Step 1 — Run the sanity check FIRST

```bash
python -m report_eval.sanity_check
```

This checks everything before you commit to the full run:

| Check | What it verifies |
|-------|-----------------|
| `config.yaml` | exists, `data_root` points to real data |
| `val_ids.txt` | exists, not empty |
| Dataset | loads correctly, first image readable |
| Checkpoints | reports FOUND / MISSING for each of the 5 models |
| Dependencies | all pip packages importable |
| Cellpose files | `cp_chromo_aug`, `cp_nucleus` etc. in `backend/models/` |
| Training logs | `train_log.csv` present for learning curves |
| Output folder | writable |

Fix any `[FAIL]` items before continuing. `[WARN]` items are non-fatal — the script will skip that model/figure and keep going.

---

## Step 2 — Run the comparison

```bash
python -m report_eval.compare_models
```

---

## What gets generated

```
report_eval/outputs/
├── metrics_summary.csv          ← Table 1 in your report
├── per_image_metrics.csv        ← full per-image data / appendix
├── statistical_tests.csv        ← Wilcoxon p-values (A1 vs A2 claim)
├── confusion_matrices.csv       ← pixel-level confusion data
└── figures/
    ├── bar_dice.png             ← main quantitative result (Dice)
    ├── bar_iou.png              ← main quantitative result (IoU)
    ├── boxplot_dice_chromocenter.png  ← variance/robustness per model
    ├── boxplot_dice_nucleoplasm.png
    ├── scatter_chrom_dice.png   ← per-image pairwise comparison
    ├── confusion_<model>.png    ← where each model fails (one per model)
    └── comparison_cell<id>.png  ← Original | GT | all models | error maps
                                    (5 cells: worst/25th/median/75th/best)
```

---

## Expected checkpoint paths

| Model | Path |
|-------|------|
| U-Net++ | `architecture_team_1/unetpp/runs_unetpp/best_unetpp.pt` |
| U-Net | `architecture_team_2/unet/runs_unet/best_unet.pt` |
| DeepLabV3 | `architecture_team_4/deeplabv3/runs_deeplabv3/best_deeplabv3.pt` |
| DeepLabV3+ | `architecture_team_5/deeplabv3plus/runs_deeplabv3plus/best_deeplabv3plus.pt` |
| Cellpose | `backend/models/cp_chromo_aug` + `backend/models/cp_nucleus` |

Missing checkpoints are skipped gracefully.

---

## What each output is for in the report

| File | Where it goes |
|------|--------------|
| `metrics_summary.csv` | **Table 1** — paste directly |
| `statistical_tests.csv` | Cite p-values in results section to satisfy the prof's "A1 vs A2" requirement |
| `bar_dice.png` | Main results figure |
| `boxplot_dice_chromocenter.png` | Shows consistency, not just mean |
| `scatter_chrom_dice.png` | Shows per-image differences between models |
| `comparison_cell*.png` | Qualitative figure: GT + all model predictions + error maps |
| `confusion_<model>.png` | Error analysis section |

---

## Colour coding in all figures

| Colour | Class |
|--------|-------|
| Near-black | Background |
| Teal `#26788E` | Nucleoplasm |
| Orange `#E2480C` | Chromocenter |
| Red | Error pixel (pred ≠ GT) |
