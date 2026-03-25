"""
eval_deeplabv3plus_qualityhat.py

this file is used for evaluating the trained DeepLabV3+ model with uncertainty-based
quality estimation, so it can output dice_hat and iou_hat without needing ground truth
at inference time.

what this file does:

- loads config.yaml to get dataset path and preprocessing settings
- loads val_ids.txt to use the exact same validation split every time
- loads the trained DeepLabV3+ checkpoint (best_deeplabv3plus.pt)
- runs stochastic inference on every validation image
- predicts 3-class segmentation (0 background, 1 nucleus, 2 chromocenter)
- extracts uncertainty features from the prediction
- converts uncertainty features into:
    dice_hat_class1
    iou_hat_class1
    dice_hat_class2
    iou_hat_class2
- if GT is available and calibration mode is on:
    also computes true Dice and IoU
    fits a quality head and saves it
- saves predicted masks (scaled for viewing: 0=black, 1=gray, 2=white)
- saves per-image metrics into metrics.csv
- prints average metrics and worst 5 cells

important before running:

- make sure you already trained the model using train_deeplabv3plus.py
- make sure best_deeplabv3plus.pt exists inside:
    architecture_team_5/deeplabv3plus/runs_deeplabv3plus/
- make sure config.yaml has correct data_root path
- make sure val_ids.txt has not been modified

how to run:

from project root folder:

1) first run on labeled validation set to fit the quality head:
python -m architecture_team_5.deeplabv3plus.eval_deeplabv3plus_qualityhat --use-gt-for-calibration

2) later run without GT, using saved quality head only:
python -m architecture_team_5.deeplabv3plus.eval_deeplabv3plus_qualityhat --no-use-gt-for-calibration

optional:
python -m architecture_team_5.deeplabv3plus.eval_deeplabv3plus_qualityhat --use-gt-for-calibration --mc-passes 8

outputs:

- metrics.csv saved to:
    architecture_team_5/deeplabv3plus/outputs_deeplabv3plus/
- predicted masks saved to:
    architecture_team_5/deeplabv3plus/outputs_deeplabv3plus/pred_masks/
- fitted quality head saved to:
    architecture_team_5/deeplabv3plus/runs_deeplabv3plus/quality_head_deeplabv3plus.npz

notes:

- this evaluates both nucleus and chromocenter
- background is not evaluated because it is trivial and would inflate metrics
- chromocenter Dice is usually the main performance number
- dice_hat / iou_hat are estimated quality scores, not true metrics
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import tifffile as tiff

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils.dataset import CellDataset
from architecture_team_5.deeplabv3plus_model import build_deeplabv3plus


EPS = 1e-8
DEFAULT_MC_PASSES = 8
DEFAULT_N_FOLDS = 5

QUALITY_HEAD_PATH = os.path.join(
    THIS_DIR,
    "runs_deeplabv3plus",
    "quality_head_deeplabv3plus.npz",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepLabV3+ and predict dice_hat / iou_hat from uncertainty."
    )

    parser.add_argument(
        "--use-gt-for-calibration",
        dest="use_gt_for_calibration",
        action="store_true",
        help="Use GT masks to fit the quality head on the validation set.",
    )

    parser.add_argument(
        "--no-use-gt-for-calibration",
        dest="use_gt_for_calibration",
        action="store_false",
        help="Do not use GT. Load an existing quality head and only output dice_hat / iou_hat.",
    )

    parser.add_argument(
        "--mc-passes",
        type=int,
        default=DEFAULT_MC_PASSES,
        help="Number of stochastic forward passes for uncertainty estimation.",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of folds for out-of-fold calibration on validation set.",
    )

    parser.set_defaults(use_gt_for_calibration=True)
    return parser.parse_args()


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repeat_to_3ch(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def dice_iou_for_class(pred, gt, cls):
    p = (pred == cls)
    g = (gt == cls)

    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    p_sum = p.sum()
    g_sum = g.sum()

    denom = p_sum + g_sum
    if denom == 0:
        dice = 1.0
    else:
        dice = (2.0 * inter) / float(denom)

    if union == 0:
        iou = 1.0
    else:
        iou = inter / float(union)

    return float(dice), float(iou)


def save_scaled_mask(mask_hw, out_path):
    vis = np.zeros_like(mask_hw, dtype=np.uint8)
    vis[mask_hw == 1] = 127
    vis[mask_hw == 2] = 255
    tiff.imwrite(out_path, vis)


def enable_dropout_only(model):
    count = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
                torch.nn.FeatureAlphaDropout,
            ),
        ):
            module.train()
            count += 1
    return count


def apply_tta(x, mode):
    if mode == "none":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "hvflip":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def undo_tta(x, mode):
    if mode == "none":
        return x
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "vflip":
        return torch.flip(x, dims=[2])
    if mode == "hvflip":
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown TTA mode: {mode}")


def stochastic_predict(model, img, n_passes=8):
    model.eval()
    dropout_count = enable_dropout_only(model)

    prob_list = []

    with torch.no_grad():
        if dropout_count > 0:
            for _ in range(n_passes):
                logits = model(img)
                probs = torch.softmax(logits, dim=1)[0].cpu()
                prob_list.append(probs)
        else:
            tta_modes = ["none", "hflip", "vflip", "hvflip"]
            n_use = min(n_passes, len(tta_modes))
            for mode in tta_modes[:n_use]:
                aug = apply_tta(img, mode)
                logits = model(aug)
                probs = torch.softmax(logits, dim=1)
                probs = undo_tta(probs, mode)[0].cpu()
                prob_list.append(probs)

    probs = torch.stack(prob_list, dim=0).numpy()   # [T, C, H, W]
    mean_prob = probs.mean(axis=0)                  # [C, H, W]
    pred = np.argmax(mean_prob, axis=0).astype(np.uint8)
    max_prob = np.max(mean_prob, axis=0)

    num_classes = mean_prob.shape[0]
    entropy = -(mean_prob * np.log(mean_prob + EPS)).sum(axis=0)
    entropy = entropy / np.log(float(num_classes))

    var_map = probs.var(axis=0)                     # [C, H, W]

    return pred, mean_prob, max_prob, entropy, var_map, dropout_count


def safe_mean(x, default=0.0):
    if x.size == 0:
        return float(default)
    return float(np.mean(x))


def safe_percentile(x, q, default=0.0):
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, q))


def quality_features_for_class(pred, mean_prob, max_prob, entropy, var_map, cls):
    mask = (pred == cls)

    area_frac = float(mask.mean())
    present = 1.0 if mask.any() else 0.0

    cls_prob = mean_prob[cls]
    cls_var = var_map[cls]

    if mask.any():
        mean_maxprob_on_cls = safe_mean(max_prob[mask], default=1.0)
        mean_clsprob_on_cls = safe_mean(cls_prob[mask], default=0.0)
        mean_entropy_on_cls = safe_mean(entropy[mask], default=0.0)
        p90_entropy_on_cls = safe_percentile(entropy[mask], 90, default=0.0)
        mean_var_on_cls = safe_mean(cls_var[mask], default=0.0)
        p90_var_on_cls = safe_percentile(cls_var[mask], 90, default=0.0)
    else:
        mean_maxprob_on_cls = 1.0
        mean_clsprob_on_cls = 0.0
        mean_entropy_on_cls = 0.0
        p90_entropy_on_cls = 0.0
        mean_var_on_cls = 0.0
        p90_var_on_cls = 0.0

    global_mean_conf = float(max_prob.mean())
    global_mean_entropy = float(entropy.mean())

    feat = np.array(
        [
            1.0,
            present,
            area_frac,
            mean_maxprob_on_cls,
            mean_clsprob_on_cls,
            mean_entropy_on_cls,
            p90_entropy_on_cls,
            mean_var_on_cls,
            p90_var_on_cls,
            global_mean_conf,
            global_mean_entropy,
        ],
        dtype=np.float32,
    )

    return feat


def fit_ridge_regression(X, y, l2=1e-4):
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    eye = np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(X.T @ X + l2 * eye, X.T @ y)
    return w.astype(np.float32)


def predict_with_head(X, w):
    y_hat = X @ w
    y_hat = np.clip(y_hat, 0.0, 1.0)
    return y_hat.astype(np.float32)


def fit_oof_and_full(X, y, n_folds=5, seed=42):
    n = len(y)
    n_folds = max(2, min(n_folds, n))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    oof = np.zeros(n, dtype=np.float32)

    for fold_idx in range(n_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        w = fit_ridge_regression(X[train_idx], y[train_idx])
        oof[test_idx] = predict_with_head(X[test_idx], w)

    w_full = fit_ridge_regression(X, y)
    return oof, w_full


def save_quality_heads(path, heads):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **heads)


def load_quality_heads(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    args = parse_args()
    use_gt_for_calibration = args.use_gt_for_calibration
    mc_passes = args.mc_passes
    n_folds = args.n_folds

    t0 = time.time()

    cfg = load_config()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    root_dir = cfg["data_root"]
    preprocess_mode = cfg.get("preprocess_mode", "basic")
    target_size = tuple(cfg.get("target_size", [256, 256]))

    val_ids = CellDataset.load_split_ids(os.path.join(PROJECT_ROOT, "val_ids.txt"))

    val_ds = CellDataset(
        root_dir=root_dir,
        preprocess_mode=preprocess_mode,
        aug_strength="none",
        target_size=target_size,
        split_ids=val_ids,
    )

    ckpt_path = os.path.join(THIS_DIR, "runs_deeplabv3plus", "best_deeplabv3plus.pt")
    out_dir = os.path.join(THIS_DIR, "outputs_deeplabv3plus")
    pred_dir = os.path.join(out_dir, "pred_masks")

    os.makedirs(pred_dir, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", ckpt_path)
    print("Out dir:", out_dir)
    print("Quality head:", QUALITY_HEAD_PATH)
    print("Use GT for calibration:", use_gt_for_calibration)
    print("MC passes:", mc_passes)
    print("N folds:", n_folds)

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    output_stride = ckpt.get("output_stride", 16)

    model = build_deeplabv3plus(
        backbone=backbone,
        pretrained_backbone=False,
        out_channels=3,
        output_stride=output_stride,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []

    x_cls1 = []
    x_cls2 = []
    y_dice1 = []
    y_iou1 = []
    y_dice2 = []
    y_iou2 = []

    warned_no_dropout = False

    i = 0
    while i < len(val_ds):
        sample = val_ds[i]
        cid = val_ds.samples[i]

        if isinstance(sample, (tuple, list)):
            if len(sample) >= 2:
                img = sample[0]
                gt_mask = sample[1]
            else:
                img = sample[0]
                gt_mask = None
        else:
            img = sample
            gt_mask = None

        img = img.unsqueeze(0).to(device)
        img = repeat_to_3ch(img)

        pred, mean_prob, max_prob, entropy, var_map, dropout_count = stochastic_predict(
            model=model,
            img=img,
            n_passes=mc_passes,
        )

        if (dropout_count == 0) and (not warned_no_dropout):
            print("Warning: no dropout layers found in model. Using TTA fallback for uncertainty.")
            warned_no_dropout = True

        feat1 = quality_features_for_class(pred, mean_prob, max_prob, entropy, var_map, cls=1)
        feat2 = quality_features_for_class(pred, mean_prob, max_prob, entropy, var_map, cls=2)

        row = {
            "cell_id": cid,
            "pred_mask_file": f"Pred_mask_{cid}.tif",
        }

        if use_gt_for_calibration:
            if gt_mask is None:
                raise RuntimeError(
                    "Calibration mode requires GT masks, but dataset did not return ground truth."
                )

            gt = gt_mask.cpu().numpy().astype(np.uint8)

            dice1, iou1 = dice_iou_for_class(pred, gt, cls=1)
            dice2, iou2 = dice_iou_for_class(pred, gt, cls=2)

            row["dice_class1"] = dice1
            row["iou_class1"] = iou1
            row["dice_class2"] = dice2
            row["iou_class2"] = iou2

            x_cls1.append(feat1)
            x_cls2.append(feat2)
            y_dice1.append(dice1)
            y_iou1.append(iou1)
            y_dice2.append(dice2)
            y_iou2.append(iou2)
        else:
            x_cls1.append(feat1)
            x_cls2.append(feat2)

        out_path = os.path.join(pred_dir, f"Pred_mask_{cid}.tif")
        save_scaled_mask(pred, out_path)

        rows.append(row)
        i += 1

    X1 = np.vstack(x_cls1).astype(np.float32)
    X2 = np.vstack(x_cls2).astype(np.float32)

    if use_gt_for_calibration:
        y_dice1 = np.asarray(y_dice1, dtype=np.float32)
        y_iou1 = np.asarray(y_iou1, dtype=np.float32)
        y_dice2 = np.asarray(y_dice2, dtype=np.float32)
        y_iou2 = np.asarray(y_iou2, dtype=np.float32)

        dice1_hat, w_dice1 = fit_oof_and_full(X1, y_dice1, n_folds=n_folds)
        iou1_hat,  w_iou1  = fit_oof_and_full(X1, y_iou1,  n_folds=n_folds)
        dice2_hat, w_dice2 = fit_oof_and_full(X2, y_dice2, n_folds=n_folds)
        iou2_hat,  w_iou2  = fit_oof_and_full(X2, y_iou2,  n_folds=n_folds)

        save_quality_heads(
            QUALITY_HEAD_PATH,
            {
                "w_dice1": w_dice1,
                "w_iou1": w_iou1,
                "w_dice2": w_dice2,
                "w_iou2": w_iou2,
            },
        )
    else:
        if not os.path.exists(QUALITY_HEAD_PATH):
            raise FileNotFoundError(
                f"Quality head not found: {QUALITY_HEAD_PATH}. "
                f"Run once with --use-gt-for-calibration first."
            )

        heads = load_quality_heads(QUALITY_HEAD_PATH)
        dice1_hat = predict_with_head(X1, heads["w_dice1"])
        iou1_hat = predict_with_head(X1, heads["w_iou1"])
        dice2_hat = predict_with_head(X2, heads["w_dice2"])
        iou2_hat = predict_with_head(X2, heads["w_iou2"])

    for idx in range(len(rows)):
        rows[idx]["dice_hat_class1"] = float(dice1_hat[idx])
        rows[idx]["iou_hat_class1"] = float(iou1_hat[idx])
        rows[idx]["dice_hat_class2"] = float(dice2_hat[idx])
        rows[idx]["iou_hat_class2"] = float(iou2_hat[idx])

    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved metrics:", csv_path)
    print("Saved predictions:", pred_dir)
    print("Seconds:", int(time.time() - t0))

    if use_gt_for_calibration:
        print("\nAverages on val:")
        print("Mean Dice class1:", float(df["dice_class1"].mean()))
        print("Mean IoU  class1:", float(df["iou_class1"].mean()))
        print("Mean Dice class2:", float(df["dice_class2"].mean()))
        print("Mean IoU  class2:", float(df["iou_class2"].mean()))

        print("\nAverages of hats on val:")
        print("Mean Dice_hat class1:", float(df["dice_hat_class1"].mean()))
        print("Mean IoU_hat  class1:", float(df["iou_hat_class1"].mean()))
        print("Mean Dice_hat class2:", float(df["dice_hat_class2"].mean()))
        print("Mean IoU_hat  class2:", float(df["iou_hat_class2"].mean()))

        print("\nMAE between true metrics and hats:")
        print("MAE Dice class1:", float(np.abs(df["dice_hat_class1"] - df["dice_class1"]).mean()))
        print("MAE IoU  class1:", float(np.abs(df["iou_hat_class1"] - df["iou_class1"]).mean()))
        print("MAE Dice class2:", float(np.abs(df["dice_hat_class2"] - df["dice_class2"]).mean()))
        print("MAE IoU  class2:", float(np.abs(df["iou_hat_class2"] - df["iou_class2"]).mean()))

        print("\nWorst 5 cells by Dice_hat (class2):")
        worst = df.sort_values("dice_hat_class2", ascending=True).head(5)
        print(
            worst[
                ["cell_id", "dice_hat_class2", "iou_hat_class2", "dice_class2", "iou_class2"]
            ].to_string(index=False)
        )
    else:
        print("\nAverages without GT:")
        print("Mean Dice_hat class1:", float(df["dice_hat_class1"].mean()))
        print("Mean IoU_hat  class1:", float(df["iou_hat_class1"].mean()))
        print("Mean Dice_hat class2:", float(df["dice_hat_class2"].mean()))
        print("Mean IoU_hat  class2:", float(df["iou_hat_class2"].mean()))

        print("\nWorst 5 cells by Dice_hat (class2):")
        worst = df.sort_values("dice_hat_class2", ascending=True).head(5)
        print(worst[["cell_id", "dice_hat_class2", "iou_hat_class2"]].to_string(index=False))


if __name__ == "__main__":
    main()