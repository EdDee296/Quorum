"""
sanity_check.py  —  Run this BEFORE compare_models.py

Checks everything the comparison script needs:
  1. config.yaml exists and data_root is reachable
  2. val_ids.txt exists and is not empty
  3. Dataset loads correctly (tries to load first image)
  4. All 5 model checkpoints — reports FOUND / MISSING
  5. All Python dependencies are importable
  6. Cellpose model files exist
  7. Training log CSVs exist (for learning curves)

How to run (from project root):
    python -m report_eval.sanity_check
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, ROOT)

PASS  = "  [OK]  "
FAIL  = "  [FAIL]"
WARN  = "  [WARN]"
SKIP  = "  [SKIP]"

all_ok = True

def ok(msg):
    print(f"{PASS} {msg}")

def fail(msg):
    global all_ok
    all_ok = False
    print(f"{FAIL} {msg}")

def warn(msg):
    print(f"{WARN} {msg}")


print("\n── 1. config.yaml ──")
config_path = os.path.join(ROOT, "config.yaml")
cfg = None
if not os.path.exists(config_path):
    fail(f"config.yaml not found at {config_path}")
else:
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        ok(f"config.yaml loaded")
        data_root = cfg.get("data_root", "")
        if not data_root:
            fail("data_root is empty in config.yaml")
        elif not os.path.exists(data_root):
            fail(f"data_root does not exist: {data_root}")
        else:
            img_dir  = os.path.join(data_root, "Microscopy_images")
            mask_dir = os.path.join(data_root, "Ground_truth_masks")
            mask_dir2= os.path.join(data_root, "Groud_truth_masks")  # typo variant
            if os.path.exists(img_dir):
                n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(".tif")])
                ok(f"Microscopy_images found: {n_imgs} .tif files")
            else:
                fail(f"Microscopy_images folder not found at {img_dir}")
            if os.path.exists(mask_dir):
                ok(f"Ground_truth_masks found")
            elif os.path.exists(mask_dir2):
                warn(f"Found 'Groud_truth_masks' (typo folder) — this is fine, dataset.py handles it")
            else:
                fail(f"Ground_truth_masks folder not found at {mask_dir}")
        print(f"       preprocess_mode : {cfg.get('preprocess_mode','basic')}")
        print(f"       target_size     : {cfg.get('target_size',[256,256])}")
        print(f"       epochs          : {cfg.get('epochs',18)}")
    except Exception as e:
        fail(f"Could not parse config.yaml: {e}")


#  val_ids.txt
print("\n── 2. val_ids.txt ──")
val_ids_path = os.path.join(ROOT, "val_ids.txt")
if not os.path.exists(val_ids_path):
    fail(f"val_ids.txt not found at {val_ids_path}")
else:
    with open(val_ids_path) as f:
        ids = [l.strip() for l in f if l.strip()]
    if not ids:
        fail("val_ids.txt is empty")
    else:
        ok(f"val_ids.txt found: {len(ids)} validation IDs")


# Dataset loads
print("\n── 3. Dataset loading ──")
if cfg is not None:
    try:
        from data_utils.dataset import CellDataset
        val_ids = CellDataset.load_split_ids(val_ids_path)
        ds = CellDataset(
            root_dir        = cfg["data_root"],
            preprocess_mode = cfg.get("preprocess_mode", "basic"),
            aug_strength    = "none",
            target_size     = tuple(cfg.get("target_size", [256, 256])),
            split_ids       = val_ids,
        )
        ok(f"Dataset loaded: {len(ds)} validation samples")
        if len(ds) == 0:
            fail("Dataset has 0 samples — check data_root and val_ids.txt")
        else:
            img, mask = ds[0]
            ok(f"First sample loaded — image shape: {tuple(img.shape)}, mask shape: {tuple(mask.shape)}")
            unique_labels = mask.unique().tolist()
            ok(f"Mask labels in first sample: {unique_labels}  (expect subset of [0, 1, 2])")
    except Exception as e:
        fail(f"Dataset failed to load: {e}")
else:
    print(f"{SKIP} Skipping dataset check (config.yaml failed)")


# Model checkpoints
print("\n── 4. Model checkpoints ──")

checkpoints = {
    "U-Net++"   : os.path.join(ROOT, "architecture_team_1", "unetpp",     "runs_unetpp",      "best_unetpp.pt"),
    "U-Net"     : os.path.join(ROOT, "architecture_team_2", "unet",       "runs_unet",        "best_unet.pt"),
    "DeepLabV3" : os.path.join(ROOT, "architecture_team_4", "deeplabv3",  "runs_deeplabv3",   "best_deeplabv3.pt"),
    "DeepLabV3+": os.path.join(ROOT, "architecture_team_5", "deeplabv3plus","runs_deeplabv3plus","best_deeplabv3plus.pt"),
}

found_count = 0
for name, path in checkpoints.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        ok(f"{name:<14} {path.replace(ROOT,'.')}  ({size_mb:.1f} MB)")
        found_count += 1
    else:
        warn(f"{name:<14} NOT FOUND — will be skipped in compare_models.py\n"
             f"             expected: {path.replace(ROOT,'.')}")

# Cellpose models
model_dir = os.path.join(ROOT, "backend", "models")
cp_names  = ["cp_chromo_aug", "cp_chromo_no_aug", "cp_nucleus"]
cp_found  = []
for cp_name in cp_names:
    p = os.path.join(model_dir, cp_name)
    if os.path.exists(p):
        ok(f"Cellpose        {cp_name}")
        cp_found.append(cp_name)
    else:
        warn(f"Cellpose        {cp_name} NOT FOUND at {p.replace(ROOT,'.')}")

if not cp_found:
    warn("No Cellpose model files found — Cellpose will use pretrained fallback if cellpose is installed")
    print(f"         (this is ok for running — pretrained nuclei model will be used)")

print(f"\n  {found_count}/4 PyTorch checkpoints found")
if found_count == 0:
    fail("No checkpoints found at all — have you trained the models? compare_models.py cannot run.")
elif found_count < 4:
    warn(f"Only {found_count}/4 checkpoints found — missing models will show as N/A in figures")


# Python dependencies
print("\n── 5. Python dependencies ──")

deps = [
    ("torch",                   "PyTorch"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("numpy",                   "numpy"),
    ("pandas",                  "pandas"),
    ("matplotlib",              "matplotlib"),
    ("seaborn",                 "seaborn"),
    ("scipy",                   "scipy"),
    ("cv2",                     "opencv-python"),
    ("tifffile",                "tifffile"),
    ("yaml",                    "pyyaml"),
    ("albumentations",          "albumentations"),
]

for module, pip_name in deps:
    try:
        __import__(module)
        ok(f"{pip_name}")
    except ImportError:
        fail(f"{pip_name} not installed  →  pip install {pip_name}")

# cellpose separately (optional but needed for full comparison)
try:
    import cellpose
    ok("cellpose")
except ImportError:
    warn("cellpose not installed — Cellpose model will be fully skipped\n"
         "         to install: pip install cellpose")

# check torch device
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        ok(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        warn("No GPU detected — running on CPU (will be slow but works fine)")
except Exception:
    pass


print("\n── 6. Training logs (for learning curves figure) ──")

train_logs = {
    "U-Net++" : os.path.join(ROOT, "architecture_team_1", "unetpp",  "runs_unetpp", "train_log.csv"),
    "U-Net"   : os.path.join(ROOT, "architecture_team_2", "unet",    "runs_unet",   "train_log.csv"),
}
logs_found = 0
for name, path in train_logs.items():
    if os.path.exists(path):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            ok(f"{name:<14} train_log.csv — {len(df)} epochs logged")
            logs_found += 1
        except Exception as e:
            warn(f"{name:<14} train_log.csv found but couldn't read: {e}")
    else:
        warn(f"{name:<14} train_log.csv NOT FOUND — learning curve for this model will be skipped\n"
             f"             expected: {path.replace(ROOT,'.')}")

if logs_found == 0:
    warn("No training logs found — learning_curves.png will be skipped entirely")


print("\n── 7. Output directory ──")
out_dir = os.path.join(THIS_DIR, "outputs")
try:
    os.makedirs(out_dir, exist_ok=True)
    test_file = os.path.join(out_dir, ".write_test")
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
    ok(f"Output directory writable: {out_dir}")
except Exception as e:
    fail(f"Cannot write to output directory: {e}")


print(f"\n{'='*55}")
if all_ok:
    print("  ALL CHECKS PASSED — you are ready to run:")
    print("  python -m report_eval.compare_models")
else:
    print("  SOME CHECKS FAILED — fix the [FAIL] items above")
    print("  before running compare_models.py")
    print("\n  [WARN] items are non-fatal — the script will still")
    print("  run but those models/figures will be skipped.")
print(f"{'='*55}\n")
