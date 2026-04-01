# Quorum: Sub-Cellular Instance Segmentation

A web app for automated segmentation of chromocenters, nuclei, and background in microscopy images.

## Table of Contents

- [Project Structure](#project-structure)
- [Model Files](#model-files)
- [Dataset](#dataset)
- [Running the App](#running-the-app)
  - [Step 1: Install Docker Desktop](#step-1-install-docker-desktop)
  - [Step 2: One-Time Memory Setup](#step-2-one-time-memory-setup)
  - [Step 3: Build and Run](#step-3-build-and-run-the-app)
  - [Stopping the App](#stopping-the-app)
  - [Starting Again](#starting-again-after-first-build)
  - [Rebuild After Code Changes](#rebuild-after-code-changes)
- [Option 2: Run Manually](#option-2-run-manually-for-developers)
- [Training the Models](#training-the-models)
- [Evaluating the Models](#evaluating-the-models)
- [Configuration](#configuration-configyaml)
- [Validation Split](#validation-split)
- [Reproducibility Rules](#reproducibility-rules)
---

## Project Structure

```
Quorum/
├── architecture_team_1/        # U-Net++ model
│   └── unetpp/
│       ├── train_unetpp.py
│       ├── eval_unetpp.py
│       └── infer_unetpp.py
├── architecture_team_2/        # U-Net model
│   └── unet/
│       ├── runs_unet/
│       │   └── best_unet.pt
│       └── outputs_unet/
│           ├── metrics.csv
│           └── pred_masks/
├── architecture_team_3/        # Cellpose model
│   └── cellpose/
│       ├── train_cellpose.py
│       ├── eval_cellpose.py
│       └── infer_cellpose.py
├── architecture_team_4/        # deepLabV3 model
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
├── architecture_team_5/         # deepLabV3+ model
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
├── backend/                    # FastAPI backend
│   ├── main.py
│   └── models/                 # Trained model files
│       ├── best_unetpp.pt
│       ├── cp_chromo_aug
│       ├── cp_chromo_no_aug
│       └── cp_nucleus
├── frontend/                   # React frontend
├── report_eval/                # Model comparison suite for report
│   ├── compare_models.py       # Runs all 5 models, generates figures + CSVs
│   ├── sanity_check.py         # Run this before compare_models.py
│   ├── README.md
│   └── outputs/                # Generated outputs (gitignored)
│       ├── metrics_summary.csv
│       ├── per_image_metrics.csv
│       ├── statistical_tests.csv
│       ├── confusion_matrices.csv
│       └── figures/
├── data_utils/                 # Shared dataset pipeline
│   └── dataset.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── config.yaml
├── requirements.txt
└── val_ids.txt
```

---

## Model Files

The following trained model files are **included in this repo** inside `backend/models/`:

| File | Description |
|------|-------------|
| `backend/models/best_unetpp.pt` | Trained U-Net++ checkpoint (~100MB) |
| `backend/models/cp_chromo_aug` | Cellpose chromocenter model (augmented) |
| `backend/models/cp_chromo_no_aug` | Cellpose chromocenter model (no augmentation) |
| `backend/models/cp_nucleus` | Cellpose nucleus model |
| `backend/models/cp_nucleus_diameter.txt` | Saved nucleus diameter for inference |

These are ready to use out of the box so no retraining needed. If you want to retrain them yourself, see the [Training the Models](#training-the-models) section below.

---

## Dataset

**Data is NOT stored in this repo.**

Expected structure on Google Drive:
```
CMPUT469_Cell/
├── Microscopy_images/
└── Ground_truth_masks/
```

Update `config.yaml` with your local data path:
```yaml
data_root: /path/to/your/CMPUT469_Cell
```

---
 
## Running the App
 
The easiest way to run Quorum is with **Docker**. You do not need to install Python, Node.js, or any other tools. Docker handles everything automatically.
 
---
 
### Step 1: Install Docker Desktop
 
Download and install Docker Desktop from **https://www.docker.com/products/docker-desktop**
 
Make sure Docker Desktop is **open and running** before continuing. You should see "Engine running" in the bottom left of the Docker Desktop window.
 
---
 
### Step 2: One-Time Memory Setup
 
The first time you build the app, Docker needs to download and unpack large ML libraries (~2-3GB). By default, some systems don't allocate enough memory for this. Follow the instructions for your operating system below.
 
#### Windows Users
1. Open **PowerShell** (search "PowerShell" in the Windows Start menu, or use the terminal inside Docker Desktop)
2. Run this command:
```powershell
@"
[wsl2]
memory=6GB
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding utf8
```
3. Then run this to restart WSL (the Linux layer Docker uses on Windows):
```powershell
wsl --shutdown
```
4. Reopen your terminal and continue to Step 3.
 
> This is a one-time setup. You never need to do this again.
 
#### Mac Users
1. Open Docker Desktop
2. Go to **Settings → Resources → Memory**
3. Set memory to **6GB**
4. Click **Apply & Restart**
 
> This is a one-time setup. You never need to do this again.
 
#### Linux Users
No extra setup needed. Go straight to Step 3.
 
---
 
### Step 3: Build and Run the App
 
We recommend using **VS Code** for the best experience. Open VS Code, then open the integrated terminal (`Ctrl+`` ` on Windows/Linux, `Cmd+`` ` on Mac) and run:
 
```bash
git clone <repo-url>
cd Quorum
docker compose up --build
```

Then open your browser and go to **http://localhost**
 
> The first build takes **10-15 minutes** as it downloads all the required libraries.
> After that, starting the app again only takes a few seconds.
 
---
 
### Stopping the App
 
Either:
- Press `Ctrl+C` in the terminal where the app is running, or
- Open **Docker Desktop → Containers** → click the **Stop ⏹** button next to **quorum**
 
---
 
### Starting Again (After First Build)
 
You do not need to rebuild every time. Either:
- Open a terminal and run:
```bash
docker compose up
```
- Or open **Docker Desktop → Containers** → click the **Play ▶** button next to **quorum**
 
> Wait **1-2 minutes** after starting before opening the browser as the backend needs time to load the ML models.
 
---
 
### Rebuild After Code Changes
 
Any time you change the code, rebuild first:
 
```bash
docker compose up --build
```
 
---
 
## Option 2: Run Manually (For Developers)
 
Use this if you prefer running the app locally without Docker
 
### Requirements
- Python 3.10+
- Node.js 20+
 
### Backend Setup
Open a terminal (VS Code recommended) and run:
 
```bash
# create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
 
# install dependencies
pip install -r requirements.txt
 
# run the backend from the project root (not from inside backend/)
uvicorn backend.main:app --reload
```
 
Backend runs at **http://localhost:8000**
 
### Frontend Setup
Open a **second terminal** in VS Code (keep the backend running in the first one) and run:
 
```bash
cd frontend
npm install
npm run dev
```
 
---
 
## Training the Models
 
Make sure your `config.yaml` has the correct `data_root` before training.
 
### Train U-Net++
```bash
python -m architecture_team_1.unetpp.train_unetpp
```
Output: `backend/models/best_unetpp.pt`
 
### Train Cellpose
```bash
python -m architecture_team_3.cellpose.train_cellpose
```
Output: `backend/models/cp_chromo_aug`, `cp_chromo_no_aug`, `cp_nucleus`, `cp_nucleus_diameter.txt`
 
---
 
## Evaluating the Models
 
### Evaluate U-Net++
```bash
python -m architecture_team_1.unetpp.eval_unetpp
```
 
### Evaluate Cellpose
```bash
python -m architecture_team_3.cellpose.eval_cellpose
```
 
---
 
## Configuration (`config.yaml`)
 
```yaml
data_root: /path/to/CMPUT469_Cell   # update this for your machine
preprocess_mode: basic               # 'basic' or 'full'
augmentation: standard               # 'none', 'light', or 'standard'
target_size: [256, 256]              # image resize dimensions
epochs: 18                           # training epochs
```
 
**Never hardcode paths in scripts or notebooks — always use `config.yaml`.**
 
---
 
## Validation Split
 
- 200 images total in which 160 train and 40 validation
- Fixed split stored in `val_ids.txt`
- **Do not modify `val_ids.txt`** as changing it breaks reproducibility across teams
 
---
 
## Reproducibility Rules
 
- **Do not modify** `dataset.py` or `val_ids.txt`
- **Always use** `config.yaml` for all paths
- **Never store** dataset files in the repository
- **Never hardcode** paths in scripts or notebooks
 
