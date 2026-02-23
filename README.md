# Quorum: Cell Segmentation Comparison

This project compares two independent model architectures on the same microscopy dataset and evaluates them against Weka predictions. The goal is reproducible and fair comparison using a shared data pipeline.

Code is versioned on GitHub. Data is stored on Google Drive. Training is typically run in Google Colab or locally.

## Project Structure

```text
quorum/
│
├── data_utils/
│   └── dataset.py
│
├── architecture_team_1/
├── architecture_team_2/
│
├── evaluation/
│   └── compare_models.ipynb
│
├── config.yaml
├── val_ids.txt
├── requirements.txt
└── README.md
```

## Dataset Location

Data lives on shared Google Drive and is not stored in this repository.

Expected folder structure:

```text
CMPUT469_Cell/
├── Microscopy_images/
├── Ground_truth_masks/
```

**Note:** Do not copy dataset files into the repo.

## Setup Instructions (Google Colab)

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone repository

```bash
!git clone https://github.com/your-username/quorum.git
```

### Step 3: Import shared dataset

```python
import sys
sys.path.append('/content/quorum')

from data_utils.dataset import CellDataset
```

### Step 4: Load dataset

```python
from torch.utils.data import DataLoader

dataset = CellDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(loader))
print(images.shape, masks.shape)
```

## Using config.yaml

`config.yaml` defines where the dataset is stored.

Default configuration for Colab:

```yaml
data_root: /content/drive/MyDrive/CMPUT469_Cell
image_size: 256
batch_size: 8
```

If running locally, update only the `data_root` path:

```yaml
data_root: /your/local/path/CMPUT469_Cell
```

Do not hardcode paths inside notebooks or scripts.

## Team Responsibilities

### Data Pipeline

- maintains `dataset.py`
- defines preprocessing
- maintains `config.yaml`

### Architecture Teams

- implement model inside their folder
- import `CellDataset` from `data_utils`
- do not modify shared modules
- save predictions to Google Drive using original Cell IDs

### Evaluation

- compares predictions from both teams
- computes Dice and IoU metrics
- uses fixed validation IDs from `val_ids.txt`

## Reproducibility Rules

- Do not modify `dataset.py`
- Use validation IDs from `val_ids.txt`
- Use `config.yaml` for all paths
- Do not store dataset files in the repository
- Save predictions with original Cell ID filenames
