# Satellite-Image-Based-Classifier
This is a basic minimal project for detecting changes in terms of infrastructure of an image of an area clicked by satellite at different times. 

# Satellite Change Detection (LEVIR-CD+)

Jupyter notebook project for **satellite image change detection** using paired images (Day 1 vs Day 30 / Time A vs Time B), producing a change mask, a Day30 overlay with red contours, and basic changed-area statistics.

---

## What this repo contains

- `satellite_detection.ipynb` — the full end-to-end notebook:
  - Loads and explores the LEVIR-CD+ dataset
  - Trains a Siamese U-Net–style change detector
  - Evaluates with IoU / F1
  - Runs inference on sample tiles or your own image pair
  - Visualizes results in a 1-1-1 layout: **Day 1 | Day 30 | Day 30 + changes**

> Note: This repo intentionally does **not** include the dataset or trained weights.

---

## Requirements

- macOS / Linux / Windows
- Python 3.10+ recommended (Apple Silicon users: MPS is supported if your PyTorch build supports it)
- Enough disk space for the dataset (LEVIR-CD+ is not tiny)

Python packages (installed via `requirements.txt` or manually):
- `torch`, `torchvision`
- `numpy`, `Pillow`
- `opencv-python`
- `matplotlib`, `tqdm`
- (optional) `gradio` for UI experiments

---

## Setup

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
dsource .venv/bin/activate # On Windows use: .venv\Scripts\activate.bat or .venv\Scripts\Activate.ps1 
```

## 2. Upgrade pip
```bash
python -m pip install -U pip
```
## 3. Install dependencies
#### If you have requirements.txt:
```bash
pip install -r requirements.txt
```

## Otherwise install manually:
```bash
pip install torch torchvision numpy pillow opencv-python matplotlib tqdm
```

## Dataset (manual download)
This project expects LEVIR-CD+ on disk, but does not ship it in the repo.
Download the dataset zip from Kaggle:

Dataset page: [mdrifaturrahman33/levir-cd-change-detection](https://www.kaggle.com/mdrifaturrahman33/levir-cd-change-detection)

Unzip it locally into the repo under:
```
data/levir_cd/LEVIR-CD+/
```
your final directory structure should look like:
```
data/levir_cd/LEVIR-CD+/ \
├── train \
│   ├── A \
│   ├── B \
│   └── label \
├── val \
│   ├── A \
│   ├── B \
│   └── label \
```
defaults to your current directory if not specified.

dataset root differs, update path variables in notebook.
download and unzip dataset manually before running notebook.
to run notebook:
```bash
source .venv/bin/activate # activate virtual env before launching jupyter notebook.
pip install jupyter
jupyter notebook # then open satellite_detection.ipynb and run cells top-to-bottom.
```

Inference with your own images:

1. In the notebook set two file paths, e.g.: DAY1_PATH = ".../day1.jpg"; DAY30_PATH = ".../day30.jpg". 
2. Run inference cell to produce overlays and masks. Ensure images are aligned and similar resolution. Resize or tile large images to avoid memory issues.
outcomes during runs may generate folders like final_outputs/, preds/, simple_buildings/. 
3. These are artifacts; do not commit them. Keep locally or move small examples into assets/ for README.
4. git/repo hygiene: uses .gitignore to exclude `.venv/`, data/, model weights (*.pt, *.pth), output folders, checkpoint files (.ipynb_checkpoints/). 
5. Remove accidental commits without deleting locally.
note on model & metrics: trained on building change detection; metrics include IoU and F1 scores on validation/test sets.

future work roadmap includes improving vegetation sensitivity, robust inference techniques, optional UI with Gradio, and detailed written reports.