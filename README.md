# Lane Detection with LaneATT — CS543/ECE549 Class Project

This repository contains our final project for **CS543/ECE549 (Computer Vision)**.  
We use the **LaneATT** model to study lane detection on the TuSimple and CULane datasets.

**Team Members:** Bozhen Zhou, Larry Liao, Manan Hitesh Masheswari, Rashiqul Alam

---

## Prerequisites

- **Python**: 3.10 or later
- **NVIDIA GPU**: Required (CUDA 12.4+ recommended)
- **CUDA Toolkit**: Must have `nvcc` compiler installed
- **Disk Space**: ~20 GB for TuSimple dataset

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:rashiqul/laneatt-classproject.git
cd laneatt-classproject
```

### 2. Initialize Git Submodules

The LaneATT implementation is included as a submodule:

```bash
git submodule update --init --recursive
```

Verify the submodule is populated:
```bash
ls external/LaneATT/
# Should show: main.py, lib/, cfgs/, etc.
```

### 3. Install Poetry

We use Poetry 1.1.12+ for dependency management:

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

### 4. Install Dependencies

```bash
poetry install --no-root
```

This installs PyTorch, OpenCV, imgaug, and all other dependencies from `pyproject.toml`.

### 5. Build the NMS CUDA Extension

LaneATT uses a custom CUDA-based NMS (Non-Maximum Suppression) module:

```bash
cd external/LaneATT/lib/nms
poetry run python setup.py install
cd ../../../..
```

**Requirements:**
- CUDA toolkit with `nvcc` compiler
- Compatible GPU (tested on Quadro T1000)

Verify the build:
```bash
poetry run python -c "from nms import nms; print('NMS loaded successfully')"
```

### 6. Fix Library Path (Important!)

Add this to your `~/.bashrc` to fix PyTorch CUDA library loading:

```bash
# Add PyTorch's CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(poetry run python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null):$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

**Why this is needed:** The compiled NMS module needs to find PyTorch's CUDA libraries (`libc10.so`, `libtorch_cuda.so`, etc.). Without this, you'll get `ImportError: libc10.so: cannot open shared object file`.

### 7. Verify Installation

```bash
poetry run python -c "import torch, cv2, imgaug, numpy; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.5.1+cu124
CUDA available: True
```

---

## Dataset Setup

### TuSimple Dataset (~20 GB)

#### 1. Download the Dataset

```bash
cd datasets

# Train & validation data (~10 GB)
mkdir tusimple
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple

# Test data (~10 GB)
mkdir tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test

# Test annotations
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/

cd ..
```

#### 2. Create Train/Validation Split

The original TuSimple dataset provides three training annotation files. Our modified version expects `label_data_train.json` and `label_data_val.json` for proper train/validation split:

```bash
cd external/LaneATT/datasets/tusimple

# Create train split (80% - combines 0313 and 0531)
cat label_data_0313.json label_data_0531.json > label_data_train.json

# Create validation split (20% - uses 0601)
cp label_data_0601.json label_data_val.json

cd ../../../..
```

**Split breakdown:**
- Training: ~4,800 images (label_data_0313.json + label_data_0531.json)
- Validation: ~1,200 images (label_data_0601.json)

#### 3. (Optional) Create Smaller Subset for Faster Experimentation

For quick prototyping, you can create a smaller subset:

```bash
poetry run python src/create_subset_data.py \
    --source-train datasets/tusimple \
    --source-test datasets/tusimple-test \
    --target-train datasets/tusimple-1k \
    --target-test datasets/tusimple-test-1k \
    --num-train 1000 \
    --num-test 500
```

This creates:
- `datasets/tusimple-1k/` with 1,000 training images
- `datasets/tusimple-test-1k/` with 500 test images

Then update your config file to point to these smaller datasets.

---

## Training

### Two Ways to Execute LaneATT

**Method 1: Class Project Wrapper** (Recommended)
```bash
# Train with custom config and WandB support
poetry run python src/train.py --config configs/tusimple_full.yaml [--wandb] [--epochs N]
```

**Method 2: Original LaneATT Entry Point** (Direct)
```bash
cd external/LaneATT
python main.py train --exp_name tusimple_resnet18 --cfg cfgs/laneatt_tusimple_resnet18.yml
python main.py test --exp_name tusimple_resnet18 --epoch 70
cd ../..
```

**Batch Training Multiple Models:**
```bash
bash run_all_models.sh  # Runs ResNet-18, 34, and 122 sequentially
```

---

## Training Details

## Training Details

### Quick Start Examples

Train on the full TuSimple dataset with the custom train/val split:

```bash
poetry run python src/train.py --config configs/tusimple_full.yaml
```

Or for quick debugging (2 epochs only):

```bash
poetry run python src/train.py --config configs/tusimple_debug.yaml
```

### Training Options

**Override epochs:**
```bash
poetry run python src/train.py --config configs/tusimple_full.yaml --epochs 50
```

**Enable WandB logging:**
```bash
# First time: Login to WandB
poetry run python -c "import wandb; wandb.login()"

# Train with tracking
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb
```

### What Happens During Training

The training script (`src/train.py`):
1. Loads your config from the specified YAML file (e.g., `configs/tusimple_full.yaml`)
2. Identifies the LaneATT config to use (e.g., `laneatt_tusimple_split_resnet18.yml`)
3. Optionally overrides epochs if `--epochs` is specified
4. Calls the LaneATT training script (`external/LaneATT/main.py`) with appropriate parameters
5. Saves checkpoints and logs to `experiments/runs/<experiment_name>/`

### Monitoring Training

Training outputs are saved to:
- **Checkpoints**: `external/LaneATT/experiments/<exp_name>/`
- **TensorBoard logs**: `external/LaneATT/tensorboard/<exp_name>/`
- **WandB logs** (if enabled): Online at wandb.ai

To view TensorBoard logs:
```bash
poetry run tensorboard --logdir external/LaneATT/tensorboard
```

### Configuration Files

- **Project configs** (`configs/`):
  - [`configs/tusimple_full.yaml`](configs/tusimple_full.yaml) - Full training (100 epochs, batch size 8)
  - [`configs/tusimple_debug.yaml`](configs/tusimple_debug.yaml) - Debug config (2 epochs, batch size 4)
  - [`configs/culane_debug.yaml`](configs/culane_debug.yaml) - CULane debug config
  - [`configs/paths.yaml`](configs/paths.yaml) - Dataset paths (currently unused)

- **LaneATT configs** (`external/LaneATT/cfgs/`):
  - [`laneatt_tusimple_split_resnet18.yml`](external/LaneATT/cfgs/laneatt_tusimple_split_resnet18.yml) - Custom train/val split config
  - Other original LaneATT configs for different datasets and backbones

---

## Key Modifications to LaneATT

Our fork includes several enhancements to the original LaneATT implementation:

### 1. Train/Validation Split Support
- **Modified**: `external/LaneATT/lib/datasets/tusimple.py`
- **Change**: Added support for `train`, `val`, and `train+val` splits
- **Files expected**:
  - `label_data_train.json` (training data)
  - `label_data_val.json` (validation data)
  - `test_label.json` (test data)

### 2. Custom Split Configuration
- **Added**: `external/LaneATT/cfgs/laneatt_tusimple_split_resnet18.yml`
- **Features**:
  - Separate train/val datasets in config
  - CosineAnnealingLR scheduler tuned for 100 epochs
  - Data augmentation enabled only for training
  - Validation every 5 epochs

### 3. Training Wrapper Script
- **Added**: `src/train.py`
- **Features**:
  - Easy-to-use command-line interface
  - Epoch override: `--epochs <N>`
  - WandB toggle: `--wandb` (disabled by default)
  - Automatic path resolution for configs
  - Temporary config generation for overrides

### 4. Subset Creation Utility
- **Added**: `src/create_subset_data.py`
- **Purpose**: Create smaller datasets for rapid prototyping
- **Usage**:
  ```bash
  poetry run python src/create_subset_data.py \
      --num-train 1000 --num-test 500
  ```

### 5. Improved Dependency Management
- **Updated**: `pyproject.toml`
- **Changes**:
  - Pinned PyTorch 2.5.1 with CUDA 12.4
  - NumPy < 2.0 for imgaug compatibility
  - Added WandB for experiment tracking
  - Added utilities: gdown, rich, p-tqdm

### 6. Environment Cleanup Script
- **Added**: `scripts/clean_env.sh`
- **Purpose**: Clean Poetry cache and reinstall from scratch
- **Handles**: Broken virtualenvs, PATH issues, corrupted caches

---

## Project Structure

```
laneatt-classproject/
├── configs/                    # Training configurations
│   ├── tusimple_full.yaml     # Full training config (100 epochs)
│   ├── tusimple_debug.yaml    # Debug config (2 epochs)
│   ├── culane_debug.yaml      # CULane debug config
│   └── paths.yaml             # Dataset paths
│   ├── tusimple/              # Training data + train/val split
│   │   ├── clips/             # Images organized by date
│   │   ├── label_data_0313.json
│   │   ├── label_data_0531.json
│   │   ├── label_data_0601.json
│   │   ├── label_data_train.json  # Train split (0313 + 0531)
│   │   └── label_data_val.json    # Val split (0601)
│   └── tusimple-test/         # Test data
│       ├── clips/
│       ├── test_tasks_0627.json
│       └── test_label.json
├── experiments/                # Training outputs and logs (gitignored)
│   └── runs/
│       ├── tusimple_full/
│       └── tusimple_debug/
├── external/
│   └── LaneATT/               # LaneATT submodule (our fork)
│       ├── main.py            # LaneATT training/testing entry point
│       ├── cfgs/              # LaneATT model configurations
│       │   ├── laneatt_tusimple_split_resnet18.yml  # ⭐ Custom split config
│       │   └── ...            # Other original configs
│       ├── lib/
│       │   ├── datasets/
│       │   │   └── tusimple.py  # ⭐ Modified for train/val split
│       │   └── models/
│       ├── experiments/       # Training checkpoints (gitignored)
│       ├── tensorboard/       # TensorBoard logs (gitignored)
│       └── wandb/             # WandB logs (gitignored)
├── src/
│   ├── train.py               # ⭐ Training wrapper script
│   └── create_subset_data.py  # ⭐ Create smaller dataset subsets
├── run_all_models.sh          # ⭐ Batch training script
├── pyproject.toml             # Poetry dependencies
└── README.md

⭐ = Modified or added by our team
```

---

## Complete Training Pipeline

This section provides step-by-step instructions to run the entire training pipeline from scratch.

### Step 1: Environment Setup

```bash
# Clone the repository
git clone --recursive git@github.com:rashiqul/laneatt-classproject.git
cd laneatt-classproject

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
poetry install --no-root

# Build NMS CUDA extension
cd external/LaneATT/lib/nms
poetry run python setup.py install
cd ../../../..

# Configure library path for CUDA
export LD_LIBRARY_PATH=$(poetry run python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Verify installation
poetry run python -c "import torch, cv2, imgaug, numpy; from nms import nms; print('✅ All dependencies loaded successfully')"
```

### Step 2: Download and Prepare Dataset

```bash
# Create datasets directory
mkdir -p datasets
cd datasets

# Download TuSimple training data (~10 GB)
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip"
unzip train_set.zip -d tusimple

# Download TuSimple test data (~10 GB)
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip"
unzip test_set.zip -d tusimple-test
wget "https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json" -P tusimple-test/

cd ..
```

### Step 3: Create Train/Validation Split

The modified TuSimple dataset loader expects separate train/val annotation files:

```bash
cd datasets/tusimple

# Create training split (combines first two dates: ~4,800 images)
cat label_data_0313.json label_data_0531.json > label_data_train.json

# Create validation split (last date: ~1,200 images)
cp label_data_0601.json label_data_val.json

# Verify files exist
ls -lh label_data_*.json
# You should see:
# label_data_0313.json
# label_data_0531.json
# label_data_0601.json
# label_data_train.json  (largest file)
# label_data_val.json

cd ../..
```

### Step 4: Run Training

#### Option A: Full Training (Recommended for Final Results)

```bash
# Train for 100 epochs with ResNet-18 backbone
poetry run python src/train.py --config configs/tusimple_full.yaml

# With WandB logging enabled (optional)
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb

# Override epochs (e.g., train for 50 epochs instead)
poetry run python src/train.py --config configs/tusimple_full.yaml --epochs 50
```

**Expected training time**: ~6-8 hours on NVIDIA Quadro T1000 for 100 epochs

#### Option B: Debug Training (Fast Iteration)

```bash
# Quick 2-epoch run for testing
poetry run python src/train.py --config configs/tusimple_debug.yaml
```

**Expected training time**: ~10-15 minutes for 2 epochs

### Step 5: Monitor Training Progress

**View TensorBoard logs:**
```bash
poetry run tensorboard --logdir external/LaneATT/tensorboard
# Open browser to http://localhost:6006
```

**Check training outputs:**
```bash
# Checkpoints and model weights
ls external/LaneATT/experiments/tusimple_full/

# Training logs
cat external/LaneATT/experiments/tusimple_full/*.log
```

### Step 6: Evaluate Model

After training completes, evaluate on the test set:

```bash
cd external/LaneATT

# Evaluate the best checkpoint (typically the last epoch)
poetry run python main.py test --exp_name tusimple_full

# Visualize predictions (creates output images)
poetry run python main.py test --exp_name tusimple_full --view all

cd ../..
```

### Step 7: View Results

Results are saved to:
- **Metrics**: `external/LaneATT/experiments/tusimple_full/test_results.json`
- **Predictions**: `external/LaneATT/experiments/tusimple_full/tusimple_predictions.json`
- **Visualizations** (if --view all): `external/LaneATT/experiments/tusimple_full/visualization/`

**Expected results** (on TuSimple test set):
- Accuracy: ~95.5%
- F1 Score: ~96.7%
- FDR (False Discovery Rate): ~3.5%
- FNR (False Negative Rate): ~3.0%

### Optional: Create Smaller Dataset for Faster Experiments

For rapid prototyping and debugging:

```bash
# Create 1k training samples and 500 test samples
poetry run python src/create_subset_data.py \
    --source-train datasets/tusimple \
    --source-test datasets/tusimple-test \
    --target-train datasets/tusimple-1k \
    --target-test datasets/tusimple-test-1k \
    --num-train 1000 \
    --num-test 500

# Update config to use smaller dataset
# Edit external/LaneATT/cfgs/laneatt_tusimple_split_resnet18.yml
# Change: root: "datasets/tusimple-1k"
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'nms'"

Build the NMS CUDA extension:
```bash
cd external/LaneATT/lib/nms
poetry run python setup.py install
cd ../../../..
```

### "ImportError: libc10.so: cannot open shared object file"

Add PyTorch's lib directory to `LD_LIBRARY_PATH` in your `~/.bashrc` (see Step 6 above), or run:
```bash
export LD_LIBRARY_PATH=$(poetry run python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

### "CUDA out of memory"

Reduce `batch_size` in the LaneATT config file:
```yaml
# In external/LaneATT/cfgs/laneatt_tusimple_split_resnet18.yml
batch_size: 4  # or 2, or 1
```

### "FileNotFoundError: label_data_train.json not found"

Make sure you've created the train/val split:
```bash
cd datasets/tusimple
cat label_data_0313.json label_data_0531.json > label_data_train.json
cp label_data_0601.json label_data_val.json
cd ../..
```

### "Split 'train' does not exist"

Your dataset path might be wrong. Check the config file:
```yaml
# In external/LaneATT/cfgs/laneatt_tusimple_split_resnet18.yml
datasets:
  train:
    parameters:
      root: "datasets/tusimple"  # Should point to correct location
```

The path is relative to `external/LaneATT/`, so use `datasets/tusimple` or an absolute path.

### Git Submodule Not Initialized

If `external/LaneATT/` is empty:
```bash
git submodule update --init --recursive
```

### Clean and Rebuild Environment

If you encounter persistent issues with Poetry or virtualenv:
```bash
./scripts/clean_env.sh
```

This removes the Poetry virtualenv and reinstalls all dependencies fresh.

### WandB Login Issues

If WandB asks for login during training and you don't want to use it:
```bash
# Disable WandB globally
poetry run wandb offline

# Or use --wandb flag only when you want logging
poetry run python src/train.py --config configs/tusimple_full.yaml  # WandB disabled
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb  # WandB enabled
```

---

## Quick Reference

### Common Commands

```bash
# Train full model (100 epochs)
poetry run python src/train.py --config configs/tusimple_full.yaml

# Quick debug run (2 epochs)
poetry run python src/train.py --config configs/tusimple_debug.yaml

# Train with custom epochs
poetry run python src/train.py --config configs/tusimple_full.yaml --epochs 50

# Train with WandB logging
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb

# Evaluate trained model
cd external/LaneATT
poetry run python main.py test --exp_name tusimple_full
poetry run python main.py test --exp_name tusimple_full --view all  # with visualization
cd ../..

# Create smaller dataset subset
poetry run python src/create_subset_data.py --num-train 1000 --num-test 500

# View TensorBoard logs
poetry run tensorboard --logdir external/LaneATT/tensorboard

# Clean and rebuild environment
./scripts/clean_env.sh
```

### Important Paths

| Component | Location |
|-----------|----------|
| Training script | `src/train.py` |
| Project configs | `configs/` |
| LaneATT configs | `external/LaneATT/cfgs/` |
| Dataset | `datasets/tusimple/` |
| Train annotations | `datasets/tusimple/label_data_train.json` |
| Val annotations | `datasets/tusimple/label_data_val.json` |
| Test annotations | `datasets/tusimple-test/test_label.json` |
| Model checkpoints | `external/LaneATT/experiments/<exp_name>/` |
| TensorBoard logs | `external/LaneATT/tensorboard/<exp_name>/` |
| WandB logs | `external/LaneATT/wandb/` |

### Dataset Statistics

| Split | Files | Images | Source |
|-------|-------|--------|--------|
| Train | `label_data_train.json` | ~4,800 | 0313 + 0531 |
| Val | `label_data_val.json` | ~1,200 | 0601 |
| Test | `test_label.json` | ~2,800 | Official test set |

---

## Documentation

- [`external/LaneATT/README.md`](external/LaneATT/README.md) - Original LaneATT documentation
- [`external/LaneATT/DATASETS.md`](external/LaneATT/DATASETS.md) - LaneATT dataset setup guide

---

## Notes

- Tested with Python 3.10, Poetry 1.1.12, PyTorch 2.5.1, CUDA 12.4
- Requires NumPy 1.x (not 2.x) due to imgaug compatibility
- GPU training only—CPU mode is not supported by LaneATT's NMS implementation
- Training time: ~6-8 hours for 100 epochs on NVIDIA Quadro T1000

---

## Citation

If you use this code or our modifications in your research, please cite the original LaneATT paper:

```bibtex
@InProceedings{tabelini2021cvpr,
  author    = {Lucas Tabelini and Rodrigo Berriel and Thiago M. Paix\~ao and Claudine Badue and Alberto Ferreira De Souza and Thiago Oliveira-Santos},
  title     = {{Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection}},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```