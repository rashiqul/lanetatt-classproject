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
git clone git@github.com:rashiqul/lanetatt-classproject.git
cd lanetatt-classproject
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

See [`docs/datasets.md`](docs/datasets.md) for more details on TuSimple and CULane datasets.

---

## Training

### Run Training Script

```bash
poetry run python src/train.py --config configs/tusimple_debug.yaml
```

**What it does:**
1. Loads your config from `configs/tusimple_debug.yaml`
2. Calls the LaneATT training script with the appropriate parameters
3. Saves results to `experiments/runs/tusimple_debug/`

### Configuration Files

- [`configs/tusimple_debug.yaml`](configs/tusimple_debug.yaml) - Debug config for quick iteration (2 epochs, batch size 4)
- [`configs/paths.yaml`](configs/paths.yaml) - Dataset paths (relative to project root)
- [`external/LaneATT/cfgs/`](external/LaneATT/cfgs/) - LaneATT's original configs

---

## Project Structure

```
laneatt-classproject/
├── configs/              # Training configurations
├── data/                 # Dataset documentation
├── datasets/             # Actual datasets (gitignored, ~20GB)
│   ├── tusimple/
│   └── tusimple-test/
├── docs/                 # Project documentation
├── experiments/          # Training outputs and logs
├── external/
│   └── LaneATT/         # LaneATT submodule
├── scripts/
│   └── clean_env.sh     # Clean and rebuild Poetry environment
├── src/
│   └── train.py         # Training wrapper script
├── pyproject.toml       # Poetry dependencies
└── README.md
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

Add PyTorch's lib directory to `LD_LIBRARY_PATH` in your `~/.bashrc` (see Step 6 above).

### "CUDA out of memory"

Reduce `batch_size` in your config file:
```yaml
train:
  batch_size: 2  # or 1
```

### Clean and Rebuild Environment

```bash
./scripts/clean_env.sh
```

This removes the Poetry virtualenv and reinstalls all dependencies fresh.

---

## Documentation

- [`docs/datasets.md`](docs/datasets.md) - Dataset details and evaluation metrics
- [`data/README.md`](data/README.md) - Data preparation guide
- [`external/LaneATT/README.md`](external/LaneATT/README.md) - LaneATT documentation

---

## Notes

- Tested with Python 3.10, Poetry 1.1.12, PyTorch 2.5.1, CUDA 12.4
- Requires NumPy 1.x (not 2.x) due to imgaug compatibility
- GPU training only—CPU mode is not supported by LaneATT's NMS implementation