# CS543 / ECE549 Final Project — Attention-Guided Lane Detection

This repository contains our final project for **CS543/ECE549 (Computer Vision)**.  
We implement **LaneATT** (Attention-based Lane Detection) to study its architecture, hyperparameters, and performance on TuSimple and CULane datasets.

## Features

- ✅ **Integrated CuLane Optimizations** - Performance improvements from zhoubozhen's fork
- ✅ **Workspace-Level Training** - Run directly from project root (no cd into submodule needed)
- ✅ **WandB Integration** - Experiment tracking and visualization
- ✅ **Optimized DataLoaders** - 16 workers, pin_memory, prefetch_factor for faster training
- ✅ **All Dependencies Fixed** - TensorBoard, NumPy 1.x compatibility, NMS CUDA extension

---

## Setup

### 1. Install Poetry 1.1.12
```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.1.12 python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

### 2. Clone the repository
```bash
git clone git@github.com:rashiqul/lanetatt-classproject.git
cd lanetatt-classproject
```

### 3. Initialize the submodule
Run these two commands once after cloning:

```bash
git submodule init
git submodule update
```

That will:
- Register the submodule
- Check out the correct commit inside `external/LaneATT`

You should then see the LaneATT code under `external/LaneATT/`.

### 4. Install dependencies
```bash
poetry install --no-root
```

**Note:** This project uses specific versions for compatibility:
- NumPy 1.26.4 (required for imgaug compatibility)
- opencv-python 4.9.x (compatible with NumPy 1.x)
- TensorBoard 2.20.0
- WandB 0.23.1 for experiment tracking

### 5. Verify installation
```bash
python -c "import torch, torchvision, numpy, cv2; print(torch.__version__, torchvision.__version__, numpy.__version__, cv2.__version__)"
```

If version numbers print without errors, the setup is complete.

### 6. Build NMS CUDA extension
The NMS (Non-Maximum Suppression) module needs to be compiled once:

```bash
cd external/LaneATT/lib/nms
poetry run python setup.py build_ext --inplace
cd ../../../..
```

**Note:** The NMS module is automatically added to the Python path via `lib/__init__.py` - no pip install needed!

## Dataset Setup

Download the datasets and place them in the appropriate directories:

```bash
# TuSimple dataset structure
external/LaneATT/datasets/tusimple/
├── clips/
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
└── test_label.json

# CuLane dataset structure  
external/LaneATT/datasets/culane/
├── driver_*_*frame/
├── laneseg_label_w16/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Running Training

### From Workspace Root (Recommended)
You can now run training directly from the project workspace:

```bash
# Simple training command
poetry run python train.py train --exp_name my_experiment --cfg external/LaneATT/cfgs/laneatt_tusimple_resnet18.yml

# Or use the convenience script with tmux
./train_laneatt.sh
```

### Command Line Examples
```bash
# Train on TuSimple dataset (ResNet18 backbone)
poetry run python train.py train --exp_name tusimple_test --cfg cfgs/laneatt_tusimple_resnet18.yml

# Train on CuLane dataset (ResNet18 backbone)
poetry run python train.py train --exp_name culane_test --cfg cfgs/laneatt_culane_resnet18.yml

# Train with different backbones
poetry run python train.py train --exp_name culane_resnet34 --cfg cfgs/laneatt_culane_resnet34.yml

# Test/evaluate a trained model
poetry run python train.py test --exp_name tusimple_test --epoch 15

# Resume training from checkpoint
poetry run python train.py train --exp_name tusimple_test --cfg cfgs/laneatt_tusimple_resnet18.yml --resume
```

### Configuration Files

Available configs in `external/LaneATT/cfgs/`:
- `laneatt_tusimple_resnet18.yml` - TuSimple with ResNet18
- `laneatt_tusimple_resnet34.yml` - TuSimple with ResNet34  
- `laneatt_culane_resnet18.yml` - CuLane with ResNet18
- `laneatt_culane_resnet34.yml` - CuLane with ResNet34

**Key optimizations in current integration:**
- DataLoader workers: 16 (increased from 8)
- pin_memory: True
- prefetch_factor: 4
- persistent_workers: True
```

## Optional: GPU Setup

If you have an NVIDIA GPU:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Project Structure

```
.
├── train.py                    # Main training entry point (workspace-level)
├── train_laneatt.sh           # Training script with tmux support
├── pyproject.toml             # Poetry dependencies
├── poetry.lock                # Locked dependency versions
├── README.md                  # This file
├── scripts/                   # Utility scripts
└── external/
    └── LaneATT/               # LaneATT submodule (rashiqul/LaneATT fork)
        ├── main.py            # Original LaneATT entry point
        ├── cfgs/              # Configuration files
        ├── lib/               # Core library code
        │   ├── models/        # Model architectures
        │   ├── datasets/      # Dataset loaders
        │   └── nms/           # CUDA NMS extension
        ├── experiments/       # Training outputs (checkpoints, logs)
        └── wandb/             # WandB experiment tracking
```

## Integration Details

This project integrates changes from multiple sources:

1. **Base:** lucastabelini/LaneATT (original implementation)
2. **Integrated:** zhoubozhen/LaneATT (CuLane optimizations)
3. **Current Fork:** rashiqul/LaneATT (merged improvements)

Branch: `integrate-zhoubozhen-culane` contains all integrated changes.

## Troubleshooting

### NMS Module Not Found
The NMS CUDA extension should be automatically loaded. If you see import errors:
```bash
cd external/LaneATT/lib/nms
poetry run python setup.py build_ext --inplace
```

### CUDA Compatibility
If you see CUDA warnings about version mismatch, they're usually safe to ignore. The code works with CUDA 12.4-12.8.

### NumPy/imgaug Issues
This project uses NumPy 1.x (not 2.x) due to imgaug compatibility. Don't upgrade NumPy to 2.x.

## Notes

- Requires Python 3.10–3.14
- Tested with Poetry 1.1.12 and PyTorch 2.5.1
- For reproducibility, all dependencies are defined in `pyproject.toml` and `poetry.lock`
- **Run from workspace root** - no need to cd into the submodule!
- Experiment results are tracked in WandB (login required for sync)

## Contributing

This is a class project repository. For the original LaneATT implementation, see:
- Original: https://github.com/lucastabelini/LaneATT
- Our fork: https://github.com/rashiqul/LaneATT
