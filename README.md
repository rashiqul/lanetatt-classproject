# CD543 / ECE549 Final Project — Attention-Guided Lane Detection

This repository contains our final project for **CD543/ECE549 (Computer Vision)**.  
We re-implement a simplified version of **LaneATT** to study its architecture, hyperparameters, and performance on small subsets of TuSimple or CULane datasets.

---

## Setup

### 1. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

### 2. Clone the repository
```bash
git clone git@github.com:YOURUSER/laneatt-classproj.git
cd laneatt-classproj
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

### 4. Install dependencies and activate the environment
```bash
poetry install
poetry shell
```

### 5. Verify installation
```bash
python -c "import torch, torchvision, numpy, cv2; print(torch.__version__, torchvision.__version__, numpy.__version__, cv2.__version__)"
```

If version numbers print without errors, the setup is complete.

## Optional: GPU Setup

If you have an NVIDIA GPU:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Notes

- Requires Python 3.10–3.14
- Tested with Poetry and PyTorch 2.5.1
- For reproducibility, all dependencies are defined in `pyproject.toml` and `poetry.lock`
