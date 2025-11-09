# WandB Experiment Tracking

This guide explains how to run experiments with Weights & Biases (WandB) tracking, similar to [Bozhen's experiments](https://wandb.ai/bozhen2-uiuc/LaneATT-TuSimple?nw=nwuserbozhen2).

## Quick Start

### 1. Login to WandB

First time setup:

```bash
poetry run python -c "import wandb; wandb.login()"
```

This will open a browser window. Copy your API key from https://wandb.ai/authorize and paste it.

### 2. Run a Single Experiment

```bash
# Quick debug run (2 epochs, ~15 minutes)
poetry run python src/train.py --config configs/tusimple_debug.yaml --wandb

# Full training with ResNet-18 (100 epochs, ~6-8 hours)
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb

# Full training with ResNet-34
poetry run python src/train.py --config configs/tusimple_resnet34.yaml --wandb

# Full training with ResNet-122
poetry run python src/train.py --config configs/tusimple_resnet122.yaml --wandb
```

### 3. Run Multiple Experiments

Use the batch script to run multiple configurations:

```bash
# Run all experiments (ResNet-18, ResNet-34)
./scripts/run_wandb_experiments.sh all

# Run specific backbone
./scripts/run_wandb_experiments.sh resnet18
./scripts/run_wandb_experiments.sh resnet34
./scripts/run_wandb_experiments.sh resnet122

# Quick test (50 epochs)
./scripts/run_wandb_experiments.sh fast

# Debug (2 epochs)
./scripts/run_wandb_experiments.sh debug
```

## View Results

After training starts, you'll see:
```
🚀 wandb: Currently logged in as: your-username
wandb: Tracking run with wandb version 0.22.3
wandb: Run data is saved locally in ./wandb/run-XXXXXXXX-XXXXXXXX
wandb: Run wandb offline to turn off syncing.
wandb: Syncing run tusimple_full
wandb: ⭐️ View project at https://wandb.ai/your-username/LaneATT-TuSimple
wandb: 🚀 View run at https://wandb.ai/your-username/LaneATT-TuSimple/runs/XXXXXXXX
```

Click the link to view your experiment in real-time!

## What Gets Logged

The WandB integration logs:

### Training Metrics (per step)
- `train/loss` - Total training loss
- `train/cls_loss` - Classification loss
- `train/reg_loss` - Regression loss
- `train/learning_rate` - Current learning rate

### Validation Metrics (per epoch)
- `val/loss` - Validation loss
- `val/accuracy` - Lane detection accuracy
- `val/fp` - False positives
- `val/fn` - False negatives
- `val/precision` - Precision score
- `val/recall` - Recall score
- `val/f1` - F1 score

### Configuration
All hyperparameters are logged:
- Model architecture (backbone)
- Learning rate
- Batch size
- Image size
- Anchor settings
- Data augmentation parameters

## Comparing Experiments

Once you have multiple runs, you can:

1. **Compare metrics** - View loss/accuracy curves side-by-side
2. **Compare configs** - See which hyperparameters differ
3. **Filter runs** - Group by backbone, batch size, etc.
4. **Create reports** - Share results with your team

## Example Experiments (Like Bozhen's)

Here are some experiment ideas:

### 1. Backbone Comparison
```bash
# Compare ResNet-18 vs ResNet-34 vs ResNet-122
./scripts/run_wandb_experiments.sh all
```

### 2. Image Size Study
Edit configs to test different resolutions:
- 360x640 (default)
- 720x1280 (high-res)
- 288x800 (different aspect ratio)

### 3. Batch Size Ablation
Test different batch sizes:
- batch_size: 4
- batch_size: 8 (default)
- batch_size: 16

### 4. Learning Rate Schedule
Compare different schedulers:
- CosineAnnealingLR (default)
- StepLR
- ReduceLROnPlateau

## Tips

### Offline Mode
If you want to train without syncing (save to sync later):
```bash
export WANDB_MODE=offline
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb
```

Then sync later:
```bash
wandb sync wandb/run-XXXXXXXX-XXXXXXXX
```

### Custom Run Names
Edit the experiment name in your config:
```yaml
log:
  out_dir: experiments/runs/my_custom_experiment_name
```

This will be used as the WandB run name.

### Resume Interrupted Runs
WandB automatically handles resume. If training crashes, just restart with the same command - it will continue from the last checkpoint.

### Group Related Runs
You can add groups in `lib/logger_wandb.py`:
```python
wandb.init(
    project="LaneATT-TuSimple",
    name=exp_name,
    group="resnet18_experiments",  # Add this
    config=cfg
)
```

## Troubleshooting

### "wandb: Network error"
Your firewall might be blocking WandB. Try:
```bash
export WANDB_MODE=offline
```

### "wandb: ERROR Run initialization failed"
Make sure you're logged in:
```bash
poetry run python -c "import wandb; wandb.login()"
```

### Don't Want WandB Logging?
Simply omit the `--wandb` flag:
```bash
poetry run python src/train.py --config configs/tusimple_full.yaml
```

## Resources

- **Your WandB Dashboard**: https://wandb.ai
- **Bozhen's Experiments**: https://wandb.ai/bozhen2-uiuc/LaneATT-TuSimple
- **WandB Documentation**: https://docs.wandb.ai
