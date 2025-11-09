# Quick Start: WandB Experiments

## Setup (One-time)

```bash
# 1. Checkout the wandb-experiments branch
git checkout wandb-experiments

# 2. Run setup script
./scripts/setup_wandb.sh
```

This will:
- Check your Poetry environment
- Verify WandB installation
- Help you login to WandB

## Running Experiments

### Single Experiment

```bash
# Quick debug (2 epochs, ~15 min)
poetry run python src/train.py --config configs/tusimple_debug.yaml --wandb

# Full training - ResNet-18 (100 epochs, ~6-8 hours)
poetry run python src/train.py --config configs/tusimple_full.yaml --wandb

# Full training - ResNet-34
poetry run python src/train.py --config configs/tusimple_resnet34.yaml --wandb

# Full training - ResNet-122
poetry run python src/train.py --config configs/tusimple_resnet122.yaml --wandb
```

### Batch Experiments

```bash
# Run all experiments (ResNet-18, 34)
./scripts/run_wandb_experiments.sh all

# Just ResNet-18
./scripts/run_wandb_experiments.sh resnet18

# Just ResNet-34
./scripts/run_wandb_experiments.sh resnet34

# Quick test (50 epochs)
./scripts/run_wandb_experiments.sh fast

# Debug (2 epochs)
./scripts/run_wandb_experiments.sh debug
```

## View Results

After starting training, you'll see a link like:
```
🚀 View run at https://wandb.ai/your-username/LaneATT-TuSimple/runs/XXXXXXXX
```

Click it to see real-time metrics!

## What You'll See

- **Loss curves**: Training and validation loss over time
- **Accuracy metrics**: F1, precision, recall
- **Learning rate**: How LR changes during training
- **System metrics**: GPU usage, CPU, memory
- **Hyperparameters**: All config settings

## Comparing with Bozhen

You can compare your results with Bozhen's:
https://wandb.ai/bozhen2-uiuc/LaneATT-TuSimple

Look for experiments like:
- `tusimple_resnet122`
- `tusimple_resnet34`
- `tusimple_fast`

## Tips

1. **Run overnight**: Full training takes 6-8 hours
2. **Monitor GPU**: Use `nvidia-smi` or WandB system metrics
3. **Save checkpoints**: Models saved to `external/LaneATT/experiments/`
4. **Compare runs**: Use WandB's comparison view

## Full Documentation

See [`docs/wandb-experiments.md`](docs/wandb-experiments.md) for complete guide.
