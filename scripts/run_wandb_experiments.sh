#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# WandB Experiment Runner - Run multiple LaneATT experiments with tracking
# =============================================================================
# This script runs various model configurations similar to Bozhen's experiments
# All runs will be logged to WandB for comparison
#
# Usage:
#   ./scripts/run_wandb_experiments.sh [experiment_name]
#
# Examples:
#   ./scripts/run_wandb_experiments.sh all          # Run all experiments
#   ./scripts/run_wandb_experiments.sh resnet18     # Run only ResNet-18
#   ./scripts/run_wandb_experiments.sh resnet34     # Run only ResNet-34
#   ./scripts/run_wandb_experiments.sh fast         # Run fast experiments
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 WandB Experiment Runner"
echo "============================"
echo ""

# Check if WandB is logged in
if ! poetry run python -c "import wandb; wandb.login()" 2>/dev/null; then
    echo "⚠️  WandB not logged in. Please run:"
    echo "   poetry run python -c 'import wandb; wandb.login()'"
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local exp_name="$1"
    local config="$2"
    local epochs="${3:-100}"
    
    echo ""
    echo "📊 Starting experiment: $exp_name"
    echo "   Config: $config"
    echo "   Epochs: $epochs"
    echo "   Time: $(date)"
    echo "---------------------------------------------------"
    
    # Run with WandB enabled
    poetry run python scripts/train.py \
        --config "$config" \
        --epochs "$epochs" \
        --wandb
    
    echo "✅ Completed: $exp_name"
    echo ""
}

# Parse argument
EXPERIMENT="${1:-all}"

case "$EXPERIMENT" in
    "all")
        echo "Running ALL experiments (this will take a while!)"
        
        # ResNet-18 experiments
        run_experiment "tusimple_resnet18" "configs/tusimple_resnet18.yaml" 100
        
        # ResNet-34 experiments
        run_experiment "tusimple_resnet34" "configs/tusimple_resnet34.yaml" 100
        
        # ResNet-122 experiments (if you want to try)
        # run_experiment "tusimple_resnet122" "configs/tusimple_resnet122.yaml" 100
        ;;
        
    "resnet18")
        echo "Running ResNet-18 experiments"
        run_experiment "tusimple_resnet18" "configs/tusimple_resnet18.yaml" 100
        ;;
        
    "resnet34")
        echo "Running ResNet-34 experiments"
        run_experiment "tusimple_resnet34" "configs/tusimple_resnet34.yaml" 100
        ;;
        
    "resnet122")
        echo "Running ResNet-122 experiments"
        run_experiment "tusimple_resnet122" "configs/tusimple_resnet122.yaml" 100
        ;;
        
    "fast")
        echo "Running FAST experiments (ResNet-18, 50 epochs)"
        run_experiment "tusimple_resnet18_fast" "configs/tusimple_resnet18.yaml" 50
        ;;
        
    "debug")
        echo "Running DEBUG experiments (ResNet-18, 2 epochs)"
        run_experiment "tusimple_resnet18_debug" "configs/tusimple_resnet18.yaml" 2
        ;;
        
    *)
        echo "❌ Unknown experiment: $EXPERIMENT"
        echo ""
        echo "Available experiments:"
        echo "  all        - Run all experiments"
        echo "  resnet18   - ResNet-18 full training"
        echo "  resnet34   - ResNet-34 full training"
        echo "  resnet122  - ResNet-122 full training"
        echo "  fast       - Quick training (50 epochs)"
        echo "  debug      - Debug run (2 epochs)"
        exit 1
        ;;
esac

echo ""
echo "🎉 All experiments completed!"
echo "📊 View results at: https://wandb.ai"
echo ""
