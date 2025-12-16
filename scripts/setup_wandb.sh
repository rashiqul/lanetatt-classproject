#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# WandB Setup Script
# =============================================================================
# This script helps you set up Weights & Biases for experiment tracking
# =============================================================================

echo "🔧 WandB Setup for LaneATT Experiments"
echo "======================================="
echo ""

# Check if Poetry environment exists
if ! poetry env info &>/dev/null; then
    echo "❌ Poetry environment not found. Please run 'poetry install' first."
    exit 1
fi

echo "✅ Poetry environment found"
echo ""

# Check if WandB is installed
if ! poetry run python -c "import wandb" 2>/dev/null; then
    echo "❌ WandB not installed. Installing..."
    poetry add wandb
else
    echo "✅ WandB is installed (version: $(poetry run python -c 'import wandb; print(wandb.__version__)'))"
fi

echo ""
echo "📝 WandB Login"
echo "-------------"
echo "You'll need a WandB account and API key."
echo "1. If you don't have an account, sign up at https://wandb.ai"
echo "2. Get your API key from https://wandb.ai/authorize"
echo ""

# Try to login
if poetry run python -c "import wandb; wandb.login()" 2>/dev/null; then
    echo ""
    echo "✅ Successfully logged in to WandB!"
else
    echo ""
    echo "⚠️  Login incomplete. You can try again with:"
    echo "   poetry run python -c 'import wandb; wandb.login()'"
    exit 1
fi

echo ""
echo "🎯 Next Steps"
echo "============="
echo ""
echo "1. Run a quick test:"
echo "   poetry run python scripts/train.py --config configs/tusimple_debug.yaml --wandb"
echo ""
echo "2. Run full training:"
echo "   poetry run python scripts/train.py --config configs/tusimple_full.yaml --wandb"
echo ""
echo "3. Run multiple experiments:"
echo "   ./scripts/run_wandb_experiments.sh all"
echo ""
echo "4. View your experiments:"
echo "   https://wandb.ai"
echo ""
echo "📖 For more details, see: docs/wandb-experiments.md"
echo ""
