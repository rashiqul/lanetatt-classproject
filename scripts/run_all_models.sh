#!/bin/bash
#
# Batch Training Script for All ResNet Models
# Runs all four experiments sequentially overnight
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/rashiqul/workspace/laneatt-project"
LANEATT_DIR="$PROJECT_DIR/external/LaneATT"
LOG_DIR="$PROJECT_DIR/training_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/batch_training.log"
}

# Function to run training
run_training() {
    local exp_name=$1
    local config_file=$2
    local log_file="$LOG_DIR/${exp_name}_$(date '+%Y%m%d_%H%M%S').log"
    
    log "=========================================="
    log "Starting: $exp_name"
    log "Config: $config_file"
    log "Log file: $log_file"
    log "=========================================="
    
    cd "$LANEATT_DIR"
    
    # Activate poetry environment and run training (must be in LaneATT dir for data paths)
    if cd "$PROJECT_DIR" && poetry run bash -c "cd external/LaneATT && python main.py train --exp_name '$exp_name' --cfg 'cfgs/$config_file'" \
        2>&1 | tee "$log_file"; then
        log "✓ SUCCESS: $exp_name completed"
    else
        log "✗ FAILED: $exp_name failed with exit code $?"
    fi
    
    log ""
}

# Main execution
log "================================================"
log "BATCH TRAINING START"
log "Project: LaneATT - TuSimple Custom Split"
log "Models: ResNet-18, ResNet-34, ResNet-122"
log "================================================"
log ""

# Run all models sequentially
run_training "tusimple_resnet18" "laneatt_tusimple_resnet18.yml"
run_training "tusimple_resnet34" "laneatt_tusimple_resnet34.yml"
run_training "tusimple_resnet122" "laneatt_tusimple_resnet122.yml"

log "================================================"
log "BATCH TRAINING COMPLETE"
log "All models finished training"
log "================================================"

# Summary
log ""
log "Training Summary:"
log "  • ResNet-18: Check $PROJECT_DIR/external/LaneATT/experiments/tusimple_resnet18/"
log "  • ResNet-34: Check $PROJECT_DIR/external/LaneATT/experiments/tusimple_resnet34/"
log "  • ResNet-122: Check $PROJECT_DIR/external/LaneATT/experiments/tusimple_resnet122/"
log ""
log "WandB Dashboard: https://wandb.ai/<your-username>/LaneATT-TuSimple"
