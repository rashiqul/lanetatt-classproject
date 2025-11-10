#!/bin/bash
#
# Start Overnight Training in tmux Session
# Run this script to start all model training in the background
#

PROJECT_DIR="/home/rashiqul/workspace/laneatt-project"

echo "================================================"
echo "Starting Overnight Training Session"
echo "================================================"
echo ""
echo "This will:"
echo "  1. Create a tmux session called 'laneatt-training'"
echo "  2. Run all three ResNet models sequentially"
echo "  3. Keep running even if you disconnect"
echo ""
echo "Models to train:"
echo "  • ResNet-18  (~1.5 hours)"
echo "  • ResNet-34  (~1.5 hours)"  
echo "  • ResNet-122 (~10 hours)"
echo "  Total: ~13 hours"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t laneatt-training"
echo ""
echo "To detach (leave running):"
echo "  Press: Ctrl+B, then D"
echo ""
read -p "Press ENTER to start training..."

# Kill existing session if it exists
tmux kill-session -t laneatt-training 2>/dev/null

# Create new tmux session and run training
tmux new-session -d -s laneatt-training "cd $PROJECT_DIR && ./run_all_models.sh; echo ''; echo 'Training complete! Press ENTER to exit...'; read"

echo ""
echo "✓ Training started in tmux session 'laneatt-training'"
echo ""
echo "To view progress:"
echo "  tmux attach -t laneatt-training"
echo ""
echo "To check if still running:"
echo "  tmux ls"
echo ""
echo "Logs will be saved to: $PROJECT_DIR/training_logs/"
echo "================================================"
