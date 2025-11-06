#!/bin/bash
# ===============================================================
# LaneATT Training Script (safe to run even after SSH disconnect)
# Author: bozhen2 setup helper
# ===============================================================

SESSION_NAME="laneatt_train"
EXP_NAME="tusimple_fast"
CFG_PATH="cfgs/laneatt_tusimple_split_resnet18.yml"

# Go to project root
cd ~/ComputerVision/lanetatt-classproject || exit

echo "🚀 Starting LaneATT training inside tmux session: $SESSION_NAME"

# If tmux session already exists, attach to it
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚙️  Session already exists, attaching..."
    tmux attach -t $SESSION_NAME
else
    # Otherwise create a new one and start training
    tmux new-session -d -s $SESSION_NAME "
        echo '✅ Environment: activating Poetry virtualenv...';
        source /home/bozhen2/.cache/pypoetry/virtualenvs/laneatt-classproj-TAaFv3Zi-py3.10/bin/activate;
        cd external/LaneATT;
        echo '🏋️  Starting training for experiment: $EXP_NAME';
        python main.py train --exp_name $EXP_NAME --cfg $CFG_PATH;
        echo '✅ Training completed! Press Ctrl+b d to detach if attached.';
        exec bash
    "
    echo "✅ Training started in background."
    echo "👉 Use 'tmux attach -t $SESSION_NAME' to view progress."
fi
