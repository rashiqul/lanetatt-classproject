#!/usr/bin/env python3
"""
LaneATT Training Entry Point
Runs from project workspace root instead of submodule directory
"""
import sys
import os
import argparse

# Get absolute paths
workspace_root = os.path.dirname(os.path.abspath(__file__))
laneatt_path = os.path.join(workspace_root, 'external', 'LaneATT')

# Add external/LaneATT to Python path
sys.path.insert(0, laneatt_path)

# Parse args to convert relative config paths to absolute
parser = argparse.ArgumentParser(description="Train lane detector")
parser.add_argument("mode", choices=["train", "test"], help="Train or test?")
parser.add_argument("--exp_name", help="Experiment name", required=True)
parser.add_argument("--cfg", help="Config file")
parser.add_argument("--resume", action="store_true", help="Resume training")
parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
parser.add_argument("--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU")
parser.add_argument("--save_predictions", action="store_true", help="Save predictions to pickle file")
parser.add_argument("--view", choices=["all", "mistakes"], help="Show predictions")
parser.add_argument("--deterministic", action="store_true", help="set cudnn.deterministic = True and cudnn.benchmark = False")
args = parser.parse_args()

# Convert config path if it's relative to workspace
if args.cfg and not os.path.isabs(args.cfg):
    # If path starts with external/LaneATT, it's relative to workspace
    if args.cfg.startswith('external/LaneATT/'):
        args.cfg = os.path.join(workspace_root, args.cfg)
    # Otherwise assume it's relative to LaneATT directory
    elif not os.path.exists(args.cfg):
        potential_path = os.path.join(laneatt_path, args.cfg)
        if os.path.exists(potential_path):
            args.cfg = potential_path

# Change working directory to LaneATT for relative path compatibility
os.chdir(laneatt_path)

# Restore arguments for main()
sys.argv = [sys.argv[0], args.mode]
sys.argv.extend(['--exp_name', args.exp_name])
if args.cfg:
    sys.argv.extend(['--cfg', args.cfg])
if args.resume:
    sys.argv.append('--resume')
if args.epoch is not None:
    sys.argv.extend(['--epoch', str(args.epoch)])
if args.cpu:
    sys.argv.append('--cpu')
if args.save_predictions:
    sys.argv.append('--save_predictions')
if args.view:
    sys.argv.extend(['--view', args.view])
if args.deterministic:
    sys.argv.append('--deterministic')

# Import and run main
from main import main

if __name__ == '__main__':
    main()
