"""
Training script for LaneATT lane detection model (class project wrapper).

This script serves as a wrapper around the external LaneATT implementation.
It handles configuration loading, path setup, and delegates actual training
to the LaneATT submodule's training function.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml

# ============================================================================
# PATH SETUP: Ensure project root and LaneATT submodule are importable
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LANEATT_ROOT = ROOT / "external" / "LaneATT"

if not LANEATT_ROOT.exists():
    raise RuntimeError(
        f"LaneATT submodule not found at {LANEATT_ROOT}. "
        "Did you run: git submodule update --init --recursive ?"
    )

if str(LANEATT_ROOT) not in sys.path:
    sys.path.insert(0, str(LANEATT_ROOT))


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(path: Path) -> dict:
    """Load a YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


# ============================================================================
# MAIN TRAINING ORCHESTRATION
# ============================================================================

def main() -> None:
    """Main entry point for the training script."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Class project runner for LaneATT"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/tusimple_full.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (modifies LaneATT config temporarily)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging (disabled by default)"
    )
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_config(cfg_path)
    print(f"[INFO] Loaded config from {cfg_path}")

    # Extract experiment name
    exp_name = cfg.get("log", {}).get("out_dir", "default_exp").split("/")[-1]
    print(f"[INFO] Starting experiment: {exp_name}")

    # Get LaneATT config name
    laneatt_cfg_name = cfg.get("laneatt", {}).get("config", "laneatt_tusimple_split_resnet18.yml")
    laneatt_cfg = LANEATT_ROOT / "cfgs" / laneatt_cfg_name

    if not laneatt_cfg.exists():
        print(f"[ERROR] LaneATT config not found: {laneatt_cfg}")
        print("[INFO] Available configs:")
        cfgs_dir = LANEATT_ROOT / "cfgs"
        for cfg_file in cfgs_dir.glob("*.yml"):
            print(f"  - {cfg_file.name}")
        return

    # Handle epochs override by modifying config in-place
    if args.epochs is not None:
        print(f"[INFO] Overriding epochs to: {args.epochs}")
        # Read the config as text and replace epochs value
        with laneatt_cfg.open('r') as f:
            config_text = f.read()
        
        # Replace epochs line
        import re
        config_text = re.sub(r'^epochs:\s*\d+', f'epochs: {args.epochs}', config_text, flags=re.MULTILINE)
        
        # Save to temporary config
        temp_cfg = LANEATT_ROOT / "cfgs" / f"temp_{exp_name}.yml"
        with temp_cfg.open('w') as f:
            f.write(config_text)
        laneatt_cfg = temp_cfg
        print(f"[INFO] Created temporary config: {temp_cfg}")
    else:
        # Try to read epochs from config for display
        try:
            with laneatt_cfg.open('r') as f:
                for line in f:
                    if line.startswith('epochs:'):
                        epochs = line.split(':')[1].strip()
                        print(f"[INFO] Using epochs from config: {epochs}")
                        break
        except:
            print(f"[INFO] Using default epochs from config")

    # Disable WandB unless explicitly enabled
    if not args.wandb:
        os.environ['WANDB_MODE'] = 'disabled'
        print(f"[INFO] WandB logging disabled (use --wandb to enable)")
    else:
        print(f"[INFO] WandB logging enabled")

    # Build command
    import subprocess
    
    laneatt_main = LANEATT_ROOT / "main.py"
    cmd = [
        sys.executable,
        str(laneatt_main),
        "train",
        "--exp_name", exp_name,
        "--cfg", str(laneatt_cfg),
    ]

    print(f"[INFO] Running command: {' '.join(cmd)}")
    print(f"[INFO] LaneATT config: {laneatt_cfg}")
    print(f"[INFO] Working directory: {LANEATT_ROOT}")
    print("-" * 80)

    # Run training
    result = subprocess.run(cmd, cwd=str(LANEATT_ROOT), env=os.environ.copy())

    # Clean up temporary config if created
    if args.epochs is not None:
        temp_cfg = LANEATT_ROOT / "cfgs" / f"temp_{exp_name}.yml"
        if temp_cfg.exists():
            temp_cfg.unlink()
            print(f"[INFO] Cleaned up temporary config")

    if result.returncode != 0:
        raise RuntimeError(f"LaneATT training failed with code {result.returncode}")

    print("-" * 80)
    print("[INFO] Training completed successfully!")


if __name__ == "__main__":
    main()