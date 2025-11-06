"""
Training script for LaneATT lane detection model (class project wrapper).

This script serves as a wrapper around the external LaneATT implementation.
It handles configuration loading, path setup, and delegates actual training
to the LaneATT submodule's training function.
"""

import argparse
import sys
from pathlib import Path

import yaml

# ============================================================================
# PATH SETUP: Ensure project root and LaneATT submodule are importable
# ============================================================================

# Resolve the project root directory (parent of parent of this file)
# __file__ = /path/to/laneatt-project/src/train.py
# parents[0] = /path/to/laneatt-project/src
# parents[1] = /path/to/laneatt-project (ROOT)
ROOT = Path(__file__).resolve().parents[1]

# Add project root to sys.path if it's not already there
# This allows us to import modules from external/LaneATT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Construct the path to the LaneATT submodule
LANEATT_ROOT = ROOT / "external" / "LaneATT"

# Verify the LaneATT submodule exists; raise error with helpful message if not
if not LANEATT_ROOT.exists():
    raise RuntimeError(
        f"LaneATT submodule not found at {LANEATT_ROOT}. "
        "Did you run: git submodule update --init --recursive ?"
    )

# Add LaneATT root to sys.path so we can import its lib module
if str(LANEATT_ROOT) not in sys.path:
    sys.path.insert(0, str(LANEATT_ROOT))


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(path: Path) -> dict:
    """
    Load a YAML configuration file and return it as a Python dictionary.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Parsed configuration dictionary with training parameters,
        model settings, dataset paths, etc.

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML syntax is invalid
    """
    # Open the file in read mode
    with path.open("r") as f:
        # Parse YAML safely (safe_load prevents arbitrary code execution)
        return yaml.safe_load(f)


# ============================================================================
# MAIN TRAINING ORCHESTRATION
# ============================================================================

def main() -> None:
    """
    Main entry point for the training script.

    This function:
    1. Parses command-line arguments to get the config file path
    2. Loads the configuration YAML
    3. Converts our config format to LaneATT's expected format
    4. Delegates training to the LaneATT Runner

    Raises:
        FileNotFoundError: If the specified config file doesn't exist
        RuntimeError: If the LaneATT modules cannot be imported
    """
    # ========================================================================
    # STEP 1: Parse command-line arguments
    # ========================================================================

    # Create argument parser with a description
    parser = argparse.ArgumentParser(
        description="Class project runner for LaneATT"
    )

    # Add required --config argument for specifying the YAML config file
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/tusimple_debug.yaml)",
    )

    # Parse the arguments from sys.argv
    args = parser.parse_args()

    # ========================================================================
    # STEP 2: Load and validate configuration
    # ========================================================================

    # Convert the config argument to a Path object
    cfg_path = Path(args.config)

    # Check if the config file exists; raise error if not
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load the YAML configuration into a dictionary
    cfg = load_config(cfg_path)

    # Print confirmation and config contents for debugging
    print(f"[INFO] Loaded config from {cfg_path}")
    print(f"[INFO] Config content: {cfg}")

    # ========================================================================
    # STEP 3: Import LaneATT modules
    # ========================================================================
    # The LaneATT repo structure:
    # - external/LaneATT/main.py is the entry point
    # - external/LaneATT/lib/ contains Config, Runner, Experiment classes
    # We'll import these and use them directly

    try:
        # Import LaneATT's Config, Runner, and Experiment classes
        from lib.config import Config
        from lib.runner import Runner
        from lib.experiment import Experiment
        import torch
    except ImportError as e:
        raise RuntimeError(
            f"Could not import LaneATT modules. Error: {e}\n"
            f"Make sure the submodule is initialized and "
            f"LANEATT_ROOT={LANEATT_ROOT} is in sys.path"
        )

    # ========================================================================
    # STEP 4: Create experiment directory and config adapter
    # ========================================================================
    # LaneATT expects experiments to have a name and uses its own Config class
    # We'll create a simple experiment name from our config

    # Extract experiment name from output directory
    # e.g., "experiments/runs/tusimple_debug" -> "tusimple_debug"
    exp_name = cfg.get("log", {}).get("out_dir", "default_exp").split("/")[-1]

    print(f"[INFO] Starting experiment: {exp_name}")

    # ========================================================================
    # STEP 5: Call LaneATT main.py as subprocess with 1k config
    # ========================================================================
    # Since LaneATT's main.py expects CLI arguments, we call it as a subprocess

    import subprocess

    # Path to LaneATT's main.py
    laneatt_main = LANEATT_ROOT / "main.py"

    # Use the new 1k subset config
    laneatt_cfg = LANEATT_ROOT / "cfgs" / "laneatt_tusimple_1k.yml"

    if not laneatt_cfg.exists():
        print(f"[ERROR] LaneATT config not found: {laneatt_cfg}")
        print("[INFO] Available configs:")
        cfgs_dir = LANEATT_ROOT / "cfgs"
        for cfg_file in cfgs_dir.glob("*.yml"):
            print(f"  - {cfg_file.name}")
        return

    # Build the command to run LaneATT's main.py
    cmd = [
        sys.executable,  # Python interpreter
        str(laneatt_main),
        "train",  # mode
        "--exp_name", exp_name,
        "--cfg", str(laneatt_cfg),
    ]

    print(f"[INFO] Running command: {' '.join(cmd)}")

    # Run the LaneATT training
    result = subprocess.run(cmd, cwd=str(LANEATT_ROOT))

    # Check if training succeeded
    if result.returncode != 0:
        raise RuntimeError(f"LaneATT training failed with code {result.returncode}")

    print("[INFO] Training completed successfully!")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

# This block ensures main() only runs when the script is executed directly,
# not when imported as a module
if __name__ == "__main__":
    main()