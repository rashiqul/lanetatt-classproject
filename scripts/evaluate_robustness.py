import os
import shutil
import subprocess
import yaml
import sys

# Paths
PROJECT_ROOT = "/home/manan/lanetatt-classproject"
LANEATT_ROOT = os.path.join(PROJECT_ROOT, "external/LaneATT")
TUSIMPLE_ROOT = os.path.join(LANEATT_ROOT, "datasets/tusimple")
AUGMENTED_ROOT = os.path.join(LANEATT_ROOT, "datasets/TuSimple_augmented/test_robust")
BASE_CONFIG = os.path.join(LANEATT_ROOT, "cfgs/laneatt_tusimple_resnet18.yml")

AUGMENTATIONS = ["blur", "brightness", "noise"]

def find_latest_checkpoint(models_dir):
    import re
    if not os.path.isdir(models_dir):
        return None, -1
    pattern = re.compile(r'model_(\d+).pt')
    last_epoch = -1
    last_ckpt = None
    for f in os.listdir(models_dir):
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            if epoch > last_epoch:
                last_epoch = epoch
                last_ckpt = os.path.join(models_dir, f)
    return last_ckpt, last_epoch

def main():
    # Source experiment (where the trained model is)
    SOURCE_EXP = "tusimple_exp"
    SOURCE_EXP_DIR = os.path.join(LANEATT_ROOT, "experiments", SOURCE_EXP)
    SOURCE_MODELS_DIR = os.path.join(SOURCE_EXP_DIR, "models")
    
    # Check for trained model
    ckpt_path, epoch = find_latest_checkpoint(SOURCE_MODELS_DIR)
    if not ckpt_path:
        print(f"Error: No trained model found in {SOURCE_MODELS_DIR}. Please train the model first or download weights.")
        return
    print(f"Using checkpoint: {ckpt_path} (Epoch {epoch})")

    # Ensure raw test labels exist
    original_label_file = os.path.join(TUSIMPLE_ROOT, "test_label.json")
    if not os.path.exists(original_label_file):
        print(f"Error: {original_label_file} not found.")
        return

    for aug in AUGMENTATIONS:
        print(f"\n========================================")
        print(f"Running evaluation for augmentation: {aug}")
        print(f"========================================")
        
        aug_dir = os.path.join(AUGMENTED_ROOT, aug)
        if not os.path.isdir(aug_dir):
            print(f"Warning: {aug_dir} does not exist. Skipping.")
            continue
            
        # 1. Copy test_label.json
        dest_label_file = os.path.join(aug_dir, "test_label.json")
        if not os.path.exists(dest_label_file):
            print(f"Copying labels to {dest_label_file}")
            shutil.copy(original_label_file, dest_label_file)

        # 1.5 Fix directory structure (move 0530, 0531, 0601 into clips/ if needed)
        # Check if 'clips' exists
        clips_dir = os.path.join(aug_dir, "clips")
        if not os.path.exists(clips_dir):
            print(f"Fixing directory structure for {aug}: creating clips/ directory")
            os.makedirs(clips_dir)
            # Move all directories that start with distinct numbers (like 0531, 0601)
            for item in os.listdir(aug_dir):
                item_path = os.path.join(aug_dir, item)
                if os.path.isdir(item_path) and item != "clips" and item[0].isdigit():
                    shutil.move(item_path, os.path.join(clips_dir, item))
            
        # 2. Setup isolated experiment directory
        exp_name = f"robustness_{aug}"
        exp_dir = os.path.join(LANEATT_ROOT, "experiments", exp_name)
        new_models_dir = os.path.join(exp_dir, "models")
        os.makedirs(new_models_dir, exist_ok=True)
        
        # Copy checkpoint if not present
        dest_ckpt = os.path.join(new_models_dir, os.path.basename(ckpt_path))
        if not os.path.exists(dest_ckpt):
            print(f"Copying checkpoint to {dest_ckpt}")
            shutil.copy(ckpt_path, dest_ckpt)
        
        # 3. Create temporary config
        with open(BASE_CONFIG, 'r') as f:
            # use unsafe_load to handle !!python/tuple
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            
        # Update root for test dataset so it points to the augmented version
        # We use relative path from LANEATT_ROOT
        rel_aug_path = f"datasets/TuSimple_augmented/test_robust/{aug}"
        config['datasets']['test']['parameters']['root'] = rel_aug_path
        
        # Point to the specific config for this run
        temp_config_path = os.path.join(exp_dir, "config.yml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
            
        # 4. Run evaluation
        # Use sys.executable to ensure we use the current virtualenv's python
        # and avoid "poetry run" issues in subdirectory without pyproject.toml
        cmd = [
            sys.executable, "main.py", "test",
            "--exp_name", exp_name,
            "--cfg", temp_config_path,
            "--epoch", str(epoch) # Explicitly use the epoch we copied
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=LANEATT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed for {aug}: {e}")
        
    print("\nRobustness evaluation script finished.")

if __name__ == "__main__":
    main()
