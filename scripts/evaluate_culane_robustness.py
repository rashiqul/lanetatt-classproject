import os
import shutil
import subprocess
import yaml
import sys

# Paths
PROJECT_ROOT = "/home/manan/lanetatt-classproject"
LANEATT_ROOT = os.path.join(PROJECT_ROOT, "external/LaneATT")
CULANE_AUG_ROOT = os.path.join(PROJECT_ROOT, "datasets/CULane_augmented")
# CULane model
MODEL_PATH = os.path.join(LANEATT_ROOT, "data/culane_res18.pt") 
BASE_CONFIG = os.path.join(LANEATT_ROOT, "cfgs/laneatt_culane_resnet18.yml")

AUGMENTATIONS = ["blur", "brightness", "lowlight", "noise", "shadow"]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    print(f"Using model: {MODEL_PATH}")
    print("Adapting TuSimple model for CULane evaluation...")

    with open(BASE_CONFIG, 'r') as f:
        # unsafe_load for python objects in yaml
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    for aug in AUGMENTATIONS:
        print(f"\n========================================")
        print(f"Evaluation CULane Robustness: {aug}")
        print(f"========================================")

        # Clear CULane cache to force reload of image paths
        # The cache is typically in the current working directory (LANEATT_ROOT) under 'cache/'
        cache_file = os.path.join(LANEATT_ROOT, "cache", "culane_test.json")
        if os.path.exists(cache_file):
            print(f"Removing cache file: {cache_file}")
            os.remove(cache_file)
            
        aug_dir = os.path.join(CULANE_AUG_ROOT, aug)
        if not os.path.exists(aug_dir):
            print(f"Skipping {aug} (not found)")
            continue
            
        exp_name = f"culane_robustness_{aug}"
        exp_dir = os.path.join(LANEATT_ROOT, "experiments", exp_name)
        new_models_dir = os.path.join(exp_dir, "models")
        os.makedirs(new_models_dir, exist_ok=True)
        
        # Copy model so main.py finds it
        dest_ckpt = os.path.join(new_models_dir, "model_0001.pt")
        if not os.path.exists(dest_ckpt):
            shutil.copy(MODEL_PATH, dest_ckpt)
            
        # Modify Config for CULane
        # We keep the model parameters (anchors) from TuSimple config
        # But we change the dataset loader to use CULane
        
        # The CULane loader parameters need to be set
        # config['datasets']['test']['parameters']['dataset'] = 'culane' # Already set in base config
        # config['datasets']['test']['parameters']['split'] = 'test' # Already 'test'
        config['datasets']['test']['parameters']['root'] = aug_dir
        
        # CULane loader might need specific img_size if it's different in defaults
        # But LaneDataset wrapper usually handles resizing to model input size
        
        # Save temp config
        temp_config_path = os.path.join(exp_dir, "config.yml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Run Test
        cmd = [
            sys.executable, "main.py", "test",
            "--exp_name", exp_name,
            "--cfg", temp_config_path,
            "--epoch", "1"
        ]
        try:
            subprocess.run(cmd, cwd=LANEATT_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed {aug}: {e}")

if __name__ == "__main__":
    main()
