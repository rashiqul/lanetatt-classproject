import os
import cv2
import albumentations as A
from tqdm import tqdm
import shutil

# Paths
PROJECT_ROOT = "/home/manan/lanetatt-classproject"
CULANE_ROOT = os.path.join(PROJECT_ROOT, "datasets/CULane")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "datasets/CULane_augmented")
TEST_LIST = os.path.join(CULANE_ROOT, "list/test.txt")

# Limit for robustness subset (to save time/space)
LIMIT = 500 

def get_augmentations():
    return {
        "blur": A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.6),
                A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            ], p=1.0)
        ]),
        "brightness": A.Compose([
             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=1.0)
        ]),
        "noise": A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=1.0)
        ]),
        "shadow": A.Compose([
             A.RandomShadow(shadow_roi=(0, 0.4, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=1.0)
        ]),
        "lowlight": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.3), contrast_limit=0.2, p=1.0),
            A.GaussNoise(var_limit=(20.0, 60.0), mean=0, per_channel=True, p=1.0)
        ]),
    }

def main():
    if not os.path.exists(TEST_LIST):
        print(f"Error: {TEST_LIST} not found")
        return

    print(f"Reading {TEST_LIST}...")
    with open(TEST_LIST, 'r') as f:
        # Filter lines to remove empty ones
        all_lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Take subset
    if LIMIT and len(all_lines) > LIMIT:
        print(f"Limiting to first {LIMIT} images out of {len(all_lines)}")
        lines = all_lines[:LIMIT]
    else:
        lines = all_lines

    transformers = get_augmentations()
    
    for aug_name, transform in transformers.items():
        print(f"\nGeneratng augmentation: {aug_name}")
        aug_root = os.path.join(OUTPUT_ROOT, aug_name)
        os.makedirs(aug_root, exist_ok=True)
        
        # 1. Create the subset list file in the augmented dir
        #    This is crucial for the dataset loader to know which files to load.
        #    Since we are creating a self-contained dataset root, we need 'list/test.txt'.
        list_dir = os.path.join(aug_root, "list")
        os.makedirs(list_dir, exist_ok=True)
        new_list_path = os.path.join(list_dir, "test.txt")
        
        with open(new_list_path, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        
        # 2. Process images and symlink labels
        for line in tqdm(lines):
            rel_path = line
            if rel_path.startswith('/'):
                rel_path = rel_path[1:]
            
            # Source Image
            src_img_path = os.path.join(CULANE_ROOT, rel_path)
            
            # Destination Image
            dst_img_path = os.path.join(aug_root, rel_path)
            
            # Source Label (.lines.txt)
            # CULane labels are usually colocated with images, with .lines.txt extension replacing .jpg
            src_label_path = os.path.splitext(src_img_path)[0] + ".lines.txt"
            dst_label_path = os.path.splitext(dst_img_path)[0] + ".lines.txt"
            
            if not os.path.exists(src_img_path):
                # print(f"Warning: Image not found {src_img_path}")
                continue
                
            # Augment Image
            img = cv2.imread(src_img_path)
            if img is None:
                continue
            
            try:
                aug_img = transform(image=img)["image"]
            except Exception as e:
                print(f"Error augmenting {src_img_path}: {e}")
                continue
            
            # Save Image
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            cv2.imwrite(dst_img_path, aug_img)
            
            # Symlink Label (if it exists)
            # We copy or symlink the label so the evaluator can find ground truth in the augmented root
            if os.path.exists(src_label_path):
                if os.path.exists(dst_label_path):
                    os.remove(dst_label_path)
                os.symlink(src_label_path, dst_label_path)

if __name__ == "__main__":
    main()
