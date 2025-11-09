"""
Create a smaller subset of TuSimple dataset for faster experimentation.

This script:
1. Samples N random images from train/test sets
2. Filters the corresponding JSON annotations
3. Maintains the proper directory structure
"""

import json
import random
import shutil
from pathlib import Path
import argparse


def sample_tusimple_train(
    source_dir: Path,
    target_dir: Path,
    num_samples: int = 1000,
    seed: int = 42
):
    """
    Sample from TuSimple training set.
    
    Args:
        source_dir: Path to original tusimple dataset
        target_dir: Path to output subset
        num_samples: Number of images to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create target directories
    target_clips = target_dir / "clips"
    target_clips.mkdir(parents=True, exist_ok=True)
    
    # Read all training annotations
    annotations = []
    json_files = [
        "label_data_0313.json",
        "label_data_0531.json", 
        "label_data_0601.json"
    ]
    
    for json_file in json_files:
        json_path = source_dir / json_file
        if json_path.exists():
            with open(json_path, 'r') as f:
                # TuSimple format: one JSON object per line
                for line in f:
                    annotations.append(json.loads(line.strip()))
    
    print(f"Total training annotations: {len(annotations)}")
    
    # Sample random annotations
    sampled = random.sample(annotations, min(num_samples, len(annotations)))
    print(f"Sampled: {len(sampled)} annotations")
    
    # Copy images and create new annotation files
    copied_count = 0
    output_annotations = {json_file: [] for json_file in json_files}
    
    for anno in sampled:
        # Image path is relative: "clips/0313-1/1.jpg"
        img_rel_path = anno['raw_file']
        src_img = source_dir / img_rel_path
        dst_img = target_dir / img_rel_path
        
        if src_img.exists():
            # Create parent directory
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            # Copy image
            shutil.copy2(src_img, dst_img)
            copied_count += 1
            
            # Determine which JSON file this annotation belongs to
            # Based on the date in the path
            if "0313" in img_rel_path:
                output_annotations["label_data_0313.json"].append(anno)
            elif "0531" in img_rel_path:
                output_annotations["label_data_0531.json"].append(anno)
            elif "0601" in img_rel_path:
                output_annotations["label_data_0601.json"].append(anno)
    
    # Write filtered annotations
    for json_file, annos in output_annotations.items():
        if annos:
            output_path = target_dir / json_file
            with open(output_path, 'w') as f:
                for anno in annos:
                    f.write(json.dumps(anno) + '\n')
            print(f"Wrote {len(annos)} annotations to {json_file}")
    
    print(f"Copied {copied_count} images to {target_dir}")


def sample_tusimple_test(
    source_dir: Path,
    target_dir: Path,
    num_samples: int = 1000,
    seed: int = 42
):
    """
    Sample from TuSimple test set.
    
    Args:
        source_dir: Path to original tusimple-test dataset
        target_dir: Path to output subset
        num_samples: Number of images to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create target directories
    target_clips = target_dir / "clips"
    target_clips.mkdir(parents=True, exist_ok=True)
    
    # Read test tasks (list of images)
    tasks_file = source_dir / "test_tasks_0627.json"
    if not tasks_file.exists():
        print(f"Warning: {tasks_file} not found")
        return
    
    with open(tasks_file, 'r') as f:
        # One JSON object per line
        tasks = [json.loads(line.strip()) for line in f]
    
    print(f"Total test tasks: {len(tasks)}")
    
    # Sample random tasks
    sampled = random.sample(tasks, min(num_samples, len(tasks)))
    print(f"Sampled: {len(sampled)} test images")
    
    # Copy images
    copied_count = 0
    for task in sampled:
        img_rel_path = task['raw_file']
        src_img = source_dir / img_rel_path
        dst_img = target_dir / img_rel_path
        
        if src_img.exists():
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)
            copied_count += 1
    
    # Write filtered test tasks
    output_tasks = target_dir / "test_tasks_0627.json"
    with open(output_tasks, 'w') as f:
        for task in sampled:
            f.write(json.dumps(task) + '\n')
    
    # Copy test labels if they exist
    test_label = source_dir / "test_label.json"
    if test_label.exists():
        # Filter test labels to match sampled images
        with open(test_label, 'r') as f:
            all_labels = [json.loads(line.strip()) for line in f]
        
        sampled_paths = {task['raw_file'] for task in sampled}
        filtered_labels = [
            label for label in all_labels 
            if label['raw_file'] in sampled_paths
        ]
        
        output_labels = target_dir / "test_label.json"
        with open(output_labels, 'w') as f:
            for label in filtered_labels:
                f.write(json.dumps(label) + '\n')
        print(f"Wrote {len(filtered_labels)} test labels")
    
    print(f"Copied {copied_count} test images to {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create smaller TuSimple dataset subset"
    )
    parser.add_argument(
        "--source-train",
        type=Path,
        default=Path("datasets/tusimple"),
        help="Path to original training set"
    )
    parser.add_argument(
        "--source-test",
        type=Path,
        default=Path("datasets/tusimple-test"),
        help="Path to original test set"
    )
    parser.add_argument(
        "--target-train",
        type=Path,
        default=Path("datasets/tusimple-1k"),
        help="Path to output training subset"
    )
    parser.add_argument(
        "--target-test",
        type=Path,
        default=Path("datasets/tusimple-test-1k"),
        help="Path to output test subset"
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=1000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Creating TuSimple dataset subset")
    print("=" * 60)
    
    # Sample training set
    if args.source_train.exists():
        print(f"\nProcessing training set from {args.source_train}")
        sample_tusimple_train(
            args.source_train,
            args.target_train,
            args.num_train,
            args.seed
        )
    else:
        print(f"Warning: Training set not found at {args.source_train}")
    
    # Sample test set
    if args.source_test.exists():
        print(f"\nProcessing test set from {args.source_test}")
        sample_tusimple_test(
            args.source_test,
            args.target_test,
            args.num_test,
            args.seed
        )
    else:
        print(f"Warning: Test set not found at {args.source_test}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Training subset: {args.target_train}")
    print(f"Test subset: {args.target_test}")


if __name__ == "__main__":
    main()