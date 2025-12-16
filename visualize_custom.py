#!/usr/bin/env python3
"""
Visualize LaneATT predictions on custom images without requiring dataset downloads.
This script allows you to test a trained model on your own road images.

Usage:
    python visualize_custom.py --model experiments/my_exp/models/model_100.pt \
                               --cfg cfgs/laneatt_tusimple_resnet18.yml \
                               --images img1.jpg img2.jpg \
                               --output results/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add LaneATT lib to path
LANEATT_DIR = Path(__file__).parent / "external" / "LaneATT"
sys.path.insert(0, str(LANEATT_DIR))

from lib.config import Config
from lib.lane import Lane

# Color scheme matching paper
GT_COLOR = (255, 0, 0)  # Blue for ground truth (not used here)
PRED_HIT_COLOR = (0, 255, 0)  # Green for predictions
PRED_MISS_COLOR = (0, 0, 255)  # Red (not used without ground truth)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LaneATT inference on custom images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image
    python visualize_custom.py --model experiments/my_exp/models/model_100.pt \\
                               --cfg external/LaneATT/cfgs/laneatt_tusimple_resnet18.yml \\
                               --images my_road_image.jpg

    # Multiple images
    python visualize_custom.py --model experiments/my_exp/models/model_100.pt \\
                               --cfg external/LaneATT/cfgs/laneatt_tusimple_resnet18.yml \\
                               --images img1.jpg img2.jpg img3.jpg \\
                               --output results/

    # All images in directory
    python visualize_custom.py --model experiments/my_exp/models/model_100.pt \\
                               --cfg external/LaneATT/cfgs/laneatt_tusimple_resnet18.yml \\
                               --images my_images/*.jpg \\
                               --output results/ \\
                               --no-display
        """
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--cfg", required=True, help="Path to config file (.yml)")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--output", help="Output directory for results (optional)")
    parser.add_argument("--display", dest="display", action="store_true", help="Display images (default)")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Don't display images")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--thickness", type=int, default=3, help="Line thickness (default: 3)")
    parser.set_defaults(display=True)

    return parser.parse_args()


def preprocess_image(img_path, target_size):
    """Load and preprocess image for model input."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    # Resize to model input size
    img_resized = cv2.resize(img, target_size)
    
    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
    
    return img_tensor.unsqueeze(0), original_img, (h, w)


def draw_lanes_on_image(img, lanes, target_size, original_size, thickness=3, conf_threshold=0.5):
    """Draw predicted lanes on image."""
    # Resize image to target size for consistent visualization
    img_resized = cv2.resize(img, target_size)
    h, w = target_size[1], target_size[0]
    
    for lane in lanes:
        # Filter by confidence
        if hasattr(lane, 'metadata') and 'conf' in lane.metadata:
            if lane.metadata['conf'] < conf_threshold:
                continue
        
        # Get lane points
        points = lane.points
        points[:, 0] *= w  # Scale x coordinates
        points[:, 1] *= h  # Scale y coordinates
        points = points.round().astype(int)
        
        # Draw lane
        for curr_p, next_p in zip(points[:-1], points[1:]):
            cv2.line(img_resized,
                    tuple(curr_p),
                    tuple(next_p),
                    color=PRED_HIT_COLOR,
                    thickness=thickness)
        
        # Optionally draw confidence score
        if hasattr(lane, 'metadata') and 'conf' in lane.metadata:
            mid_idx = len(points) // 2
            if mid_idx < len(points):
                conf_text = f"{lane.metadata['conf']:.2f}"
                cv2.putText(img_resized,
                           conf_text,
                           tuple(points[mid_idx]),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           PRED_HIT_COLOR,
                           1)
    
    return img_resized


def main():
    args = parse_args()
    
    # Convert paths to absolute before changing directory
    original_dir = os.getcwd()
    args.model = str(Path(args.model).resolve())
    args.images = [str(Path(img).resolve()) for img in args.images]
    if args.output:
        args.output = str(Path(args.output).resolve())
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to LaneATT directory so relative paths in config work
    os.chdir(LANEATT_DIR)
    
    # Load config (now relative paths in config will work)
    print(f"Loading config from {args.cfg}...")
    cfg = Config(args.cfg)
    
    # Get model input size
    img_size = cfg['datasets']['train']['parameters']['img_size']  # (height, width)
    target_size = (img_size[1], img_size[0])  # (width, height) for cv2
    
    # Load model
    print(f"Loading model from {args.model}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = cfg.get_model()
    checkpoint = torch.load(args.model, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model'])
        except RuntimeError as e:
            print(f"Warning: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint['model'], strict=False)
    else:
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print(f"Warning: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model = model.to(device)
    
    # Get test parameters
    test_params = cfg.get_test_parameters()
    
    print(f"\nProcessing {len(args.images)} images...")
    print(f"Model input size: {img_size[1]}x{img_size[0]}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Device: {device}\n")
    
    # Process each image
    with torch.no_grad():
        for img_path in tqdm(args.images, desc="Processing images"):
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            try:
                # Preprocess
                img_tensor, original_img, original_size = preprocess_image(img_path, target_size)
                img_tensor = img_tensor.to(device)
                
                # Run inference
                output = model(img_tensor, **test_params)
                predictions = model.decode(output, as_lanes=True)[0]  # Get first batch item
                
                # Filter by confidence
                filtered_preds = [
                    lane for lane in predictions
                    if not (hasattr(lane, 'metadata') and 'conf' in lane.metadata and 
                           lane.metadata['conf'] < args.conf_threshold)
                ]
                
                print(f"{img_path.name}: Detected {len(filtered_preds)}/{len(predictions)} lanes " 
                      f"(conf > {args.conf_threshold})")
                
                # Draw lanes
                result_img = draw_lanes_on_image(
                    original_img,
                    predictions,
                    target_size,
                    original_size,
                    thickness=args.thickness,
                    conf_threshold=args.conf_threshold
                )
                
                # Save if output directory specified
                if args.output:
                    output_path = output_dir / f"result_{img_path.name}"
                    cv2.imwrite(str(output_path), result_img)
                
                # Display if requested
                if args.display:
                    cv2.imshow(f"LaneATT Result - {img_path.name}", result_img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        print("Quit requested by user")
                        break
                    cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"\n✓ Processing complete!")
    if args.output:
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
