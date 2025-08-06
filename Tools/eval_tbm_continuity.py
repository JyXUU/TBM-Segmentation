import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import argparse
import os

def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0

def save_skeleton_image(skeleton, base_path, label):
    """Save the skeleton as a PNG overlay."""
    overlay = (skeleton * 255).astype(np.uint8)
    save_path = f"{base_path}_{label}_skeleton.png"
    Image.fromarray(overlay).save(save_path)
    print(f"Saved {label} skeleton image to: {save_path}")

def evaluate_continuity(pred_mask_path, gt_mask_path):
    # Load binary masks
    pred_mask = load_mask(pred_mask_path)
    gt_mask = load_mask(gt_mask_path)

    # Skeletonize both masks
    pred_skeleton = skeletonize(pred_mask).astype(np.uint8)
    gt_skeleton = skeletonize(gt_mask).astype(np.uint8)

    # Count connected components
    pred_cc = label(pred_skeleton)
    gt_cc = label(gt_skeleton)
    num_pred_segments = np.max(pred_cc)
    num_gt_segments = np.max(gt_cc)

    # Compute distance transform from GT skeleton
    dist_transform = distance_transform_edt(~gt_skeleton)

    # Get coordinates of predicted skeleton pixels
    pred_coords = np.argwhere(pred_skeleton)

    # Compute average distance to GT skeleton
    if len(pred_coords) > 0:
        avg_distance = np.mean([dist_transform[y, x] for y, x in pred_coords])
    else:
        avg_distance = float('inf')

    # Save visualizations
    base_pred_path = os.path.splitext(pred_mask_path)[0]
    save_skeleton_image(gt_skeleton, base_pred_path, "gt")
    save_skeleton_image(pred_skeleton, base_pred_path, "pred")

    # Print results
    print("=== TBM Continuity Evaluation ===")
    print(f"GT skeleton segments      : {num_gt_segments}")
    print(f"Predicted skeleton segments: {num_pred_segments}")
    print(f"Avg. distance (pred → GT)  : {avg_distance:.4f} pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TBM continuity in segmentation predictions.")
    parser.add_argument("--pred_mask", required=True, help="Path to predicted TBM binary mask (.png)")
    parser.add_argument("--gt_mask", required=True, help="Path to ground truth TBM binary mask (.png)")
    args = parser.parse_args()

    if not os.path.exists(args.pred_mask):
        raise FileNotFoundError(f"Predicted mask not found: {args.pred_mask}")
    if not os.path.exists(args.gt_mask):
        raise FileNotFoundError(f"Ground truth mask not found: {args.gt_mask}")

    evaluate_continuity(args.pred_mask, args.gt_mask)

    
"""
python Tools/eval_tbm_continuity.py \
  --pred_mask /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop_refined_tbm_mask.png \
  --gt_mask /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks/S-2103-004858_PAS_1of2_mask.png
"""
