import numpy as np
import cv2
from scipy.ndimage import label
import os
import argparse
import matplotlib.pyplot as plt

def binarize_mask(mask, threshold=127):
    return (mask > threshold).astype(np.uint8)

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return binarize_mask(mask)

def compute_per_tbm_coverage(gt_mask, pred_mask, min_size=50):
    labeled_gt, num_gt = label(gt_mask)
    coverage_scores = []

    for i in range(1, num_gt + 1):
        tbm_obj = (labeled_gt == i)
        tbm_area = np.sum(tbm_obj)

        if tbm_area < min_size:
            continue  # Skip small/noise regions

        predicted_overlap = np.logical_and(tbm_obj, pred_mask)
        overlap_area = np.sum(predicted_overlap)

        coverage = overlap_area / tbm_area
        coverage_scores.append(coverage)

    return coverage_scores

def summarize_coverage(coverage_scores):
    coverage_array = np.array(coverage_scores)
    stats = {
        "Num_TBMs": len(coverage_array),
        "Mean_Coverage": np.mean(coverage_array),
        "Median_Coverage": np.median(coverage_array),
        "Min_Coverage": np.min(coverage_array),
        "Max_Coverage": np.max(coverage_array),
        "Coverage_>=50%": np.sum(coverage_array >= 0.5) / len(coverage_array),
        "Coverage_>=80%": np.sum(coverage_array >= 0.8) / len(coverage_array),
    }
    return stats

def plot_histogram(coverage_scores, output_path=None):
    plt.figure(figsize=(8, 4))
    plt.hist(coverage_scores, bins=20, range=(0,1), color='skyblue', edgecolor='black')
    plt.title("Per-TBM Coverage Histogram")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Number of TBMs")
    plt.grid(True)
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TBM coverage per ground truth object.")
    parser.add_argument("--gt", required=True, help="Path to ground truth mask")
    parser.add_argument("--pred", required=True, help="Path to predicted mask")
    parser.add_argument("--plot", help="Optional path to save coverage histogram")
    args = parser.parse_args()

    gt = load_mask(args.gt)
    pred = load_mask(args.pred)

    coverages = compute_per_tbm_coverage(gt, pred)
    summary = summarize_coverage(coverages)

    print("=== Per-TBM Coverage Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if args.plot:
        plot_histogram(coverages, args.plot)
    else:
        plot_histogram(coverages)

"""
python Tools/per_tbm_coverage_eval.py --gt /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks/S-2103-004858_PAS_1of2_mask.png --pred /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop_refined_tbm_mask.png --plot coverage_histogram.png
"""