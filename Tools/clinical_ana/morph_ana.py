import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_fill_holes, binary_closing
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import openslide
from joblib import Parallel, delayed
import torch

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Constants
PIXEL_SIZE_UM = 0.252
THICKNESS_THRESHOLD_UM = 1.2
sns.set(style="whitegrid", context="talk")

# === Helper functions for .svs ===

def extract_roi_from_svs(svs_path, top_left=(0, 0), size=(3000, 3000), level=0):
    slide = openslide.OpenSlide(svs_path)
    region = slide.read_region(top_left, level, size).convert("RGB")
    slide.close()
    return np.array(region)

def save_rgb_image(np_img_rgb, save_path):
    bgr = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)

def run_segmentation_model(image_rgb):
    raise NotImplementedError("Segmentation model inference not implemented.")

# === I/O and preprocessing ===

def load_binary_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return (mask > 0).astype(np.uint8)

def micronsq_to_pixels(area_um2, pixel_size_um=PIXEL_SIZE_UM):
    return int(np.round(area_um2 / (pixel_size_um**2)))

def clean_binary_mask(binary_mask, min_area=100):
    labeled = measure.label(binary_mask)
    cleaned = morphology.remove_small_objects(labeled, min_size=min_area)
    return (cleaned > 0).astype(np.uint8)

def visualize_discarded_regions(original_mask, cleaned_mask, save_path):
    discarded = ((original_mask > 0) & (cleaned_mask == 0)).astype(np.uint8) * 255
    cv2.imwrite(save_path, discarded)

def save_cleaned_mask(cleaned_mask, save_path):
    vis_mask = (cleaned_mask > 0).astype(np.uint8) * 255
    cv2.imwrite(save_path, vis_mask)

# === Feature extraction ===

def process_region(p):
    area = p.area
    perimeter = p.perimeter
    circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
    eccentricity = p.eccentricity
    inner_area = p.convex_area - area
    tbm_to_tubule_ratio = area / (area + inner_area) if inner_area > 0 else np.nan
    return {
        "Area": area,
        "Perimeter": perimeter,
        "Circularity": circularity,
        "Eccentricity": eccentricity,
        "TBM-to-Tubule Ratio": tbm_to_tubule_ratio
    }

def extract_shape_descriptors(binary_mask, n_jobs=4):  # n_jobs is ignored now
    labeled = measure.label(binary_mask)
    props_table = measure.regionprops_table(labeled, properties=[
        "area", "perimeter", "eccentricity", "convex_area"
    ])
    df = pd.DataFrame(props_table)

    # Vectorized feature calculations
    circularity = 4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-6)
    inner_area = df["convex_area"] - df["area"]
    tbm_to_tubule_ratio = df["area"] / (df["area"] + inner_area.replace(0, np.nan))

    df["Circularity"] = circularity
    df["TBM-to-Tubule Ratio"] = tbm_to_tubule_ratio

    return df.to_dict(orient="records"), labeled


def measure_thickness(binary_mask):
    bbox = np.argwhere(binary_mask)
    y0, x0 = bbox.min(axis=0)
    y1, x1 = bbox.max(axis=0) + 1
    roi = binary_mask[y0:y1, x0:x1]
    skeleton = morphology.skeletonize(roi)
    distance_map = distance_transform_edt(roi)
    thickness_values = distance_map[skeleton > 0] * 2
    full_distance_map = np.zeros_like(binary_mask, dtype=np.float32)
    full_distance_map[y0:y1, x0:x1] = distance_map
    return thickness_values, skeleton, full_distance_map

def compute_pas_intensity(pas_path, binary_mask):
    h, w = binary_mask.shape

    if pas_path.endswith(".svs"):
        region = extract_roi_from_svs(pas_path, top_left=(0, 0), size=(w, h))  # match mask shape
        image = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(pas_path)
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))

    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[binary_mask > 0]
    return np.mean(pixels), np.std(pixels)

def overlay_thickness_heatmap(pas_path, thickness_map, mask, output_path, alpha=0.6):
    h, w = mask.shape

    if pas_path.endswith(".svs"):
        region = extract_roi_from_svs(pas_path, top_left=(0, 0), size=(w, h))  # (width, height)
        image = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)  # Convert to BGR for consistency
    else:
        image = cv2.imread(pas_path)
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {pas_path}")

    # Convert thickness map to µm
    thickness_um = thickness_map * 2 * PIXEL_SIZE_UM

    # Create color overlay
    heatmap_rgb = np.zeros_like(image)

    # Define color thresholds
    green = (0, 255, 0)      # Thin
    yellow = (0, 255, 255)   # Moderate
    red = (0, 0, 255)        # Thick

    thin_mask = (thickness_um < 1.2) & (mask > 0)
    moderate_mask = (thickness_um >= 1.2) & (thickness_um < 2.0) & (mask > 0)
    thick_mask = (thickness_um >= 2.0) & (mask > 0)

    heatmap_rgb[thin_mask] = green
    heatmap_rgb[moderate_mask] = yellow
    heatmap_rgb[thick_mask] = red

    # Blend only where mask is present
    blended = image.copy()
    mask_3ch = np.stack([mask]*3, axis=-1).astype(bool)
    blended[mask_3ch] = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)[mask_3ch]

    cv2.imwrite(output_path, blended)

# === Interpretation ===

def interpret_statistics(summary):
    statements = []
    if "Thickness_um" in summary.index:
        mean_thickness = summary.loc["Thickness_um", "mean"]
        std_thickness = summary.loc["Thickness_um", "std"]
        if mean_thickness > THICKNESS_THRESHOLD_UM:
            statements.append(f"The average TBM thickness is {mean_thickness:.2f} um, which exceeds the clinical threshold of {THICKNESS_THRESHOLD_UM} µm.")
        else:
            statements.append(f"The average TBM thickness is {mean_thickness:.2f} um, within normal range.")
        statements.append(f"The standard deviation of thickness is {std_thickness:.2f} um.")

    if "Circularity" in summary.index:
        circ = summary.loc["Circularity", "mean"]
        shape_desc = "mostly round" if circ > 0.7 else "irregular"
        statements.append(f"The mean TBM circularity is {circ:.2f}, suggesting tubules are {shape_desc}.")

    return "\n".join(statements)

# === Main analysis pipeline ===

def analyze(mask_path, pas_path=None):
    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_dir = os.path.join(os.path.dirname(mask_path), f"{base_name}_tbm_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if mask_path.endswith(".svs"):
        print("[INFO] Detected .svs input. Extracting PAS ROI and generating mask...")
        roi_rgb = extract_roi_from_svs(mask_path, top_left=(0, 0), size=(3000, 3000))
        pas_path = os.path.join(output_dir, "pas_patch.png")
        save_rgb_image(roi_rgb, pas_path)

        try:
            binary_mask = run_segmentation_model(roi_rgb)
        except NotImplementedError:
            raise RuntimeError("Please implement run_segmentation_model().")

        mask_path = os.path.join(output_dir, "predicted_mask.png")
        cv2.imwrite(mask_path, binary_mask.astype(np.uint8) * 255)

    print("[INFO] Loading and preprocessing mask...")
    original_mask = load_binary_mask(mask_path)

    shape_data_unfiltered, _ = extract_shape_descriptors(original_mask)
    df_unfiltered = pd.DataFrame(shape_data_unfiltered)
    df_unfiltered["Thickness_um"] = (df_unfiltered["area"] / df_unfiltered["perimeter"]) * PIXEL_SIZE_UM
    summary_unfiltered = df_unfiltered.describe().T
    summary_unfiltered["median"] = df_unfiltered.median(numeric_only=True)
    summary_unfiltered["IQR"] = df_unfiltered.quantile(0.75) - df_unfiltered.quantile(0.25)
    summary_unfiltered = summary_unfiltered.round(3)
    summary_unfiltered.to_csv(os.path.join(output_dir, "summary_table_unfiltered.csv"))

    min_area_px = micronsq_to_pixels(area_um2=25.0)
    cleaned_mask = clean_binary_mask(original_mask, min_area=min_area_px)
    visualize_discarded_regions(original_mask, cleaned_mask, os.path.join(output_dir, "discarded_regions.png"))
    save_cleaned_mask(cleaned_mask, os.path.join(output_dir, "cleaned_mask.png"))

    shape_data, _ = extract_shape_descriptors(cleaned_mask)
    thickness_values, _, distance_map = measure_thickness(cleaned_mask)

    df = pd.DataFrame(shape_data)
    df["Thickness_um"] = (df["area"] / df["perimeter"]) * PIXEL_SIZE_UM
    df["TBM_Thickened"] = df["Thickness_um"] > THICKNESS_THRESHOLD_UM

    if pas_path:
        mean_intensity, std_intensity = compute_pas_intensity(pas_path, cleaned_mask)
        df["Mean_PAS_Intensity"] = mean_intensity
        df["Std_PAS_Intensity"] = std_intensity

    feature_csv = os.path.join(output_dir, "tbm_features.csv")
    df.to_csv(feature_csv, index=False)

    numeric_df = df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    summary["median"] = numeric_df.median()
    summary["IQR"] = numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
    summary = summary.round(3)
    summary.loc["Region Count"] = [df.shape[0]] + [np.nan] * (summary.shape[1] - 1)
    summary.to_csv(os.path.join(output_dir, "summary_table.csv"))

    interpretation = interpret_statistics(summary)
    with open(os.path.join(output_dir, "clinical_interpretation.txt"), "w") as f:
        f.write(f"Note: Small TBM segments < 25 µm² (approx. {min_area_px} px) were excluded.\n\n")
        f.write(interpretation)

    # === Plotting TBM Thickness Distribution and Stratified Boxplot ===
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Left: Histogram with KDE
    sns.histplot(df["Thickness_um"], kde=True, bins=30, ax=axes[0], color="steelblue", alpha=0.6)
    axes[0].set_title("Distribution of Thickness_um")
    axes[0].set_xlabel("Thickness_um")
    axes[0].set_ylabel("Frequency")

    # Right: Boxplot grouped by TBM thickening
    sns.boxplot(x="TBM_Thickened", y="Thickness_um", data=df, ax=axes[1], palette="pastel")
    axes[1].set_title("Thickness_um by TBM Thickening")
    axes[1].set_xlabel("TBM Thickened")
    axes[1].set_ylabel("Thickness_um")

    # Rightmost: Pie chart of thickness categories
    thin_count = (df["Thickness_um"] < 1.2).sum()
    moderate_count = ((df["Thickness_um"] >= 1.2) & (df["Thickness_um"] < 2.0)).sum()
    thick_count = (df["Thickness_um"] >= 2.0).sum()

    counts = [thin_count, moderate_count, thick_count]
    labels = ['Thin (<1.2um)', 'Moderate (1.2–2.0um)', 'Thick (≥2.0um)']
    colors = ['green', 'yellow', 'red']

    axes[2].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    axes[2].set_title("TBM Thickness Proportion")

    # Save plot
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "tbm_thickness_distribution.png")
    plt.savefig(plot_path)
    
    plt.close()
    print(f"Saved: {plot_path}")
    
    if pas_path and os.path.exists(pas_path):
        heatmap_path = os.path.join(output_dir, "tbm_thickness_heatmap_overlay.png")
        overlay_thickness_heatmap(pas_path, distance_map, cleaned_mask, heatmap_path)
        print(f"Saved: {heatmap_path}")

    print(f"Saved: {feature_csv}")
    print(f"Saved: {os.path.join(output_dir, 'summary_table.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'clinical_interpretation.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TBM segmentation clinical analysis.")
    parser.add_argument("--mask", required=True, help="Path to TBM binary mask")
    parser.add_argument("--pas", default=None, help="PAS image")
    args = parser.parse_args()
    analyze(args.mask, args.pas)

"""
python Tools/clinical_ana/morph_ana.py \
    --mask "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/K1900221_6_PAS_WU-001_refined_tbm_mask.png" \
    --pas "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/WSIs/K1900221_6_PAS_WU-001.svs"
"""
"""
srun --job-name=tbm_cpu_test \
     --partition=hpg-default \
     --cpus-per-task=1 \
     --mem=12gb \
     --time=8:00:00 \
     --pty bash
"""