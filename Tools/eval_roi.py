import os
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.morphology import skeletonize
from sklearn.metrics import roc_curve, auc

# === Paths ===
gt_path = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks/S-2103-004858_PAS_1of2_mask.png"
pred_path = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop_refined_tbm_mask.png"
orig_img_path = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI/S-2103-004858_PAS_1of2_crop.tif"
metrics_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/S-2103-004858_PAS_1of2_crop/4round2/overlay/"
save_vis_path = os.path.join(metrics_dir, "S-2103-004858_PAS_1of2_crop.png")

os.makedirs(metrics_dir, exist_ok=True)

# === Load & Resize Images ===
gt_mask = np.array(Image.open(gt_path).convert("L"))
pred_mask = np.array(Image.open(pred_path).convert("L"))
orig_img = cv2.imread(orig_img_path)

assert gt_mask.shape == pred_mask.shape, "Shape mismatch between GT and Prediction"
if orig_img.shape[:2] != gt_mask.shape:
    orig_img = cv2.resize(orig_img, (gt_mask.shape[1], gt_mask.shape[0]))

# === Binarize Masks ===
gt_bin = (gt_mask > 127).astype(np.uint8)
pred_bin = (pred_mask > 127).astype(np.uint8)

# === Overlay Generation ===
overlay = orig_img.copy()
overlay = cv2.addWeighted(overlay, 1.0, np.stack([np.zeros_like(gt_bin), gt_bin * 255, np.zeros_like(gt_bin)], axis=2), 0.5, 0)
overlay = cv2.addWeighted(overlay, 1.0, np.stack([pred_bin * 255, np.zeros_like(gt_bin), np.zeros_like(gt_bin)], axis=2), 0.5, 0)
cv2.imwrite(save_vis_path, overlay)
print(f"Overlay saved to: {save_vis_path}")

# === Core Metrics ===
tp = np.logical_and(gt_bin, pred_bin).sum()
fp = pred_bin.sum() - tp
fn = gt_bin.sum() - tp
tn = ((gt_bin == 0) & (pred_bin == 0)).sum()

# === Matthews Correlation Coefficient ===
numerator = (tp * tn) - (fp * fn)
denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
mcc = numerator / denominator

dice = 2 * tp / (gt_bin.sum() + pred_bin.sum() + 1e-8)
iou = tp / (tp + fp + fn + 1e-8)
precision = tp / (tp + fp + 1e-8)
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)

# === clDice Score ===
def compute_cldice(pred, gt):
    skel_pred = skeletonize(pred).astype(np.uint8)
    skel_gt = skeletonize(gt).astype(np.uint8)
    tprec = np.sum(skel_pred * gt) / (np.sum(skel_pred) + 1e-8)
    tsens = np.sum(skel_gt * pred) / (np.sum(skel_gt) + 1e-8)
    return 2 * tprec * tsens / (tprec + tsens + 1e-8)

cldice_score = compute_cldice(pred_bin, gt_bin)

# === Print Metrics ===
print("\n=== Evaluation Metrics ===")
print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU:              {iou:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Sensitivity:      {sensitivity:.4f}")
print(f"Specificity:      {specificity:.4f}")
print(f"clDice Score:     {cldice_score:.4f}")
print(f"MCC:              {mcc:.4f}")

# === Annotated Overlay ===
metrics_text = [
    f"Dice: {dice:.3f}", f"IoU: {iou:.3f}", f"Prec: {precision:.3f}",
    f"Rec: {sensitivity:.3f}", f"Spec: {specificity:.3f}",
    f"clDice: {cldice_score:.3f}", f"MCC: {mcc:.3f}"
]
annotated = overlay.copy()
for i, text in enumerate(metrics_text):
    cv2.putText(annotated, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

annotated_path = save_vis_path.replace(".png", "_metrics_overlay.png")
cv2.imwrite(annotated_path, annotated)
print(f"Overlay with metrics saved to: {annotated_path}")

# === Confusion Matrix ===
conf_matrix = np.array([[tn, fp], [fn, tp]])
conf_matrix_norm = conf_matrix.astype(np.float64)
conf_matrix_norm /= (conf_matrix_norm.sum(axis=1, keepdims=True) + 1e-8)

# === Unnormalized Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix.astype(int), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background", "TBM"], yticklabels=["Background", "TBM"])
plt.title("Unnormalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
conf_matrix_unnorm_path = os.path.join(metrics_dir, "confusion_matrix_unnormalized.png")
plt.savefig(conf_matrix_unnorm_path, dpi=300)
plt.close()
print(f"Saved unnormalized confusion matrix: {conf_matrix_unnorm_path}")

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Background", "TBM"], yticklabels=["Background", "TBM"])
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
conf_matrix_path = os.path.join(metrics_dir, "confusion_matrix_normalized.png")
plt.savefig(conf_matrix_path, dpi=300)
plt.close()
print(f"Saved normalized confusion matrix: {conf_matrix_path}")


# === Radar Plot with Annotations ===
metrics = [dice, iou, precision, sensitivity, specificity, cldice_score, mcc]
labels = ['Dice', 'IoU', 'Precision', 'Sensitivity', 'Specificity', 'clDice', 'MCC']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
metrics += [metrics[0]]  # Close the loop
angles += angles[:1]

# Color settings
radar_color = '#FF6F00'

# Plot
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
ax.plot(angles, metrics, linewidth=2, color=radar_color)
ax.fill(angles, metrics, color=radar_color, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_ylim(0, 1)
ax.set_title("TBM Segmentation Metrics", y=1.1, fontsize=14)

# Annotate metric values
for i, (angle, score) in enumerate(zip(angles[:-1], metrics[:-1])):
    x = angle
    y = score
    offset = 0.05  # small radial offset for label
    ax.text(x, y + offset, f"{score:.2f}", ha='center', va='center', fontsize=11, color='black')

radar_path = os.path.join(metrics_dir, "metrics_radar_plot.png")
plt.tight_layout()
plt.savefig(radar_path, dpi=300)
plt.close()
print(f"Saved annotated radar plot: {radar_path}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC and AUC
fpr, tpr, _ = roc_curve(gt_bin.ravel(), pred_bin.ravel())
roc_auc = auc(fpr, tpr)

# Create a clean and stylish ROC plot
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#FF6F00', lw=3, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')

# Set fonts and sizes
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid and legend
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()

# Save high-quality figure
roc_path = os.path.join(metrics_dir, "roc_curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved improved ROC curve: {roc_path}")

from scipy.ndimage import distance_transform_edt

# === Thickness Distribution ===
def compute_thickness_map(binary_mask):
    """Computes thickness (diameter) as 2 * distance from skeleton to nearest edge."""
    skel = skeletonize(binary_mask)
    dist_map = distance_transform_edt(binary_mask)
    thickness_map = dist_map * 2 * skel
    thickness_values = thickness_map[thickness_map > 0]
    return thickness_values

gt_thickness = compute_thickness_map(gt_bin)
pred_thickness = compute_thickness_map(pred_bin)

microns_per_pixel = 0.25  # Adjust if different
gt_thickness_um = gt_thickness * microns_per_pixel
pred_thickness_um = pred_thickness * microns_per_pixel

# Plot thickness histogram
plt.figure(figsize=(7, 5))
plt.hist(gt_thickness_um, bins=30, alpha=0.7, label='GT', color='tab:blue')
plt.hist(pred_thickness_um, bins=30, alpha=0.7, label='Prediction', color='tab:orange')
plt.xlabel("Thickness (um)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("TBM Thickness Distribution", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
thickness_plot_path = os.path.join(metrics_dir, "thickness_distribution.png")
plt.savefig(thickness_plot_path, dpi=300)
plt.close()
print(f"Saved thickness distribution plot: {thickness_plot_path}")

# === Skeleton Overlap by Length ===
skel_gt = skeletonize(gt_bin)
skel_pred = skeletonize(pred_bin)

gt_in_pred = np.logical_and(skel_gt, pred_bin).sum()
pred_in_gt = np.logical_and(skel_pred, gt_bin).sum()

skel_sensitivity = gt_in_pred / (skel_gt.sum() + 1e-8)
skel_precision = pred_in_gt / (skel_pred.sum() + 1e-8)

print("\n=== Skeleton Overlap by Length ===")
print(f"Skeleton Sensitivity: {skel_sensitivity:.4f}")
print(f"Skeleton Precision:   {skel_precision:.4f}")

# === Save Metrics to File ===
metrics_txt_path = os.path.join(metrics_dir, "metrics_summary.txt")
with open(metrics_txt_path, "w") as f:
    f.write("=== Evaluation Metrics ===\n")
    f.write(f"Dice Coefficient:      {dice:.4f}\n")
    f.write(f"IoU:                   {iou:.4f}\n")
    f.write(f"Precision:             {precision:.4f}\n")
    f.write(f"Sensitivity (Recall):  {sensitivity:.4f}\n")
    f.write(f"Specificity:           {specificity:.4f}\n")
    f.write(f"clDice Score:          {cldice_score:.4f}\n")
    f.write(f"Matthews Corr Coef:    {mcc:.4f}\n\n")
    
    f.write("=== Skeleton Overlap by Length ===\n")
    f.write(f"Skeleton Sensitivity:  {skel_sensitivity:.4f}\n")
    f.write(f"Skeleton Precision:    {skel_precision:.4f}\n")

print(f"Saved metrics summary to: {metrics_txt_path}")


"""
=== Evaluation Metrics ===
Dice Coefficient: 0.7005 How well does my predicted mask fill in the right area
IoU:              0.5391 How big is the overlapping part compared to the total area covered by both
Precision:        0.7771 How many were actually TBM. High precision = few false positive
Sensitivity:      0.6377 Of all actual TBM pixels, how many were correctly predicted (Recall / TP rate). High sensitivity = model detects TBM well
Specificity:      0.9708 Of all non-TBM pixels, how many were correctly predicted as background
clDice Score:     0.8104 How well the model match the structure AND its path

=== Skeleton Overlap by Length ===
Skeleton Sensitivity: 0.7296
Skeleton Precision:   0.9112
"""



