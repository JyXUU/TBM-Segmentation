import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ====== Paths ======
base_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/wssal_output/round_1/labeled"
image_dir = os.path.join(base_dir, "images")
mask_dir = os.path.join(base_dir, "masks")
output_dir = os.path.join(base_dir, "overlays")

os.makedirs(output_dir, exist_ok=True)

# ====== Overlay Settings ======
alpha = 0.5  # Mask transparency
mask_color = [255, 0, 0]  # Red color for TBM regions

# ====== Process all masks ======
for fname in tqdm(os.listdir(mask_dir)):
    if not fname.endswith(".png"):
        continue

    mask_path = os.path.join(mask_dir, fname)
    image_path = os.path.join(image_dir, fname.replace("_pseudo.png", ".png"))

    if not os.path.exists(image_path):
        print(f"Image not found for: {fname}")
        continue

    # Load image and mask
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Resize mask if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create color overlay
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = mask_color

    overlay = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)

    # Save overlay
    out_path = os.path.join(output_dir, fname.replace("_pseudo.png", "_overlay.png"))
    Image.fromarray(overlay).save(out_path)

print(f"Overlays saved to: {output_dir}")
