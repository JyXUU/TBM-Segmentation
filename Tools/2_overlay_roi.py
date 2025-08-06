from PIL import Image
import os
from tqdm import tqdm

# Define directories
CROPPED_IMG_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI"
MASK_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks"
OVERLAY_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/overlay_ROI"

# Iterate over each mask
for fname in tqdm(os.listdir(MASK_DIR)):
    if not fname.endswith(".png"):
        continue

    base = os.path.splitext(fname)[0].replace("_mask", "")
    img_path = os.path.join(CROPPED_IMG_DIR, f"{base}_crop.tif")
    mask_path = os.path.join(MASK_DIR, fname)
    overlay_path = os.path.join(OVERLAY_DIR, f"{base}_overlay.png")

    if not os.path.exists(img_path):
        print(f"Corresponding image not found for: {base}")
        continue

    # Load images
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Create an alpha mask: white areas become fully transparent (0), black remains opaque (255)
    alpha = mask.point(lambda p: 255 if p == 255 else 0)

    # Convert original image to RGBA and apply alpha mask
    image.putalpha(alpha)
    image.save(overlay_path)
