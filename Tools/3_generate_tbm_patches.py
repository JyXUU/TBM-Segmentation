import os
from PIL import Image
from tqdm import tqdm

ROI_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI"
MASK_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks"
PATCH_IMG_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train/images"
PATCH_MASK_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train/masks"

PATCH_SIZE = 256
STRIDE = 256 

os.makedirs(PATCH_IMG_DIR, exist_ok=True)
os.makedirs(PATCH_MASK_DIR, exist_ok=True)

def save_patch(img, mask, base_name):
    w, h = img.size
    for i in range(0, h - PATCH_SIZE + 1, STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch_img = img.crop((j, i, j + PATCH_SIZE, i + PATCH_SIZE))
            patch_mask = mask.crop((j, i, j + PATCH_SIZE, i + PATCH_SIZE))

            patch_id = f"{base_name}_x{j}_y{i}"
            patch_img.save(os.path.join(PATCH_IMG_DIR, f"{patch_id}.png"))
            patch_mask.save(os.path.join(PATCH_MASK_DIR, f"{patch_id}.png"))

for fname in tqdm(os.listdir(ROI_DIR)):
    if not fname.endswith(".tif"):
        continue
    base = os.path.splitext(fname)[0].replace("_crop", "")
    roi_path = os.path.join(ROI_DIR, fname)
    mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")

    if not os.path.exists(mask_path):
        print(f"Missing mask for {base}")
        continue

    img = Image.open(roi_path)
    mask = Image.open(mask_path).convert("L")

    save_patch(img, mask, base)
