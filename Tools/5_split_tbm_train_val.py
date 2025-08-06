import os
import shutil
import random
from pathlib import Path

BASE = Path("/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm")
IMG_DIR = BASE / "train/images"
MASK_DIR = BASE / "train/masks"

TRAIN_IMG_OUT = BASE / "train_split/images"
TRAIN_MASK_OUT = BASE / "train_split/masks"
VAL_IMG_OUT = BASE / "val/images"
VAL_MASK_OUT = BASE / "val/masks"

# Make output dirs
for p in [TRAIN_IMG_OUT, TRAIN_MASK_OUT, VAL_IMG_OUT, VAL_MASK_OUT]:
    p.mkdir(parents=True, exist_ok=True)

# Get all image filenames
images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith((".png", ".tif"))])
random.seed(42)
random.shuffle(images)

# Split
split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]

def copy_files(file_list, src_img, src_mask, dst_img, dst_mask):
    for f in file_list:
        shutil.copy(src_img / f, dst_img / f)
        shutil.copy(src_mask / f, dst_mask / f)

copy_files(train_imgs, IMG_DIR, MASK_DIR, TRAIN_IMG_OUT, TRAIN_MASK_OUT)
copy_files(val_imgs, IMG_DIR, MASK_DIR, VAL_IMG_OUT, VAL_MASK_OUT)

print(f"Done. {len(train_imgs)} training + {len(val_imgs)} validation patches.")
