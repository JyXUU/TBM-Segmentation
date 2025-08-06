from PIL import Image
import os
import numpy as np

mask_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train_split/masks"

for fname in os.listdir(mask_dir):
    if not fname.endswith(".png"):
        continue

    path = os.path.join(mask_dir, fname)
    mask = np.array(Image.open(path))

    # Convert 255 → 1, leave 0 as is
    mask[mask == 255] = 1

    # Save back
    Image.fromarray(mask.astype(np.uint8)).save(path)