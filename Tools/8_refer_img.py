import openslide
import os
import cv2
import numpy as np
from PIL import Image
import random

# === Configuration ===
WSI_PATH = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/WSIs/S-2006-004952_PAS_2of2.svs"
PATCH_SIZE = 1024
NUM_TRIES = 100
SAVE_PATH = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/reference/reference_img.png"

def is_informative(patch, threshold=0.85):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return np.mean(gray < 220) > (1 - threshold)  # Not too bright (white)

def extract_reference_patch(wsi_path, save_path):
    slide = openslide.OpenSlide(wsi_path)
    level = 0  # full res

    width, height = slide.level_dimensions[level]

    for _ in range(NUM_TRIES):
        x = random.randint(0, width - PATCH_SIZE)
        y = random.randint(0, height - PATCH_SIZE)

        patch = slide.read_region((x, y), level, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        patch_np = np.array(patch)

        if is_informative(patch_np):
            patch.save(save_path)
            print(f"[INFO] Saved reference patch to: {save_path}")
            return

    print("[ERROR] Could not find a good reference patch. Try increasing NUM_TRIES or inspecting slide.")

if __name__ == "__main__":
    assert os.path.exists(WSI_PATH), f"WSI not found at {WSI_PATH}"
    extract_reference_patch(WSI_PATH, SAVE_PATH)
