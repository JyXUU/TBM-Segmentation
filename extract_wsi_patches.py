import openslide
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import random

def is_background(patch, threshold=220, coverage=0.4):
    gray = patch.convert("L")
    np_gray = np.array(gray)
    white_pixels = np_gray > threshold
    white_ratio = np.sum(white_pixels) / white_pixels.size
    print(f"White ratio: {white_ratio:.2f}")  # Add this line
    return white_ratio > coverage

def extract_patches(svs_path, output_dir, patch_size=256, stride=256, level=0, max_patches=200):
    slide = openslide.OpenSlide(svs_path)
    svs_name = os.path.splitext(os.path.basename(svs_path))[0]
    width, height = slide.level_dimensions[level]

    x_tiles = (width - patch_size) // stride + 1
    y_tiles = (height - patch_size) // stride + 1

    candidate_coords = []

    for i in range(x_tiles):
        for j in range(y_tiles):
            x = i * stride
            y = j * stride
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            if not is_background(patch):
                candidate_coords.append((x, y))

    print(f"{svs_name}: Found {len(candidate_coords)} tissue patches")

    selected_coords = random.sample(candidate_coords, min(len(candidate_coords), max_patches))

    os.makedirs(output_dir, exist_ok=True)
    for x, y in selected_coords:
        patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
        patch_name = f"{svs_name}_x{x}_y{y}.png"
        patch.save(os.path.join(output_dir, patch_name))

    print(f"{svs_name}: Saved {len(selected_coords)} patches")

    slide.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", type=str, required=True, help="Folder containing .svs files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save patches")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    svs_files = [f for f in os.listdir(args.wsi_dir) if f.endswith(".svs")]

    for svs_file in tqdm(svs_files, desc="Processing WSIs"):
        svs_path = os.path.join(args.wsi_dir, svs_file)
        extract_patches(
            svs_path,
            args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            level=args.level
        )

if __name__ == "__main__":
    main()

"""
python wssal_pipeline/extract_wsi_patches.py \
  --wsi_dir /orange/pinaki.sarder/Davy_Jones_Locker/Brandon_IFTA_segmentation/Training/WSIs \
  --output_dir /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/wssal_pipeline/Data/patches/unlabeled \
  --patch_size 256 \
  --stride 256 \
  --level 0
"""