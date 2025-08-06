import os
import json
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm

# ====== USER CONFIGURATION ======
PATCH_SIZE = 256
WSI_JSON_FOLDER = '/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train/JSON'  # Folder containing JSONs
WSI_IMAGE_FOLDER = '/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/WSIs/'  # Folder containing .svs files
OUTPUT_IMAGE_FOLDER = '/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train/images'
OUTPUT_MASK_FOLDER = '/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train/masks'

# Create output folders if not exist
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)

# ====== Helper functions ======
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_center_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_center = (min(x_coords) + max(x_coords)) / 2
    y_center = (min(y_coords) + max(y_coords)) / 2
    return int(x_center), int(y_center)

def crop_patch(slide, center_x, center_y, patch_size):
    half_size = patch_size // 2
    x = int(center_x - half_size)
    y = int(center_y)
    if x < 0 or y < 0 or (x + patch_size) > slide.dimensions[0] or (y + patch_size) > slide.dimensions[1]:
        return None  # Patch would go outside bounds
    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
    return patch

def save_blank_mask(save_path, patch_size):
    blank = Image.fromarray(np.zeros((patch_size, patch_size), dtype=np.uint8))
    blank.save(save_path)

# ====== Main processing ======

# Counters for filenames
glom_counter = 0
artery_counter = 0

wsi_files = [f for f in os.listdir(WSI_JSON_FOLDER) if f.endswith('.svs')]

for wsi_file in tqdm(wsi_files, desc="Processing WSIs"):
    slide_path = os.path.join(WSI_IMAGE_FOLDER, wsi_file)
    slide = openslide.OpenSlide(slide_path)

    # Paths to JSON files
    globally_glom_json = os.path.join(WSI_JSON_FOLDER, wsi_file, 'globally_sclerotic_glomeruli.json')
    non_globally_glom_json = os.path.join(WSI_JSON_FOLDER, wsi_file, 'non_globally_sclerotic_glomeruli.json')
    artery_json = os.path.join(WSI_JSON_FOLDER, wsi_file, 'arteries_arterioles.json')

    # Process Glomeruli (both globally and non-globally)
    for glom_json in [globally_glom_json, non_globally_glom_json]:
        if not os.path.exists(glom_json):
            continue
        data = load_json(glom_json)
        elements = data.get('annotation', {}).get('elements', [])
        print(f"[INFO] {glom_json}: {len(elements)} glomeruli found")

        np.random.shuffle(elements)  # Shuffle the elements randomly
        for elem in elements:
            if glom_counter >= 400:
                break  # Stop after 400 glomeruli patches
            points = elem.get('points', [])
            if not points:
                continue
            cx, cy = extract_center_bbox(points)
            patch = crop_patch(slide, cx, cy, PATCH_SIZE)
            if patch:
                img_save_path = os.path.join(OUTPUT_IMAGE_FOLDER, f'glom_{glom_counter:04d}.png')
                mask_save_path = os.path.join(OUTPUT_MASK_FOLDER, f'glom_{glom_counter:04d}.png')
                patch.save(img_save_path)
                save_blank_mask(mask_save_path, PATCH_SIZE)
                glom_counter += 1


    # Process Arteries
    if os.path.exists(artery_json):
        data = load_json(artery_json)
        elements = data.get('annotation', {}).get('elements', [])
        print(f"[INFO] {artery_json}: {len(elements)} arteries found")

        np.random.shuffle(elements)  # Shuffle the elements randomly
        for elem in elements:
            if artery_counter >= 400:
                break  # Stop after 400 artery patches
            points = elem.get('points', [])
            if not points:
                continue
            cx, cy = extract_center_bbox(points)
            patch = crop_patch(slide, cx, cy, PATCH_SIZE)
            if patch:
                img_save_path = os.path.join(OUTPUT_IMAGE_FOLDER, f'artery_{artery_counter:04d}.png')
                mask_save_path = os.path.join(OUTPUT_MASK_FOLDER, f'artery_{artery_counter:04d}.png')
                patch.save(img_save_path)
                save_blank_mask(mask_save_path, PATCH_SIZE)
                artery_counter += 1

    slide.close()

print(f"Extraction completed: {glom_counter} glomeruli patches, {artery_counter} artery patches saved.")