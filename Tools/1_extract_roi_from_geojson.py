import os
import json
import openslide
from shapely.geometry import shape, box
from shapely.affinity import translate
from PIL import Image, ImageDraw
from tqdm import tqdm

WSI_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/WSIs"
GEOJSON_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/GT"
OUTPUT_IMG_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI"
OUTPUT_MASK_DIR = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI_masks"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

def find_rectangular_roi(features):
    for feat in features:
        geom = shape(feat["geometry"])
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
            if len(coords) == 5:  # Rectangle (4 corners + closing point)
                return geom
    raise ValueError("No rectangular ROI (with 5 points) found in annotations.")

def crop_region(slide_path, geojson_path, out_img_path, out_mask_path):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.dimensions

    with open(geojson_path, 'r') as f:
        geo = json.load(f)

    if not geo["features"]:
        raise ValueError(f"No annotations found in {geojson_path}")

    # === Step 1: Find rectangular ROI annotation ===
    roi_shape = find_rectangular_roi(geo["features"])
    crop_rect = box(*roi_shape.bounds)  # enforce rectangle
    minx, miny, maxx, maxy = crop_rect.bounds
    x0, y0 = int(max(minx, 0)), int(max(miny, 0))
    x1, y1 = int(min(maxx, w)), int(min(maxy, h))
    crop_w, crop_h = x1 - x0, y1 - y0

    # === Step 2: Create full-size mask ===
    full_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(full_mask)

    # === Step 3: Draw TBM annotations (exclude the ROI rectangle) ===
    for feat in geo["features"]:
        geom = shape(feat["geometry"])
        coords = list(geom.exterior.coords) if geom.geom_type == "Polygon" else None
        if coords and len(coords) == 5:
            continue  # Skip the rectangular ROI
        if geom.is_empty or not geom.is_valid:
            continue
        if geom.geom_type == "Polygon":
            if len(list(geom.exterior.coords)) >= 3:
                draw.polygon(list(geom.exterior.coords), fill=255)
            for interior in geom.interiors:
                ic = list(interior.coords)
                if len(ic) >= 3:
                    draw.polygon(ic, fill=0)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                if len(list(poly.exterior.coords)) >= 3:
                    draw.polygon(list(poly.exterior.coords), fill=255)
                for interior in poly.interiors:
                    ic = list(interior.coords)
                    if len(ic) >= 3:
                        draw.polygon(ic, fill=0)

    # === Step 4: Crop WSI and mask ===
    cropped_img = slide.read_region((x0, y0), 0, (crop_w, crop_h)).convert("RGB")
    cropped_mask = full_mask.crop((x0, y0, x1, y1))

    # === Step 5: Save outputs ===
    cropped_img.save(out_img_path, format="TIFF")
    cropped_mask.save(out_mask_path, format="PNG")

# === Main loop ===
for fname in tqdm(os.listdir(GEOJSON_DIR)):
    if not fname.endswith(".geojson"):
        continue
    base = os.path.splitext(fname)[0]
    wsi_path = os.path.join(WSI_DIR, base + ".svs")
    geojson_path = os.path.join(GEOJSON_DIR, fname)

    if not os.path.exists(wsi_path):
        print(f"WSI not found for: {base}")
        continue

    out_img_path = os.path.join(OUTPUT_IMG_DIR, f"{base}_crop.tif")
    out_mask_path = os.path.join(OUTPUT_MASK_DIR, f"{base}_mask.png")

    try:
        crop_region(wsi_path, geojson_path, out_img_path, out_mask_path)
    except Exception as e:
        print(f"Failed to process {base}: {e}")
