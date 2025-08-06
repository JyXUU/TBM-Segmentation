import json
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None 

def load_geojson_polygons(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    polygons = []
    for elem in data['annotation']['elements']:
        if elem['type'] == 'polyline' and elem.get('points'):
            points = [(x, y) for x, y, _ in elem['points']]
            polygons.append(Polygon(points))
    return polygons

def polygons_to_mask(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        # Handle MultiPolygon
        if isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                coords = np.array(p.exterior.coords).round().astype(np.int32)
                cv2.fillPoly(mask, [coords], color=1)
        else:  # Single Polygon
            coords = np.array(poly.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [coords], color=1)
    return mask

def refine_tbm_mask(tbm_mask_path, refined_save_path, geojson_dir, output_shape=None):
    # Load TBM mask
    tbm_mask = np.array(Image.open(tbm_mask_path).convert('L')) > 0
    if output_shape:
        tbm_mask = cv2.resize(tbm_mask.astype(np.uint8), output_shape, interpolation=cv2.INTER_NEAREST)

    # Load all exclusion polygons
    non_sclerotic = load_geojson_polygons(os.path.join(geojson_dir, "non_globally_sclerotic_glomeruli.json"))
    sclerotic = load_geojson_polygons(os.path.join(geojson_dir, "globally_sclerotic_glomeruli.json"))
    arteries = load_geojson_polygons(os.path.join(geojson_dir, "arteries_arterioles.json"))

    all_exclusions = non_sclerotic + sclerotic + arteries

    # Create exclusion mask
    exclusion_mask = polygons_to_mask(all_exclusions, tbm_mask.shape)

    # Refine TBM prediction
    refined_mask = np.where(exclusion_mask == 1, 0, tbm_mask).astype(np.uint8)

    # Save result
    Image.fromarray((refined_mask * 255).astype(np.uint8)).save(refined_save_path)
    print(f"Saved refined TBM mask to: {refined_save_path}")

refine_tbm_mask(
    tbm_mask_path='/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop.png',
    refined_save_path='/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop_refined_tbm_mask.png',
    geojson_dir='/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/S-2103-004858_PAS_1of2/mc_seg_crop'
)

