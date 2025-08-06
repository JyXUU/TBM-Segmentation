import json
import numpy as np
from PIL import Image
import cv2

mask_path = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop_refined_tbm_mask.png"
output_geojson = mask_path.replace(".png", "_tbm_rings.geojson")

Image.MAX_IMAGE_PIXELS = None
mask = np.array(Image.open(mask_path).convert("L"))

_, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

features = []
if hierarchy is not None:
    for idx, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
        if hier[3] != -1:
            continue  # skip holes

        # Outer contour
        outer = cnt[:, 0, :].tolist()
        if len(outer) < 3:
            continue
        if outer[0] != outer[-1]:
            outer.append(outer[0])

        # Holes
        holes = []
        child = hier[2]
        while child != -1:
            hole_cnt = contours[child][:, 0, :].tolist()
            if len(hole_cnt) >= 3:
                if hole_cnt[0] != hole_cnt[-1]:
                    hole_cnt.append(hole_cnt[0])
                holes.append(hole_cnt)
            child = hierarchy[0][child][0]

        coords = [outer] + holes
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": {
                "classification": {
                    "name": "TBM",
                    "color": [0, 255, 0]
                },
                "isLocked": False
            }
        }
        features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(output_geojson, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"TBM ring polygons with holes saved to: {output_geojson}")
