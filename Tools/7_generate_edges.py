import os
import cv2
import numpy as np
from glob import glob

mask_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train_split/masks"
edge_dir = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/train_split/edges"
os.makedirs(edge_dir, exist_ok=True)

for path in glob(os.path.join(mask_dir, "*.png")):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(mask, threshold1=50, threshold2=150)  # or use cv2.Sobel
    save_path = os.path.join(edge_dir, os.path.basename(path))
    cv2.imwrite(save_path, edges)
