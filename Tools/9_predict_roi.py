import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import openslide
import xml.etree.ElementTree as ET

# SyncBatchNorm workaround
torch.nn.SyncBatchNorm = torch.nn.BatchNorm2d

import _init_paths
import models
from config import config, update_config
from utils.utils import create_logger
from skimage.morphology import skeletonize

def parse_args():
    parser = argparse.ArgumentParser(description='Patch-wise WSI TBM segmentation')
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--threshold', type=float, default=0.20)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--patch_size', type=int, default=3000)
    parser.add_argument('--stride_ratio', type=float, default=0.5)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def make_gaussian_weight(tile_size):
    x = np.linspace(-1, 1, tile_size)
    xv, yv = np.meshgrid(x, x)
    dist = np.sqrt(xv**2 + yv**2)
    sigma = 0.5
    return np.exp(-(dist**2) / (2 * sigma**2)).astype(np.float32)

def clean_state_dict(state_dict):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return {k.replace('module.', '').replace('model.', ''): v for k, v in state_dict.items()}

def predict_softmax(model, patch, device):
    print("[Predict] Processing patch with shape:", patch.shape)
    inp = patch.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = torch.from_numpy(inp).unsqueeze(0).to(device)

    if inp.shape[2] < 32 or inp.shape[3] < 32:
        print("[Predict] Patch too small, returning zeros")
        return np.zeros((model.final_layer.out_channels, inp.shape[2], inp.shape[3]), dtype=np.float32)

    with torch.no_grad():
        preds = []
        pred = model(inp)
        pred = F.interpolate(pred, size=patch.shape[:2], mode='bilinear', align_corners=True)
        preds.append(pred)

        pred_flip = model(torch.flip(inp, dims=[3]))
        pred_flip = torch.flip(pred_flip, dims=[3])
        pred_flip = F.interpolate(pred_flip, size=patch.shape[:2], mode='bilinear', align_corners=True)
        preds.append(pred_flip)

        pred_vflip = model(torch.flip(inp, dims=[2]))
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        pred_vflip = F.interpolate(pred_vflip, size=patch.shape[:2], mode='bilinear', align_corners=True)
        preds.append(pred_vflip)

        pred = torch.mean(torch.stack(preds), dim=0)
        return F.softmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

def postprocess_tbm(softmax_pred, threshold, min_area, raw_image, output_prefix=None):
    print("[Postprocess] Applying threshold and morphological operations")

    tbm_probs = softmax_pred[1, :, :]

    binary = (tbm_probs > threshold).astype(np.uint8)
    if output_prefix:
        Image.fromarray((tbm_probs * 255).astype(np.uint8)).save(f"{output_prefix}_probs.png")
        Image.fromarray(binary * 255).save(f"{output_prefix}_binary_thresh.png")

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    if output_prefix:
        Image.fromarray(closed * 255).save(f"{output_prefix}_closed.png")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    final = np.zeros_like(binary)

    print(f"[Postprocess] Found {num_labels - 1} connected components")
    for i in tqdm(range(1, num_labels), desc="[Postprocess] Filtering components"):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final[labels == i] = 1
    if output_prefix:
        Image.fromarray(final * 255).save(f"{output_prefix}_area_filtered.png")

    skeleton = skeletonize(final).astype(np.uint8)
    if output_prefix:
        Image.fromarray(skeleton * 255).save(f"{output_prefix}_skeleton.png")

    combined = np.logical_or(final, skeleton).astype(np.uint8)
    if output_prefix:
        Image.fromarray(combined * 255).save(f"{output_prefix}_combined.png")

    print(f"[Postprocess] Kept {np.sum(final)} TBM pixels, added {np.sum(skeleton)} skeleton pixels")
    
    ### === [Color-based TBM Gap Filling] === ###
    print("[Postprocess] Performing color-based gap filling")

    # Dilate mask to find potential border regions
    dilated = cv2.dilate(final, np.ones((3, 3), np.uint8), iterations=1)
    gap_candidates = (dilated == 1) & (final == 0)

    # Convert raw image to HSV
    hsv = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Define white pixels: all channels high 
    r, g, b = raw_image[..., 0], raw_image[..., 1], raw_image[..., 2]
    white_mask = (r > 170) & (g > 170) & (b > 170)

    # Remove white regions from final TBM mask
    filtered = final.copy()
    filtered[white_mask] = 0  # erase TBM in white-looking areas


    if output_prefix:
        Image.fromarray((white_mask * 255).astype(np.uint8)).save(f"{output_prefix}_white_regions.png")
        Image.fromarray((filtered * 255).astype(np.uint8)).save(f"{output_prefix}_final_filtered.png")

    return filtered

def save_as_xml(pred_map, xml_path):
    print(f"[XML] Saving XML to {xml_path}")
    binary = (pred_map == 1).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    root = ET.Element("Annotations")
    annotation = ET.SubElement(root, "Annotation", Id="1", Incremental="0", LineColor="65280", Type="4", Visible="1")
    regions = ET.SubElement(annotation, "Regions")
    for rid, contour in enumerate(contours):
        if len(contour) < 3: continue
        region = ET.SubElement(regions, "Region", Id=str(rid + 1), Type="0")
        vertices = ET.SubElement(region, "Vertices")
        for x, y in contour.squeeze(1):
            ET.SubElement(vertices, "Vertex", X=str(float(x)), Y=str(float(y)), Z="0")
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

def main():
    args = parse_args()
    update_config(config, args)
    logger, _, _ = create_logger(config, args.cfg, 'predict_wsi')
    logger.info(vars(args))

    print("[Main] Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.get_seg_model(config)
    ckpt = torch.load(config.TEST.MODEL_FILE, map_location=device)
    model.load_state_dict(clean_state_dict(ckpt), strict=False)
    model.to(device).eval()
    logger.info(f"Model loaded on {device}.")
    print(f"[Main] Model loaded on {device}.")

    ext = os.path.splitext(args.image_path)[-1].lower()
    if ext in ['.svs', '.ndpi', '.vms', '.vmu']:
        slide = openslide.OpenSlide(args.image_path)
        w, h = slide.level_dimensions[0]
        read_patch = lambda x, y, w, h: np.array(slide.read_region((x, y), 0, (w, h)).convert('RGB'))
        print(f"[Main] Loaded SVS with size {w}x{h}")
    elif ext in ['.tif', '.tiff']:
        full_img = Image.open(args.image_path).convert('RGB')
        full_img_np = np.array(full_img)
        h, w = full_img_np.shape[:2]
        read_patch = lambda x, y, w_, h_: full_img_np[y:y+h_, x:x+w_]
        print(f"[Main] Loaded TIF with size {w}x{h}")
    else:
        raise ValueError(f"Unsupported image format: {ext}")


    patch_size = args.patch_size
    stride = int(patch_size * args.stride_ratio)
    num_classes = config.DATASET.NUM_CLASSES
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    gaussian = make_gaussian_weight(patch_size)

    for y in tqdm(range(0, h, stride), desc="[Main] Patch Y"):
        for x in range(0, w, stride):
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)

            if x_end - x < 32 or y_end - y < 32:
                print(f"[Main] Skipping small patch at ({x}, {y})")
                continue

            print(f"[Main] Reading region ({x}, {y}, {x_end - x}, {y_end - y})")
            patch = read_patch(x, y, x_end - x, y_end - y)


            softmax_pred = predict_softmax(model, patch, device)

            for c in range(num_classes):
                prob_map[c, y:y_end, x:x_end] += softmax_pred[c, :y_end - y, :x_end - x] * gaussian[:y_end - y, :x_end - x]
            weight_map[y:y_end, x:x_end] += gaussian[:y_end - y, :x_end - x]

        print("[Main] Normalizing probability map")
        weight_map = np.clip(weight_map, a_min=1e-5, a_max=None)
        for c in range(num_classes):
            prob_map[c] /= weight_map

        print("[Main] Generating prediction map")

        base_name = os.path.splitext(os.path.basename(args.output_path))[0]
        output_prefix = os.path.join(os.path.dirname(args.output_path), base_name)

        pred_map = np.argmax(prob_map, axis=0).astype(np.uint8)
        Image.fromarray((pred_map * 127).astype(np.uint8)).save(f"{output_prefix}_raw_argmax.png")  # 0: background, 1: TBM

        raw_img_np = np.array(full_img) if ext in ['.tif', '.tiff'] else read_patch(0, 0, w, h)
        final_tbm = postprocess_tbm(prob_map, threshold=args.threshold, min_area=args.min_area, raw_image=raw_img_np, output_prefix=output_prefix)

        print(f"[Main] Saving final PNG to {args.output_path}")
        Image.fromarray((final_tbm * 255).astype(np.uint8)).save(args.output_path)
        save_as_xml(final_tbm, args.output_path.replace('.png', '.xml'))

        logger.info("[Main] Final TBM prediction saved.")
        print("Patch-by-patch TBM segmentation complete.")

if __name__ == '__main__':
    main()

"""
python Tools/9_predict_roi.py \
  --cfg /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/experiments/tbm/seg_hrnet_ocr_tbm.yaml \
  --image_path /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/cropped_ROI/S-2103-004858_PAS_1of2_crop.tif \
  --output_path /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/Data/tbm/test/4round2_S-2103-004858_PAS_1of2_crop.png \
"""
