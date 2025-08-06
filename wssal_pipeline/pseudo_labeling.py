import os
import numpy as np
from PIL import Image
import argparse
import pickle
import matplotlib.pyplot as plt

def save_mask(prob_map, threshold=0.97):
    return (prob_map > threshold).astype(np.uint8) * 255

def save_debug_maps(prob, entropy, out_dir, name, cmap='viridis'):
    os.makedirs(out_dir, exist_ok=True)
    prob_img = (prob * 255).astype(np.uint8)
    entropy_img = (entropy / np.max(entropy + 1e-6) * 255).astype(np.uint8)

    Image.fromarray(prob_img).save(os.path.join(out_dir, name.replace('.png', '_prob.png')))
    Image.fromarray(entropy_img).save(os.path.join(out_dir, name.replace('.png', '_entropy.png')))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pseudo_labels_from_mc_output(pkl_path, out_dir_mask, out_dir_debug=None, threshold=0.97):
    os.makedirs(out_dir_mask, exist_ok=True)
    data = load_pickle(pkl_path)  # list of dicts

    for item in data:
        name = item['name']
        prob = item['prob']
        entropy = item['entropy']

        # Save binary mask
        mask = save_mask(prob, threshold)
        mask_path = os.path.join(out_dir_mask, name.replace('.png', '_pseudo.png'))
        Image.fromarray(mask).save(mask_path)

        # Save probability/entropy maps if needed
        if out_dir_debug:
            save_debug_maps(prob, entropy, out_dir_debug, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, required=True, help="Path to .pkl from MC dropout inference")
    parser.add_argument('--mask_out', type=str, required=True, help="Path to save binary pseudo masks")
    parser.add_argument('--debug_out', type=str, default=None, help="Path to save prob/entropy images")
    parser.add_argument('--threshold', type=float, default=0.97, help="Confidence threshold")
    args = parser.parse_args()

    save_pseudo_labels_from_mc_output(args.pkl, args.mask_out, args.debug_out, args.threshold)

if __name__ == '__main__':
    main()

"""
python wssal_pipeline/pseudo_labeling.py \
  --pkl mc_output.pkl \
  --mask_out Data/patches/pseudo_labels/ \
  --debug_out Data/patches/debug_maps/
"""