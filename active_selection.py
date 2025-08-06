import os
import argparse
import pickle
import numpy as np

def load_mc_output(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def compute_mean_entropy(entropy_map):
    return np.mean(entropy_map)

def select_topk_uncertain(mc_data, k=100):
    scores = []
    for item in mc_data:
        entropy = item['entropy']
        mean_ent = compute_mean_entropy(entropy)
        scores.append((item['name'], mean_ent))
    scores.sort(key=lambda x: x[1], reverse=True)  # descending
    return [name for name, _ in scores[:k]]

def save_selected_names(selected_names, out_path):
    with open(out_path, 'w') as f:
        for name in selected_names:
            f.write(name + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, required=True, help='Path to mc_output.pkl')
    parser.add_argument('--k', type=int, default=100, help='Number of most uncertain patches to select')
    parser.add_argument('--out_txt', type=str, required=True, help='Where to save selected patch names')
    args = parser.parse_args()

    mc_data = load_mc_output(args.pkl)
    selected = select_topk_uncertain(mc_data, k=args.k)
    save_selected_names(selected, args.out_txt)

    print(f"Saved top {args.k} uncertain patch names to {args.out_txt}")

if __name__ == '__main__':
    main()

"""
python wssal_pipeline/active_selection.py \
  --pkl wssal_output/round_1/mc_output.pkl \
  --k 100 \
  --out_txt wssal_output/round_1/to_annotate.txt
"""
