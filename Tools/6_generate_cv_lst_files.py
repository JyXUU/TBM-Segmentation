import os
import glob
import random
from collections import defaultdict
from sklearn.model_selection import KFold

# === CONFIG ===
root = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/"
image_dir = os.path.join(root, "Data/tbm/train_split/images")
mask_dir = os.path.join(root, "Data/tbm/train_split/masks")
output_dir = os.path.join(root, "cv_lst/tbm/folds")
os.makedirs(output_dir, exist_ok=True)

k_folds = 5
seed = 42

# === Group patches by WSI prefix ===
def extract_prefix(filename):
    return filename.split("_patch")[0]  # e.g., "K1800212_2_PAS_WU"

groups = defaultdict(list)
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
print(f"[INFO] Found {len(image_paths)} image patches.")

for img_path in image_paths:
    filename = os.path.basename(img_path)
    prefix = extract_prefix(filename)
    mask_path = os.path.join(mask_dir, filename)
    if os.path.exists(mask_path):
        rel_img = os.path.relpath(img_path, root)
        rel_mask = os.path.relpath(mask_path, root)
        groups[prefix].append((rel_img, rel_mask))
    else:
        print(f"[WARNING] Missing mask for {filename}")

# === Create folds based on WSI groups ===
wsis = sorted(groups.keys())
random.seed(seed)
random.shuffle(wsis)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(wsis)):
    fold_train = []
    fold_val = []
    for i in train_idx:
        fold_train.extend(groups[wsis[i]])
    for i in val_idx:
        fold_val.extend(groups[wsis[i]])
    
    train_out = os.path.join(output_dir, f"train_{fold_idx}.lst")
    val_out = os.path.join(output_dir, f"val_{fold_idx}.lst")

    def write_lst(pairs, out_path):
        with open(out_path, 'w') as f:
            for img, mask in pairs:
                f.write(f"{img} {mask}\n")
        print(f"[FOLD {fold_idx}] Wrote {len(pairs)} to {out_path}")

    write_lst(fold_train, train_out)
    write_lst(fold_val, val_out)
