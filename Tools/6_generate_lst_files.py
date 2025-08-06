import os
import glob

# === CONFIG ===
root = "/blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/"

train_image_dir = os.path.join(root, "Data/tbm/train_split/images")
train_mask_dir = os.path.join(root, "Data/tbm/train_split/masks")

val_image_dir = os.path.join(root, "Data/tbm/val/images")
val_mask_dir = os.path.join(root, "Data/tbm/val/masks")

output_train_lst = os.path.join(root, "lst/tbm/train.lst")
output_val_lst = os.path.join(root, "lst/tbm/val.lst")

# === Helper function to generate pairs ===
def generate_pairs(image_dir, mask_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print(f"[INFO] Found {len(image_paths)} images in {image_dir}")
    pairs = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, filename)
        if os.path.exists(mask_path):
            rel_img = os.path.relpath(img_path, root)
            rel_mask = os.path.relpath(mask_path, root)
            pairs.append((rel_img, rel_mask))
        else:
            print(f"[WARNING] Mask not found for {filename}")
    return pairs

# === Write .lst files ===
def write_lst(pairs, output_path):
    with open(output_path, 'w') as f:
        for img, mask in pairs:
            f.write(f"{img} {mask}\n")
    print(f"[INFO] Wrote {len(pairs)} entries to {output_path}")

# === Generate and Save ===
train_pairs = generate_pairs(train_image_dir, train_mask_dir)
val_pairs = generate_pairs(val_image_dir, val_mask_dir)

write_lst(train_pairs, output_train_lst)
write_lst(val_pairs, output_val_lst)
