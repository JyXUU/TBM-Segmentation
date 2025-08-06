import os
import shutil
import argparse

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def move_to_labeled_pool(txt_path, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, train_list_path):
    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    with open(txt_path, 'r') as f:
        names = [line.strip() for line in f.readlines()]

    with open(train_list_path, 'a') as train_file:
        for name in names:
            img_src = os.path.join(src_img_dir, name)
            mask_src = os.path.join(src_mask_dir, name.replace('.png', '_pseudo.png'))
            img_dst = os.path.join(dst_img_dir, name)
            mask_dst = os.path.join(dst_mask_dir, name)

            # Copy both image and corresponding pseudo-mask
            shutil.copyfile(img_src, img_dst)
            shutil.copyfile(mask_src, mask_dst)

            # Add to train.lst
            train_file.write(f"wssal_output/round_1/labeled/images/{name} labeled/masks/{name}\n")

    print(f"Added {len(names)} new samples to {train_list_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, required=True, help='Path to to_annotate.txt')
    parser.add_argument('--src_img', type=str, required=True)
    parser.add_argument('--src_mask', type=str, required=True)
    parser.add_argument('--dst_img', type=str, required=True)
    parser.add_argument('--dst_mask', type=str, required=True)
    parser.add_argument('--train_lst', type=str, required=True)
    args = parser.parse_args()

    move_to_labeled_pool(
        txt_path=args.txt,
        src_img_dir=args.src_img,
        src_mask_dir=args.src_mask,
        dst_img_dir=args.dst_img,
        dst_mask_dir=args.dst_mask,
        train_list_path=args.train_lst
    )

if __name__ == "__main__":
    main()

"""
python wssal_pipeline/update_labeled_pool.py \
            --txt wssal_output/round_1/to_annotate.txt \
            --src_img wssal_pipeline/Data/patches/unlabeled \
            --src_mask wssal_output/round_1/pseudo_labels \
            --dst_img wssal_output/round_1/labeled/images \
            --dst_mask wssal_output/round_1/labeled/masks \
            --train_lst wssal_output/round_1/train_wssal.lst
"""
