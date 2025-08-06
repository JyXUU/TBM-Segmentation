import os
import subprocess
import argparse
from datetime import datetime


def run_command(cmd):
    full_cmd = f"PYTHONPATH=$(pwd):$(pwd)/lib {cmd}" # prepend PYTHONPATH
    print(f"\n[Running] {full_cmd}")
    subprocess.run(full_cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run WSSAL active learning iterations")
    parser.add_argument('--iterations', type=int, default=5, help='Number of active learning iterations')
    parser.add_argument('--select_k', type=int, default=100, help='Number of patches to select each round')
    parser.add_argument('--base_cfg', type=str, required=True, help='Base YAML config for training')
    parser.add_argument('--image_dir', type=str, required=True, help='Unlabeled image patch directory')
    parser.add_argument('--output_root', type=str, required=True, help='Root output dir for logs, models, lists')
    parser.add_argument('--initial_sup_lst', type=str, required=True, help='Path to initial labeled train.lst')
    args = parser.parse_args()

    pseudo_mask_dir = os.path.join(args.output_root, 'pseudo_labels')
    mc_output_path = os.path.join(args.output_root, 'mc_output.pkl')
    to_annotate_txt = os.path.join(args.output_root, 'to_annotate.txt')
    labeled_img_dir = os.path.join(args.output_root, 'labeled', 'images')
    labeled_mask_dir = os.path.join(args.output_root, 'labeled', 'masks')
    train_lst_path = os.path.join(args.output_root, 'train_wssal.lst')
    os.makedirs(args.output_root, exist_ok=True)

    # Copy initial sup list to working train list
    if not os.path.exists(train_lst_path):
        os.system(f"cp {args.initial_sup_lst} {train_lst_path}")

    for i in range(args.iterations):
        print(f"\n===== WSSAL Iteration {i+1}/{args.iterations} =====")
        iter_output_dir = os.path.join(args.output_root, f"wssal_iter_{i+1}")
        os.makedirs(iter_output_dir, exist_ok=True)

        # Step 1: MC-Dropout Inference
        run_command(
            f"python wssal_pipeline/inference_mc_dropout.py \
            --cfg {args.base_cfg} \
            --image_dir {args.image_dir} \
            --output_dir {pseudo_mask_dir} \
            --mc_output_pkl {mc_output_path}"
        )


        # Step 2: Active Selection
        run_command(
            f"python wssal_pipeline/active_selection.py \
             --pkl {mc_output_path} \
             --k {args.select_k} \
             --out_txt {to_annotate_txt}"
        )

        # Step 3: Update Labeled Pool
        run_command(
            f"python wssal_pipeline/update_labeled_pool.py \
            --txt wssal_output/round_1/to_annotate.txt \
            --src_img wssal_pipeline/Data/patches/unlabeled \
            --src_mask wssal_output/round_1/pseudo_labels \
            --dst_img wssal_output/round_1/labeled/images \
            --dst_mask wssal_output/round_1/labeled/masks \
            --train_lst wssal_output/round_1/train_wssal.lst"
        )

        # Step 4: Train Student with Teacher
        run_command(
            f"python wssal_pipeline/train_wssal_loop.py \
             --cfg {args.base_cfg} \
             --sup_lst {train_lst_path} \
             --unsup_lst {train_lst_path} \
             --output_dir {iter_output_dir}"
        )

    print("\nWSSAL Loop Completed")


if __name__ == '__main__':
    main()
    
"""
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
PYTHONPATH=$(pwd):$(pwd)/lib \
python wssal_pipeline/run_wssal_loop.py \
  --iterations 5 \
  --select_k 100 \
  --base_cfg experiments/tbm/seg_hrnet_ocr_tbm.yaml \
  --image_dir wssal_pipeline/Data/patches/unlabeled \
  --output_root wssal_output/round_1 \
  --initial_sup_lst wssal_pipeline/lst/train_init.lst
"""