import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

from lib.config import config as base_config, update_config
from wssal_pipeline.models.hrnet_wssal import HRNetWSSAL

import pickle

# ====== Configurable parameters ======
MC_STEPS = 5
THRESHOLD = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Dataset loader (expects image paths only) ======
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, os.path.basename(self.image_paths[idx])

# ====== Monte Carlo Dropout Prediction ======
@torch.no_grad()
def mc_dropout_predict(model, dataloader, T=5):
    model.eval()
    model.to(DEVICE)

    results = []

    for images, names in tqdm(dataloader):
        images = images.to(DEVICE)
        B, _, H, W = images.shape

        all_probs = []
        for _ in range(T):
            logits = model(images, mc_dropout=True)
            probs = logits  # Already probabilities
            all_probs.append(probs)

        probs_stack = torch.stack(all_probs, dim=0)       # [T, B, C, H, W]
        mean_probs = probs_stack.mean(dim=0)              # [B, C, H, W]
        entropy_map = -mean_probs * torch.log(mean_probs + 1e-6) - (1 - mean_probs) * torch.log(1 - mean_probs + 1e-6)
        entropy_map = entropy_map.sum(dim=1)              # [B, H, W]

        for i in range(B):
            results.append({
                'name': names[i],
                'prob': mean_probs[i, 0].cpu().numpy(),
                'entropy': entropy_map[i].cpu().numpy()
            })

    return results

# ====== Save binary pseudo-masks over threshold ======
def save_pseudo_labels(results, out_dir, threshold=0.9):
    os.makedirs(out_dir, exist_ok=True)
    for res in results:
        prob_map = res['prob']
        mask = (prob_map > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)

        # Resize to 256×256 using nearest neighbor interpolation
        mask_resized = mask_img.resize((256, 256), resample=Image.NEAREST)

        save_path = os.path.join(out_dir, res['name'].replace('.png', '_pseudo.png'))
        mask_resized.save(save_path)

# ====== Main routine ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to HRNet YAML config")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to unlabeled image patches")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save pseudo-label masks")
    parser.add_argument("--mc_output_pkl", type=str, default="mc_output.pkl", help="Where to save the .pkl with prob & entropy maps")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options")
    args = parser.parse_args()

    config = base_config
    update_config(config, args)

    model = HRNetWSSAL(config, use_ocr=True)
    
    if os.path.isfile(config.TEST.MODEL_FILE):
        print(f"[INFO] Loading model weights from {config.TEST.MODEL_FILE}")
        state_dict = torch.load(config.TEST.MODEL_FILE, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"[ERROR] Cannot find model weights at {config.TEST.MODEL_FILE}")


    image_paths = [os.path.join(args.image_dir, f)
                   for f in os.listdir(args.image_dir)
                   if f.endswith('.png') or f.endswith('.jpg')]

    dataset = PatchDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    results = mc_dropout_predict(model, dataloader, T=MC_STEPS)

    # Save binary pseudo-masks
    save_pseudo_labels(results, args.output_dir, threshold=THRESHOLD)

    # Save prob + entropy maps for later filtering
    with open(args.mc_output_pkl, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
    
"""
python wssal_pipeline/inference_mc_dropout.py \
  --cfg experiments/tbm/seg_hrnet_ocr_tbm.yaml \
  --image_dir wssal_pipeline/Data/patches/unlabeled \
  --output_dir wssal_output/round_1/pseudo_labels \
  --mc_output_pkl wssal_output/round_1/mc_output.pkl \
  --opts TEST.MODEL_FILE /blue/pinaki.sarder/jingyixu/TBM/HRNet-Semantic-Segmentation/output/tbm_fold_4/tbm/seg_hrnet_ocr_tbm/best.pth
"""
