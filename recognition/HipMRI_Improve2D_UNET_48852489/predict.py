# predict.py
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import random as rd
from dataset import *

from module import ImprovedUNet2D     # your model


def plot_triplet(img_hw: np.ndarray, pred_hw: np.ndarray, gt_hw: np.ndarray, case_id: str,
                 alpha_overlay: float = 0.65, save_fig: str | None = None):
    """
    Show 1×3: original (gray), prediction (colored), ground truth (colored).
    """
    # Pick a fixed colormap w/ distinct colors
    cmap = "tab20"  # works for up to ~20 classes

    plt.figure(figsize=(12, 4))

    # 1) Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_hw, cmap="gray")
    plt.title(f"Original ({case_id})")
    plt.axis("off")

    # 2) Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(pred_hw, cmap=cmap, interpolation="nearest")
    plt.title("Prediction")
    plt.axis("off")

    # 3) Ground truth
    plt.subplot(1, 3, 3)
    plt.imshow(gt_hw, cmap=cmap, interpolation="nearest")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"✓ saved figure to {save_fig}")
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--case_root",type = str, default = "1")
    ap.add_argument("--image_subdir", type=str, default="original")
    ap.add_argument("--label_subdir", type=str, default="label")
    ap.add_argument("--image_name", type=str, default="t2.nii.gz")
    ap.add_argument("--label_name", type=str, default="label.nii.gz")

    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--num_classes", type=int, default=6)

    ap.add_argument("--model_path", type=str, required=True)

    # how to choose the case
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--case_idx", type=int, default=0,
                   help="0-based index into discovered cases")
    g.add_argument("--case_id", type=str, default=None,
                   help="case folder name to select (overrides --case_idx)")

    ap.add_argument("--save_fig", type=str, default=None,
                    help="optional path to save the 1x3 figure (e.g., fig/pred_case001.png)")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> device: {device}")

    img_test, lab_test = discover_paired(args.data_root, "test")

    if args.case_root == "1":
        lab_idx = 1
        args.case_root = img_test[lab_idx]
    try:
        lab_idx = img_test.index(args.case_root)
    except:
        print("Cannot find this test file, ensure you inlcude the whole path")

    print(f"  image: {args.case_root}")
    print(f"  label: {lab_test[lab_idx]}")

    # 3) load one case arrays
    out_size = (args.height, args.width)
    img_hw, gt_hw, affine = load_one_case(args.case_root, lab_test[lab_idx], out_size, num_classes=args.num_classes)

    # 4) prepare model input
    x = torch.from_numpy(img_hw).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)

    # 5) load model + weights
    ckpt = torch.load(args.model_path, map_location=device)
    num_classes = ckpt.get("num_classes", args.num_classes)
    net = ImprovedUNet2D(in_channels=1, num_classes=num_classes, base=32, deep_supervision=True).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # 6) inference
    with torch.no_grad(), torch.amp.autocast('cuda',enabled=args.amp):
        logits = net(x)  # (1,K,H,W)
        pred = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy()  # (H,W)

    # 7) show figure (original / pred / ground truth)
    plot_triplet(img_hw, pred, gt_hw, lab_idx, save_fig=args.save_fig)


if __name__ == "__main__":
    main()
