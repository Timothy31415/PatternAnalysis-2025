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
def dice_per_class_np(preds, gts, num_classes):
    """
    preds: (N,H,W) predicted integer masks
    gts:   (N,H,W) ground truth integer masks
    returns: np.array of per-class dice
    """
    dice_scores = []
    for c in range(num_classes):
        p = (preds == c).astype(np.uint8)
        g = (gts == c).astype(np.uint8)
        inter = np.sum(p * g)
        union = np.sum(p) + np.sum(g)
        d = (2. * inter) / (union + 1e-6)
        dice_scores.append(d)
    return np.array(dice_scores)

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

    ap.add_argument("--save_fig", type=str, default=None,
                    help="optional path to save the 1x3 figure (e.g., fig/pred_case001.png)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval_all",action="store_true")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> device: {device}")

    img_test, lab_test = discover_paired(args.data_root, "test")

    if args.eval_all:
        # 1) load paired arrays (images z-scored, labels nearest)
        X_np, Y_oh = load_paired_2D(
            img_test, lab_test,
            normImage=True,
            out_size=(args.height, args.width),
            num_classes=args.num_classes,
            augment=False
        )
        # 2)ground truth as class ids (N,H,W)
        GT_np = np.argmax(Y_oh, axis=-1)

        # 3) tensor batch (N,1,H,W)
        X = torch.from_numpy(X_np).unsqueeze(1).float().to(device)

        # 4) load model once
        ckpt = torch.load(args.model_path, map_location=device)
        num_classes = ckpt.get("num_classes", args.num_classes)
        net = ImprovedUNet2D(in_channels=1, num_classes=num_classes, base=32, deep_supervision=True).to(device)
        net.load_state_dict(ckpt["model_state"])
        net.eval()

        # 5) batched inference
        preds_chunks = []
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=args.amp):
            for i in range(0, X.shape[0], 4):
                xb = X[i:i+4]
                logits = net(xb)                              # (B,K,H,W)
                pred = torch.softmax(logits, dim=1).argmax(1) # (B,H,W)
                preds_chunks.append(pred.cpu().numpy())
        PRED_np = np.concatenate(preds_chunks, axis=0)       # (N,H,W)

        # 6) per-class Dice
        dpc = dice_per_class_np(PRED_np, GT_np, num_classes=args.num_classes)
        print("\n=== Per-class Dice on test set ===")
        for k, s in enumerate(dpc):
            print(f"Class {k}: {s:.4f}")
        print(f"Mean Dice: {dpc.mean():.4f}")

        # if you only want eval_all, stop here:
        return
    
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
