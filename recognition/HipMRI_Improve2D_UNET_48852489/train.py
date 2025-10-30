# train.py
import os, argparse, random, math, time
from typing import List, Tuple
from dataset import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_paired_2D  # uses (H,W)=(256,128), no skimage
from module import ImprovedUNet2D, DiceLoss, dice_score


# ----------------------- utils -----------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------- training -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--image_subdir", type=str, default="original")
    p.add_argument("--label_subdir", type=str, default="label")
    p.add_argument("--image_name", type=str, default="t2.nii.gz")
    p.add_argument("--label_name", type=str, default="label.nii.gz")

    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=6)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_path", type=str, default="improved_unet2d.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> device: {device}")


    #Load the dataset for train and validate
    imgs_train, lab_train = discover_paired(args.data_root, "test" )
    imgs_val, lab_val = discover_paired(args.data_root, "validate" )

    # ---- load arrays (NumPy) and convert to tensors ----
    out_size = (args.height, args.width)

    Xtr_np, Ytr_np = load_paired_2D(
        imgs_train, lab_train,
        normImage=True,
        out_size=out_size,
        num_classes=args.num_classes,
        augment=args.augment,  # flips on train
        flip_p=0.5
    )
    
    Xval_np, Yval_np = load_paired_2D(
        imgs_val, lab_val,
        normImage=True,
        out_size=out_size,
        num_classes=args.num_classes,
        augment=False
    )

    # X: (N,H,W) -> (N,1,H,W), float32
    Xtr = torch.from_numpy(Xtr_np).unsqueeze(1).float()
    Xval = torch.from_numpy(Xval_np).unsqueeze(1).float()
    # Y one-hot -> indices: (N,H,W,C) -> (N,H,W), long
    Ytr = torch.from_numpy(Ytr_np).argmax(dim=-1).long()
    Yval = torch.from_numpy(Yval_np).argmax(dim=-1).long()

    tr_ds  = TensorDataset(Xtr, Ytr)
    val_ds = TensorDataset(Xval, Yval)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model / loss / optim ----
    net = ImprovedUNet2D(in_channels=1, num_classes=args.num_classes, base=32, deep_supervision=True).to(device)
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        tr_loss, n_batches = 0.0, 0

        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = net(xb)                     # (B,K,H,W)
                loss = 0.5 * ce(logits, yb) + 0.5 * dice(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
            n_batches += 1

        # ---- validation ----
        net.eval()
        val_loss, dices, m = 0.0, [], 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = net(xb)
                loss = 0.5 * ce(logits, yb) + 0.5 * dice(logits, yb)
                val_loss += loss.item()
                dices.append(dice_score(logits, yb, ignore_bg=False))
                m += 1

        epoch_tr = tr_loss / max(n_batches, 1)
        epoch_val = val_loss / max(m, 1)
        epoch_dice = float(np.mean(dices)) if dices else 0.0
        dt = time.time() - t0
        print(f"[{epoch:03d}] train {epoch_tr:.4f} | val {epoch_val:.4f} | dice {epoch_dice:.4f} | {dt:.1f}s")

        # save best by dice
        if epoch_dice > best_val:
            best_val = epoch_dice
            torch.save({
                "model_state": net.state_dict(),
                "epoch": epoch,
                "dice": epoch_dice,
                "height": args.height,
                "width": args.width,
                "num_classes": args.num_classes,
            }, args.save_path)
            print(f"  ✓ saved best to {args.save_path} (dice={epoch_dice:.4f})")


if __name__ == "__main__":
    main()
