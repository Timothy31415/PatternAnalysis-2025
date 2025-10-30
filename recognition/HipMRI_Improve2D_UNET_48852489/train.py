# train.py
import os, argparse, random, math, time
from typing import List, Tuple
from dataset import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt 

from dataset import load_paired_2D  # uses (H,W)=(256,128), no skimage
from module import ImprovedUNet2D, DiceLoss, dice_score

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
    p.add_argument("--verbose", action ="store_true")
    p.add_argument("--eval_all",action="store_true")

    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=6)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--augment", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_path", type=str, default="improved_unet2d.pt")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> device: {device}")


    #Load the dataset for train and validate
    imgs_train, lab_train = discover_paired(args.data_root, "train" )
    imgs_val, lab_val = discover_paired(args.data_root, "validate" )
    img_test, lab_test = discover_paired(args.data_root, "test")

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

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=0, pin_memory=False, persistent_workers=False)
    val_dl = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                   num_workers=0, pin_memory=False, persistent_workers=False)
    
    # ---- model / loss / optim ----
    net = ImprovedUNet2D(in_channels=1, num_classes=args.num_classes, base=32, deep_supervision=True).to(device)
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda',enabled=args.amp)

    best_val = -1.0
    dice_plot = [] #Plotting purpose
    tr_loss_plot = []
    val_loss_plot =[]
    for epoch in range(1, args.epochs + 1):
        print(f"Training: {epoch}")
        net.train()
        t0 = time.time()
        tr_loss, n_batches = 0.0, 0

        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=args.amp):
                logits = net(xb)       
                loss = 0.5 * ce(logits, yb) + 0.5 * dice(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
            n_batches += 1

        
        # ---- validation ----
        net.eval()
        val_loss, dices, m = 0.0, [], 0
        with torch.no_grad(), torch.amp.autocast('cuda',enabled=args.amp):
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = net(xb)
                loss = 0.5 * ce(logits, yb) + 0.5 * dice(logits, yb) 
                val_loss += loss.item()
                dices.append(dice_score(logits, yb, ignore_bg=False))
                m += 1

        epoch_tr = tr_loss / max(n_batches, 1)
        tr_loss_plot.append(epoch_tr)

        epoch_val = val_loss / max(m, 1)
        val_loss_plot.append(epoch_val)

        epoch_dice = float(np.mean(dices)) if dices else 0.0
        dt = time.time() - t0
        dice_plot.append(epoch_dice) #plotting purpose
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


    if args.verbose:
        plt.figure(figsize=(12, 4))
        plt.plot(range(1,args.epochs+1),dice_plot, 'r-' )
        plt.xlabel("Epochs No")
        plt.ylabel("Dice score (Validation)")
        plt.title("Epochs No vs Dice score" )
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(range(1,args.epochs+1),tr_loss_plot, 'r-', label = "Training loss" )
        plt.plot(range(1,args.epochs+1),val_loss_plot, 'b-', label = "Validation loss" )
        plt.legend(loc ='upper right')
        plt.xlabel("Epochs No")
        plt.ylabel("Loss")
        plt.title("Epochs No vs Training and validation loss" )
        plt.show()


if __name__ == "__main__":
    main()
