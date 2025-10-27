"""
This file contains the data loader for loading and preprocessing data.
It is used in train.py and predict.py.  (No skimage dependency.)
"""

import os, glob
from typing import List, Tuple, Optional, Sequence
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ------------------------------
# Utilities
# ------------------------------

def to_channels(arr: np.ndarray, num_classes: int = 5, dtype=np.uint8) -> np.ndarray:
    """
    One-hot encode a label map into 'num_classes' channels.
    Any label >= num_classes will be clipped.
    arr: (H,W) integers
    returns: (H,W,C)
    """
    h, w = arr.shape
    res = np.zeros((h, w, num_classes), dtype=dtype)
    lab = np.clip(arr.astype(np.int64), 0, num_classes - 1)
    res[np.arange(h)[:, None], np.arange(w)[None, :], lab] = 1
    return res

def _zscore(img: np.ndarray, clip: Optional[float] = 5.0) -> np.ndarray:
    mu, sd = img.mean(), img.std() + 1e-8
    out = (img - mu) / sd
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out

def _resize2d_torch(img2d: np.ndarray, out_hw: Tuple[int,int], is_label: bool) -> np.ndarray:
    """
    Resize a 2D numpy array via torch.interpolate.
    - Images: bilinear + align_corners=False
    - Labels: nearest
    Keeps values in original dtype range (uses float32 internally).
    """
    H, W = img2d.shape
    t = torch.from_numpy(img2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    mode = "nearest" if is_label else "bilinear"
    t_res = F.interpolate(t, size=out_hw, mode=mode, align_corners=False if mode=="bilinear" else None)
    out = t_res.squeeze(0).squeeze(0).cpu().numpy()
    return out

# ------------------------------
# Loaders (single-list and paired)
# ------------------------------

def load_data_2D(
    imageNames: Sequence[str],
    normImage: bool = False,
    categorical: bool = False,
    dtype=np.float32,
    getAffines: bool = False,
    early_stop: bool = False,
    out_size: Tuple[int,int] = (256, 128),
    num_classes: int = 5
):
    """
    Load 2D NIfTI images into a preallocated numpy array (no skimage).
    If 'categorical' is True, one-hot encode with 'num_classes' channels.

    Returns:
        images  -> (N,H,W) or (N,H,W,C)
        [affines] -> list of affines if getAffines=True
    """
    affines = []

    # probe first case to preallocate
    first = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if first.ndim == 3:
        first = first[:, :, 0]  # HipMRI sometimes has a dummy 3rd dim
    first = _resize2d_torch(first, out_size, is_label=categorical)

    if categorical:
        first = to_channels(first.astype(np.int64), num_classes=num_classes, dtype=dtype)
        rows, cols, channels = first.shape
        images = np.zeros((len(imageNames), rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first.shape
        images = np.zeros((len(imageNames), rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, desc="Loading 2D")):
        ni = nib.load(inName)
        arr = ni.get_fdata(caching="unchanged")
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        arr = _resize2d_torch(arr, (rows, cols), is_label=categorical).astype(dtype)

        if normImage and not categorical:
            arr = _zscore(arr)

        if categorical:
            arr = to_channels(arr.astype(np.int64), num_classes=num_classes, dtype=dtype)
            images[i, ...] = arr
        else:
            images[i, ...] = arr

        affines.append(ni.affine)
        if early_stop and i > 20:
            break

    return (images, affines) if getAffines else images


def load_paired_2D(
    img_paths: Sequence[str],
    lab_paths: Sequence[str],
    normImage: bool = True,
    out_size: Tuple[int,int] = (256, 128),
    num_classes: int = 2,
    dtype_img=np.float32,
    dtype_lab=np.uint8
):
    """
    Load paired image+label NIfTIs into aligned 2D arrays (no skimage).

    Returns:
        X -> (N,H,W) float32
        Y -> (N,H,W,C) one-hot uint8 (C=num_classes)
    """
    assert len(img_paths) == len(lab_paths), "img_paths and lab_paths must align"
    X, Y = [], []

    for ip, lp in tqdm(list(zip(img_paths, lab_paths)), total=len(img_paths), desc="Loading paired 2D"):
        ni = nib.load(ip); li = nib.load(lp)

        img = ni.get_fdata(caching="unchanged")
        if img.ndim == 3: img = img[:, :, 0]
        lab = li.get_fdata(caching="unchanged")
        if lab.ndim == 3: lab = lab[:, :, 0]

        img = _resize2d_torch(img, out_size, is_label=False).astype(dtype_img)
        lab = _resize2d_torch(lab, out_size, is_label=True).astype(np.int64)

        if normImage:
            img = _zscore(img)

        X.append(img)
        Y.append(to_channels(lab, num_classes=num_classes, dtype=dtype_lab))

    X = np.stack(X, axis=0)  # (N,H,W)
    Y = np.stack(Y, axis=0)  # (N,H,W,C)
    return X, Y

# ------------------------------
# Discovery helpers
# ------------------------------

def discover_cases_flat(folder: str, pattern: str = "*.nii*") -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))

def discover_paired(
    root: str,
    image_name: str = "t2.nii.gz",
    label_name: str = "label.nii.gz"
) -> Tuple[List[str], List[str]]:
    imgs, labs = [], []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d): 
            continue
        ip = os.path.join(d, image_name)
        lp = os.path.join(d, label_name)
        if os.path.exists(ip) and os.path.exists(lp):
            imgs.append(ip); labs.append(lp)
    return imgs, labs

# ------------------------------
# Visualization
# ------------------------------

def visualize_example(
    img: np.ndarray,
    lab_oh: Optional[np.ndarray] = None,
    title: str = "Sample (H×W)",
    alpha: float = 0.4
):
    """
    Show a single 2D image and (optional) one-hot label overlay.
    """
    plt.figure(figsize=(8, 4))
    if lab_oh is None:
        plt.imshow(img, cmap="gray"); plt.title(title); plt.axis("off")
    else:
        lab = np.argmax(lab_oh, axis=-1)
        plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray"); plt.title(title); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(img, cmap="gray"); plt.imshow(lab, alpha=alpha, interpolation="nearest")
        plt.title("Overlay"); plt.axis("off")
    plt.tight_layout(); plt.show()

# ------------------------------
# Minimal CLI demo
# ------------------------------

if __name__ == "__main__":
    demo_root = "/path/to/HipMRI"  # <-- change me
    imgs, labs = discover_paired(demo_root, image_name="t2.nii.gz", label_name="label.nii.gz")
    if len(imgs) == 0:
        print("No cases found in demo_root; please update the path.")
    else:
        X, Y = load_paired_2D(imgs[:8], labs[:8], normImage=True, out_size=(256,128), num_classes=2)
        print("X:", X.shape, X.dtype, "Y:", Y.shape, Y.dtype)
        visualize_example(X[0], Y[0], title="T2 slice with mask")
