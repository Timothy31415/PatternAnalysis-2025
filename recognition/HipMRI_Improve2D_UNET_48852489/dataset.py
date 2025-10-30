"""
This file contains the data loader for loading and preprocessing data.
It is used in train.py and predict.py.  (No skimage dependency.)
"""

import os, glob, random
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

def to_channels(arr: np.ndarray, num_classes: int = 6, dtype=np.uint8) -> np.ndarray:
    """One-hot encode a label map into 'num_classes' channels."""
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
    """Resize 2D numpy array using torch.interpolate."""
    H, W = img2d.shape
    t = torch.from_numpy(img2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mode = "nearest" if is_label else "bilinear"
    t_res = F.interpolate(t, size=out_hw, mode=mode, align_corners=False if mode=="bilinear" else None)
    return t_res.squeeze(0).squeeze(0).cpu().numpy()

def _random_flip(img: np.ndarray, mask: Optional[np.ndarray] = None, p: float = 0.5):
    """Apply random horizontal and vertical flips (same to mask if given)."""
    # Horizontal (left-right)
    if random.random() < p:
        img = np.flip(img, axis=1)
        if mask is not None:
            mask = np.flip(mask, axis=1)
    # Vertical (top-bottom)
    if random.random() < p:
        img = np.flip(img, axis=0)
        if mask is not None:
            mask = np.flip(mask, axis=0)
    return img, mask

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
    num_classes: int = 6,
    augment: bool = False,
    flip_p: float = 0.5
):
    """
    Load 2D NIfTI images. If augment=True, applies random flips per slice.
    Returns (N,H,W) or (N,H,W,C)
    """
    affines = []
    first = nib.load(imageNames[0]).get_fdata(caching="unchanged")
    if first.ndim == 3:
        first = first[:, :, 0]
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
        if augment:
            arr, _ = _random_flip(arr, None, p=flip_p)
        if categorical:
            arr = to_channels(arr.astype(np.int64), num_classes=num_classes, dtype=dtype)
        images[i, ...] = arr
        affines.append(ni.affine)
        if early_stop and i > 20:
            break

    return (images, affines) if getAffines else images

def load_one_case(img_path: str, lab_path: str, out_size=(256,128), num_classes=6):
    """
    Load single image+label as numpy arrays with your preprocessing:
      - take first slice if 3D
      - resize via torch interpolate (bilinear for image / nearest for label)
      - z-score normalize image
    Returns: img_hw (float32), lab_hw (int64)
    """
    ni, li = nib.load(img_path), nib.load(lab_path)
    img, lab = ni.get_fdata(caching="unchanged"), li.get_fdata(caching="unchanged")
    if img.ndim == 3: img = img[:, :, 0]
    if lab.ndim == 3: lab = lab[:, :, 0]

    img = _resize2d_torch(img, out_size, is_label=False).astype(np.float32)
    lab = _resize2d_torch(lab, out_size, is_label=True).astype(np.int64)
    img = _zscore(img)
    # clip label values to [0, num_classes-1] just in case
    lab = np.clip(lab, 0, num_classes - 1).astype(np.int64)
    return img, lab, ni.affine


def load_paired_2D(
    img_paths: Sequence[str],
    lab_paths: Sequence[str],
    normImage: bool = True,
    out_size: Tuple[int,int] = (256, 128),
    num_classes: int = 6,
    dtype_img=np.float32,
    dtype_lab=np.uint8,
    augment: bool = False,
    flip_p: float = 0.5
):
    """
    Load paired image+label NIfTIs.
    If augment=True, applies random vertical/horizontal flips to both.
    """
    assert len(img_paths) == len(lab_paths)
    X, Y = [], []

    for ip, lp in tqdm(list(zip(img_paths, lab_paths)), total=len(img_paths), desc="Loading paired 2D"):
        ni, li = nib.load(ip), nib.load(lp)
        img, lab = ni.get_fdata(caching="unchanged"), li.get_fdata(caching="unchanged")
        if img.ndim == 3: img = img[:, :, 0]
        if lab.ndim == 3: lab = lab[:, :, 0]
        img = _resize2d_torch(img, out_size, is_label=False).astype(dtype_img)
        lab = _resize2d_torch(lab, out_size, is_label=True).astype(np.int64)
        if normImage:
            img = _zscore(img)
        if augment:
            img, lab = _random_flip(img, lab, p=flip_p)
        X.append(img)
        Y.append(to_channels(lab, num_classes=num_classes, dtype=dtype_lab))

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y

# ------------------------------
# Discovery helpers
# ------------------------------

def discover_cases_flat(folder: str, pattern: str = "*.nii*") -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))

def subfolder_list( root:str):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            paths.append(full_path)
    return paths



def discover_paired(root: str, type: str) -> Tuple[List[str], List[str]]:
    if type == "train":
        path_img = os.path.join(root,"keras_slices_train" )
        path_lab = os.path.join(root,"keras_slices_seg_train" )

    if type == "validate":
        path_img = os.path.join(root,"keras_slices_validate" )
        path_lab = os.path.join(root,"keras_slices_seg_validate" )

    if type == "test":
        print("dddd")
        path_img = os.path.join(root,"keras_slices_test" )
        path_lab = os.path.join(root,"keras_slices_seg_test" )

    print(path_img)
    imgs = subfolder_list(path_img)
    labs = subfolder_list(path_lab)
    return imgs, labs

# ------------------------------
# Visualization
# ------------------------------

def visualize_example(img: np.ndarray, lab_oh: Optional[np.ndarray] = None, title="Sample (H×W)", alpha=0.4):
    plt.figure(figsize=(8, 4))
    if lab_oh is None:
        plt.imshow(_random_flip(img)[0], cmap="gray"); plt.title(title); plt.axis("off")
    else:
        lab = np.argmax(lab_oh, axis=-1)
        plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title(title); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(img, cmap="gray"); plt.imshow(lab, alpha=alpha, interpolation="nearest")
        plt.title("Overlay"); plt.axis("off")
    plt.tight_layout(); plt.show()

# ------------------------------
# Minimal CLI demo
# ------------------------------

if __name__ == "__main__":
    demo_root = os.getcwd()
    data_root = os.path.join(demo_root,"keras_slices_data")
    imgs, labs = discover_paired(data_root, "train")
    if len(imgs) == 0:
        print("No cases found in demo_root; please update the path.")
    else:
        X, Y = load_paired_2D(imgs[:8], labs[:8], normImage=True, out_size=(256,128), num_classes=6, augment=True)
        print("X:", X.shape, X.dtype, "Y:", Y.shape, Y.dtype)
        visualize_example(X[0], Y[0], title="Augmented slice")
