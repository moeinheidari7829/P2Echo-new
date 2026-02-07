"""
Multi-task echocardiography dataset for P2Echo-new.

Ported from original P2Echo/sam3_echo/data/dataset.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


PLANE_TO_ID: Dict[str, int] = {
    "2CH": 0,
    "3CH": 1,
    "4CH": 2,
    "PSAX": 3,
}


@dataclass
class EchoSample:
    """Dataclass representing a single echo sample."""
    image: torch.Tensor
    mask: torch.Tensor
    plane_label: torch.Tensor
    plane: str
    dataset: str
    stem: str
    image_path: str
    mask_path: str


def normalize_image(img: torch.Tensor, eps: float = 1e-8, use_foreground_only: bool = True) -> torch.Tensor:
    """
    Z-score normalize an image tensor.

    If ``use_foreground_only`` is True, statistics are computed only on foreground pixels
    (assumed to be non-zero), and background pixels are kept at 0. This matches the common
    nnU-Net setting where normalization is computed inside the valid (non-zero) region.

    Args:
        img: Image tensor, typically ``[C, H, W]`` in ``[0, 1]`` (or ``[H, W]``).
        eps: Minimum std for numerical stability.
        use_foreground_only: If True, compute mean/std only on non-zero pixels.

    Returns:
        Normalized image tensor (float32).
    """
    img = img.to(torch.float32)

    if img.ndim == 2:
        if use_foreground_only:
            mask = img.ne(0)
            if mask.any():
                mean = img[mask].mean()
                std = img[mask].std(unbiased=False).clamp_min(eps)
                out = img.clone()
                out[mask] = (out[mask] - mean) / std
                out[~mask] = 0
                return out

        mean = img.mean()
        std = img.std(unbiased=False).clamp_min(eps)
        return (img - mean) / std

    if img.ndim != 3:
        raise ValueError(f"Expected img with shape [C,H,W] or [H,W], got {tuple(img.shape)}")

    if use_foreground_only:
        mask2d = img.ne(0).any(dim=0)  # foreground where any channel is non-zero
        if mask2d.any():
            mask = mask2d.unsqueeze(0).expand_as(img)
            mean = img[mask].mean()
            std = img[mask].std(unbiased=False).clamp_min(eps)
            out = img.clone()
            out[mask] = (out[mask] - mean) / std
            out[~mask] = 0
            return out

    mean = img.mean()
    std = img.std(unbiased=False).clamp_min(eps)
    return (img - mean) / std



class MultiTaskEchoDataset(Dataset):
    """
    Multi-task echocardiography dataset.

    Expects a pandas.DataFrame with columns:
      - image_path
      - mask_path
      - plane
      - dataset
      - stem
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        resize: Optional[Tuple[int, int]] = (256, 256),
        to_rgb: bool = True,
        image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.df = frame.reset_index(drop=True)
        self.resize = resize
        self.to_rgb = to_rgb
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        img = img.convert("RGB") if self.to_rgb else img.convert("L")
        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)
        return img

    def _load_mask(self, path: str) -> Image.Image:
        m = Image.open(path).convert("L")
        if self.resize is not None:
            m = m.resize(self.resize, Image.NEAREST)
        return m

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, copy=True)
        if img.mode == "RGB":
            arr = arr.transpose(2, 0, 1)
            t = torch.from_numpy(arr).float() / 255.0
        else:
            t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
        return t

    def _mask_to_tensor(self, mask_img: Image.Image) -> torch.Tensor:
        arr = np.array(mask_img, copy=True)
        return torch.from_numpy(arr).long()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img = self._load_image(row.image_path)
        msk = self._load_mask(row.mask_path)

        img_t = self._pil_to_tensor(img)   # [C,H,W], float in [0,1]
        msk_t = self._mask_to_tensor(msk)  # [H,W], long

        # Apply optional transform (e.g., normalization)
        if self.image_transform is not None:
            img_t = self.image_transform(img_t)

        plane_str = str(row.plane).upper()
        plane_id = PLANE_TO_ID.get(plane_str, -1)
        plane_label = torch.tensor(plane_id, dtype=torch.long)

        return {
            "image": img_t,
            "mask": msk_t,
            "plane_label": plane_label,
            "plane": plane_str,
            "dataset": row.dataset,
            "stem": row.stem,
            "image_path": row.image_path,
            "mask_path": row.mask_path,
        }


# =============================================================================
# Splits loading
# =============================================================================

def _fix_path(path_str: str, data_root: str) -> str:
    """
    Fix relative paths from JSON to use the data_root.
    """
    if path_str.startswith("./data/"):
        relative_part = path_str[7:]
        return os.path.join(data_root, relative_part)
    elif path_str.startswith("data/"):
        relative_part = path_str[5:]
        return os.path.join(data_root, relative_part)
    elif not os.path.isabs(path_str):
        return os.path.join(data_root, path_str)
    return path_str


def load_splits_json(
    path: str,
    data_root: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load train/val/test(+external) splits from JSON and return as DataFrames.

    JSON format:
      {
        "train":    [ {row1}, {row2}, ... ],
        "val":      [ ... ],
        "test":     [ ... ],
        "external": [ ... ]   # optional
      }

    Each row must contain:
      - image_path, mask_path, plane, dataset, stem
    """
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)

    if data_root is None:
        data_root = os.environ.get("SAM3_DATA_ROOT", "")

    def fix_paths_in_list(items: List[Dict]) -> List[Dict]:
        if not data_root:
            return items
        for item in items:
            if "image_path" in item:
                item["image_path"] = _fix_path(item["image_path"], data_root)
            if "mask_path" in item:
                item["mask_path"] = _fix_path(item["mask_path"], data_root)
        return items

    train_df = pd.DataFrame(fix_paths_in_list(data["train"]))
    val_df = pd.DataFrame(fix_paths_in_list(data["val"]))
    test_df = pd.DataFrame(fix_paths_in_list(data["test"]))

    external_df = None
    if "external" in data and len(data["external"]) > 0:
        external_df = pd.DataFrame(fix_paths_in_list(data["external"]))

    return train_df, val_df, test_df, external_df


# =============================================================================
# DataLoader factory
# =============================================================================

def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    external_df: Optional[pd.DataFrame] = None,
    batch_size: int = 8,
    num_workers: int = 2,
    resize: Tuple[int, int] = (256, 256),
    to_rgb: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for train/val/test/external from DataFrames.
    
    Returns:
        (train_loader, val_loader, test_loader, external_loader)
    """
    img_transform = normalize_image

    train_set = MultiTaskEchoDataset(
        train_df,
        resize=resize,
        to_rgb=to_rgb,
        image_transform=img_transform,
    )
    val_set = MultiTaskEchoDataset(
        val_df,
        resize=resize,
        to_rgb=to_rgb,
        image_transform=img_transform,
    )

    test_set = MultiTaskEchoDataset(
        test_df,
        resize=resize,
        to_rgb=to_rgb,
        image_transform=img_transform,
    ) if test_df is not None else None

    external_set = MultiTaskEchoDataset(
        external_df,
        resize=resize,
        to_rgb=to_rgb,
        image_transform=img_transform,
    ) if external_df is not None else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    external_loader = None
    if external_set is not None:
        external_loader = DataLoader(
            external_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    return train_loader, val_loader, test_loader, external_loader
