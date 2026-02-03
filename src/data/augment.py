"""
nnU-Net style data augmentation for P2Echo-new.

Ported from original P2Echo/voxtell/echo/augment.py.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class SafeGammaTransform:
    """
    Gamma correction that is safe for negative-valued inputs (won't create NaNs).

    batchgenerators' GammaTransform can produce NaNs if it ever sees negative values and a
    non-integer gamma. This custom transform shifts inputs to be non-negative before
    applying gamma, then shifts back.
    """

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        invert_image: bool = False,
        per_channel: bool = True,
        p_per_sample: float = 0.3,
    ) -> None:
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.invert_image = bool(invert_image)
        self.per_channel = bool(per_channel)
        self.p_per_sample = float(p_per_sample)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data = data_dict.get("data")
        if data is None:
            return data_dict

        data = data.astype(np.float32, copy=False)
        b, c = int(data.shape[0]), int(data.shape[1])

        for bi in range(b):
            if np.random.random() >= self.p_per_sample:
                continue

            if self.invert_image:
                data[bi] = -data[bi]

            if self.per_channel:
                for ci in range(c):
                    gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
                    x = data[bi, ci]
                    min_x = float(np.min(x))
                    x0 = x - min_x
                    max_x0 = float(np.max(x0))
                    if max_x0 > 0:
                        x01 = x0 / (max_x0 + 1e-8)
                        x01 = np.power(x01, gamma, dtype=np.float32)
                        data[bi, ci] = x01 * max_x0 + min_x
            else:
                gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
                x = data[bi]
                min_x = float(np.min(x))
                x0 = x - min_x
                max_x0 = float(np.max(x0))
                if max_x0 > 0:
                    x01 = x0 / (max_x0 + 1e-8)
                    x01 = np.power(x01, gamma, dtype=np.float32)
                    data[bi] = x01 * max_x0 + min_x

        data_dict["data"] = data
        return data_dict


def build_nnunet_2d_train_transforms(
    patch_size: Tuple[int, int],
    disable_intensity: bool = False,
) -> Any:
    """
    Build an nnU-Net-style 2D augmentation pipeline using batchgenerators.

    Args:
        patch_size: (H, W) patch size
        disable_intensity: If True, only spatial transforms are applied

    Returns:
        A callable (Compose) that can be applied to a dict with keys:
        - 'data': np.ndarray [B, C, H, W] float32
        - 'seg' : np.ndarray [B, 1, H, W] int/float
    """
    try:
        from batchgenerators.transforms.abstract_transforms import Compose
        from batchgenerators.transforms.color_transforms import (
            BrightnessMultiplicativeTransform,
            ContrastAugmentationTransform,
        )
        from batchgenerators.transforms.noise_transforms import (
            GaussianBlurTransform,
            GaussianNoiseTransform,
        )
        from batchgenerators.transforms.resample_transforms import (
            SimulateLowResolutionTransform,
        )
        from batchgenerators.transforms.spatial_transforms import SpatialTransform
    except Exception as e:
        raise ImportError(
            "batchgenerators is required for nnU-Net-style augmentation. "
            "Install it with: pip install batchgenerators"
        ) from e

    # nnU-Net-style defaults for 2D
    rot = (-15.0 / 180.0 * np.pi, 15.0 / 180.0 * np.pi)
    scale = (0.7, 1.4)

    transforms = [
        SpatialTransform(
            patch_size=patch_size,
            patch_center_dist_from_border=None,
            do_elastic_deform=False,
            alpha=(0.0, 0.0),
            sigma=(0.0, 0.0),
            do_rotation=True,
            angle_x=rot,
            angle_y=(0.0, 0.0),
            angle_z=(0.0, 0.0),
            do_scale=True,
            scale=scale,
            border_mode_data="constant",
            border_cval_data=0.0,
            order_data=3,
            border_mode_seg="constant",
            border_cval_seg=0,
            order_seg=0,
            random_crop=False,
            p_el_per_sample=0.0,
            p_scale_per_sample=0.2,
            p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
        ),
    ]

    if not bool(disable_intensity):
        transforms.extend([
            GaussianNoiseTransform(p_per_sample=0.1),
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
            ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.7, 1.3),
                p_per_sample=0.15,
            ),
            ContrastAugmentationTransform(
                contrast_range=(0.65, 1.5),
                p_per_sample=0.15,
            ),
            SafeGammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=False,
                per_channel=True,
                p_per_sample=0.3,
            ),
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1.0),
                per_channel=True,
                p_per_sample=0.25,
                order_downsample=0,
                order_upsample=3,
            ),
        ])

    return Compose(transforms)


def apply_batchgenerators_transforms(
    *,
    trf: Any,
    data: np.ndarray,
    seg: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Apply batchgenerators Compose to arrays.

    Args:
        trf: Transform pipeline (Compose)
        data: [B, C, H, W] float32 image data
        seg: [B, 1, H, W] int/float segmentation

    Returns:
        Dict with 'data' and 'seg' keys
    """
    if trf is None:
        return {"data": data, "seg": seg}
    out = trf(data=data, seg=seg)
    return {"data": out["data"], "seg": out["seg"]}
