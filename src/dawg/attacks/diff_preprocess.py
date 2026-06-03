"""Differentiable reimplementation of MolmoWeb's image preprocessing.

Maps screenshot pixels -> the (n_crops, n_patches, 588) `images` tensor the
vision backbone consumes, in torch, so gradients flow pixel -> images for
pixel-space PGD. Only the pixel-carrying path is reproduced; the geometry
(tiling, token_pooling, masks, tokens) is pixel-INDEPENDENT and is taken from
the real preprocessor's output (reuse those fields verbatim in the attack).

Faithful port of olmo's `overlap-and-resize-c2` path
(olmo/preprocessing/{image_preprocessor,multicrop_preprocessor}.py).

Verified MolmoWeb-4B-Native config (checkpoints/.../config.yaml):
    crop_mode = overlap-and-resize-c2,  max_crops = 8,  overlap_margins = (4, 4),
    image_patch_size = 14,  resize/normalize = siglip,  normalize_on_gpu = True.
    base_image_input_size is read from the live preprocessor (expected 378).

### Normalization contract (read carefully)
`normalize_on_gpu=True` => the real preprocessor emits `images` as uint8
[0,255]; the model's `vision_backbone.normalize_image_tensor` then siglip-
normalizes. That function only divides by 255 when the input dtype is uint8;
for a float input it does just `x*2 - 1`. So:

  * This module outputs images in **[0,255] float** (to match the real uint8
    `images` for validation).
  * To feed the model a differentiable tensor and get the SAME [-1,1] the uint8
    path yields, pass `images / 255.0` (float in [0,1]) to the model, whose
    float-branch `x*2-1` then lands in [-1,1]. (Done in pixel_pgd.py.)
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF


def select_tiling(h: int, w: int, patch_window_size: int, max_crops: int) -> np.ndarray:
    """Port of olmo image_preprocessor.select_tiling. Integer geometry only —
    depends on image shape, never on pixel values. Returns [rows, cols]."""
    tilings = []
    for i in range(1, max_crops + 1):
        for j in range(1, max_crops + 1):
            if i * j <= max_crops:
                tilings.append((i, j))
    tilings.sort(key=lambda x: (x[0] * x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)          # [n, 2]
    candidate_resolutions = candidate_tilings * patch_window_size  # [n, 2]
    original_size = np.stack([h, w]).astype(np.float32)            # [2]
    with np.errstate(divide="ignore"):
        required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n, 1]
    if np.all(required_scale < 1):
        ix = int(np.argmax(required_scale))
    else:
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = int(np.argmin(required_scale))
    return candidate_tilings[ix]


def siglip_resize(img_hwc: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Differentiable siglip resize: torchvision BILINEAR, antialias=False, on
    float [0,255] (the same call siglip_resize_and_pad uses, minus the uint8
    round-trip). Bilinear weights are convex so values stay in range; we clamp
    anyway for exact fidelity. img_hwc: (H,W,3) float -> (out_h,out_w,3) float."""
    x = img_hwc.permute(2, 0, 1)  # CHW
    x = TF.resize(x, [out_h, out_w],
                  interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
    x = x.clamp(0.0, 255.0)
    return x.permute(1, 2, 0)  # HWC


def pixels_to_patches(crops_nhwc: torch.Tensor, patch: int) -> torch.Tensor:
    """Port of batch_pixels_to_patches (4D): (n,h,w,3) -> (n, h_p*w_p, patch*patch*3)."""
    n, h, w, c = crops_nhwc.shape
    hp, wp = h // patch, w // patch
    x = crops_nhwc.reshape(n, hp, patch, wp, patch, c)
    x = x.permute(0, 1, 3, 2, 4, 5)          # n, hp, wp, patch, patch, c
    x = x.reshape(n, hp * wp, patch * patch * c)
    return x


class DiffPreprocessor:
    """Differentiable pixel -> `images` tensor for one screenshot.

    Parameters mirror the live preprocessor (read them off it at runtime rather
    than hard-coding). __call__ takes a float (H,W,3) tensor in [0,255] and
    returns images (n_crops, n_patches, 588) in [0,255] float — same ordering
    and range as the real preprocessor's `images`, but differentiable.
    """

    def __init__(self, base_size: int, patch_size: int,
                 overlap_margins: tuple[int, int], max_crops: int,
                 crop_mode: str = "overlap-and-resize-c2"):
        self.base = int(base_size)
        self.patch = int(patch_size)
        self.left_margin, self.right_margin = int(overlap_margins[0]), int(overlap_margins[1])
        self.max_crops = int(max_crops)
        self.crop_mode = crop_mode

    def tiling_for(self, H: int, W: int) -> np.ndarray:
        crop_patches = self.base // self.patch
        crop_window_patches = crop_patches - (self.left_margin + self.right_margin)
        crop_window_size = crop_window_patches * self.patch
        total_margin = self.patch * (self.left_margin + self.right_margin)
        return select_tiling(max(H - total_margin, 1), max(W - total_margin, 1),
                             crop_window_size, self.max_crops)

    def __call__(self, img_hwc: torch.Tensor) -> torch.Tensor:
        base, patch = self.base, self.patch
        H, W = int(img_hwc.shape[0]), int(img_hwc.shape[1])

        # --- global crop: whole image squished to base x base (no aspect keep) ---
        global_crop = siglip_resize(img_hwc, base, base)               # (base,base,3)
        global_patches = pixels_to_patches(global_crop.unsqueeze(0), patch)  # (1,729,588)

        if self.crop_mode == "resize":
            return global_patches

        if self.crop_mode not in ("overlap-and-resize-c2", "overlap-and-resize"):
            raise NotImplementedError(self.crop_mode)

        crop_window_patches = (base // patch) - (self.left_margin + self.right_margin)
        crop_window_size = crop_window_patches * patch
        total_margin = patch * (self.left_margin + self.right_margin)

        tiling = self.tiling_for(H, W)
        rows, cols = int(tiling[0]), int(tiling[1])
        src = siglip_resize(img_hwc,
                            rows * crop_window_size + total_margin,
                            cols * crop_window_size + total_margin)     # (src_h,src_w,3)

        crops = []
        for i in range(rows):
            y0 = i * crop_window_size
            for j in range(cols):
                x0 = j * crop_window_size
                crops.append(src[y0:y0 + base, x0:x0 + base])
        crops = torch.stack(crops, 0)                                   # (n,base,base,3)
        crop_patches = pixels_to_patches(crops, patch)                  # (n,729,588)

        if self.crop_mode == "overlap-and-resize":
            return crop_patches
        # c2: global image goes FIRST, then the overlapping crops
        return torch.cat([global_patches, crop_patches], 0)            # (1+n,729,588)


def find_multicrop_preprocessor(preprocessor):
    """Walk a built MultimodalPreprocessor to find the MultiCropImagePreprocessor
    and return (base_size, patch_size, overlap_margins, max_crops, crop_mode).

    The nesting isn't part of a stable public API, so we search attributes for
    the object that carries `crop_mode` + an `image_preprocessor`."""
    seen = set()
    stack = [preprocessor]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "crop_mode") and hasattr(obj, "image_preprocessor"):
            ip = obj.image_preprocessor
            base = ip.base_image_input_size
            base = base[0] if isinstance(base, (tuple, list)) else base
            return {
                "base_size": int(base),
                "patch_size": int(ip.image_patch_size),
                "overlap_margins": tuple(obj.overlap_margins),
                "max_crops": int(obj.max_crops),
                "crop_mode": obj.crop_mode,
            }
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if hasattr(child, "__dict__") and not isinstance(child, (str, bytes, np.ndarray, torch.Tensor)):
                stack.append(child)
    raise RuntimeError("Could not locate MultiCropImagePreprocessor on the preprocessor")
