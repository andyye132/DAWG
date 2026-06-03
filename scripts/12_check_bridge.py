"""Validate the differentiable preprocessing bridge against the REAL preprocessor.

Builds MolmoWeb's preprocessor from config (no model weights -> runs on a login/
CPU node in seconds), runs it on an image to get the reference `images` tensor,
runs dawg.attacks.diff_preprocess.DiffPreprocessor on the same pixels, and
reports the max/mean abs difference (expected ~rounding-level, since the real
path casts to uint8 and ours stays float). Also prints the discovered
preprocessor params and the crop count / tiling so we confirm geometry.

Run in the dawg conda env (has olmo + torch + torchvision):
    python scripts/12_check_bridge.py --image external/molmoweb/assets/test_screenshot.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))

CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"
os.environ.setdefault("MOLMO_DATA_DIR",
                      os.path.join(os.environ.get("TMPDIR", "/tmp"), "molmo_data"))


def build_preprocessor():
    from olmo.models.model_config import BaseModelConfig
    from olmo.util import resource_path
    cfg_path = resource_path(str(CKPT), "config.yaml")
    model_cfg = BaseModelConfig.load(cfg_path, key="model", validate_paths=False)
    return model_cfg.build_preprocessor(for_inference=True, is_training=False)


def main() -> int:
    import torch
    from dawg.attacks.diff_preprocess import DiffPreprocessor, find_multicrop_preprocessor

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(CKPT.parent.parent / "assets" / "test_screenshot.png"))
    ap.add_argument("--question", default="What is shown on this page?")
    args = ap.parse_args()

    print(f"[bridge] building preprocessor from {CKPT}/config.yaml ...")
    preproc = build_preprocessor()
    params = find_multicrop_preprocessor(preproc)
    print(f"[bridge] discovered params: {params}")

    img_pil = Image.open(args.image).convert("RGB")
    W, H = img_pil.size
    img_np = np.array(img_pil)  # (H,W,3) uint8
    print(f"[bridge] image: {args.image}  size(WxH)={img_pil.size}  np={img_np.shape}")

    # --- REAL preprocessor ---
    batch = preproc(dict(image=img_pil, style="demo", question=args.question))
    images_ref = np.asarray(batch["images"])
    print(f"[bridge] REAL images: shape={images_ref.shape} dtype={images_ref.dtype} "
          f"min={images_ref.min()} max={images_ref.max()}")

    # --- DIFFERENTIABLE bridge ---
    bridge = DiffPreprocessor(**params)
    tiling = bridge.tiling_for(H, W)
    expected_crops = (1 if params["crop_mode"] == "overlap-and-resize-c2" else 0) + int(tiling[0]) * int(tiling[1])
    print(f"[bridge] tiling(rows,cols)={tuple(int(t) for t in tiling)}  expected_crops={expected_crops}")

    img_t = torch.tensor(img_np, dtype=torch.float32)  # (H,W,3) [0,255]
    images_mine = bridge(img_t).detach().cpu().numpy()
    print(f"[bridge] MINE images: shape={images_mine.shape} "
          f"min={images_mine.min():.2f} max={images_mine.max():.2f}")

    # --- compare ---
    if images_mine.shape != images_ref.shape:
        print(f"[bridge] !! SHAPE MISMATCH: mine={images_mine.shape} ref={images_ref.shape}")
        return 1
    ref_f = images_ref.astype(np.float32)
    diff = np.abs(images_mine - ref_f)
    print(f"[bridge] abs diff (in [0,255] units): max={diff.max():.4f} "
          f"mean={diff.mean():.4f}  p99={np.percentile(diff, 99):.4f}")
    # per-crop, to localize any single-crop logic bug
    per_crop = diff.reshape(diff.shape[0], -1).max(axis=1)
    print(f"[bridge] per-crop max diff: {np.round(per_crop, 3).tolist()}")
    # normalized-space equivalent (what the model sees: /255 then *2)
    norm_max = diff.max() / 255.0 * 2.0
    print(f"[bridge] max diff in model [-1,1] space: {norm_max:.5f}")

    if diff.max() <= 2.0:
        print("[bridge] PASS — differences are rounding-level; bridge matches the real preprocessor.")
        return 0
    print("[bridge] WARN — differences exceed rounding; inspect per-crop diffs above.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
