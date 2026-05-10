"""Project gate: confirm pixel gradients flow through MolmoWeb's vision backbone.

The ViT consumes already-patchified input (B, num_patch, n_pixels). The
canonical path is preprocessor → batch dict → vision_backbone. We use that
path here, swap in a batch["images"] that requires_grad, and verify gradients
flow back to it. Once this passes, building a fully differentiable pixel→
patch bridge for PGD is straightforward (just torch.nn.functional.unfold).

Run after activating the dawg conda env on a GPU node:
    python scripts/00_check_grad_vit.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))

CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"
TEST_IMG = DAWG_ROOT / "external" / "molmoweb" / "assets" / "test_screenshot.png"
DEVICE = "cuda:0"


def main() -> int:
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] No CUDA — allocate a GPU node first.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading MolmoWeb-4B-Native...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device=DEVICE)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  ViT class: {type(pred.model.vision_backbone.image_vit).__name__}")

    # --- Build batch via the canonical preprocessor path (mirrors predict()) ---
    img = Image.open(TEST_IMG).convert("RGB")
    print(f"\nInput PIL image: {img.size} (W, H)")

    batch = pred.preprocessor(dict(
        image=img,
        style="demo",
        question="What is shown on this page?",
    ))
    batch["input_ids"] = batch.pop("input_tokens")
    batch.pop("metadata", None)

    print(f"\nPreprocessor batch keys:")
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v if not hasattr(v, '__len__') or len(str(v)) < 80 else '...'}")

    # Add batch dim and move to device
    batch_t = {
        k: torch.as_tensor(np.expand_dims(v, 0), device=DEVICE)
        for k, v in batch.items()
        if isinstance(v, np.ndarray)
    }

    print(f"\nBatch tensors on device:")
    for k, v in batch_t.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # --- Make `images` require grad ---
    if "images" not in batch_t:
        print("[FAIL] No 'images' key in batch.")
        return 2

    batch_t["images"] = batch_t["images"].float().detach().requires_grad_(True)
    print(f"\nimages tensor now requires_grad: {batch_t['images'].requires_grad}")
    print(f"images stats: min={batch_t['images'].min().item():.3f} "
          f"max={batch_t['images'].max().item():.3f}")

    # --- Try the vision_backbone forward with whichever kwargs are in the batch ---
    vb = pred.model.vision_backbone
    print(f"\nCalling vision_backbone (type: {type(vb).__name__})...")

    # Build kwargs to match the call inside Molmo.forward (line 385 of molmo.py):
    #   self.vision_backbone(images, image_masks, token_pooling)
    # where image_masks is Optional (None when preprocessor doesn't emit it),
    # and the third positional arg `pooled_patches_idx` is named `token_pooling`
    # in the preprocessor batch.
    vb_kwargs = {
        "image_masks": batch_t.get("image_masks"),
        "pooled_patches_idx": batch_t["token_pooling"],
    }
    # Optional kwargs the backbone also accepts:
    for k in ["cum_token_pooling_bounds", "cum_image_bounds"]:
        if k in batch_t:
            vb_kwargs[k] = batch_t[k]
    print(f"  vb_kwargs: { {k: (tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in vb_kwargs.items()} }")

    try:
        out = vb(images=batch_t["images"], **vb_kwargs)
    except TypeError as e:
        print(f"  TypeError: {e}")
        import inspect
        print(f"  vb.forward signature: {inspect.signature(vb.forward)}")
        return 3

    if isinstance(out, tuple):
        feat = out[0]
        print(f"  vb returned tuple of {len(out)}; out[0] shape: {tuple(feat.shape)}")
    else:
        feat = out
        print(f"  vb returned tensor shape: {tuple(feat.shape)}")

    # --- Backward ---
    loss = feat.sum()
    loss.backward()

    g = batch_t["images"].grad
    if g is None:
        print("[FAIL] images.grad is None.")
        return 4

    mean_abs = g.abs().mean().item()
    max_abs = g.abs().max().item()
    print(f"\nGradient on images tensor: shape={tuple(g.shape)} "
          f"mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}")

    if mean_abs <= 0:
        print("[FAIL] Mean abs gradient is 0.")
        return 5

    print(f"\n[PASS] Gradient flows from preprocessor output through "
          f"vision_backbone. Patch-token-space PGD is feasible. Next step: "
          f"build a differentiable pixel→patch bridge so PGD works in "
          f"viewport (1280×720) space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
