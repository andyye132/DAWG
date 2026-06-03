"""Pixel-space PGD L1 — 1-example SANITY CHECK (no attack run).

Loads MolmoWeb-4B-Native, picks one SyntheticQA page + its first question, reuses
the color-block crop bbox from attack_meta.json, and does a SINGLE forward+
backward through the differentiable bridge to confirm:
  - the clean answer comes back,
  - the CE loss is finite,
  - gradients flow to the pixel delta and are localized to the bbox,
  - the bridge still matches the real preprocessor for this image.

It does NOT iterate PGD (per "set it up but don't run it yet").

Run on a GPU node in the dawg env:
    python scripts/13_pixel_pgd_sanity.py --page data/syntheticQA/adidas/page000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))

CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"
DEVICE = "cuda:0"


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, default=DAWG_ROOT / "data" / "syntheticQA" / "adidas" / "page000")
    ap.add_argument("--qa-index", type=int, default=0)
    args = ap.parse_args()

    print(f"PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[FAIL] No CUDA — allocate a GPU node.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    page = args.page
    meta = json.loads((page / "meta.json").read_text())
    qa = meta["qa"][args.qa_index]
    question, gt_answer = qa["question"], qa["answer"]
    bbox = json.loads((page / "attack_meta.json").read_text())["patch_bbox"]
    img_np = np.array(Image.open(page / "screenshot.png").convert("RGB"))

    print(f"\npage={page.relative_to(DAWG_ROOT)}  image={img_np.shape[1]}x{img_np.shape[0]}")
    print(f"question: {question}")
    print(f"GT answer: {gt_answer}")
    print(f"crop bbox [x,y,w,h]: {bbox}")

    print(f"\nLoading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device=DEVICE)
    print(f"  loaded in {time.time() - t0:.1f}s  (model dtype={next(pred.model.parameters()).dtype})")

    from dawg.attacks.pixel_pgd import PixelPGDSetup
    print("\nBuilding pixel-PGD setup (clean inference + batch + bridge) ...")
    t0 = time.time()
    setup = PixelPGDSetup(pred, img_np, question, bbox, device=DEVICE)
    print(f"  clean answer: {setup.clean_answer!r}")
    print(f"  setup in {time.time() - t0:.1f}s")

    print("\nRunning 1-example sanity (single forward + backward) ...")
    t0 = time.time()
    r = setup.sanity_check()
    dt = time.time() - t0

    print("\n================= SANITY REPORT =================")
    for k in ["images_shape", "n_crops", "n_prompt_tokens", "n_target_tokens",
              "n_perturbable_pixels", "loss", "loss_finite", "grad_max_abs",
              "grad_mean_abs_in_box", "grad_abs_sum_outside_box",
              "bridge_max_diff_0to255"]:
        print(f"  {k:26s}: {r[k]}")
    print(f"  {'fwd+bwd_seconds':26s}: {dt:.2f}")

    ok = (r["loss_finite"] and r["loss"] > 0 and r["grad_max_abs"] > 0
          and r["grad_abs_sum_outside_box"] < 1e-3 and r["bridge_max_diff_0to255"] <= 2.0)
    print("\n[RESULT] " + ("PASS — pixel->loss->grad chain works; gradient localized to bbox; "
                           "ready to run PGD." if ok else
                           "CHECK — see values above."))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
