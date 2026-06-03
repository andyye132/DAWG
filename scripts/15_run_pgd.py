"""Run pixel-space PGD (L1 untargeted) on one page, end-to-end.

  PGD optimizes the bbox pixels to MAXIMIZE cross-entropy against MolmoWeb's
  clean answer -> save the adversarial screenshot -> re-query MolmoWeb on it ->
  score the answer drift with eval.equiv.

Greedy decoding by default so the clean vs adversarial comparison isn't muddied
by sampling. Saves: pgd_adversarial.png, pgd_diff.png (amplified perturbation),
pgd_sidebyside.png, pgd_meta.json (answers + loss curve + drift) in the page dir.

Run on a GPU node in the dawg env (see sbatch preamble).
    python scripts/15_run_pgd.py --page 9animetv/page000 --eps 16 --iters 80
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
DATA = DAWG_ROOT / "data" / "syntheticQA"


def diff_viz(clean: np.ndarray, adv: np.ndarray) -> Image.Image:
    """Amplified |adv-clean|, normalized to 0-255 so the perturbation is visible."""
    d = np.abs(adv.astype(np.float32) - clean.astype(np.float32))
    span = d.max() - d.min()
    n = np.zeros_like(d, dtype=np.uint8) if span < 1e-6 else \
        ((d - d.min()) / span * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(n)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="9animetv/page000")
    ap.add_argument("--qa-index", type=int, default=0)
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--bbox", default=None, help="override 'x,y,w,h'")
    ap.add_argument("--sample", action="store_true", help="use sampling instead of greedy")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] No CUDA — allocate a GPU node.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    page = DATA / args.page
    meta = json.loads((page / "meta.json").read_text())
    qa = meta["qa"][args.qa_index]
    question, gt = qa["question"], qa["answer"]
    if args.bbox:
        bbox = [int(v) for v in args.bbox.split(",")]
    else:
        bbox = json.loads((page / "attack_meta.json").read_text())["patch_bbox"]
    img_np = np.array(Image.open(page / "screenshot.png").convert("RGB"))

    print(f"\npage={args.page}  image={img_np.shape[1]}x{img_np.shape[0]}  bbox={bbox}")
    print(f"Q : {question}")
    print(f"GT: {gt}")
    print(f"PGD: eps={args.eps} iters={args.iters} lr={args.lr} "
          f"decode={'sampling' if args.sample else 'greedy'}")

    print("\nLoading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0",
                                 top_p=(0.8 if args.sample else None))
    print(f"  loaded in {time.time() - t0:.1f}s")

    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift

    setup = PixelPGDSetup(pred, img_np, question, bbox, device="cuda:0")
    clean_answer = setup.clean_answer
    base = answer_drift(gt, clean_answer)
    print(f"\nCLEAN answer: {clean_answer}")
    print(f"  clean-vs-GT: similarity={base['similarity']} "
          f"(clean {'MATCHES' if base['same_meaning'] else 'DIFFERS from'} GT)")

    print(f"\nRunning PGD ({args.iters} steps) ...")
    t0 = time.time()
    adv_np, hist = setup.pgd_l1(eps=args.eps, n_iter=args.iters, lr=args.lr, verbose=True)
    print(f"  PGD done in {time.time() - t0:.1f}s  loss {hist[0]:.4f} -> {hist[-1]:.4f} "
          f"(+{hist[-1]-hist[0]:.4f})")

    # save artifacts
    Image.fromarray(adv_np).save(page / "pgd_adversarial.png")
    Image.fromarray(img_np).save(page / "pgd_clean.png")
    diff_viz(img_np, adv_np).save(page / "pgd_diff.png")
    H, W = img_np.shape[:2]
    sbs = Image.new("RGB", (W * 2 + 20, H), (0, 0, 0))
    sbs.paste(Image.fromarray(img_np), (0, 0))
    sbs.paste(Image.fromarray(adv_np), (W + 20, 0))
    sbs.resize((int((W * 2 + 20) * 1100 / (W * 2 + 20)), int(H * 1100 / (W * 2 + 20)))).save(page / "pgd_sidebyside.png")

    print("\nRe-querying MolmoWeb on the adversarial screenshot ...")
    adv_answer = pred.predict(question, adv_np)
    d_gt = answer_drift(gt, adv_answer)
    d_clean = answer_drift(clean_answer, adv_answer)

    (page / "pgd_meta.json").write_text(json.dumps({
        "page": args.page, "question": question, "gt_answer": gt,
        "clean_answer": clean_answer, "adv_answer": adv_answer,
        "eps": args.eps, "iters": args.iters, "lr": args.lr, "bbox": bbox,
        "decode": "sampling" if args.sample else "greedy",
        "loss_first": hist[0], "loss_last": hist[-1], "loss_history": hist,
        "drift_vs_gt": d_gt, "drift_vs_clean": d_clean,
    }, indent=2))

    print("\n=================== ATTACK REPORT ===================")
    print(f"  Q              : {question}")
    print(f"  GT answer      : {gt}")
    print(f"  CLEAN answer   : {clean_answer}")
    print(f"  ADVERSARIAL    : {adv_answer}")
    print(f"  loss           : {hist[0]:.4f} -> {hist[-1]:.4f}")
    print(f"  drift clean->adv: distance={d_clean['distance']}  similarity={d_clean['similarity']}")
    print(f"  drift  GT  ->adv: distance={d_gt['distance']}  similarity={d_gt['similarity']}")
    flipped = d_clean["attack_success"]
    print(f"\n  [RESULT] {'ATTACK SUCCEEDED — answer meaning changed!' if flipped else 'no flip (try larger --eps / --iters or a bbox over the answer content)'}")
    print(f"  artifacts in {page}/: pgd_adversarial.png, pgd_diff.png, pgd_sidebyside.png, pgd_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
