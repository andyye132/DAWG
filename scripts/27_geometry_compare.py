"""Visual comparison: single 10% patch vs three 3.3% patches on the SAME page at
the SAME eps. For each page emit clean / adv_single / adv_multi PNGs, amplified
perturbation-diff images (to show WHERE and how visible the perturbation is), and
a labeled side-by-side. Lets us judge the stealth argument (is one big box more
noticeable than three small ones?).

    python scripts/27_geometry_compare.py --pages adidas/page000,accuweather/page000 --eps 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))
CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"


def amplified_diff(clean: np.ndarray, adv: np.ndarray, gain: float = 6.0) -> np.ndarray:
    d = np.abs(adv.astype("float32") - clean.astype("float32")) * gain
    return np.clip(d, 0, 255).astype("uint8")


def label(img: np.ndarray, text: str) -> Image.Image:
    pil = Image.fromarray(img)
    bar = Image.new("RGB", (pil.width, 34), (20, 20, 20))
    d = ImageDraw.Draw(bar); d.text((8, 9), text, fill=(255, 255, 255))
    out = Image.new("RGB", (pil.width, pil.height + 34), (20, 20, 20))
    out.paste(bar, (0, 0)); out.paste(pil, (0, 34))
    return out


def hstack(panels: list[Image.Image]) -> Image.Image:
    h = max(p.height for p in panels); w = sum(p.width for p in panels) + 8 * (len(panels) - 1)
    out = Image.new("RGB", (w, h), (40, 40, 40)); x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 8
    return out


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "geometry_compare")
    ap.add_argument("--pages", default="adidas/page000,accuweather/page000,allrecipes/page000")
    ap.add_argument("--eps", type=float, default=16.0)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] no CUDA"); return 1

    print("Loading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)
    print(f"  loaded in {time.time()-t0:.1f}s")
    from dawg.attacks.pointing import point_to_answer, centered_bbox_at, content_bboxes
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift
    import json

    args.out.mkdir(parents=True, exist_ok=True)
    for rel in args.pages.split(","):
        pd = args.root / rel
        meta = json.loads((pd / "meta.json").read_text())
        q = meta["qa"][0]["question"]
        img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        H, W = img.shape[:2]
        clean = pred.predict(q, img)
        pt, _ = point_to_answer(pred, img, q)
        cx, cy = pt if pt else (W // 2, H // 2)

        # full-area placement (shift-to-fit, NOT clipped) so single and multi are
        # matched at ~10% total area for a fair visibility comparison.
        def full_box_at(cx, cy, area_frac, aspect=1.5):
            a = area_frac * W * H
            w = max(14, min(int(round((a * aspect) ** 0.5)), W))
            h = max(14, min(int(round((a / aspect) ** 0.5)), H))
            x = max(0, min(cx - w // 2, W - w)); y = max(0, min(cy - h // 2, H - h))
            return [x, y, w, h]
        single_box = [full_box_at(cx, cy, 0.10)]
        s = sum((b[2] * b[3]) for b in single_box) / (W * H) * 100
        multi_box = [full_box_at(cx, cy, 0.0333)]
        for (bx, by, bw, bh) in content_bboxes(img, 2, 0.0333, exclude=multi_box[0]):
            multi_box.append(full_box_at(bx + bw // 2, by + bh // 2, 0.0333))
        m = sum((b[2] * b[3]) for b in multi_box) / (W * H) * 100

        tag = rel.replace("/", "__")
        Image.fromarray(img).save(args.out / f"{tag}__clean.png")
        results = {}
        for name, boxes, area in [("single10", single_box, s), ("multi3x33", multi_box, m)]:
            setup = PixelPGDSetup(pred, img, q, boxes, device="cuda:0", clean_answer=clean)
            adv, _ = setup.pgd_l1(eps=args.eps, n_iter=args.iters, verbose=False)
            d = answer_drift(clean, pred.predict(q, adv))
            Image.fromarray(adv).save(args.out / f"{tag}__adv_{name}.png")
            Image.fromarray(amplified_diff(img, adv)).save(args.out / f"{tag}__diff_{name}.png")
            results[name] = (adv, area, d["distance"], d["attack_success"])
            print(f"  {rel} {name}: area={area:.1f}% eps={args.eps:.0f} -> d={d['distance']:.2f} "
                  f"{'BROKE' if d['attack_success'] else 'held'}")
        # labeled side-by-side: clean | single | multi
        panels = [label(img, f"CLEAN  ({W}x{H})"),
                  label(results["single10"][0], f"SINGLE 10%  eps{args.eps:.0f}  d={results['single10'][2]:.2f}"),
                  label(results["multi3x33"][0], f"3x3.3%  eps{args.eps:.0f}  d={results['multi3x33'][2]:.2f}")]
        hstack(panels).save(args.out / f"{tag}__sidebyside.png")
        print(f"  wrote {tag}__sidebyside.png")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
