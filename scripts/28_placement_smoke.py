"""Smoke-compare 3 patch-placement strategies on a few pages (incl. redundancy-
robust ones) to see which breaks MolmoWeb best at matched ~10% mass:

  A single10  : 1 patch, 10% area, on the single POI.
  B top3      : 3 patches (~3.3% each) on the TOP-3 distinct POIs MolmoWeb points
                to when we black out each previous hit (point_to_top_k).
  C heuristic : 1 patch on the POI + 2 on content-dense regions (current method).

For B it also saves a marker image showing where the 3 points landed (to see if
it finds repeated copies of the answer on robust pages).

    python scripts/28_placement_smoke.py --pages amazon/page000,247sports/page000,... --eps 16
"""
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "placement_smoke")
    ap.add_argument("--pages", default="amazon/page000,247sports/page000,auctionninja/page000,"
                                       "9animetv/page000,arxiv/page000,airbnb/page000")
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
    from dawg.attacks.pointing import (point_to_answer, point_to_top_k,
                                       centered_bbox_at, content_bboxes)
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in args.pages.split(","):
        pd = args.root / rel
        try:
            meta = json.loads((pd / "meta.json").read_text())
            q, gt = meta["qa"][0]["question"], meta["qa"][0]["answer"]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        except Exception as e:
            print(f"  [skip] {rel}: {e}"); continue
        H, W = img.shape[:2]
        clean = pred.predict(q, img)
        clean_ok = answer_drift(gt, clean)["same_meaning"]
        poi, _ = point_to_answer(pred, img, q)
        cx, cy = poi if poi else (W // 2, H // 2)
        top3, _ = point_to_top_k(pred, img, q, k=3, blackout_frac=0.05)

        # save marker image for strategy B's points
        mk = Image.fromarray(img).copy(); d = ImageDraw.Draw(mk)
        for i, (px, py) in enumerate(top3):
            d.ellipse([px-16, py-16, px+16, py+16], outline=(255, 0, 0), width=5)
            d.text((px+18, py-8), str(i+1), fill=(255, 0, 0))
        mk.save(args.out / f"{rel.replace('/', '__')}__top3_points.png")

        strategies = {
            "A_single10": [centered_bbox_at(cx, cy, W, H, 0.10)],
            "B_top3": [centered_bbox_at(px, py, W, H, 0.0333) for (px, py) in top3] or
                      [centered_bbox_at(cx, cy, W, H, 0.10)],
            "C_heuristic": [centered_bbox_at(cx, cy, W, H, 0.0333)] +
                           content_bboxes(img, 2, 0.0333, exclude=centered_bbox_at(cx, cy, W, H, 0.0333)),
        }
        rec = {"page": rel, "q": q[:50], "clean_correct": clean_ok,
               "poi": [cx, cy], "top3": top3, "n_top3": len(top3)}
        for name, boxes in strategies.items():
            area = sum(b[2]*b[3] for b in boxes) / (W*H) * 100
            setup = PixelPGDSetup(pred, img, q, boxes, device="cuda:0", clean_answer=clean)
            adv, _ = setup.pgd_l1(eps=args.eps, n_iter=args.iters, verbose=False)
            dd = answer_drift(clean, pred.predict(q, adv))
            rec[name] = {"area_pct": round(area, 1), "distance": dd["distance"],
                         "success": dd["attack_success"]}
            print(f"  {rel:24} {name:12} area={area:4.1f}% d={dd['distance']:.2f} "
                  f"{'BROKE' if dd['attack_success'] else 'held'}")
        rows.append(rec)
        print(f"    -> {rel} top3 points: {top3}  clean_correct={clean_ok}")

    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    print("\n=== SUMMARY (success by strategy) ===")
    for name in ["A_single10", "B_top3", "C_heuristic"]:
        ok = sum(1 for r in rows if r.get(name, {}).get("success"))
        okc = sum(1 for r in rows if r["clean_correct"] and r.get(name, {}).get("success"))
        nc = sum(1 for r in rows if r["clean_correct"])
        print(f"  {name:12} broke {ok}/{len(rows)} pages  ({okc}/{nc} clean-correct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
