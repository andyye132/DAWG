"""Comprehensive placement-strategy comparison (sharded). For N random pages,
compare 3 patch-placement strategies at matched ~10% mass:

  A_single10  : 1 patch, 10% area, on the POI.
  B_top3      : 3 patches (~3.3% each) on the TOP-3 distinct POIs (point_to_top_k
                — query MolmoWeb, black out the hit, re-query x3).
  C_heuristic : 1 patch on the POI + 2 on content-dense regions (current method).

Writes one JSON per page (24/25-compatible cell schema: arm=strategy, eps,
success, distance, degenerate) + clean_correct + the top-3 points, so
scripts/25_aggregate_scale.py reduces it to per-strategy ASR with CIs.

Boots MolmoWeb once per shard; page i -> shard i%num_shards.

    sbatch --array=0-7 scripts/sbatch_strategy.sh
"""
from __future__ import annotations

import argparse
import json
import random
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


def resolve_pages(root: Path, spec: str) -> list[Path]:
    if spec.startswith("rand:"):
        _, n, seed = spec.split(":")
        units = sorted(p.parent for p in root.glob("*/page*/screenshot.png"))
        rng = random.Random(int(seed)); rng.shuffle(units)
        return units[: int(n)]
    return [root / s for s in spec.split(",")]


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "strategy_v1")
    ap.add_argument("--pages", default="rand:60:7")
    ap.add_argument("--eps", default="8,16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] no CUDA"); return 1

    pages = resolve_pages(args.root, args.pages)
    shard = [p for i, p in enumerate(pages) if i % args.num_shards == args.shard_id]
    epslist = [float(e) for e in args.eps.split(",")]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(pages)} pages "
          f"eps={epslist}  GPU={torch.cuda.get_device_name(0)}")

    print("Loading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)
    print(f"  loaded in {time.time()-t0:.1f}s")
    from dawg.attacks.pointing import (point_to_answer, point_to_top_k,
                                       centered_bbox_at, content_bboxes)
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift, is_degenerate

    args.out.mkdir(parents=True, exist_ok=True)
    for pd in shard:
        site, pageid = pd.parent.name, pd.name
        out_path = args.out / f"{site}__{pageid}.json"
        if out_path.exists():
            print(f"  [skip] {site}/{pageid}"); continue
        try:
            meta = json.loads((pd / "meta.json").read_text())
            q, gt = meta["qa"][0]["question"], meta["qa"][0]["answer"]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        except Exception as e:
            print(f"  [page-FAIL] {site}/{pageid}: {e}"); continue
        H, W = img.shape[:2]
        try:
            clean = pred.predict(q, img)
            clean_ok = answer_drift(gt, clean)["same_meaning"]
            poi, _ = point_to_answer(pred, img, q)
            cx, cy = poi if poi else (W // 2, H // 2)
            top3, _ = point_to_top_k(pred, img, q, k=3, blackout_frac=0.05)
        except Exception as e:
            print(f"  [page-FAIL] {site}/{pageid}: {e}"); continue
        print(f"\n  {site}/{pageid} clean_correct={clean_ok} poi=({cx},{cy}) top3={top3}")

        strategies = {
            "A_single10": [centered_bbox_at(cx, cy, W, H, 0.10)],
            "B_top3": [centered_bbox_at(px, py, W, H, 0.0333) for (px, py) in top3]
                      or [centered_bbox_at(cx, cy, W, H, 0.10)],
            "C_heuristic": [centered_bbox_at(cx, cy, W, H, 0.0333)]
                           + content_bboxes(img, 2, 0.0333, exclude=centered_bbox_at(cx, cy, W, H, 0.0333)),
        }
        cells = []
        for arm, boxes in strategies.items():
            area = sum(b[2] * b[3] for b in boxes) / (W * H) * 100
            setup = PixelPGDSetup(pred, img, q, boxes, device="cuda:0", clean_answer=clean)
            for e in epslist:
                adv, _ = setup.pgd_l1(eps=e, n_iter=args.iters, lr=args.lr, verbose=False)
                ans = pred.predict(q, adv)
                d = answer_drift(clean, ans)
                cells.append({"arm": arm, "eps": e, "npatch": len(boxes),
                              "realized_area_pct": round(area, 1),
                              "distance": d["distance"], "success": d["attack_success"],
                              "degenerate": is_degenerate(ans), "adv_answer": ans})
                print(f"    {arm:12} eps={e:>3.0f} n={len(boxes)} area={area:4.1f}% "
                      f"d={d['distance']:.2f} {'BROKE' if d['attack_success'] else 'held'}")
        out_path.write_text(json.dumps({
            "site": site, "page": pageid, "question": q, "gt": gt, "clean": clean,
            "clean_correct": clean_ok, "poi": [cx, cy], "top3": top3,
            "image": [W, H], "cells": cells}, indent=2))
        print(f"  wrote {out_path.name}")
    print(f"\nshard {args.shard_id} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
