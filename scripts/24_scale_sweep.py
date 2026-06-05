"""Scale / placement sweep: at a FIXED ~10% perturbation mass, compare how much
the answer breaks as a function of potency (eps) and how the mass is distributed.

Three arms per page (all centered on MolmoWeb's pointed POI, patch CLIPPED to stay
on the POI — see pointing.centered_bbox_at):
  - single10     : 1 patch, ~10% area               (mass concentrated)
  - multi_even   : 3 patches, ~3.33% each (~10%)     (same mass, spread evenly)
  - multi_random : 3 patches, random sizes ~10% tot  (same mass, uneven split)

Each arm is swept over eps in {2,4,8,12,16,24,32}. For every cell we LOG the raw
adversarial answer, similarity/distance, realized patch boxes + realized area, a
degeneracy flag, and clean_correct — so the aggregator can condition ASR on
clean_correct and exclude gibberish without ever re-running the attack.

Page sampling fixes the old auto:N alphabetical bias: `rand:N:SEED` samples N
(site,page) units uniformly across ALL sites and ALL pages with a fixed seed.

    python scripts/24_scale_sweep.py --pages rand:20:0 \
        --eps 2,4,8,12,16,24,32 --total-area 10 --npatch 3 --iters 50 \
        --shard-id $SLURM_ARRAY_TASK_ID --num-shards 10
"""
from __future__ import annotations

import argparse
import hashlib
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
    """Resolve a --pages spec to a list of page dirs (each holding screenshot.png + meta.json)."""
    if spec.startswith("file:"):
        # frozen page list (one page dir path per line) -- decouples sharding
        # from a data dir that may be growing (e.g. a concurrent fetch).
        return [Path(l) for l in Path(spec[5:]).read_text().split() if l]
    if spec.startswith("rand:"):
        _, n, seed = spec.split(":")
        n, seed = int(n), int(seed)
        units = []
        for s in sorted(p for p in root.iterdir() if p.is_dir()):
            units += [pg.parent for pg in sorted(s.glob("page*/screenshot.png"))]
        rng = random.Random(seed)
        rng.shuffle(units)
        return units[:n]
    if spec.startswith("auto:"):
        n = int(spec.split(":")[1])
        sites = sorted(p for p in root.iterdir() if p.is_dir())
        step = max(1, len(sites) // n)
        return [pg[0].parent for s in sites[::step][:n]
                if (pg := sorted(s.glob("page*/screenshot.png")))]
    return [root / sp for sp in spec.split(",")]


def page_rng(site: str, pageid: str) -> random.Random:
    """Deterministic per-page RNG (so multi_random sizes are reproducible)."""
    h = hashlib.sha256(f"{site}/{pageid}".encode()).hexdigest()[:8]
    return random.Random(int(h, 16))


def build_patches(img: np.ndarray, cx: int, cy: int, sizes: list[float], content_bboxes,
                  centered_bbox_at) -> list[list[int]]:
    """Build len(sizes) patches: the first centered on the POI, the rest on the
    most content-dense regions (variance ranked), each with its own area_frac (%)."""
    H, W = img.shape[:2]
    boxes = [centered_bbox_at(cx, cy, W, H, sizes[0] / 100.0)]
    if len(sizes) > 1:
        mean_extra = sum(sizes[1:]) / len(sizes[1:]) / 100.0
        cand = content_bboxes(img, len(sizes) - 1, mean_extra, exclude=boxes[0])
        for i, (bx, by, bw, bh) in enumerate(cand):
            ccx, ccy = bx + bw // 2, by + bh // 2
            boxes.append(centered_bbox_at(ccx, ccy, W, H, sizes[i + 1] / 100.0))
    return boxes


def realized_area_pct(boxes: list[list[int]], W: int, H: int) -> float:
    return 100.0 * sum(w * h for (_x, _y, w, h) in boxes) / (W * H)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "scale_v1")
    ap.add_argument("--pages", default="rand:20:0")
    ap.add_argument("--qa-index", type=int, default=0)
    ap.add_argument("--total-area", type=float, default=10.0, help="total %% area per arm")
    ap.add_argument("--npatch", type=int, default=3, help="patch count for the multi arms")
    ap.add_argument("--eps", default="2,4,8,12,16,24,32")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--restarts", type=int, default=3,
                    help="PGD random restarts/cell; success = best-of-k (denoises "
                         "the single-seed eps non-monotonicity)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] No CUDA."); return 1

    all_pages = resolve_pages(args.root, args.pages)
    shard = [p for i, p in enumerate(all_pages) if i % args.num_shards == args.shard_id]
    epslist = [float(e) for e in args.eps.split(",")]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(all_pages)} pages  "
          f"total_area={args.total_area}% npatch={args.npatch} eps={epslist} "
          f"iters={args.iters}  GPU={torch.cuda.get_device_name(0)}")

    print("Loading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)
    print(f"  loaded in {time.time()-t0:.1f}s")

    from dawg.attacks.pointing import point_to_answer, centered_bbox_at, content_bboxes
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift, is_degenerate

    args.out.mkdir(parents=True, exist_ok=True)
    tot, n = args.total_area, args.npatch

    for pd in shard:
        site, pageid = pd.parent.name, pd.name
        out_path = args.out / f"{site}__{pageid}.json"
        if out_path.exists():
            print(f"  [skip] {site}/{pageid} (done)"); continue
        try:
            meta = json.loads((pd / "meta.json").read_text())
            qa = meta["qa"][args.qa_index]
            question, gt = qa["question"], qa["answer"]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
            H, W = img.shape[:2]
            clean = pred.predict(question, img)
            pt, raw = point_to_answer(pred, img, question)
        except Exception as e:
            print(f"  [page-FAIL] {site}/{pageid}: {type(e).__name__}: {e}"); continue

        point_parsed = pt is not None
        cx, cy = pt if pt else (W // 2, H // 2)
        clean_ok = answer_drift(gt, clean)["same_meaning"]
        print(f"\n  {site}/{pageid}  poi=({cx},{cy}) parsed={point_parsed}  "
              f"clean_correct={clean_ok}  Q={question[:55]}")

        # The three arms (sizes are %% of image area, summing to ~total per arm).
        rng = page_rng(site, pageid)
        w_ = [rng.random() + 0.25 for _ in range(n)]
        rand_sizes = [tot * x / sum(w_) for x in w_]
        arms = [
            ("single10", [tot]),
            ("multi_even", [tot / n] * n),
            ("multi_random", rand_sizes),
        ]

        cells = []
        for arm, sizes in arms:
            boxes = build_patches(img, cx, cy, sizes, content_bboxes, centered_bbox_at)
            r_area = realized_area_pct(boxes, W, H)
            setup = PixelPGDSetup(pred, img, question, boxes, device="cuda:0", clean_answer=clean)
            for e in epslist:
                # best-of-k random restarts: restart 0 is the deterministic zero
                # start; the rest start from a random point in the eps-ball. Take
                # the run with the largest drift; record how many of k broke.
                trials = []
                for r in range(max(1, args.restarts)):
                    if r == 0:
                        delta0 = None
                    else:
                        seed = int(hashlib.sha256(
                            f"{site}/{pageid}/{arm}/{int(e)}/{r}".encode()).hexdigest()[:8], 16)
                        gen = torch.Generator(device=setup.device).manual_seed(seed)
                        delta0 = (torch.rand(setup.screenshot.shape, generator=gen,
                                             device=setup.device) * 2 - 1) * e
                    adv, _hist = setup.pgd_l1(eps=e, n_iter=args.iters, lr=args.lr,
                                              verbose=False, delta0=delta0)
                    adv_ans = pred.predict(question, adv)
                    dd = answer_drift(clean, adv_ans)
                    trials.append((dd["distance"], dd["attack_success"], adv_ans, dd["similarity"]))
                n_break = sum(1 for _d, s, _a, _s in trials if s)
                dist, success, adv_ans, sim = max(trials, key=lambda t: t[0])
                degen = is_degenerate(adv_ans)
                cells.append({
                    "arm": arm, "eps": e, "npatch": len(boxes),
                    "req_area_pct": round(sum(sizes), 3), "realized_area_pct": round(r_area, 3),
                    "sizes_pct": [round(s, 3) for s in sizes], "bboxes": boxes,
                    "distance": dist, "similarity": sim,
                    "success": success, "n_break": n_break, "n_restarts": len(trials),
                    "degenerate": degen, "adv_answer": adv_ans,
                })
                tag = "BROKE" if success else "held"
                if success and degen:
                    tag += "*gibberish"
                print(f"    {arm:13} eps={e:>4.0f} n={len(boxes)} area={r_area:4.1f}% "
                      f"-> d={dist:.2f} {tag} ({n_break}/{len(trials)})")

        out_path.write_text(json.dumps({
            "site": site, "page": pageid, "question": question, "gt": gt, "clean": clean,
            "clean_correct": clean_ok, "point_parsed": point_parsed,
            "poi": [cx, cy], "poi_raw": raw, "image": [W, H],
            "total_area_pct": tot, "cells": cells}, indent=2))
        print(f"  wrote {out_path.name}")
    print(f"\nshard {args.shard_id} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
