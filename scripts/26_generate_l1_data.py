"""Generate the L1 adversarial dataset: for every (page, question), produce a
diversified-strong multi-patch attack and save the (clean, adversarial) pair +
label. This is the ground data for (a) adversarial-fine-tuning MolmoWeb or
(b) training the discriminator.

Design (chosen from the scale sweep: 3x3.3% multi-patch beats one 10% patch):
  - 3 patches, ~3.3% area each (~10% total), patch #0 on MolmoWeb's pointed POI,
    the other 2 on content-dense regions, with mild per-unit size jitter.
  - eps drawn ~U(eps_min, eps_max) per unit (default 12-24). Imperceptibility is
    NOT a constraint (only the agent sees the page), so we optimize for success
    and diversity, not minimal perturbation.
  - one attack (R=1) per (page, question); KEEP ALL rows labeled by success.

Boots MolmoWeb ONCE per shard and grinds every page assigned to the shard
(shard handles page i iff i % num_shards == shard_id) -- use FEW shards on good
GPUs so the ~3-min model load is amortized over many pages.

Output under <out>/:
  <site>/<page>/clean.png                      # shared clean screenshot
                /adv_q{k}.png                   # adversarial image for QA k
  shard{id}.jsonl                               # one row per (page, qa)

Each JSONL row: site,page,qa_index,question,question_type,gt,clean_answer,
adv_answer,success(vs clean),success_vs_gt,distance_clean,distance_gt,
degenerate,eps,total_area_pct,bboxes,clean_path,adv_path,image[w,h].

    python scripts/26_generate_l1_data.py --root data/syntheticqa_full \
        --out results/l1_dataset --shard-id $SLURM_ARRAY_TASK_ID --num-shards 12
"""
from __future__ import annotations

import argparse
import hashlib
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


def all_pages(root: Path) -> list[Path]:
    """Every page dir (site/pageNNN) under root that has a screenshot, sorted."""
    return sorted(p.parent for p in root.glob("*/page*/screenshot.png"))


def unit_rng(site: str, pageid: str, qi: int):
    import random
    h = hashlib.sha256(f"{site}/{pageid}/{qi}".encode()).hexdigest()[:8]
    return random.Random(int(h, 16))


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "l1_dataset")
    ap.add_argument("--pages-file", type=Path, default=None,
                    help="frozen list of page dirs (one per line); decouples sharding "
                         "from a growing data dir. Falls back to all pages under --root.")
    ap.add_argument("--qa-per-page", type=int, default=5, help="max QA attacked per page")
    ap.add_argument("--npatch", type=int, default=3)
    ap.add_argument("--placement", default="poi_content", choices=["poi_content", "topk"],
                    help="multi-patch placement: POI+content-dense (default) or top-k pointed POIs")
    ap.add_argument("--optim", default="sign", choices=["sign", "momentum", "adam"],
                    help="PGD optimizer (sign=Linf-PGD default; momentum=MI-FGSM)")
    ap.add_argument("--total-area", type=float, default=10.0, help="total %% area")
    ap.add_argument("--eps-min", type=float, default=12.0)
    ap.add_argument("--eps-max", type=float, default=24.0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] No CUDA."); return 1

    if args.pages_file:
        pages = [Path(l) for l in args.pages_file.read_text().split() if l]
    else:
        pages = all_pages(args.root)
    shard = [p for i, p in enumerate(pages) if i % args.num_shards == args.shard_id]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(pages)} pages  "
          f"npatch={args.npatch} area={args.total_area}% eps~U({args.eps_min},{args.eps_max}) "
          f"iters={args.iters}  GPU={torch.cuda.get_device_name(0)}")

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
    jsonl = args.out / f"shard{args.shard_id}.jsonl"
    done = set()
    if jsonl.exists():  # resume: skip (page,qa) already written
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["site"], r["page"], r["qa_index"]))
            except Exception:
                pass
    fout = jsonl.open("a")

    n_rows = n_success = 0
    for pd in shard:
        site, pageid = pd.parent.name, pd.name
        try:
            meta = json.loads((pd / "meta.json").read_text())
            qas = meta["qa"][: args.qa_per_page]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        except Exception as e:
            print(f"  [page-FAIL] {site}/{pageid}: {type(e).__name__}: {e}"); continue
        H, W = img.shape[:2]
        outdir = args.out / site / pageid
        outdir.mkdir(parents=True, exist_ok=True)
        clean_path = outdir / "clean.png"
        if not clean_path.exists():
            Image.fromarray(img).save(clean_path)

        # clean answer + POI computed ONCE per page (reused across this page's QA
        # only when the question matches; QA differ, so recompute per QA).
        for qi, qa in enumerate(qas):
            if (site, pageid, qi) in done:
                continue
            question, gt = qa.get("question", ""), qa.get("answer", "")
            if not question:
                continue
            try:
                clean = pred.predict(question, img)
                pt, raw = point_to_answer(pred, img, question)
                cx, cy = pt if pt else (W // 2, H // 2)
                rng = unit_rng(site, pageid, qi)
                eps = rng.uniform(args.eps_min, args.eps_max)
                # 3x3.3% with mild jitter, summing to ~total_area
                w_ = [rng.uniform(0.8, 1.2) for _ in range(args.npatch)]
                sizes = [args.total_area * x / sum(w_) for x in w_]
                if args.placement == "topk" and args.npatch > 1:
                    # patches on the top-k distinct POIs (query, black out hit, re-query)
                    pts, _ = point_to_top_k(pred, img, question, k=args.npatch, blackout_frac=0.08)
                    boxes = [centered_bbox_at(px, py, W, H, sizes[i] / 100.0)
                             for i, (px, py) in enumerate(pts)]
                    if len(boxes) < args.npatch:  # collapse/parse-fail -> pad on content-dense
                        ex = boxes[0] if boxes else centered_bbox_at(cx, cy, W, H, sizes[0] / 100.0)
                        for (bx, by, bw, bh) in content_bboxes(img, args.npatch - len(boxes),
                                                               sizes[-1] / 100.0, exclude=ex):
                            boxes.append(centered_bbox_at(bx + bw // 2, by + bh // 2, W, H,
                                                          sizes[len(boxes)] / 100.0))
                    if not boxes:
                        boxes = [centered_bbox_at(cx, cy, W, H, args.total_area / 100.0)]
                else:
                    boxes = [centered_bbox_at(cx, cy, W, H, sizes[0] / 100.0)]
                    if args.npatch > 1:
                        cand = content_bboxes(img, args.npatch - 1,
                                              sum(sizes[1:]) / len(sizes[1:]) / 100.0, exclude=boxes[0])
                        for j, (bx, by, bw, bh) in enumerate(cand):
                            boxes.append(centered_bbox_at(bx + bw // 2, by + bh // 2, W, H,
                                                          sizes[j + 1] / 100.0))
                setup = PixelPGDSetup(pred, img, question, boxes, device="cuda:0", clean_answer=clean)
                adv, _ = setup.pgd_l1(eps=eps, n_iter=args.iters, lr=args.lr, verbose=False, optim=args.optim)
                adv_ans = pred.predict(question, adv)
            except Exception as e:
                print(f"  [qa-FAIL] {site}/{pageid} q{qi}: {type(e).__name__}: {e}"); continue

            dc = answer_drift(clean, adv_ans)       # drift from clean behavior
            dg = answer_drift(gt, adv_ans)          # drift from ground truth
            adv_path = outdir / f"adv_q{qi}.png"
            Image.fromarray(adv).save(adv_path)
            row = {
                "site": site, "page": pageid, "qa_index": qi,
                "question": question, "question_type": qa.get("question_type", ""),
                "gt": gt, "clean_answer": clean, "adv_answer": adv_ans,
                "success": dc["attack_success"], "success_vs_gt": dg["attack_success"],
                "distance_clean": dc["distance"], "distance_gt": dg["distance"],
                "degenerate": is_degenerate(adv_ans),
                "eps": round(eps, 2), "total_area_pct": round(sum(sizes), 2),
                "npatch": len(boxes), "bboxes": boxes,
                "clean_path": str(clean_path.relative_to(args.out)),
                "adv_path": str(adv_path.relative_to(args.out)),
                "image": [W, H],
            }
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n_rows += 1; n_success += int(dc["attack_success"])
        print(f"  {site}/{pageid}: {len(qas)} qa done  (running: {n_success}/{n_rows} broke)")
    fout.close()
    print(f"\nshard {args.shard_id} done. rows={n_rows} success={n_success}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
