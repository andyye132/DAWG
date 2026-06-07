"""Generate the L2 (stealth-targeted) adversarial dataset over OCR pages.

For each OCR (page, question): make a deterministic minimal-lie target, run masked
targeted PGD to pull MolmoWeb onto it, re-query, and SAVE the (clean, adversarial)
pair + answers. Detection/scoring is a SEPARATE offline pass (scripts/33) so this
GPU job stays focused on the slow part (the attack) and we can re-score with tuned
thresholds without re-attacking.

Sharded (page i -> shard i%num_shards), resumable (per-row JSONL). Boots MolmoWeb
once per shard. OCR-only v1; non-OCR / non-perturbable / clean-wrong pages skipped.

    sbatch --array=0-3 scripts/sbatch_l2gen_a100.sh        # start on a100s
    # later, when L1 frees a40s:  sbatch --array=4-15 scripts/sbatch_l2gen_a40.sh
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))
CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-file", type=Path, default=DAWG_ROOT / "data" / "chunk_l2.txt")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "l2_dataset")
    ap.add_argument("--eps", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=16)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] no CUDA"); return 1

    pages = [Path(l) for l in args.pages_file.read_text().split() if l]
    shard = [p for i, p in enumerate(pages) if i % args.num_shards == args.shard_id]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard)}/{len(pages)} pages  "
          f"eps={args.eps} iters={args.iters} restarts={args.restarts}  GPU={torch.cuda.get_device_name(0)}")

    print("Loading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)
    print(f"  loaded in {time.time()-t0:.1f}s")
    from dawg.attacks.pointing import point_to_answer, centered_bbox_at
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.attacks.l2_targets import make_ocr_target
    from dawg.eval.equiv import answer_drift, _unwrap_json_action, similarity

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = args.out / f"shard{args.shard_id}.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["site"], r["page"]))
            except Exception:
                pass
    fout = jsonl.open("a")

    n_attacked = n_hit = 0
    for pd in shard:
        site, pageid = pd.parent.name, pd.name
        if (site, pageid) in done:
            continue
        try:
            meta = json.loads((pd / "meta.json").read_text())
            # Scan ALL QAs for an OCR one with a perturbable answer — the old
            # qa[0]-only pick left 2.3x attackable pages on the table (see
            # scripts/34_l2_yield_analysis.py: 24% -> 57% of chunk_l2). Among
            # eligible QAs prefer rare lie kinds so number_perturb doesn't
            # dominate the dataset.
            _rank = {"yesno_flip": 0, "year_perturb": 1, "list_drop": 2, "number_perturb": 3}
            cands = []
            for cqa in meta["qa"]:
                if cqa.get("question_type", "").upper() != "OCR":
                    continue
                ctg = make_ocr_target(cqa["answer"], seed=0)
                if ctg is not None:
                    cands.append((_rank[ctg[1]], cqa))
            if not cands:
                continue
            qa = min(cands, key=lambda t: t[0])[1]
            q, gt = qa["question"], qa["answer"]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        except Exception as e:
            print(f"  [page-FAIL] {site}/{pageid}: {type(e).__name__}: {e}"); continue
        H, W = img.shape[:2]
        try:
            clean = pred.predict(q, img)
            if not answer_drift(gt, clean)["same_meaning"]:
                continue   # need a correct clean answer to corrupt
            msg = _unwrap_json_action(clean)
            tg = make_ocr_target(msg, seed=0)
            if tg is None:
                continue
            corrupt_msg, kind = tg
            target_answer = clean.replace(msg, corrupt_msg)
            if target_answer == clean:
                continue
            poi, _ = point_to_answer(pred, img, q)
            cx, cy = poi if poi else (W // 2, H // 2)
            bbox = centered_bbox_at(cx, cy, W, H, 0.10)
            setup = PixelPGDSetup(pred, img, q, bbox, device="cuda:0",
                                  clean_answer=clean, target_answer=target_answer)
            best = None
            for r in range(max(1, args.restarts)):
                d0 = None if r == 0 else (torch.rand(setup.screenshot.shape,
                     generator=torch.Generator(device="cuda:0").manual_seed(r),
                     device="cuda:0") * 2 - 1) * args.eps
                adv, hist = setup.pgd_l1(eps=args.eps, n_iter=args.iters, lr=args.lr,
                                         verbose=False, optim="momentum", minimize=True, delta0=d0)
                if best is None or hist[-1] < best[1]:
                    best = (adv, hist[-1])
            adv_ans = pred.predict(q, best[0])
        except Exception as e:
            print(f"  [attack-FAIL] {site}/{pageid}: {type(e).__name__}: {e}"); continue

        outdir = args.out / site / pageid
        outdir.mkdir(parents=True, exist_ok=True)
        if not (outdir / "clean.png").exists():
            Image.fromarray(img).save(outdir / "clean.png")
        Image.fromarray(best[0]).save(outdir / "adv.png")
        adv_msg = _unwrap_json_action(adv_ans)
        hit = similarity(corrupt_msg, adv_msg) > 0.9
        n_attacked += 1; n_hit += int(hit)
        fout.write(json.dumps({
            "site": site, "page": pageid, "question": q, "question_type": "OCR",
            "gt": gt, "clean_answer": clean, "clean_msg": msg, "target_msg": corrupt_msg,
            "adv_answer": adv_ans, "adv_msg": adv_msg, "lie_kind": kind, "hit_target": hit,
            "eps": args.eps, "clean_path": str((outdir / "clean.png").relative_to(args.out)),
            "adv_path": str((outdir / "adv.png").relative_to(args.out)), "image": [W, H],
        }) + "\n"); fout.flush()
        print(f"  {site}/{pageid} [{kind}] '{msg[:24]}'->'{corrupt_msg[:24]}' adv='{adv_msg[:24]}' hit={hit}")
    fout.close()
    print(f"\nshard {args.shard_id} done. attacked={n_attacked} hit_target={n_hit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
