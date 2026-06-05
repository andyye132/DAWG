"""Smoke: geometry x optimizer. Compares, on a few pages at one eps:
  single10_sign      : 1x10% patch, sign-PGD (current method)
  single10_adam      : 1x10% patch, Adam-PGD
  single10_momentum  : 1x10% patch, MI-FGSM momentum
  two5_sign          : 2x5% patches on the top-2 POIs, sign-PGD (Andy's "area matters" idea)
  three33_sign       : 3x3.3% (1 POI + 2 content), sign-PGD (reference)

    python scripts/30_optim_geom_smoke.py --eps 12
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
    ap.add_argument("--root", type=Path, default=DAWG_ROOT / "data" / "syntheticqa_full")
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "optim_geom_smoke")
    ap.add_argument("--pages", default="amazon/page000,247sports/page000,auctionninja/page000,"
                                       "9animetv/page000,arxiv/page000,airbnb/page000,"
                                       "accuweather/page000,adidas/page000")
    ap.add_argument("--eps", type=float, default=12.0)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("[FAIL] no CUDA"); return 1
    print("Loading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)
    print(f"  loaded in {time.time()-t0:.1f}s")
    from dawg.attacks.pointing import point_to_answer, point_to_top_k, centered_bbox_at, content_bboxes
    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in args.pages.split(","):
        pd = args.root / rel
        try:
            meta = json.loads((pd / "meta.json").read_text())
            q = meta["qa"][0]["question"]
            img = np.array(Image.open(pd / "screenshot.png").convert("RGB"))
        except Exception as e:
            print(f"  [skip] {rel}: {e}"); continue
        H, W = img.shape[:2]
        clean = pred.predict(q, img)
        clean_ok = answer_drift(meta["qa"][0]["answer"], clean)["same_meaning"]
        poi, _ = point_to_answer(pred, img, q)
        cx, cy = poi if poi else (W // 2, H // 2)
        top2, _ = point_to_top_k(pred, img, q, k=2, blackout_frac=0.05)

        single = [centered_bbox_at(cx, cy, W, H, 0.10)]
        two5 = [centered_bbox_at(px, py, W, H, 0.05) for (px, py) in top2] or single
        three33 = [centered_bbox_at(cx, cy, W, H, 0.0333)] + \
                  content_bboxes(img, 2, 0.0333, exclude=centered_bbox_at(cx, cy, W, H, 0.0333))

        configs = [
            ("single10_sign", single, "sign"),
            ("single10_adam", single, "adam"),
            ("single10_momentum", single, "momentum"),
            ("two5_sign", two5, "sign"),
            ("three33_sign", three33, "sign"),
        ]
        rec = {"page": rel, "clean_correct": clean_ok}
        for name, boxes, opt in configs:
            setup = PixelPGDSetup(pred, img, q, boxes, device="cuda:0", clean_answer=clean)
            adv, hist = setup.pgd_l1(eps=args.eps, n_iter=args.iters, lr=2.0, verbose=False, optim=opt)
            d = answer_drift(clean, pred.predict(q, adv))
            rec[name] = {"distance": d["distance"], "success": d["attack_success"],
                         "loss0": round(hist[0], 3), "lossN": round(hist[-1], 3)}
            print(f"  {rel:22} {name:18} d={d['distance']:.2f} loss {hist[0]:.2f}->{hist[-1]:.2f} "
                  f"{'BROKE' if d['attack_success'] else 'held'}")
        rows.append(rec)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    print("\n=== SUMMARY (success / mean-dist over clean-correct pages) ===")
    cc = [r for r in rows if r["clean_correct"]]
    for name in ["single10_sign", "single10_adam", "single10_momentum", "two5_sign", "three33_sign"]:
        s = sum(r[name]["success"] for r in cc)
        md = sum(r[name]["distance"] for r in cc) / max(len(cc), 1)
        print(f"  {name:18} broke {s}/{len(cc)}  mean_d={md:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
