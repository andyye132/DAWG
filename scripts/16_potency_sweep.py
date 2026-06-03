"""Potency (eps) sweep on one page: attack strength vs. visibility vs. effect.

Loads MolmoWeb ONCE, builds one PixelPGDSetup (fixed page / question / bbox /
clean answer), then runs PGD at several eps levels expressed as % of a max
budget (100/75/50/25% -> eps 16/12/8/4 by default). For each level it saves the
full adversarial screenshot, re-queries MolmoWeb, and scores answer_drift.

Outputs to results/pgd_runs/<page>_potency/:
  adv_<pct>.png          full adversarial screenshot at each level
  potency_grid.png       zoomed patch crop: clean + each level, captioned
  comparison.json        answers + distances + loss per level
  summary.txt            human-readable table

Run on a GPU node (dawg env).
    python scripts/16_potency_sweep.py --page 9animetv/page000 \
        --bbox 560,240,240,160 --max-eps 16 --levels 100,75,50,25 --iters 60
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
DATA = DAWG_ROOT / "data" / "syntheticQA"
OUT_ROOT = DAWG_ROOT / "results" / "pgd_runs"


def crop_zoom(img: np.ndarray, bbox, disp_w: int = 240) -> Image.Image:
    """Crop the bbox region and upscale (NEAREST) so the perturbation is visible."""
    x, y, w, h = bbox
    patch = Image.fromarray(img[y:y + h, x:x + w])
    disp_h = max(1, round(disp_w * h / w))
    return patch.resize((disp_w, disp_h), Image.Resampling.NEAREST)


def build_grid(clean_np, levels, bbox, out_path):
    """clean + each level's patch-crop side by side, captioned underneath."""
    cells = [("CLEAN", crop_zoom(clean_np, bbox), None)]
    for lv in levels:
        cap = f"eps={lv['eps']:.0f} ({lv['pct']}%)\nd={lv['distance']:.2f} {'BROKE' if lv['success'] else 'held'}"
        cells.append((cap, crop_zoom(lv["adv_np"], bbox), lv["success"]))
    pad, cap_h = 12, 46
    cw = max(c[1].width for c in cells)
    ch = max(c[1].height for c in cells)
    W = len(cells) * (cw + pad) + pad
    H = ch + cap_h + 2 * pad
    grid = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(grid)
    for i, (cap, im, ok) in enumerate(cells):
        x0 = pad + i * (cw + pad)
        grid.paste(im, (x0, pad))
        color = (20, 20, 20) if ok is None else ((0, 130, 0) if ok else (170, 0, 0))
        for j, line in enumerate(cap.split("\n")):
            d.text((x0, pad + ch + 4 + j * 14), line, fill=color)
    grid.save(out_path)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="9animetv/page000")
    ap.add_argument("--qa-index", type=int, default=0)
    ap.add_argument("--bbox", default="560,240,240,160", help="x,y,w,h (smaller default)")
    ap.add_argument("--max-eps", type=float, default=16.0)
    ap.add_argument("--levels", default="100,75,50,25", help="potency %% of max-eps")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] No CUDA — allocate a GPU node.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    page = DATA / args.page
    meta = json.loads((page / "meta.json").read_text())
    qa = meta["qa"][args.qa_index]
    question, gt = qa["question"], qa["answer"]
    bbox = [int(v) for v in args.bbox.split(",")]
    pcts = [int(p) for p in args.levels.split(",")]
    img_np = np.array(Image.open(page / "screenshot.png").convert("RGB"))
    area_pct = 100.0 * bbox[2] * bbox[3] / (img_np.shape[0] * img_np.shape[1])

    print(f"\npage={args.page} image={img_np.shape[1]}x{img_np.shape[0]}")
    print(f"bbox={bbox}  (patch area = {area_pct:.1f}% of screen)")
    print(f"Q : {question}\nGT: {gt}")
    print(f"sweep: max_eps={args.max_eps} levels={pcts}% iters={args.iters}")

    print("\nLoading MolmoWeb-4B-Native ...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0", top_p=None)  # greedy
    print(f"  loaded in {time.time() - t0:.1f}s")

    from dawg.attacks.pixel_pgd import PixelPGDSetup
    from dawg.eval.equiv import answer_drift

    setup = PixelPGDSetup(pred, img_np, question, bbox, device="cuda:0")
    clean_answer = setup.clean_answer
    print(f"\nCLEAN answer: {clean_answer}")

    out_dir = OUT_ROOT / (args.page.replace("/", "_") + "_potency")
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = []
    for pct in pcts:
        eps = args.max_eps * pct / 100.0
        print(f"\n--- potency {pct}%  (eps={eps:.1f}) ---")
        t0 = time.time()
        adv_np, hist = setup.pgd_l1(eps=eps, n_iter=args.iters, lr=args.lr, verbose=False)
        adv_answer = pred.predict(question, adv_np)
        d = answer_drift(clean_answer, adv_answer)
        Image.fromarray(adv_np).save(out_dir / f"adv_{pct}.png")
        print(f"  loss {hist[0]:.3f}->{hist[-1]:.3f}  dist={d['distance']}  "
              f"success={d['attack_success']}  ({time.time()-t0:.0f}s)")
        print(f"  adv answer: {adv_answer[:120]}")
        levels.append({"pct": pct, "eps": eps, "adv_np": adv_np, "adv_answer": adv_answer,
                       "distance": d["distance"], "similarity": d["similarity"],
                       "success": d["attack_success"], "loss_first": hist[0], "loss_last": hist[-1]})

    build_grid(img_np, levels, bbox, out_dir / "potency_grid.png")

    comp = {"page": args.page, "question": question, "gt_answer": gt,
            "clean_answer": clean_answer, "bbox": bbox, "patch_area_pct": round(area_pct, 2),
            "max_eps": args.max_eps, "iters": args.iters,
            "levels": [{k: v for k, v in lv.items() if k != "adv_np"} for lv in levels]}
    (out_dir / "comparison.json").write_text(json.dumps(comp, indent=2))

    lines = [f"POTENCY SWEEP — {args.page}",
             f"patch {bbox[2]}x{bbox[3]} ({area_pct:.1f}% of screen)  iters={args.iters}",
             f"Q : {question}", f"GT: {gt}", f"clean: {clean_answer}", "",
             f"{'pct':>4} {'eps':>5} {'dist':>6} {'ok':>5}  adversarial answer"]
    for lv in levels:
        lines.append(f"{lv['pct']:>3}% {lv['eps']:>5.1f} {lv['distance']:>6.2f} "
                     f"{'BROKE' if lv['success'] else 'held':>5}  {lv['adv_answer'][:80]}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")

    print("\n=================== POTENCY SWEEP ===================")
    print("\n".join(lines))
    print(f"\nartifacts in {out_dir}/  (potency_grid.png, adv_*.png, comparison.json, summary.txt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
