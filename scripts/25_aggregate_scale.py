"""Aggregate the scale/placement sweep (scripts/24) into ASR-vs-eps curves per arm.

Fixes the reporting flaws the audit flagged: ASR is conditioned on clean_correct
(pages MolmoWeb already got wrong don't count), reported with a Wilson 95% CI and
an explicit n, and also reported EXCLUDING degenerate "gibberish" successes.

    python scripts/25_aggregate_scale.py --dir results/scale_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent

ARMS = ["single10", "multi_even", "multi_random"]


def wilson(succ: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval -> (point, lo, hi) as fractions."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = succ / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--no-clean-filter", action="store_true",
                    help="include pages where MolmoWeb was already wrong clean")
    args = ap.parse_args()

    pages = [json.loads(open(f).read()) for f in sorted(glob.glob(str(args.dir / "*.json")))]
    n_clean = sum(p.get("clean_correct") for p in pages)
    n_parsed = sum(p.get("point_parsed", True) for p in pages)
    print(f"pages: {len(pages)}  clean_correct: {n_clean}  point_parsed: {n_parsed}")
    used = pages if args.no_clean_filter else [p for p in pages if p.get("clean_correct")]
    print(f"counting ASR over {len(used)} pages "
          f"({'ALL' if args.no_clean_filter else 'clean_correct only'})\n")

    # derive the arm names from the data (works for single/multi OR strategy runs)
    global ARMS
    found = sorted({c["arm"] for p in pages for c in p.get("cells", [])})
    if found:
        ARMS = found

    # (arm, eps) -> counts
    agg = defaultdict(lambda: {"succ": 0, "succ_nd": 0, "n": 0, "dist": 0.0, "area": 0.0,
                               "brk": 0, "trials": 0})
    epsset = set()
    for p in used:
        for c in p["cells"]:
            epsset.add(c["eps"])
            a = agg[(c["arm"], c["eps"])]
            a["n"] += 1
            a["succ"] += int(c["success"])
            a["succ_nd"] += int(c["success"] and not c.get("degenerate"))
            a["dist"] += c["distance"]
            a["area"] += c.get("realized_area_pct", 0.0)
            a["brk"] += c.get("n_break", int(c["success"]))
            a["trials"] += c.get("n_restarts", 1)
    epslist = sorted(epsset)

    for arm in ARMS:
        if not any((arm, e) in agg for e in epslist):
            continue
        print(f"=== {arm} ===  (best-of-k ASR%% [Wilson95] | excl-gibberish | per-seed%% | mean-d | area%%)")
        for e in epslist:
            a = agg[(arm, e)]
            if not a["n"]:
                continue
            p, lo, hi = wilson(a["succ"], a["n"])
            p_nd = a["succ_nd"] / a["n"]
            seed = a["brk"] / a["trials"] if a["trials"] else 0
            print(f"  eps={e:>4.0f}  n={a['n']:>2}  "
                  f"ASR={100*p:5.1f}% [{100*lo:4.0f},{100*hi:4.0f}]  "
                  f"| excl={100*p_nd:5.1f}%  | seed={100*seed:5.1f}%  "
                  f"| d={a['dist']/a['n']:.2f}  | {a['area']/a['n']:4.1f}%")
        print()

    # Arm comparison at each eps (does spreading the same mass help?)
    print("=== arm comparison: ASR%% (clean_correct) at each eps ===")
    hdr = "  eps  " + "".join(f"{arm:>14}" for arm in ARMS)
    print(hdr)
    for e in epslist:
        line = f"  {e:>4.0f} "
        for arm in ARMS:
            a = agg[(arm, e)]
            line += f"{(100*a['succ']/a['n'] if a['n'] else 0):>13.0f}%"
        print(line)

    # Heatmap: rows = arm, cols = eps
    cw, ch, lab = 96, 48, 110
    im = Image.new("RGB", (lab + len(epslist) * cw + 12, lab + len(ARMS) * ch + 12), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.text((6, 6), "arm \\ eps", fill=(0, 0, 0))
    for j, e in enumerate(epslist):
        d.text((lab + j * cw + 8, 6), f"{e:.0f}", fill=(0, 0, 0))
    for i, arm in enumerate(ARMS):
        d.text((6, lab + i * ch + 16), arm, fill=(0, 0, 0))
        for j, e in enumerate(epslist):
            a = agg[(arm, e)]
            asr = a["succ"] / a["n"] if a["n"] else 0
            x0, y0 = lab + j * cw, lab + i * ch
            g = int(60 + 160 * asr)
            d.rectangle([x0, y0, x0 + cw - 4, y0 + ch - 4], fill=(int(230 - 150 * asr), g, 90))
            d.text((x0 + 8, y0 + 8), f"{100*asr:.0f}%", fill=(0, 0, 0))
            d.text((x0 + 8, y0 + 28), f"n{a['n']}" if a["n"] else "-", fill=(0, 0, 0))
    out = args.dir / "scale_heatmap.png"
    im.save(out)
    print(f"\nheatmap -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
