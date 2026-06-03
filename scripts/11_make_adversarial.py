"""Apply the MVP color-block mask to every SyntheticQA page under a root.

For each <root>/<website>/page###/screenshot.png:
  - sample a random crop (5-15% of area by default), recorded as patch_bbox
  - paint a solid color block over it (placeholder for the PGD patch)
  - recomposite at the same coords -> adversarial_site.png
  - write attack_meta.json (bbox, color, area_frac)

Deterministic: a global --seed plus a per-page sha256 seed, so reruns and
parallel workers reproduce identical crops. CPU-only; runs on a laptop. The
resulting adversarial_site.png's are what you copy to the cluster to query
MolmoWeb against (clean screenshot.png vs adversarial_site.png).

Usage (from repo root, with the project .venv):
    .venv/bin/python scripts/11_make_adversarial.py --root data/syntheticQA
    .venv/bin/python scripts/11_make_adversarial.py --root data/syntheticQA \
        --area-min 0.05 --area-max 0.15 --color random --seed 0 --overwrite
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_color(s: str):
    if s == "random":
        return None
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--color must be 'random' or 'R,G,B'")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("color components must be ints") from e
    if not all(0 <= c <= 255 for c in rgb):
        raise argparse.ArgumentTypeError("color components must be in 0..255")
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "syntheticQA")
    ap.add_argument("--area-min", type=float, default=0.02)
    ap.add_argument("--area-max", type=float, default=0.06)
    ap.add_argument("--color", type=parse_color, default=None,
                    help="'random' (default) or fixed 'R,G,B'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from dawg.masking.crop import make_adversarial_for_page, page_seed

    root = args.root
    if not root.exists():
        print(f"[make-adv] root not found: {root}", file=sys.stderr)
        return 1

    page_dirs = sorted(p.parent for p in root.glob("*/page*/screenshot.png"))
    print(f"[make-adv] root={root}  pages={len(page_dirs)}  "
          f"area=[{args.area_min},{args.area_max}]  "
          f"color={'random' if args.color is None else args.color}  seed={args.seed}")

    n_done = n_skip = n_fail = 0
    for pd in page_dirs:
        key = f"{pd.parent.name}/{pd.name}"
        rng = random.Random(page_seed(args.seed, key))
        try:
            meta = make_adversarial_for_page(
                pd, rng=rng,
                area_frac_range=(args.area_min, args.area_max),
                color=args.color,
                overwrite=args.overwrite,
            )
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {key}: {type(e).__name__}: {e}")
            continue
        if meta is None:
            n_skip += 1
        else:
            n_done += 1

    print(f"[make-adv] DONE  written={n_done}  skipped={n_skip}  failed={n_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
