"""Fetch a small, diverse subset of MolmoWeb-SyntheticQA to disk.

Writes <out>/<website>/page###/{screenshot.png, meta.json} + <out>/index.json.
See dawg.data.syntheticqa for the layout/schema. CPU-only; streams (no full
download). Intended to run on a laptop, then the adversarial outputs get
copied to the cluster for MolmoWeb testing.

Usage (from repo root, with the project .venv):
    .venv/bin/python scripts/10_fetch_syntheticqa.py \
        --out data/syntheticQA --n-sites 8 --pages-per-site 15 --max-scan 2500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "syntheticQA",
                    help="output root (default: data/syntheticQA)")
    ap.add_argument("--n-sites", type=int, default=8)
    ap.add_argument("--pages-per-site", type=int, default=15)
    ap.add_argument("--max-scan", type=int, default=2500,
                    help="max rows to stream before stopping (bounds bandwidth)")
    args = ap.parse_args()

    from dawg.data.syntheticqa import fetch_subset

    fetch_subset(
        out_root=args.out,
        n_sites=args.n_sites,
        pages_per_site=args.pages_per_site,
        max_scan=args.max_scan,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
