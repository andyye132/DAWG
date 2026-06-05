"""Partition fetched SyntheticQA sites into disjoint L1/L2/L3 sets (~150 each,
target) and emit the L1 wave-2 page list. Run once the fetch has finished.

L1 = the 28 wave-1 sites already being attacked + enough new sites to reach the
per-level target. L2/L3 sites are RESERVED (left untouched on disk) for the
later L2 (stealth-targeted) and L3 (action-hijack) attacks. Nothing is deleted;
this only writes a manifest + a page list.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

BASE = Path("/gscratch/raivn/andy132/dawg")
DATA = BASE / "data" / "syntheticqa_full"
PER_LEVEL = 150


def main() -> None:
    wave1 = (BASE / "data" / "pagelist_l1_wave1.txt").read_text().split()
    l1_existing = sorted({Path(p).parent.name for p in wave1 if p})  # the 28 wave-1 sites
    all_sites = sorted(d.name for d in DATA.iterdir() if d.is_dir())
    new = [s for s in all_sites if s not in set(l1_existing)]
    random.Random(0).shuffle(new)

    need_l1 = max(0, PER_LEVEL - len(l1_existing))
    l1_new = new[:need_l1]
    l2 = new[need_l1: need_l1 + PER_LEVEL]
    l3 = new[need_l1 + PER_LEVEL: need_l1 + 2 * PER_LEVEL]
    spare = new[need_l1 + 2 * PER_LEVEL:]

    split = {
        "target_per_level": PER_LEVEL,
        "l1_existing": sorted(l1_existing),
        "l1_new": sorted(l1_new),
        "l1": sorted(l1_existing) + sorted(l1_new),
        "l2": sorted(l2),
        "l3": sorted(l3),
        "spare": sorted(spare),
        "counts": {"l1": len(l1_existing) + len(l1_new), "l2": len(l2),
                   "l3": len(l3), "spare": len(spare), "total_sites": len(all_sites)},
    }
    (BASE / "data" / "attack_split.json").write_text(json.dumps(split, indent=2))

    pages = sorted(str(p.parent) for s in l1_new
                   for p in (DATA / s).glob("page*/screenshot.png"))
    (BASE / "data" / "pagelist_l1_wave2.txt").write_text("\n".join(pages) + "\n")

    c = split["counts"]
    print(f"[split] total_sites={c['total_sites']}  "
          f"L1={c['l1']} (28 existing + {len(l1_new)} new)  "
          f"L2={c['l2']}  L3={c['l3']}  spare={c['spare']}")
    print(f"[split] L1 wave-2 page list: {len(pages)} pages -> data/pagelist_l1_wave2.txt")
    print(f"[split] manifest -> data/attack_split.json")


if __name__ == "__main__":
    main()
