"""Offline L2 yield analysis: why are we only attacking ~17% of pages?

Replays the static eligibility filters of 32_generate_l2_data.py over a pages
file (no GPU, metadata only) and compares QA-selection policies:

  A) CURRENT   qa[0] only: qa[0] is OCR AND make_ocr_target(qa[0].answer) != None
  B) PROPOSED  scan all QAs: any OCR qa with a perturbable answer
  C) UPPER     any qa of ANY type with a perturbable answer (if we relax OCR-only)

Uses gt answers as a proxy for clean answers (valid: targets are built from the
clean msg only when it matches gt anyway). The remaining gap between (A) and the
observed attack rate is the clean-correct filter + attack-time failures, which
hit all policies proportionally.

    python3 scripts/34_l2_yield_analysis.py data/chunk_l2.txt data/chunk_extra.txt
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
from dawg.attacks.l2_targets import make_ocr_target  # noqa: E402


def analyze(pages_file: Path) -> None:
    pages = [Path(l) for l in pages_file.read_text().split() if l]
    n = Counter()
    qa0_kinds, best_kinds = Counter(), Counter()
    qtypes = Counter()
    eligible_per_page = Counter()  # how many OCR+perturbable QAs each page has
    sites_any = set()

    for pd in pages:
        try:
            meta = json.loads((pd / "meta.json").read_text())
            qas = meta["qa"]
        except Exception:
            n["meta_fail"] += 1
            continue
        n["pages"] += 1
        for qa in qas:
            qtypes[qa.get("question_type", "?").upper()] += 1

        # A) current policy: qa[0] only
        qa0 = qas[0]
        if qa0.get("question_type", "").upper() == "OCR":
            n["A_qa0_ocr"] += 1
            tg = make_ocr_target(qa0["answer"], seed=0)
            if tg:
                n["A_qa0_attackable"] += 1
                qa0_kinds[tg[1]] += 1

        # B) proposed: scan all QAs for OCR + perturbable
        good = []
        for qa in qas:
            if qa.get("question_type", "").upper() != "OCR":
                continue
            tg = make_ocr_target(qa["answer"], seed=0)
            if tg:
                good.append(tg[1])
        eligible_per_page[len(good)] += 1
        if good:
            n["B_any_ocr_attackable"] += 1
            sites_any.add(pd.parent.name)
            # kind balance if we prefer rare kinds (yesno > year > list > number)
            rank = {"yesno_flip": 0, "year_perturb": 1, "list_drop": 2, "number_perturb": 3}
            best_kinds[min(good, key=lambda k: rank[k])] += 1

        # C) upper bound: any QA type
        if any(make_ocr_target(qa["answer"], seed=0) for qa in qas):
            n["C_anytype_attackable"] += 1

    p = n["pages"] or 1
    print(f"\n=== {pages_file.name}: {n['pages']} pages ({n['meta_fail']} meta-fail) ===")
    print(f"  question_type mix (all QAs): {dict(qtypes.most_common())}")
    print(f"  A) CURRENT  qa[0] OCR:              {n['A_qa0_ocr']:5d}  ({100*n['A_qa0_ocr']//p}%)")
    print(f"     CURRENT  qa[0] OCR+perturbable:  {n['A_qa0_attackable']:5d}  ({100*n['A_qa0_attackable']//p}%)  <- static yield today")
    print(f"  B) PROPOSED any OCR qa perturbable: {n['B_any_ocr_attackable']:5d}  ({100*n['B_any_ocr_attackable']//p}%)  across {len(sites_any)} sites")
    print(f"  C) UPPER    any qa type perturbable:{n['C_anytype_attackable']:5d}  ({100*n['C_anytype_attackable']//p}%)")
    mult = n["B_any_ocr_attackable"] / max(n["A_qa0_attackable"], 1)
    print(f"  B/A multiplier: {mult:.2f}x")
    print(f"  eligible-QAs-per-page dist: {dict(sorted(eligible_per_page.items()))}")
    print(f"  lie kinds  A (qa[0]):        {dict(qa0_kinds.most_common())}")
    print(f"  lie kinds  B (rare-pref):    {dict(best_kinds.most_common())}")


if __name__ == "__main__":
    for f in sys.argv[1:] or [str(DAWG_ROOT / "data" / "chunk_l2.txt")]:
        analyze(Path(f))
