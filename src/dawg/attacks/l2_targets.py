"""Deterministic L2 target generation for OCR-type answers.

Corrupt exactly ONE load-bearing fact while keeping the rest identical -> a
"minimum lie, maximum truth". Tries (in order): yes/no flip, year/date perturb,
money/percent/number perturb, list-item drop. Returns (target_answer, kind) or
None if nothing OCR-perturbable is found (caller skips / falls back).

The load-bearing token mask for the masked targeted-CE loss is derived in
pixel_pgd by diffing clean vs target token ids, so we only need to return the
corrupted string here.
"""
from __future__ import annotations

import random
import re

_YESNO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
# money like $184.99 / 184.99 / 1,234 / 4.6 / 50% (captures the numeric core)
_NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")


def _perturb_number(s: str, rng: random.Random) -> str:
    """Return a plausibly-different number string, preserving decimal precision."""
    raw = s.replace(",", "")
    if "." in raw:
        prec = len(raw.split(".")[1])
        delta = rng.choice([5.0, 10.0, -10.0, 1.0, -1.0, 0.10, 2.0])
        v = float(raw) + delta
        if v <= 0:
            v = float(raw) + abs(delta)
        return f"{v:.{prec}f}"
    v = int(raw) + rng.choice([1, 2, 5, 10, -1, -5])
    if v <= 0:
        v = int(raw) + 5
    return str(v)


def make_ocr_target(answer: str, *, seed: int | None = None):
    """Return (target_answer, kind) or None. Deterministic given (answer, seed)."""
    rng = random.Random(seed if seed is not None else (hash(answer) & 0x7FFFFFFF))
    a = answer

    # 1) yes/no flip
    m = _YESNO.search(a)
    if m:
        repl = "No" if m.group(1).lower() == "yes" else "Yes"
        return a[:m.start()] + repl + a[m.end():], "yesno_flip"

    # 2) year perturb
    m = _YEAR.search(a)
    if m:
        ny = int(m.group(0)) + rng.choice([-1, 1, -2, 2])
        return a[:m.start()] + str(ny) + a[m.end():], "year_perturb"

    # 3) money/percent/number perturb (the dominant OCR case)
    nums = list(_NUM.finditer(a))
    if nums:
        m = rng.choice(nums)
        try:
            new = _perturb_number(m.group(1), rng)
            if new != m.group(1):
                return a[:m.start(1)] + new + a[m.end(1):], "number_perturb"
        except Exception:
            pass

    # 4) list: drop the last comma/and-separated item
    parts = re.split(r",\s*|\s+and\s+", a.strip())
    if len(parts) >= 3:
        dropped = ", ".join(parts[:-1])
        return dropped, "list_drop"

    return None


if __name__ == "__main__":
    for s in ["$184.99", "4.6 out of 5", "yes, it is available", "NCAAF rankings 2026",
              "Homes, Experiences, and Services", "Watch Anime Online Free", "50% off"]:
        print(f"{s!r:45} -> {make_ocr_target(s, seed=0)}")
