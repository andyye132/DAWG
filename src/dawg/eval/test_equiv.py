"""Smoke test for `dawg.eval.equiv.same_meaning`.

Run directly:
    python -m dawg.eval.test_equiv
or
    PYTHONPATH=src python src/dawg/eval/test_equiv.py

Prints each pair's normalized form, the path taken (exact / jaccard / mpnet),
the score, and pass/fail vs the expected outcome. Exits 0 if all pass, 1
otherwise.
"""

from __future__ import annotations

import sys

from dawg.eval.equiv import same_meaning


# (a, b, expected_same)
PAIRS: list[tuple[str, str, bool]] = [
    ("$184.99", "184.99 dollars", True),
    ("$184.99", "$185.00", False),  # nearby numbers should NOT match
    ("Marcus Chen", "By Marcus Chen", True),
    ("Marcus Chen", "Sarah Whitfield", False),
    (
        "City Council Approves Riverfront Expansion in Late-Night Vote",
        "Council OKs riverfront expansion",
        True,
    ),
    ('{"name": "send_msg_to_user", "msg": "$184.99"}', "184.99 dollars", True),
    (
        '{"name": "send_msg_to_user", "msg": "Marcus Chen"}',
        "Author is Marcus Chen",
        True,
    ),
    ("4.6 out of 5", "4.5 stars", False),  # different rating shouldn't match
    (
        "The Trail Runner Pro 9 Hiking Boots in Slate Grey",
        "Trail Runner Pro 9 Hiking Boots, Slate Grey",
        True,
    ),
    ("best hiking boots 2026", "best hiking boots 2025", False),  # different year
]


def main() -> int:
    passes = 0
    fails: list[tuple[int, dict, bool]] = []

    print(f"Running {len(PAIRS)} pair(s) through same_meaning(...)")
    print("=" * 80)

    for i, (a, b, expected) in enumerate(PAIRS):
        details = same_meaning(a, b, return_details=True)
        got = details["match"]
        ok = got is expected
        marker = "PASS" if ok else "FAIL"
        print(f"[{i:02d}] {marker}  path={details['path']:<7} "
              f"score={details['score']:.3f}  "
              f"expected={expected}  got={got}")
        print(f"     a_raw : {a!r}")
        print(f"     b_raw : {b!r}")
        print(f"     a_norm: {details['a_norm']!r}")
        print(f"     b_norm: {details['b_norm']!r}")
        print()
        if ok:
            passes += 1
        else:
            fails.append((i, details, expected))

    print("=" * 80)
    print(f"Result: {passes} / {len(PAIRS)} passed")
    if fails:
        print(f"Failed indices: {[i for i, _, _ in fails]}")
        for i, d, exp in fails:
            print(f"  [{i}] path={d['path']} score={d['score']:.3f} "
                  f"expected={exp} got={d['match']}")
            print(f"       a_norm={d['a_norm']!r}")
            print(f"       b_norm={d['b_norm']!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
