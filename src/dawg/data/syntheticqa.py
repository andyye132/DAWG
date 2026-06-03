"""Stream a small, diverse subset of allenai/MolmoWeb-SyntheticQA to disk.

The full dataset is ~343k screenshots across 645 websites, stored as 5 large
HF Arrow shards with images embedded — far too big to land whole on a laptop.
We instead *stream* (`streaming=True`, no on-disk cache) and keep a capped,
diverse subset.

On-disk layout written under `out_root`:

    <out_root>/<website>/page000/screenshot.png   # the clean page, NATIVE res
                                /meta.json          # this page's QA + provenance
    <out_root>/<website>/page001/...
    <out_root>/index.json                           # manifest of all pages

`meta.json` schema (one per page = one SyntheticQA row):

    {
      "website": "247sports",
      "url": "https://247sports.com/",
      "page": "page000",
      "source_index": 0,                  # row position in the stream
      "image": {"file": "screenshot.png", "width": 1920, "height": 1080},
      "qa": [
        {"question": "...", "answer": "...",
         "question_type": "OCR", "question_form": "first_person"},
        ...                               # ~5 per page
      ]
    }

Screenshots are kept at NATIVE resolution (SyntheticQA images are
1366x768 / 1536x864 / 1920x1080 — never MolmoWeb's 1280x720) because each
row's QA answers were authored against the original screenshot. The masking
stage writes its outputs (adversarial_site.png, attack_meta.json) alongside
these without mutating meta.json.

Website ordering in the stream is alphabetical and contiguous, so "first N
distinct websites" is a cheap, deterministic subset. Big sites still cost
bandwidth to stream past once their page cap is hit; `max_scan` bounds that.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

REPO_ID = "allenai/MolmoWeb-SyntheticQA"
SPLIT = "train"

_SLUG_BAD = re.compile(r"[^a-z0-9._-]+")


def slugify_site(name: str) -> str:
    """Make a website name safe as a directory name.

    SyntheticQA websites are already clean slugs ("247sports", "32degrees",
    "9animetv"), but a stray "/" or space would break the layout, so we
    normalize defensively.
    """
    s = (name or "unknown").strip().lower()
    s = _SLUG_BAD.sub("_", s)
    s = s.strip("_.-")
    return s or "unknown"


def page_dirname(idx: int) -> str:
    """Zero-padded page directory name, e.g. 0 -> 'page000'."""
    return f"page{idx:03d}"


@dataclass
class FetchStats:
    scanned: int = 0
    written: int = 0
    sites: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"scanned": self.scanned, "written": self.written,
                "sites": dict(self.sites)}


def _write_page(page_dir: Path, row: dict, source_index: int) -> dict:
    """Write screenshot.png + meta.json for one row. Returns the manifest entry."""
    page_dir.mkdir(parents=True, exist_ok=True)
    img = row["image"]  # PIL Image (RGB) when streamed via `datasets`
    w, h = img.size
    img.save(page_dir / "screenshot.png")  # native res, lossless

    qa = [
        {
            "question": m.get("question", ""),
            "answer": m.get("answer", ""),
            "question_type": m.get("question_type", ""),
            "question_form": m.get("question_form", ""),
        }
        for m in row.get("messages", [])
    ]
    md = row.get("metadata", {}) or {}
    meta = {
        "website": md.get("website", ""),
        "url": md.get("url", ""),
        "page": page_dir.name,
        "source_index": source_index,
        "image": {"file": "screenshot.png", "width": w, "height": h},
        "qa": qa,
    }
    (page_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {
        "website": meta["website"],
        "page": page_dir.name,
        "path": str(page_dir.relative_to(page_dir.parent.parent)),
        "n_qa": len(qa),
        "image": meta["image"],
        "source_index": source_index,
    }


def fetch_subset(
    out_root: Path,
    *,
    n_sites: int = 8,
    pages_per_site: int = 15,
    max_scan: int = 2500,
    stale_limit: int = 200,
    repo_id: str = REPO_ID,
    split: str = SPLIT,
    progress_every: int = 100,
    log=print,
) -> FetchStats:
    """Stream `repo_id` and write the first `n_sites` distinct websites.

    Greedy + contiguous: accept the first `n_sites` websites encountered, keep
    up to `pages_per_site` rows each, writing each row to disk immediately
    (so we never buffer decoded images).

    Stops on the first of: every accepted site hit its cap; the roster is full
    and no page has been written for `stale_limit` consecutive scanned rows
    (handles tiny sites that can never reach the cap, e.g. a site with 4 rows);
    or `max_scan` rows have been streamed (hard bandwidth bound).

    Returns FetchStats. Also writes `<out_root>/index.json`.
    """
    from datasets import load_dataset

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"[fetch] streaming {repo_id}:{split} "
        f"(n_sites={n_sites}, pages_per_site={pages_per_site}, max_scan={max_scan})")
    ds = load_dataset(repo_id, split=split, streaming=True)

    accepted: dict[str, int] = {}   # slug -> pages written so far
    manifest: list[dict] = []
    stats = FetchStats()
    since_write = 0                  # consecutive scanned rows with no write

    for row in ds:
        if stats.scanned >= max_scan:
            log(f"[fetch] hit max_scan={max_scan}; stopping scan")
            break
        stats.scanned += 1
        wrote_before = stats.written

        site_raw = (row.get("metadata", {}) or {}).get("website", "")
        slug = slugify_site(site_raw)

        if slug in accepted:
            if accepted[slug] >= pages_per_site:
                pass  # cap reached; keep streaming to reach the next site
            else:
                idx = accepted[slug]
                entry = _write_page(out_root / slug / page_dirname(idx),
                                    row, stats.scanned - 1)
                accepted[slug] = idx + 1
                manifest.append(entry)
                stats.written += 1
        elif len(accepted) < n_sites:
            accepted[slug] = 0
            entry = _write_page(out_root / slug / page_dirname(0),
                                row, stats.scanned - 1)
            accepted[slug] = 1
            manifest.append(entry)
            stats.written += 1
            log(f"[fetch] + new site #{len(accepted)}: {slug}")
        # else: site not accepted and roster full -> skip

        since_write = 0 if stats.written > wrote_before else since_write + 1

        if progress_every and stats.scanned % progress_every == 0:
            filled = sum(1 for c in accepted.values() if c >= pages_per_site)
            log(f"[fetch]   scanned={stats.scanned} written={stats.written} "
                f"sites={len(accepted)} filled={filled}")

        roster_full = len(accepted) >= n_sites
        if roster_full and all(c >= pages_per_site for c in accepted.values()):
            log("[fetch] all accepted sites filled; stopping")
            break
        if roster_full and since_write >= stale_limit:
            log(f"[fetch] roster full and no writes for {stale_limit} rows; stopping")
            break

    stats.sites = dict(accepted)
    index = {
        "repo_id": repo_id,
        "split": split,
        "params": {"n_sites": n_sites, "pages_per_site": pages_per_site,
                   "max_scan": max_scan},
        "stats": stats.as_dict(),
        "pages": manifest,
    }
    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    log(f"[fetch] DONE  sites={len(accepted)}  pages={stats.written}  "
        f"scanned={stats.scanned}  -> {out_root}")
    for slug, c in accepted.items():
        log(f"[fetch]    {slug:24s} {c} pages")
    return stats
