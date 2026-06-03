"""Crop selection + placeholder masks for the dataset-driven attack pipeline.

This is the geometry half of the attack. For the MVP the "mask" is a solid
color block — a stand-in for the eventual PGD-optimized patch. Everything here
(random crop, recorded bbox, recompositing at the same coords) is exactly what
the real attack will use; only `make_color_block` gets swapped for the
optimizer output later, so the on-disk contract (adversarial_site.png +
attack_meta.json with patch_bbox) stays stable.

CPU-only. Reuses dawg.masking.composite (Path B, pixel paste) — appropriate
here because SyntheticQA gives us screenshots, not HTML, so the threat-faithful
Playwright overlay (Path A) isn't available for these pages.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from dawg.masking.composite import composite_patch


@dataclass(frozen=True)
class Bbox:
    """Axis-aligned box in image pixel coords: top-left (x, y) + size (w, h)."""
    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @property
    def area(self) -> int:
        return self.w * self.h


def sample_crop_bbox(
    img_w: int,
    img_h: int,
    *,
    area_frac_range: tuple[float, float] = (0.02, 0.06),
    aspect_range: tuple[float, float] = (0.5, 2.0),
    rng: random.Random | None = None,
) -> Bbox:
    """Sample a random rectangle covering `area_frac_range` of the image.

    Area fraction and aspect ratio (w/h) are sampled uniformly; the top-left is
    then placed so the box fits fully inside the image. Falls back to a centered
    square at the minimum area fraction if sampling somehow fails to fit.
    """
    rng = rng or random.Random()
    img_area = img_w * img_h
    for _ in range(100):
        frac = rng.uniform(*area_frac_range)
        target_area = frac * img_area
        aspect = rng.uniform(*aspect_range)  # w / h
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w < 1 or h < 1 or w > img_w or h > img_h:
            continue
        x = rng.randint(0, img_w - w)
        y = rng.randint(0, img_h - h)
        return Bbox(x, y, w, h)

    side = max(1, min(int((area_frac_range[0] * img_area) ** 0.5), img_w, img_h))
    return Bbox((img_w - side) // 2, (img_h - side) // 2, side, side)


def random_color(rng: random.Random | None = None) -> tuple[int, int, int]:
    rng = rng or random.Random()
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def make_color_block(
    size: tuple[int, int],
    color: tuple[int, int, int] | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Solid RGB block of `size` = (w, h). Placeholder for the PGD patch."""
    w, h = int(size[0]), int(size[1])
    if color is None:
        color = random_color(rng)
    return Image.new("RGB", (w, h), tuple(color))


def page_seed(global_seed: int, key: str) -> int:
    """Deterministic per-page seed from a global seed + a stable key.

    Uses sha256 (not Python's salted hash) so reruns and parallel workers
    reproduce the same crops.
    """
    h = hashlib.sha256(f"{global_seed}:{key}".encode()).hexdigest()
    return int(h[:16], 16)


def make_adversarial_for_page(
    page_dir: str | Path,
    *,
    rng: random.Random,
    area_frac_range: tuple[float, float] = (0.02, 0.06),
    color: tuple[int, int, int] | None = None,
    source_name: str = "screenshot.png",
    out_name: str = "adversarial_site.png",
    meta_name: str = "attack_meta.json",
    overwrite: bool = False,
) -> dict | None:
    """Produce one adversarial page: sample a crop, paint a color block, recomposite.

    Reads `<page_dir>/screenshot.png`, writes `<page_dir>/adversarial_site.png`
    and `<page_dir>/attack_meta.json`. Returns the attack meta dict, or None if
    skipped (outputs exist and `overwrite` is False). Does NOT touch meta.json.
    """
    page_dir = Path(page_dir)
    shot = page_dir / source_name
    if not shot.exists():
        raise FileNotFoundError(shot)

    out_img = page_dir / out_name
    out_meta = page_dir / meta_name
    if out_img.exists() and out_meta.exists() and not overwrite:
        return None

    img = Image.open(shot).convert("RGB")
    W, H = img.size
    bbox = sample_crop_bbox(W, H, area_frac_range=area_frac_range, rng=rng)
    use_color = tuple(color) if color is not None else random_color(rng)
    block = make_color_block((bbox.w, bbox.h), color=use_color)

    # Recomposite the block at the recorded corner coords (Path B pixel paste).
    composite_patch(img, block, (bbox.x, bbox.y), out_path=out_img)

    meta = {
        "method": "color_block",        # placeholder; PGD swaps in here later
        "source_image": source_name,
        "adversarial_image": out_name,
        "patch_bbox": bbox.as_list(),   # [x, y, w, h] in screenshot px
        "image_size": [W, H],
        "area_frac": round(bbox.area / (W * H), 4),
        "color": list(use_color),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    return meta
