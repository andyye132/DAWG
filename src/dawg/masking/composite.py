"""Path B: PIL composite of a patch onto a pre-rendered screenshot.

Pixel-perfect — no browser rendering, no DPR or anti-aliasing losses, no
z-index battles. Useful for fast dataset generation and PGD iteration, but
NOT a real attack: a real attacker can't paint pixels onto the agent's
screenshot; they can only serve HTML. For the real-attack pathway through
Playwright, see overlay.py (Path A).
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

ImageLike = bytes | bytearray | str | Path | Image.Image


def composite_patch(
    screenshot: ImageLike,
    patch: ImageLike,
    xy: tuple[int, int],
    *,
    size: tuple[int, int] | None = None,
    out_path: str | Path | None = None,
) -> Image.Image:
    """Paste `patch` onto `screenshot` at top-left coords `xy`.

    Args:
        screenshot: PNG bytes, a filesystem path, or an open PIL Image.
        patch: a path or open PIL Image. PNG preferred to avoid re-encoding.
        xy: (x, y) top-left of the patch in screenshot pixel coords.
        size: optional (w, h) to resize the patch (NEAREST) before pasting.
            Default uses the patch's native size, which preserves PGD pixels.
        out_path: if given, save the composite PNG here.

    Returns:
        The composited PIL Image (RGB).
    """
    base = _to_image(screenshot)
    p = _to_image(patch)
    if size is not None:
        p = p.resize(size, Image.Resampling.NEAREST)
    base.paste(p, xy)
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        base.save(out)
    return base


def _to_image(x: ImageLike) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.copy().convert("RGB")
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    raise TypeError(f"unsupported input type for image: {type(x).__name__}")
