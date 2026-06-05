"""Locate a query's answer region on a CLEAN screenshot via MolmoWeb's own grounding.

MolmoWeb (style="demo", the predict() default) answers
    "Click on the element that answers the following question: <q>"
with a JSON click action whose x,y are PERCENTAGES (0-100) of the ORIGINAL image
dims. The vision encoder letterboxes internally, but the model emits coords
relative to the original unpadded image, so we DO NOT undo any padding — we
multiply the fraction straight by W,H. (Verified against olmo/agent code:
multimodal_agent.py _pct_to_coord, point_formatter.py; workflow
'molmoweb-pointing-recipe', 2026-06-03.)

We point on the CLEAN image, pre-attack, then center the adversarial patch on
that point — the query's genuine point of interest.
"""
from __future__ import annotations

import json
import re

POINT_PROMPT = "Click on the element that answers the following question: {q}"


def point_to_pixel(raw_output: str, W: int, H: int) -> tuple[int, int]:
    """Parse a MolmoWeb grounding output -> integer pixel (x, y) on the W x H image.

    Handles the JSON click action (x,y are 0-100 percentages) and a <points> XML
    fallback (0-1000 or 0-100 scale). No letterbox/padding inverse is applied.
    Raises ValueError if no coordinate can be parsed.
    """
    text = (raw_output or "").strip()
    fx = fy = None

    # Path A: JSON action (default predict() output)
    try:
        obj = json.loads(text)
        action = obj.get("action", obj) if isinstance(obj, dict) else obj
        if isinstance(action, dict) and "x" in action and "y" in action:
            x_pct, y_pct = float(action["x"]), float(action["y"])
            if action.get("name") == "gemini_type_text_at":  # known 10x quirk
                x_pct /= 10.0
                y_pct /= 10.0
            fx, fy = x_pct / 100.0, y_pct / 100.0
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Path B: <points coords="idx x y">label</points>
    if fx is None:
        m = re.search(r'coords="([0-9\t :;,.]+)"', text)
        if m:
            nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", m.group(1))
            if len(nums) >= 3:
                xv, yv = float(nums[1]), float(nums[2])
                scale = 1000.0 if (xv > 100.0 or yv > 100.0) else 100.0
                fx, fy = xv / scale, yv / scale

    # Path C: last-resort bare "x y" (treat as 0-100 percentages)
    if fx is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(nums) >= 2:
            fx, fy = float(nums[0]) / 100.0, float(nums[1]) / 100.0

    if fx is None or fy is None:
        raise ValueError(f"no coordinate parsed from: {raw_output!r}")

    x = max(0, min(int(round(fx * W)), W - 1))
    y = max(0, min(int(round(fy * H)), H - 1))
    return x, y


def point_to_answer(pred, image_np, question: str):
    """Ask MolmoWeb where the answer is. Returns ((x,y) or None, raw_output_str)."""
    raw = pred.predict(POINT_PROMPT.format(q=question), image_np)
    H, W = image_np.shape[:2]
    try:
        return point_to_pixel(raw, W, H), raw
    except ValueError:
        return None, raw


def point_to_top_k(pred, image_np, question: str, k: int = 3,
                   blackout_frac: float = 0.05):
    """Iteratively ask MolmoWeb to click the most relevant element, blacking out
    a `blackout_frac`-area rect around each returned point so the NEXT query is
    forced to pick a different region. Returns (points, raws) with up to k
    distinct (x,y) POIs.

    Rationale: a redundant answer (repeated headline, label + table, etc.) keeps
    an un-attacked copy when we patch only the single POI. Re-pointing after
    masking each hit surfaces the top-k distinct evidence regions so the attack
    can cover all the copies. Costs k forward passes (cheap vs the PGD)."""
    import numpy as np
    img = np.array(image_np, copy=True)
    H, W = img.shape[:2]
    pts, raws = [], []
    for _ in range(k):
        pt, raw = point_to_answer(pred, img, question)
        raws.append(raw)
        if pt is None:
            break
        pts.append(pt)
        x, y, w, h = centered_bbox_at(pt[0], pt[1], W, H, blackout_frac)
        img[y:y + h, x:x + w] = 0  # mask this hit so the next query moves on
    return pts, raws


def centered_bbox_at(cx: int, cy: int, W: int, H: int,
                     area_frac: float, aspect: float = 1.5) -> list[int]:
    """Bbox of up to `area_frac` of the image, CENTERED on (cx,cy), CLIPPED to bounds.

    We clip (shrink) rather than shift. The old version did
    `x = max(0, min(cx - w//2, W - w))`, which slides the whole box in-bounds when
    the POI is near an edge — so the patch comes OFF the POI into the opposite
    margin, and the offset GROWS with patch size (a big patch ends up mostly on
    blank background instead of the answer). Clipping keeps the patch centred on
    the answer location; the cost is a smaller realized area near edges, so
    callers should record the realized w*h (not the requested `area_frac`).
    """
    a = area_frac * W * H
    w = max(14, min(int(round((a * aspect) ** 0.5)), W))
    h = max(14, min(int(round((a / aspect) ** 0.5)), H))
    left = max(0, cx - w // 2)
    right = min(W, cx + (w - w // 2))
    top = max(0, cy - h // 2)
    bottom = min(H, cy + (h - h // 2))
    return [left, top, right - left, bottom - top]


def content_bboxes(image_np, n: int, area_frac: float, aspect: float = 1.5,
                   exclude: list[int] | None = None, grid=(6, 8)) -> list[list[int]]:
    """Up to `n` content-dense, non-overlapping boxes ranked by local pixel variance.

    A cheap, model-free way to place EXTRA patches (for multi-patch attacks) on
    parts of the page that actually have content, avoiding blank margins and the
    `exclude` box (e.g. the query POI patch).
    """
    import numpy as np
    H, W = image_np.shape[:2]
    gy, gx = grid
    ch, cw = max(1, H // gy), max(1, W // gx)
    g = image_np.astype(np.float32)
    cells = []
    for i in range(gy):
        for j in range(gx):
            block = g[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            if block.size:
                cells.append((float(block.std()), j * cw + cw // 2, i * ch + ch // 2))
    cells.sort(reverse=True)  # highest-variance (most content) first

    a = area_frac * W * H
    w = max(14, min(int((a * aspect) ** 0.5), W))
    h = max(14, min(int((a / aspect) ** 0.5), H))
    avoid = [exclude] if exclude else []
    chosen: list[list[int]] = []

    def far_enough(cx, cy, boxes):
        for (bx, by, bw, bh) in boxes:
            if abs(cx - (bx + bw / 2)) < 0.8 * (w + bw) / 2 and abs(cy - (by + bh / 2)) < 0.8 * (h + bh) / 2:
                return False
        return True

    for _std, cx, cy in cells:
        if len(chosen) >= n:
            break
        if not far_enough(cx, cy, avoid + chosen):
            continue
        x = max(0, min(cx - w // 2, W - w))
        y = max(0, min(cy - h // 2, H - h))
        chosen.append([x, y, w, h])
    return chosen
