"""DAWG discriminator: whole-image noise-residual CNN that flags adversarially
masked screenshots. Per DISCRIMINATOR_SPEC.md, tailored to the team's choices:

  - WHOLE-IMAGE (not tiled): letterbox to 512x512.
  - Residual front-end: fixed SRM high-pass kernels + a learnable Bayar
    constrained conv, concatenated with RGB -> EfficientNet-B0 (torchvision).
  - Positives = L1 (single10/two5/three33) + L2 adversarial screenshots.
  - Negatives = clean screenshots whose (site,page) was NEVER attacked
    (the "don't reuse an attacked page's clean twin" dedup) drawn from
    data/syntheticqa_deep, across all sites.
  - SITE-DISJOINT train/test split (learn the patch, not the site).

Reports held-out AUROC + TPR@1%FPR + per-geometry breakdown; saves best ckpt.

    python scripts/40_train_discriminator.py --epochs 14 --img 512 --bs 24
"""
from __future__ import annotations
import argparse, json, hashlib, random, time, sys
from pathlib import Path
import numpy as np

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Data discovery
# --------------------------------------------------------------------------- #
def discover(root: Path):
    """Return (positives, negatives) lists of dicts {path, site, page, geom}."""
    pos = []
    for geom, dirn, advname in [("single10", "l1_single10", "adv_q0.png"),
                                ("two5", "l1_two5", "adv_q0.png"),
                                ("three33", "l1_three33", "adv_q0.png"),
                                ("L2", "l2_dataset", "adv.png")]:
        base = root / "results" / dirn
        for p in base.glob(f"*/page*/{advname}"):
            pos.append({"path": str(p), "site": p.parent.parent.name,
                        "page": p.parent.name, "geom": geom, "label": 1})
    attacked = {(d["site"], d["page"]) for d in pos}

    neg = []
    for p in (root / "data" / "syntheticqa_deep").glob("*/page*/screenshot.png"):
        site, page = p.parent.parent.name, p.parent.name
        if (site, page) in attacked:
            continue                       # dedup: skip attacked pages' clean twins
        neg.append({"path": str(p), "site": site, "page": page, "geom": "clean", "label": 0})
    return pos, neg


def site_split(sites, frac_test=0.2, seed=0):
    """Deterministic site-disjoint split by hashing the site name."""
    test = set()
    for s in sorted(sites):
        h = int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) % 1000
        if h < frac_test * 1000:
            test.add(s)
    return test


# --------------------------------------------------------------------------- #
# Residual front-end (FIXED SRM high-pass only — no learnable Bayar)
# --------------------------------------------------------------------------- #
# NOTE: the earlier learnable Bayar constrained-conv was numerically unstable
# (constrain() divides by a near-zero neighbour-sum -> inf -> NaN cascaded through
# the whole net; "AUROC 0.8578" was a degenerate artifact of AUROC on all-NaN
# scores). FIXED SRM kernels are bounded and stable. SRM gives 3 residual maps ->
# fed straight into the UNMODIFIED 3-channel EfficientNet stem (no surgery). The
# high-pass strips page CONTENT so the net can't shortcut on which page it is.
def build_model():
    import torch
    import torch.nn as nn
    import torchvision

    SRM = torch.tensor([  # 3 classic SRM high-pass residual kernels (5x5)
        [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
        [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
        [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
    ], dtype=torch.float32)
    SRM = SRM / SRM.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1)

    class ResidualCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("srm", SRM.unsqueeze(1))  # (3,1,5,5)
            net = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            net.classifier[0].p = 0.4                       # dropout vs memorization
            net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
            self.net = net
        def residuals(self, x):
            gray = x.mean(1, keepdim=True)
            srm = torch.nn.functional.conv2d(gray, self.srm, padding=2)  # (B,3,H,W)
            return torch.clamp(srm, -4.0, 4.0)              # bounded high-pass residual
        def forward(self, x):
            return self.net(self.residuals(x)).squeeze(1)

    return ResidualCNN()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def make_dataset(items, img, train):
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import io

    class DS(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def _load(self, path):
            im = Image.open(path).convert("RGB")
            w, h = im.size
            s = img / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            canvas = Image.new("RGB", (img, img), (128, 128, 128))  # letterbox
            canvas.paste(im, ((img - im.size[0]) // 2, (img - im.size[1]) // 2))
            return canvas
        def __getitem__(self, i):
            d = self.items[i]
            im = self._load(d["path"])
            if train:
                if random.random() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:  # mild JPEG (q>=88), identical on both classes
                    b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(88, 100))
                    im = Image.open(b).convert("RGB")
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            if train and random.random() < 0.5:
                x = x * (0.9 + 0.2 * random.random())  # brightness jitter
                x = x.clamp(0, 1)
            return x, float(d["label"]), d["geom"]

    return DS(items)


def auroc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # rank-based fallback
        order = np.argsort(scores)
        ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
        pos = np.array(labels) == 1; npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0 or nneg == 0:
            return float("nan")
        return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tpr_at_fpr(labels, scores, fpr=0.01):
    labels, scores = np.array(labels), np.array(scores)
    neg = scores[labels == 0]
    if len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    pos = scores[labels == 1]
    return float((pos >= thr).mean()) if len(pos) else float("nan")


def main() -> int:
    import torch
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DAWG_ROOT)
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "discriminator")
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--neg-ratio", type=float, default=1.5, help="negatives per positive")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    pos, neg = discover(args.root)
    pos_sites = {d["site"] for d in pos}
    test_sites = site_split(pos_sites | {d["site"] for d in neg}, 0.2, seed=0)
    # cap negatives to neg_ratio * positives, sampled across sites
    random.shuffle(neg)
    neg = neg[:int(args.neg_ratio * len(pos))]
    tr = [d for d in pos + neg if d["site"] not in test_sites]
    te = [d for d in pos + neg if d["site"] in test_sites]
    random.shuffle(tr)
    print(f"positives={len(pos)} negatives={len(neg)} | train={len(tr)} test={len(te)} "
          f"| test_sites={len(test_sites)} (held-out)", flush=True)
    print(f"  train pos/neg = {sum(d['label'] for d in tr)}/{sum(1-d['label'] for d in tr)} "
          f"| test pos/neg = {sum(d['label'] for d in te)}/{sum(1-d['label'] for d in te)}", flush=True)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()  # fp32 throughout: SRM/Bayar residuals overflow fp16 -> nan
    dl_tr = DataLoader(make_dataset(tr, args.img, True), batch_size=args.bs,
                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    dl_te = DataLoader(make_dataset(te, args.img, False), batch_size=args.bs,
                       shuffle=False, num_workers=8, pin_memory=True)

    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), y * 0.9 + 0.05)   # label smoothing vs over-confidence
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            li = loss.item()
            if li == li:               # skip nan in the running average
                tot += li
        sched.step()
        # eval
        model.eval(); ys, ss, gs = [], [], []
        with torch.no_grad():
            for x, y, g in dl_te:
                s = torch.sigmoid(model(x.to(dev))).float().cpu().numpy()
                ys += y.tolist(); ss += s.tolist(); gs += list(g)
        a = auroc(ys, ss); t = tpr_at_fpr(ys, ss, 0.01)
        # per-geometry AUROC (positives of that geom vs all negatives)
        pg = {}
        negmask = [i for i, yy in enumerate(ys) if yy == 0]
        for geom in ("single10", "two5", "three33", "L2"):
            idx = [i for i, gg in enumerate(gs) if gg == geom] + negmask
            if any(ys[i] == 1 for i in idx):
                pg[geom] = round(auroc([ys[i] for i in idx], [ss[i] for i in idx]), 3)
        print(f"ep{ep:02d} loss={tot/max(len(dl_tr),1):.4f} AUROC={a:.4f} TPR@1%FPR={t:.3f} "
              f"per-geom={pg} ({time.time()-t0:.0f}s)", flush=True)
        if a > best:
            best = a
            torch.save({"model": model.state_dict(), "auroc": a, "tpr@1fpr": t,
                        "per_geom": pg, "epoch": ep, "img": args.img},
                       args.out / "best.pt")
            (args.out / "metrics.json").write_text(json.dumps(
                {"best_auroc": a, "tpr@1fpr": t, "per_geom": pg, "epoch": ep,
                 "n_pos": len(pos), "n_neg": len(neg), "test_sites": len(test_sites)}, indent=2))
    print(f"\nDONE best AUROC={best:.4f} -> {args.out}/best.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
