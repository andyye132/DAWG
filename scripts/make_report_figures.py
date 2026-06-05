"""Generate figures for the DAWG L1 research report -> results/figures/*.png."""
import glob, json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/gscratch/raivn/andy132/dawg"
if not os.path.isdir(BASE):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "results", "figures")
os.makedirs(FIG, exist_ok=True)


def load_cells(d):
    """dir of per-page JSONs -> list of (arm, eps, success, distance, clean_correct)."""
    out = []
    for f in glob.glob(os.path.join(BASE, "results", d, "*.json")):
        try:
            p = json.load(open(f))
        except Exception:
            continue
        cc = p.get("clean_correct", True)
        for c in p.get("cells", []):
            out.append((c["arm"], c["eps"], int(c["success"]), c["distance"], cc))
    return out


def asr_by(cells, arm=None, eps=None, cc_only=True):
    sel = [c for c in cells if (arm is None or c[0] == arm) and (eps is None or c[1] == eps)
           and (c[4] or not cc_only)]
    return (100.0 * sum(c[2] for c in sel) / len(sel)) if sel else 0.0, len(sel)


# ---- Fig 1: placement-strategy comparison (breadth eps8/16 + depth eps12) ----
sv1 = load_cells("strategy_v1")      # A/B/C x {8,16}
sd = load_cells("strategy_depth")    # A/B/C x {12}
arms = ["A_single10", "B_top3", "C_heuristic"]
labels = ["single 10%", "top-3", "heuristic 3x3.3%"]
groups = [("breadth eps8", sv1, 8.0), ("breadth eps16", sv1, 16.0), ("depth eps12", sd, 12.0)]
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(groups)); w = 0.25
colors = ["#2c7fb8", "#d95f0e", "#7fbf7b"]
for i, arm in enumerate(arms):
    vals = [asr_by(cells, arm, eps)[0] for (_, cells, eps) in groups]
    ax.bar([xi + (i-1)*w for xi in x], vals, w, label=labels[i], color=colors[i])
    for xi, v in zip(x, vals):
        ax.text(xi + (i-1)*w, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("Attack Success Rate (%)"); ax.set_ylim(0, 75)
ax.set_title("Placement strategy comparison (210 pages total)\nsingle 10% is best-or-tied in every condition")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_strategy.png"), dpi=120); plt.close()

# ---- Fig 2: potency curve (char_r3) single vs multi, ASR vs eps ----
ch = load_cells("char_r3")
epss = sorted({c[1] for c in ch})
fig, ax = plt.subplots(figsize=(8, 5))
for arm, lab, col in [("single10", "single 10%", "#2c7fb8"),
                      ("multi_even", "multi-even 3x3.3%", "#d95f0e"),
                      ("multi_random", "multi-random", "#7fbf7b")]:
    ys = [asr_by(ch, arm, e)[0] for e in epss]
    ax.plot(epss, ys, "-o", label=lab, color=col)
ax.set_xlabel("eps (Linf budget, /255)"); ax.set_ylabel("ASR (%)")
ax.set_title("Potency curve (char_r3, R=3, 15 pages)\nthreshold ~eps8, saturates eps12-24, ~73% ceiling")
ax.axvspan(12, 24, alpha=0.08, color="green"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_potency.png"), dpi=120); plt.close()

# ---- Fig 3: L1 dataset break-rate + distance histogram ----
rows = []
for f in glob.glob(os.path.join(BASE, "results", "l1_dataset", "shard*.jsonl")) + \
         glob.glob(os.path.join(BASE, "results", "l1_dataset_new", "shard*.jsonl")):
    for line in open(f):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
if rows:
    nb = sum(r["success"] for r in rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(["broke", "held"], [nb, len(rows)-nb], color=["#d95f0e", "#bbbbbb"])
    a1.set_title(f"L1 fallback dataset: {nb}/{len(rows)} broke ({100*nb/len(rows):.0f}%)")
    a1.set_ylabel("(clean, adversarial) pairs")
    for i, v in enumerate([nb, len(rows)-nb]):
        a1.text(i, v+20, str(v), ha="center")
    dists = [r["distance_clean"] for r in rows]
    a2.hist(dists, bins=20, color="#2c7fb8")
    a2.axvline(0.25, color="red", ls="--", label="success threshold (0.25)")
    a2.set_xlabel("semantic drift from clean answer"); a2.set_ylabel("count")
    a2.set_title("Drift distribution (bimodal: held ~0 vs broke ~1)"); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_l1_dataset.png"), dpi=120); plt.close()

# ---- Fig 4: per-page robustness (depth run) shows redundancy bimodality ----
perpage = defaultdict(lambda: [0, 0])
for f in glob.glob(os.path.join(BASE, "results", "strategy_depth", "*.json")):
    p = json.load(open(f))
    if not p.get("clean_correct"):
        continue
    site = p["site"]
    for c in p["cells"]:
        if c["arm"] == "A_single10":
            perpage[(site, p["page"])][0] += int(c["success"]); perpage[(site, p["page"])][1] += 1
fracs = [s/n for (s, n) in perpage.values() if n]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(fracs, bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], color="#7fbf7b", edgecolor="k")
ax.set_xlabel("per-page break fraction (single 10%)"); ax.set_ylabel("# pages")
ax.set_title("Per-page robustness is bimodal\n(some pages always break, redundant ones never do)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_robustness.png"), dpi=120); plt.close()

print("figures written:")
for f in sorted(glob.glob(os.path.join(FIG, "*.png"))):
    print(" ", f, os.path.getsize(f), "bytes")
