"""Load the trained discriminator, re-run the site-disjoint test split, and plot
ROC + clean-vs-adversarial score histogram -> dawg/poster_figs/. Needs a GPU."""
import importlib.util, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/gscratch/raivn/andy132/dawg")
OUT = ROOT / "poster_figs"; OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("disc", ROOT / "scripts" / "40_train_discriminator.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

random.seed(0); np.random.seed(0); torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
pos, neg = m.discover(ROOT)
test_sites = m.site_split({d["site"] for d in pos} | {d["site"] for d in neg}, 0.2, seed=0)
random.shuffle(neg); neg = neg[:int(1.5 * len(pos))]
te = [d for d in pos + neg if d["site"] in test_sites]
print(f"test items: {len(te)}", flush=True)

model = m.build_model().to(dev)
model.load_state_dict(torch.load(ROOT / "results" / "discriminator" / "best.pt", map_location=dev)["model"])
model.eval()
dl = DataLoader(m.make_dataset(te, 512, False), batch_size=24, num_workers=8, pin_memory=True)
ys, ss = [], []
with torch.no_grad():
    for x, y, g in dl:
        s = torch.sigmoid(model(x.to(dev))).float().cpu().numpy()
        ys += y.tolist(); ss += s.tolist()
ys, ss = np.array(ys), np.array(ss)
np.savez(OUT / "disc_scores.npz", y=ys, s=ss)

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(ys, ss); auc = roc_auc_score(ys, ss)

fig, ax = plt.subplots(figsize=(4.3, 3.6))
ax.plot(fpr, tpr, lw=2.2, color="#2a7fff", label=f"AUROC = {auc:.4f}")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Discriminator ROC (site-disjoint test)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.savefig(OUT / "fig_disc_roc.png", dpi=200, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.hist(ss[ys == 0], bins=40, alpha=0.65, label="clean", color="#2a9d4a", log=True)
ax.hist(ss[ys == 1], bins=40, alpha=0.65, label="adversarial", color="#c0392b", log=True)
ax.set_xlabel("discriminator score"); ax.set_ylabel("count (log)")
ax.set_title("Clean vs. adversarial scores (cleanly separated)"); ax.legend()
fig.savefig(OUT / "fig_disc_hist.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"wrote fig_disc_roc.png + fig_disc_hist.png  AUROC={auc:.4f}", flush=True)
