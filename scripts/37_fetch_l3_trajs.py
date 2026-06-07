"""Download MolmoWeb-SyntheticTrajs (screenshot -> action) steps for L3 action-hijack.

Streams the dataset (LOGIN NODE only — compute nodes have no internet), extracts
individual attackable steps (a screenshot + the ground-truth browser action), and
saves ~2.5k of them with full provenance. We keep screenshots + the action (incl.
click bbox / node_properties / target url) so L3 can: (a) pick a believable attacker
target element, (b) get MolmoWeb's NATIVE clean action via pred.predict() at attack
time (the dataset action is bid-grounded, not MolmoWeb's pixel format — see L3 memo).

    python 37_fetch_l3_trajs.py --config from_template --max-steps 2500 --out results/l3_trajs

Resumable: appends to steps.jsonl, skips already-saved (traj,step) ids.
"""
from __future__ import annotations
import argparse, io, json, os, sys, time
from pathlib import Path
from urllib.parse import urlparse

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent

ATTACKABLE = {"click", "mouse_click", "goto", "keyboard_type", "type",
              "scroll", "keyboard_press", "send_msg_to_user"}


def site_of(url: str) -> str:
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""


def decode_img(im):
    from PIL import Image
    if isinstance(im, Image.Image):
        return im
    if isinstance(im, dict):
        if im.get("bytes"):
            return Image.open(io.BytesIO(im["bytes"]))
        if im.get("path") and os.path.exists(im["path"]):
            return Image.open(im["path"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    # comma-separated configs; pulled round-robin so sites stay diverse.
    # from_template is single-site (allrecipes) — avoid; task_seeded_* are diverse.
    ap.add_argument("--configs", default="task_seeded_wv,task_seeded_om2w")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--out", type=Path, default=DAWG_ROOT / "results" / "l3_trajs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "steps.jsonl"
    configs = [c for c in args.configs.split(",") if c]

    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["traj_id"], r["step"]))
            except Exception:
                pass
    print(f"configs={configs} target={args.max_steps} already_have={len(done)} -> {args.out}", flush=True)
    if len(done) >= args.max_steps:
        print("target already reached."); return 0

    from datasets import load_dataset
    fout = manifest.open("a")
    n = len(done); t0 = time.time(); seen_traj = 0
    kinds: dict[str, int] = {}
    per_cfg_cap = -(-args.max_steps // len(configs))  # ceil: even share per config

    for cfg in configs:
      cfg_start = n
      ds = load_dataset("allenai/MolmoWeb-SyntheticTrajs", cfg, split="train", streaming=True)
      for row in ds:
        if n >= args.max_steps or (n - cfg_start) >= per_cfg_cap:
            break
        seen_traj += 1
        tid = row.get("sample_id") or f"row{seen_traj}"
        try:
            traj = json.loads(row["trajectory"])
            goal = json.loads(row["instruction"]).get("goal", "") if isinstance(row.get("instruction"), str) else ""
        except Exception:
            continue
        images = row.get("images") or []
        # site = first non-blank url in the trajectory
        site = ""
        for k in sorted(traj, key=lambda x: int(x)):
            u = (traj[k].get("other_obs") or {}).get("url", "")
            if u and u != "about:blank":
                site = site_of(u); break

        for k in sorted(traj, key=lambda x: int(x)):
            if n >= args.max_steps:
                break
            step = int(k)
            if (tid, step) in done:
                continue
            node = traj[k]
            act = node.get("action") or {}
            ao = act.get("action_output") or {}
            name = (ao.get("action_name") or act.get("action_str", "").split("(")[0]).strip()
            if name not in ATTACKABLE:
                continue
            img = images[step - 1] if step - 1 < len(images) else None
            pil = decode_img(img)
            if pil is None:
                continue
            pil = pil.convert("RGB")
            tdir = args.out / tid
            tdir.mkdir(parents=True, exist_ok=True)
            png = tdir / f"step{step:02d}.png"
            if not png.exists():
                pil.save(png)
            inner = ao.get("action") or {}
            rec = {
                "traj_id": tid, "step": step, "site": site,
                "url": (node.get("other_obs") or {}).get("url", ""),
                "goal": goal,
                "action_str": act.get("action_str", ""),
                "action_name": name,
                "action_params": inner,
                "bbox": inner.get("bbox"),
                "node_properties": inner.get("node_properties"),
                "thought": ao.get("thought", ""),
                "image": [pil.width, pil.height],
                "png": str(png.relative_to(args.out)),
            }
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            n += 1; kinds[name] = kinds.get(name, 0) + 1
            if n % 100 == 0:
                print(f"  {n}/{args.max_steps} steps  ({seen_traj} trajs, {time.time()-t0:.0f}s)  kinds={kinds}", flush=True)

    fout.flush(); fout.close()
    print(f"DONE: {n} steps from {seen_traj} trajectories. kinds={kinds}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # avoid streaming-iterator GIL-cleanup abort masking success


if __name__ == "__main__":
    main()
