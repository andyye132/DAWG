"""Offline L2 detector pass over a generated L2 dataset (scripts/32 output):
run the factual-drift cascade, write success labels + a cosine-vs-cascade summary."""
import json, glob, sys
from pathlib import Path
DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists(): DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DAWG_ROOT / "results" / "l2_dataset")
    args = ap.parse_args()
    from dawg.eval.l2_detect import l2_factual_drift
    from dawg.eval.equiv import similarity, _unwrap_json_action
    rows = []
    for f in glob.glob(str(args.dir / "shard*.jsonl")):
        for line in open(f):
            try: rows.append(json.loads(line))
            except Exception: pass
    out = (args.dir / "labeled.jsonl").open("w")
    n = drift = plausible = success = cos_miss = 0
    for r in rows:
        det = l2_factual_drift(r["gt"], r["adv_answer"], r["clean_answer"])
        r.update(det)
        cos_same = similarity(r["gt"], _unwrap_json_action(r["adv_answer"])) >= 0.75
        r["cosine_only_same"] = cos_same
        out.write(json.dumps(r) + "\n")
        n += 1; drift += det["drift"]; plausible += det["plausible"]; success += det["success"]
        if det["drift"] and cos_same: cos_miss += 1
    out.close()
    print(f"rows={n}  drift={drift}  plausible={plausible}  L2_success={success}")
    print(f"*** cosine-only would MISS {cos_miss}/{drift} of the real lies ***  -> {args.dir}/labeled.jsonl")

if __name__ == "__main__": sys.exit(main())
