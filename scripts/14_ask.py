"""Ask MolmoWeb-4B-Native a question about a screenshot (single-image QA).

Loads the model ONCE (~3 min). Two modes:

  one-shot:
      python scripts/14_ask.py --image data/syntheticQA/adidas/page000/screenshot.png \
                               --question "What is the promo near the top?"
      python scripts/14_ask.py --page adidas/page000            # uses that page's 1st stored question

  interactive REPL (recommended — pay the load cost once, ask many):
      python scripts/14_ask.py
    then at the prompt:
      <text>                 ask <text> about the current image
      :img <path>            switch image (a screenshot.png, or a page dir)
      :page <site/pageNNN>   shortcut into data/syntheticQA
      :list                  list available SyntheticQA pages
      :meta                  show the current page's stored Q&A (ground truth)
      :q                     quit

Must run on a GPU node in the dawg conda env (see the header of this file's
companion docs, or scripts/sbatch_sanity.sh for the env preamble).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

DAWG_ROOT = Path("/gscratch/raivn/andy132/dawg")
if not DAWG_ROOT.exists():
    DAWG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DAWG_ROOT / "src"))
sys.path.insert(0, str(DAWG_ROOT / "external" / "molmoweb"))

CKPT = DAWG_ROOT / "external" / "molmoweb" / "checkpoints" / "MolmoWeb-4B-Native"
DATA = DAWG_ROOT / "data" / "syntheticQA"


def resolve_image(spec: str):
    """Resolve a screenshot path, a page dir, or 'site/pageNNN' -> (np_image, page_dir|None)."""
    p = Path(spec)
    candidates = [p, p / "screenshot.png", DATA / spec, DATA / spec / "screenshot.png"]
    for c in candidates:
        if c.is_file():
            page_dir = c.parent if c.name == "screenshot.png" else None
            return np.array(Image.open(c).convert("RGB")), page_dir
    raise FileNotFoundError(f"no screenshot found for {spec!r}")


def unwrap(answer: str) -> str:
    """MolmoWeb replies with an action JSON; show the user-facing msg if present."""
    s = (answer or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("msg"), str):
                return obj["msg"]
        except ValueError:
            pass
    return answer


def list_pages():
    pages = sorted(str(p.parent.relative_to(DATA)) for p in DATA.glob("*/page*/screenshot.png"))
    print(f"{len(pages)} pages under {DATA}:")
    for pg in pages:
        print(f"  {pg}")


def show_meta(page_dir: Path | None):
    if page_dir is None or not (page_dir / "meta.json").exists():
        print("  (no stored meta for the current image)")
        return
    meta = json.loads((page_dir / "meta.json").read_text())
    print(f"  {meta['website']}  url={meta['url']}")
    for i, qa in enumerate(meta["qa"]):
        print(f"  [{i}] Q: {qa['question']}\n      A: {qa['answer']}  ({qa['question_type']})")


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="screenshot.png path or a page dir")
    ap.add_argument("--page", default=None, help="'site/pageNNN' under data/syntheticQA")
    ap.add_argument("--question", default=None)
    ap.add_argument("--qa-index", type=int, default=0,
                    help="if --question omitted but a page is given, use this stored question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] No CUDA — allocate a GPU node first (srun ... --gres=gpu:1 --pty bash).")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}  | loading MolmoWeb-4B-Native (~3 min)...")
    t0 = time.time()
    from agent.model_backends import NativeActionPredictor
    pred = NativeActionPredictor(checkpoint=str(CKPT), device="cuda:0",
                                 temperature=args.temperature, top_p=args.top_p)
    print(f"  loaded in {time.time() - t0:.1f}s\n")

    def ask(question, img_np):
        t = time.time()
        raw = pred.predict(question, img_np)
        print(f"  answer: {unwrap(raw)}")
        print(f"  (raw: {raw}   [{time.time() - t:.1f}s])")

    # ---- one-shot ----
    spec = args.image or args.page
    if args.question or (spec and args.page):
        img_np, page_dir = resolve_image(spec) if spec else (None, None)
        question = args.question
        if question is None and page_dir is not None:
            meta = json.loads((page_dir / "meta.json").read_text())
            qa = meta["qa"][args.qa_index]
            question, gt = qa["question"], qa["answer"]
            print(f"Q: {question}\nGT: {gt}")
        if img_np is None or question is None:
            print("Need an image and a question (or a --page with stored Q&A).")
            return 2
        print(f"Q: {question}")
        ask(question, img_np)
        return 0

    # ---- interactive ----
    cur_img, cur_dir, cur_name = None, None, None
    if spec:
        cur_img, cur_dir = resolve_image(spec)
        cur_name = spec
    print("Interactive MolmoWeb QA. Type a question, or :help for commands.")
    if cur_name:
        print(f"current image: {cur_name}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line == ":help":
            print("  <text> ask | :img <path> | :page <site/pageNNN> | :list | :meta | :q")
            continue
        if line == ":list":
            list_pages(); continue
        if line == ":meta":
            show_meta(cur_dir); continue
        if line.startswith(":img ") or line.startswith(":page "):
            spec = line.split(" ", 1)[1].strip()
            try:
                cur_img, cur_dir = resolve_image(spec); cur_name = spec
                print(f"  current image: {cur_name}  ({cur_img.shape[1]}x{cur_img.shape[0]})")
            except FileNotFoundError as e:
                print(f"  {e}")
            continue
        if cur_img is None:
            print("  no image selected — use :img <path> or :page <site/pageNNN>")
            continue
        ask(line, cur_img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
