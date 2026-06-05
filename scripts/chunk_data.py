import glob, json, random
from pathlib import Path
BASE = Path('/gscratch/raivn/andy132/dawg')
deep = BASE / 'data' / 'syntheticqa_deep'
site_pages = {}
for sd in sorted(deep.iterdir()):
    if not sd.is_dir():
        continue
    pages = sorted(str(p.parent) for p in sd.glob('page*/screenshot.png'))
    if pages:
        site_pages[sd.name] = pages
order = sorted(site_pages)
random.Random(0).shuffle(order)  # diversify site order

# assign WHOLE sites to chunks until each hits its page target
targets = [('l1', 5000), ('l2', 2500), ('extra', 2500)]
chunks = {n: [] for n, _ in targets}
chunk_sites = {n: [] for n, _ in targets}
ti = 0
for s in order:
    if ti >= len(targets):
        break
    name, tgt = targets[ti]
    chunks[name] += site_pages[s]
    chunk_sites[name].append(s)
    if len(chunks[name]) >= tgt and ti < len(targets) - 1:
        ti += 1

# split L1 into a (single-10% now) and b (TBD), by WHOLE site at ~2500 pages
l1_a, l1_b, acc = [], [], 0
for s in chunk_sites['l1']:
    if acc < 2500:
        l1_a += site_pages[s]; acc += len(site_pages[s])
    else:
        l1_b += site_pages[s]

for name, pages in chunks.items():
    (BASE / f'data/chunk_{name}.txt').write_text('\n'.join(pages) + '\n')
(BASE / 'data/chunk_l1_a_single10.txt').write_text('\n'.join(l1_a) + '\n')
(BASE / 'data/chunk_l1_b.txt').write_text('\n'.join(l1_b) + '\n')

manifest = {n: {'pages': len(p), 'sites': len(chunk_sites[n])} for n, p in chunks.items()}
manifest['l1_a_single10'] = {'pages': len(l1_a)}
manifest['l1_b_TBD'] = {'pages': len(l1_b)}
json.dump(manifest, open(BASE / 'data/chunks_manifest.json', 'w'), indent=2)
print(json.dumps(manifest, indent=2))
