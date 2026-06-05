from pathlib import Path
from collections import OrderedDict
BASE = Path('/gscratch/raivn/andy132/dawg')
pages = [p for p in (BASE/'data/chunk_l1_b.txt').read_text().split() if p]
bysite = OrderedDict()
for p in pages:
    bysite.setdefault(p.split('/')[-2], []).append(p)
half = len(pages)//2
two5, three33, acc = [], [], 0
for s, ps in bysite.items():
    if acc < half: two5 += ps; acc += len(ps)
    else: three33 += ps
(BASE/'data/chunk_l1_b_two5.txt').write_text('\n'.join(two5)+'\n')
(BASE/'data/chunk_l1_b_three33.txt').write_text('\n'.join(three33)+'\n')
print(f"two5={len(two5)} pages, three33={len(three33)} pages")
