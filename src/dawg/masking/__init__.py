"""DAWG masking: overlay adversarial patches onto webpage screenshots.

Two pathways with different threat-model fidelity:

- `composite.py` (Path B): PIL composite onto a pre-rendered screenshot.
  Pixel-perfect, no browser rendering. Fast for dataset generation but not a
  real attack — an attacker can't paint pixels onto the agent's screenshot.

- `overlay.py` (Path A): inject the patch as a positioned <img> in the page
  HTML, then render via Playwright. Matches the threat model. Subject to DPR,
  anti-aliasing, and z-index pitfalls.
"""
