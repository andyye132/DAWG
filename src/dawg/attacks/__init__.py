"""DAWG adversarial attacks against MolmoWeb.

Pixel-space PGD: optimize real screenshot pixels through a differentiable
preprocessing bridge so gradients flow pixel -> images tensor -> MolmoWeb.

  - diff_preprocess.py : differentiable port of MolmoWeb's image preprocessing
  - pixel_pgd.py       : pixel-space PGD setup (L1 untargeted); entry point
                         `PixelPGDSetup`

Submodules are imported directly (e.g. `from dawg.attacks.pixel_pgd import
PixelPGDSetup`); this package __init__ stays import-light on purpose so it
doesn't pull torch just to touch the package.
"""
