"""DAWG evaluation utilities.

`equiv.same_meaning(a, b)` decides whether two MolmoWeb responses are
semantically equivalent for the L1 attack-success metric. Backed by
sentence-transformers (all-mpnet-base-v2), lazy-loaded on first call.
"""
