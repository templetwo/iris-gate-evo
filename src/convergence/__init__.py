"""Convergence metrics — Jaccard, Cosine, JSD, Kappa. Server-side, never self-reported."""

from src.convergence.convergence import compute, delta, ConvergenceSnapshot
from src.convergence.claim_tuples import extract_tuples, ClaimTuple, group_relation
