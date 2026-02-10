"""C0 — Compiler. Detects domain, loads priors, builds the unified prompt."""

from src.compiler.compiler import (
    compile, detect_domains, load_priors,
    DOMAIN_MATURITY, MATURITY_TYPE01_THRESHOLD, compute_type01_threshold,
)
