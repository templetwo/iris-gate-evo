"""
Cross-run claim matcher — embeds claims, finds semantic matches across runs,
reclassifies TYPE based on cross-run evidence.

Reuses the same embedding model and normalization as within-run synthesis.
"""

import numpy as np
from dataclasses import dataclass, field

from src.cross_run.loader import RunData
from src.convergence.convergence import _get_embed_model
from src.stages.synthesis import _normalize_claim_text


@dataclass
class CrossMatch:
    """A semantic match between claims from different runs."""
    claim_a: dict
    claim_b: dict
    run_a: str
    run_b: str
    cosine: float
    type_a: int
    type_b: int
    classification: str


@dataclass
class CrossRunResult:
    """Complete result of cross-run analysis."""
    runs: list[RunData]
    matches: list[CrossMatch]
    structural_patterns: dict  # pattern_name -> [run_ids]
    cosine_distribution: list[float] = field(default_factory=list)


# TYPE reclassification rules
def _classify_cross_match(type_a: int, type_b: int) -> str:
    """Classify a cross-run match based on the TYPE of each claim."""
    a, b = sorted([type_a, type_b])  # normalize order

    if type_a == 3 and type_b == 3:
        return "CONVERGENT SINGULAR"
    if type_a == 3 or type_b == 3:
        return "CROSS-VALIDATED SINGULAR"
    if a <= 1 and b <= 1:
        return "INDEPENDENT REPLICATION"
    if (a == 2 and b <= 1) or (b == 2 and a <= 1):
        return "CROSS-PROMOTED"
    return "CROSS-MATCH"


# Structural pattern keywords — require 2+ claims matching before flagging
STRUCTURAL_PATTERNS = {
    "two_pathway": ["two-pathway", "biphasic", "dual", "dose-dependent switch"],
    "threshold_crossover": ["threshold", "crossover", "inflection", "phase transition"],
    "dose_dependent": ["dose-response", "dose-dependent", "occupancy"],
    "tissue_context": ["tissue-specific", "cell-type", "tissue determines"],
    "kinetic": ["kinetic", "first-order", "rate constant", "k_int", "k_rec"],
}


def _detect_structural_patterns(runs: list[RunData]) -> dict[str, list[str]]:
    """Detect structural patterns across runs via keyword matching.

    Requires 2+ claims in a run matching a pattern before flagging.
    """
    patterns_by_run: dict[str, dict[str, int]] = {}

    for run in runs:
        counts: dict[str, int] = {p: 0 for p in STRUCTURAL_PATTERNS}
        for claim in run.claims:
            text = (claim.get("statement", "") + " " + claim.get("mechanism", "")).lower()
            for pattern_name, keywords in STRUCTURAL_PATTERNS.items():
                if any(kw in text for kw in keywords):
                    counts[pattern_name] += 1
        patterns_by_run[run.session_id] = counts

    # Aggregate: pattern -> [run_ids] where count >= 2
    result: dict[str, list[str]] = {}
    for pattern_name in STRUCTURAL_PATTERNS:
        run_ids = [
            rid for rid, counts in patterns_by_run.items()
            if counts[pattern_name] >= 2
        ]
        if len(run_ids) >= 2:
            result[pattern_name] = run_ids

    return result


def cross_match(runs: list[RunData], threshold: float = 0.75) -> CrossRunResult:
    """Find semantic matches between claims from different runs.

    1. Collect all claims tagged with run_id
    2. Embed: statement + mechanism, synonym-normalized
    3. Pairwise cosine across different runs
    4. Record matches >= threshold, classify by TYPE
    5. Detect structural patterns
    """
    if len(runs) < 2:
        return CrossRunResult(runs=runs, matches=[], structural_patterns={})

    model = _get_embed_model()

    # Step 1: Collect all claims with provenance
    all_claims = []  # (run_id, claim_index, claim_dict)
    all_texts = []

    for run in runs:
        for i, claim in enumerate(run.claims):
            all_claims.append((run.session_id, i, claim))
            text = claim.get("statement", "")
            mechanism = claim.get("mechanism", "")
            if mechanism:
                text = f"{text}. {mechanism}"
            all_texts.append(_normalize_claim_text(text))

    if len(all_texts) < 2:
        return CrossRunResult(runs=runs, matches=[], structural_patterns={})

    # Step 2: Embed
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    # Step 3: Pairwise cosine (dot product on normalized)
    sim_matrix = embeddings @ embeddings.T
    n = len(all_claims)

    # Collect all cross-run cosines for distribution logging
    cross_cosines = []
    matches = []

    for i in range(n):
        for j in range(i + 1, n):
            run_a = all_claims[i][0]
            run_b = all_claims[j][0]
            if run_a == run_b:
                continue  # skip within-run

            cos = float(sim_matrix[i, j])
            cross_cosines.append(cos)

            if cos >= threshold:
                claim_a = all_claims[i][2]
                claim_b = all_claims[j][2]
                type_a = claim_a.get("type", 3)
                type_b = claim_b.get("type", 3)

                matches.append(CrossMatch(
                    claim_a=claim_a,
                    claim_b=claim_b,
                    run_a=run_a,
                    run_b=run_b,
                    cosine=round(cos, 4),
                    type_a=type_a,
                    type_b=type_b,
                    classification=_classify_cross_match(type_a, type_b),
                ))

    # Sort by cosine descending
    matches.sort(key=lambda m: -m.cosine)

    # Step 5: Structural patterns
    structural = _detect_structural_patterns(runs)

    return CrossRunResult(
        runs=runs,
        matches=matches,
        structural_patterns=structural,
        cosine_distribution=sorted(cross_cosines),
    )
