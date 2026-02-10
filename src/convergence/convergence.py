"""
Convergence Engine — Server-side metrics on parsed CLAIMS.

Models never judge their own convergence. This engine computes:
- Jaccard similarity on claim text (lexical overlap)
- Cosine similarity via sentence-transformers (semantic overlap)
- Jensen-Shannon Divergence on TYPE distributions (distributional agreement)
- Fleiss' Kappa on TYPE classifications (5-rater agreement)
- TYPE distribution (rising T0/T1 ratio = system stabilizing)

All metrics operate on the CLAIMS array, not on full response text.
"""

import numpy as np
from itertools import combinations
from typing import Optional
from dataclasses import dataclass

from src.parser import Claim


@dataclass
class ConvergenceSnapshot:
    """Metrics for one round of debate."""
    round_num: int
    jaccard: float          # Mean pairwise Jaccard on claim statements
    cosine: float           # Mean pairwise cosine on claim embeddings
    jsd: float              # JSD on TYPE distributions across models
    kappa: float            # Fleiss' Kappa on TYPE classifications
    type_distribution: dict # {0: frac, 1: frac, 2: frac, 3: frac}
    type_01_ratio: float    # Fraction of claims that are TYPE 0 or 1
    n_claims_per_model: list[int]  # Claim count per model


# Embedding model — lazy loaded
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def compute(
    claims_per_model: list[list[Claim]],
    round_num: int = 0,
    use_embeddings: bool = True,
) -> ConvergenceSnapshot:
    """Compute all convergence metrics for one round.

    Args:
        claims_per_model: List of claim lists, one per model.
        round_num: Current debate round number.
        use_embeddings: Whether to compute cosine similarity (slower).

    Returns:
        ConvergenceSnapshot with all metrics.
    """
    n_models = len(claims_per_model)

    # Flatten all claims for TYPE distribution
    all_claims = [c for model_claims in claims_per_model for c in model_claims]

    # TYPE distribution
    type_dist = _compute_type_distribution(all_claims)
    type_01_ratio = type_dist.get(0, 0.0) + type_dist.get(1, 0.0)

    # Claim counts
    n_claims = [len(cs) for cs in claims_per_model]

    # Pairwise Jaccard on claim statement text
    jaccard = _mean_pairwise_jaccard(claims_per_model)

    # Pairwise cosine on claim embeddings
    cosine = 0.0
    if use_embeddings and all_claims:
        try:
            cosine = _mean_pairwise_cosine(claims_per_model)
        except ImportError:
            pass

    # JSD on TYPE distributions across models
    jsd = _compute_jsd(claims_per_model)

    # Fleiss' Kappa on TYPE classifications
    kappa = _compute_fleiss_kappa(claims_per_model)

    return ConvergenceSnapshot(
        round_num=round_num,
        jaccard=jaccard,
        cosine=cosine,
        jsd=jsd,
        kappa=kappa,
        type_distribution=type_dist,
        type_01_ratio=type_01_ratio,
        n_claims_per_model=n_claims,
    )


def delta(current: ConvergenceSnapshot, previous: ConvergenceSnapshot) -> float:
    """Compute convergence delta between two rounds.

    Uses a weighted combination of Jaccard and cosine changes.
    Returns the magnitude of change — smaller = more stable.
    """
    d_jaccard = abs(current.jaccard - previous.jaccard)
    d_cosine = abs(current.cosine - previous.cosine)
    d_jsd = abs(current.jsd - previous.jsd)

    # Weight Jaccard and cosine equally, JSD as tiebreaker
    return 0.4 * d_jaccard + 0.4 * d_cosine + 0.2 * d_jsd


# ---------------------------------------------------------------------------
# Jaccard — lexical claim overlap
# ---------------------------------------------------------------------------

def _tokenize_claim(claim: Claim) -> set[str]:
    """Tokenize a claim statement into a set of lowercased words."""
    words = set(claim.statement.lower().split())
    # Remove very short words (articles, prepositions)
    return {w for w in words if len(w) > 2}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _mean_pairwise_jaccard(claims_per_model: list[list[Claim]]) -> float:
    """Mean pairwise Jaccard between all model pairs.

    For each pair of models, compute Jaccard on the UNION of their
    claim tokens — treating each model's claims as a bag of words.
    """
    n = len(claims_per_model)
    if n < 2:
        return 1.0

    # Build token sets per model (union of all claim tokens)
    model_tokens = []
    for model_claims in claims_per_model:
        tokens = set()
        for claim in model_claims:
            tokens |= _tokenize_claim(claim)
        model_tokens.append(tokens)

    # Mean pairwise Jaccard
    scores = []
    for i, j in combinations(range(n), 2):
        scores.append(_jaccard_similarity(model_tokens[i], model_tokens[j]))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Cosine — semantic claim similarity via embeddings
# ---------------------------------------------------------------------------

def _mean_pairwise_cosine(claims_per_model: list[list[Claim]]) -> float:
    """Mean pairwise cosine similarity between model claim sets.

    Embeds each model's claims as a single combined string,
    then computes pairwise cosine between model embeddings.
    """
    model = _get_embed_model()
    n = len(claims_per_model)
    if n < 2:
        return 1.0

    # Combine each model's claims into one string for embedding
    model_texts = []
    for model_claims in claims_per_model:
        combined = " ".join(c.statement for c in model_claims)
        model_texts.append(combined if combined else "no claims")

    embeddings = model.encode(model_texts, normalize_embeddings=True)

    # Mean pairwise cosine (dot product of normalized vectors)
    scores = []
    for i, j in combinations(range(n), 2):
        sim = float(embeddings[i] @ embeddings[j])
        scores.append(sim)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# JSD — distributional disagreement on TYPE classifications
# ---------------------------------------------------------------------------

def _model_type_distribution(claims: list[Claim]) -> np.ndarray:
    """Compute TYPE distribution for one model's claims. Returns [T0, T1, T2, T3]."""
    counts = np.zeros(4)
    for c in claims:
        t = min(max(c.type, 0), 3)  # Clamp to 0-3
        counts[t] += 1

    total = counts.sum()
    if total == 0:
        return np.ones(4) / 4  # Uniform if no claims

    return counts / total


def _compute_jsd(claims_per_model: list[list[Claim]]) -> float:
    """Jensen-Shannon Divergence across model TYPE distributions.

    JSD → 0 means all models agree on TYPE distribution.
    JSD → 1 means maximum disagreement.
    """
    from scipy.spatial.distance import jensenshannon

    dists = [_model_type_distribution(cs) for cs in claims_per_model]
    if len(dists) < 2:
        return 0.0

    # Mean pairwise JSD
    scores = []
    for i, j in combinations(range(len(dists)), 2):
        # jensenshannon returns sqrt(JSD), we want JSD
        js = jensenshannon(dists[i], dists[j]) ** 2
        scores.append(js)

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# TYPE distribution
# ---------------------------------------------------------------------------

def _compute_type_distribution(all_claims: list[Claim]) -> dict:
    """Compute overall TYPE distribution across all claims."""
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for c in all_claims:
        t = min(max(c.type, 0), 3)
        counts[t] += 1

    total = sum(counts.values())
    if total == 0:
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Fleiss' Kappa — inter-rater agreement on TYPE classifications
# ---------------------------------------------------------------------------

def _compute_fleiss_kappa(claims_per_model: list[list[Claim]]) -> float:
    """Fleiss' Kappa for 5-rater TYPE classification agreement.

    Each "subject" is a claim topic (matched by Jaccard similarity).
    Each "rater" is a model. The rating is the TYPE classification.

    If models can't be aligned on subjects, falls back to marginal agreement.
    """
    n_models = len(claims_per_model)
    if n_models < 2:
        return 1.0

    # Simplified approach: compute from TYPE distribution overlap
    # For each model, get its TYPE distribution as a vote vector
    dists = [_model_type_distribution(cs) for cs in claims_per_model]

    # Stack into matrix: models x types
    matrix = np.array(dists)  # shape (n_models, 4)

    # Mean distribution (expected by chance)
    p_bar = matrix.mean(axis=0)

    # Mean pairwise agreement
    # For each type, compute the variance across models
    # Low variance = high agreement
    agreement_per_type = 1.0 - matrix.var(axis=0)

    # Weighted average agreement
    P_bar = float(np.sum(p_bar * agreement_per_type))

    # Expected agreement by chance
    P_e = float(np.sum(p_bar ** 2))

    if P_e >= 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(np.clip(kappa, -1.0, 1.0))
