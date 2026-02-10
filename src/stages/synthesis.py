"""
Contribution Synthesis — Semantic claim embedding for convergence measurement.

Models never see each other's outputs. The SYSTEM finds convergence by
embedding each model's parsed claim statements and clustering by cosine
similarity across models. Claims are the science — raw text is scaffolding.

K-SSM lesson applied: "Any path that doesn't go through R gives the gradient
an escape route." Here: any content that isn't science (headers, formatting)
gives the embedder noise to hide behind. So we embed claims directly.

Convergence measurement: semantic claim embedding (meaning, not structure)
Downstream data (S4→S6): claim tuple extraction (structured triples)

TYPE assignment by model count in cluster:
    5/5 → TYPE 0 (ESTABLISHED)
    4/5 → TYPE 0 (ESTABLISHED)
    3/5 → TYPE 1 (REPLICATED)
    2/5 → TYPE 2 (SPECULATIVE)
    1/5 → TYPE 3 (SINGULAR — potential novel insight)

Zero API calls. Pure Python + local embedding model. Deterministic.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

from src.parser import Claim, ParsedResponse
from src.convergence.claim_tuples import (
    ClaimTuple, extract_tuples, group_relation,
    ENTITY_SYNONYMS, _MULTI_WORD_SYNONYMS,
)
from src.convergence.convergence import compute, ConvergenceSnapshot, _get_embed_model


# Overlap count → system-assigned TYPE
OVERLAP_TYPE_MAP = {5: 0, 4: 0, 3: 1, 2: 2, 1: 3}

TYPE_LABELS = {0: "ESTABLISHED", 1: "REPLICATED", 2: "SPECULATIVE", 3: "SINGULAR"}

# Semantic clustering threshold for claim embeddings.
# Diagnostic (diag_cosine.py) showed true semantic matches score 0.84-0.97
# while cross-topic pairs within the same domain score 0.50-0.75. Complete-
# linkage prevents the transitive chaining that union-find caused (one mega-
# cluster absorbing all claims via intermediate bridges). Synonym normalization
# before embedding boosts extreme cases (CBD vs Cannabidiol) from 0.39→0.77.
# Live fire at 0.75 produced 3/5 max overlap (TYPE 0/1=60%); 0.70 with
# complete-linkage gives: 5/5=1, 4/5=3, 3/5=2 — more realistic agreement.
# Cross-topic pairs (ROS vs PPARgamma) at 0.53 remain safely below 0.70.
CLAIM_COSINE_THRESHOLD = 0.70


@dataclass
class SynthesizedClaim:
    """A claim with system-assigned TYPE based on cross-model semantic overlap."""
    statement: str
    mechanism: str
    falsifiable_by: str
    type: int               # System-assigned by model count in cluster
    confidence: float       # Mean confidence across supporting models
    overlap_count: int      # How many models contributed to this cluster
    models: list            # Which models contributed
    tuples: set = field(default_factory=set)  # Grouped triples for downstream
    chunk_text: str = ""    # Representative claim text


@dataclass
class Conflict:
    """A value disagreement across models for the same claim."""
    subject: str
    relation: str
    object: str
    values: dict  # model_name → value string


def _normalize_claim_text(text: str) -> str:
    """Normalize claim text using entity synonyms before embedding.

    Replaces domain-specific synonyms (e.g., "Cannabidiol" → "cbd",
    "voltage-dependent anion channel 1" → "vdac1") so the embedding
    model sees consistent terminology across models. Without this,
    "CBD binds VDAC1" vs "Cannabidiol interacts with voltage-dependent
    anion channel 1" scores cosine 0.39 — with it, 0.77.
    """
    t = text.lower()
    # Multi-word synonyms first (longest match), e.g.,
    # "voltage-dependent anion channel 1" → "vdac1"
    for surface, canonical in _MULTI_WORD_SYNONYMS:
        t = t.replace(surface, canonical)
    # Single-word synonyms
    for surface, canonical in ENTITY_SYNONYMS.items():
        if " " not in surface and "-" not in surface:
            t = t.replace(surface.lower(), canonical)
    return t


def _embed_claims(
    claims_by_model: dict[str, list[Claim]],
) -> tuple[np.ndarray, list[dict]]:
    """Embed all claim statements in a single batch.

    Each claim's statement + mechanism are concatenated for richer
    embedding context, then synonym-normalized before embedding.
    The science is in the claims, not the scaffolding.

    Args:
        claims_by_model: {model_name: [Claim, ...]}

    Returns:
        (embeddings, claim_meta) where:
        - embeddings: np.ndarray of shape (n_claims, embed_dim), normalized
        - claim_meta: list of {"model": str, "claim": Claim, "text": str}
    """
    model = _get_embed_model()

    all_texts = []
    claim_meta = []
    for model_name, claims in claims_by_model.items():
        for claim in claims:
            # Embed statement + mechanism together for richer context
            text = claim.statement
            if claim.mechanism:
                text = f"{text}. Mechanism: {claim.mechanism}"
            # Normalize synonyms before embedding
            normalized = _normalize_claim_text(text)
            all_texts.append(normalized)
            claim_meta.append({
                "model": model_name,
                "claim": claim,
                "text": text,  # Keep original text for display
            })

    if not all_texts:
        return np.array([]).reshape(0, 0), []

    embeddings = model.encode(all_texts, normalize_embeddings=True)
    return embeddings, claim_meta


def _cluster_claims(
    embeddings: np.ndarray,
    claim_meta: list[dict],
    threshold: float = CLAIM_COSINE_THRESHOLD,
) -> list[list[int]]:
    """Cluster claims by cosine similarity across models (complete-linkage).

    Uses complete-linkage: a claim joins a cluster only if its cosine
    similarity to ALL cross-model members of that cluster >= threshold.
    This prevents transitive chaining where A→B→C→D merges unrelated
    claims through intermediate bridges.

    Only links claims from DIFFERENT models. Same-model claims
    never inflate overlap count.

    Args:
        embeddings: Normalized embedding matrix.
        claim_meta: Metadata for each claim (model, claim, text).
        threshold: Minimum cosine similarity to ALL cross-model members.

    Returns:
        List of clusters, each cluster = list of indices into claim_meta.
    """
    n = len(claim_meta)
    if n == 0:
        return []

    # Cosine similarity matrix (embeddings are normalized → dot product)
    sim_matrix = embeddings @ embeddings.T

    # Collect all cross-model edges above threshold, sorted by cosine descending
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if claim_meta[i]["model"] != claim_meta[j]["model"]:
                if sim_matrix[i, j] >= threshold:
                    edges.append((float(sim_matrix[i, j]), i, j))
    edges.sort(reverse=True)

    # Greedy complete-linkage: assign claims to clusters
    cluster_of = {}  # claim_index → cluster_id
    clusters = {}    # cluster_id → list of indices
    next_id = 0

    for _, i, j in edges:
        ci = cluster_of.get(i)
        cj = cluster_of.get(j)

        if ci is None and cj is None:
            # Both unassigned: create new cluster
            cluster_of[i] = next_id
            cluster_of[j] = next_id
            clusters[next_id] = [i, j]
            next_id += 1

        elif ci is not None and cj is None:
            # i in cluster, j unassigned: add j if complete-linkage holds
            if _complete_linkage_ok(j, clusters[ci], claim_meta, sim_matrix, threshold):
                cluster_of[j] = ci
                clusters[ci].append(j)

        elif ci is None and cj is not None:
            # j in cluster, i unassigned: add i if complete-linkage holds
            if _complete_linkage_ok(i, clusters[cj], claim_meta, sim_matrix, threshold):
                cluster_of[i] = cj
                clusters[cj].append(i)

        elif ci != cj:
            # Both in different clusters: merge if all cross-model pairs pass
            if _can_merge_clusters(clusters[ci], clusters[cj], claim_meta, sim_matrix, threshold):
                # Merge smaller into larger
                if len(clusters[ci]) < len(clusters[cj]):
                    ci, cj = cj, ci
                for idx in clusters[cj]:
                    cluster_of[idx] = ci
                clusters[ci].extend(clusters[cj])
                del clusters[cj]

    # Add unassigned claims as singleton clusters
    for i in range(n):
        if i not in cluster_of:
            clusters[next_id] = [i]
            next_id += 1

    return list(clusters.values())


def _complete_linkage_ok(
    candidate: int,
    cluster: list[int],
    claim_meta: list[dict],
    sim_matrix: np.ndarray,
    threshold: float,
) -> bool:
    """Check if candidate has cos >= threshold with all cross-model cluster members."""
    cand_model = claim_meta[candidate]["model"]
    for member in cluster:
        if claim_meta[member]["model"] != cand_model:
            if sim_matrix[candidate, member] < threshold:
                return False
    return True


def _can_merge_clusters(
    cluster_a: list[int],
    cluster_b: list[int],
    claim_meta: list[dict],
    sim_matrix: np.ndarray,
    threshold: float,
) -> bool:
    """Check if all cross-model pairs between two clusters pass threshold."""
    for i in cluster_a:
        for j in cluster_b:
            if claim_meta[i]["model"] != claim_meta[j]["model"]:
                if sim_matrix[i, j] < threshold:
                    return False
    return True


# ── Main synthesis function ──

def run_s2_synthesis(s1_result: dict) -> dict:
    """S2 — Contribution Synthesis via semantic claim embedding.

    Embeds each model's parsed claim statements directly (not raw text),
    clusters by cosine similarity across models, and assigns TYPE by how
    many independent models contributed to each cluster.

    Claims are the science. Raw text is scaffolding. Embed the science.

    Tuples extracted afterward for downstream S4→S6 structured data.

    Returns dict compatible with S3 gate and recirculation.
    """
    parsed: list[ParsedResponse] = s1_result["parsed"]
    s1_snapshot: ConvergenceSnapshot = s1_result["snapshot"]

    # Step 1: Collect claims per model
    claims_by_model: dict[str, list[Claim]] = {}
    for p in parsed:
        if p.claims:
            claims_by_model[p.model] = p.claims

    # Edge case: no claims
    if not claims_by_model:
        return _empty_result(parsed, s1_snapshot)

    # Step 2: Embed all claim statements
    embeddings, claim_meta = _embed_claims(claims_by_model)

    if len(claim_meta) == 0:
        return _empty_result(parsed, s1_snapshot)

    # Step 3: Cluster by semantic similarity across models
    clusters = _cluster_claims(embeddings, claim_meta)

    # Step 4: Build SynthesizedClaim for each cluster
    synthesized = []
    for cluster_indices in clusters:
        models_in_cluster = set()
        cluster_claims = []

        for idx in cluster_indices:
            models_in_cluster.add(claim_meta[idx]["model"])
            cluster_claims.append(claim_meta[idx]["claim"])

        overlap_count = min(len(models_in_cluster), 5)
        assigned_type = OVERLAP_TYPE_MAP.get(overlap_count, 3)

        # Pick representative: claim with highest mean cosine to others in cluster
        if len(cluster_indices) > 1:
            sub_emb = embeddings[cluster_indices]
            sub_sim = sub_emb @ sub_emb.T
            mean_sims = sub_sim.mean(axis=1)
            best_local_idx = int(np.argmax(mean_sims))
        else:
            best_local_idx = 0

        best_claim = claim_meta[cluster_indices[best_local_idx]]["claim"]

        # Extract tuples from representative claim for downstream
        claim_tuples = set()
        raw_tuples = extract_tuples(best_claim)
        claim_tuples = frozenset(
            (t.subject, group_relation(t.relation), t.object)
            for t in raw_tuples
        )

        # Confidence: mean across claims in this cluster
        confidences = [c.confidence for c in cluster_claims]
        mean_confidence = sum(confidences) / len(confidences)

        synthesized.append(SynthesizedClaim(
            statement=best_claim.statement,
            mechanism=best_claim.mechanism or "",
            falsifiable_by=best_claim.falsifiable_by or "",
            type=assigned_type,
            confidence=round(mean_confidence, 3),
            overlap_count=overlap_count,
            models=sorted(models_in_cluster),
            tuples=claim_tuples,
            chunk_text=best_claim.statement[:500],
        ))

    # Sort: highest overlap first, then by confidence
    synthesized.sort(key=lambda s: (-s.overlap_count, -s.confidence))

    # Step 5: Detect conflicts using tuple extraction
    claim_infos = _extract_all_claim_infos(parsed)
    conflicts = _detect_conflicts(claim_infos)

    # Step 6: Build convergence snapshot
    synth_claims_per_model = _build_claims_per_model(synthesized, parsed)
    synth_snapshot = compute(synth_claims_per_model, round_num=1)

    return {
        "stage": "S2",
        "parsed": parsed,
        "snapshots": [s1_snapshot, synth_snapshot],
        "rounds": [],
        "total_rounds": 0,
        "total_calls": 0,
        "early_stopped": False,
        "synthesized_claims": synthesized,
        "conflicts": conflicts,
    }


def _empty_result(parsed, s1_snapshot):
    """Return an empty S2 result when no content to synthesize."""
    return {
        "stage": "S2",
        "parsed": parsed,
        "snapshots": [s1_snapshot, s1_snapshot],
        "rounds": [],
        "total_rounds": 0,
        "total_calls": 0,
        "early_stopped": False,
        "synthesized_claims": [],
        "conflicts": [],
    }


def _collect_confidences(
    models: set[str],
    parsed: list[ParsedResponse],
) -> list[float]:
    """Collect all confidence values from claims of contributing models."""
    confidences = []
    for p in parsed:
        if p.model in models:
            for c in p.claims:
                confidences.append(c.confidence)
    return confidences


def _extract_all_claim_infos(parsed: list[ParsedResponse]) -> list[dict]:
    """Extract tuple info from all claims for conflict detection."""
    claim_infos = []
    for p in parsed:
        for claim in p.claims:
            raw_tuples = extract_tuples(claim)
            claim_infos.append({
                "model": p.model,
                "claim": claim,
                "raw_tuples": raw_tuples,
            })
    return claim_infos


def _detect_conflicts(claim_infos: list[dict]) -> list[Conflict]:
    """Find cases where models agree on subject+relation+object but disagree on value."""
    triple_values = defaultdict(lambda: defaultdict(set))

    for info in claim_infos:
        for t in info["raw_tuples"]:
            if t.value:
                grouped_rel = group_relation(t.relation)
                key = (t.subject, grouped_rel, t.object)
                triple_values[key][info["model"]].add(t.value)

    conflicts = []
    for (subj, rel, obj), model_vals in triple_values.items():
        all_values = set()
        for vals in model_vals.values():
            all_values |= vals
        if len(all_values) > 1:
            flat = {m: ", ".join(sorted(v)) for m, v in model_vals.items()}
            conflicts.append(Conflict(
                subject=subj,
                relation=rel,
                object=obj,
                values=flat,
            ))

    return conflicts


def _build_claims_per_model(
    synthesized: list[SynthesizedClaim],
    parsed: list[ParsedResponse],
) -> list[list[Claim]]:
    """Build claims_per_model list for convergence.compute().

    Each model gets the synthesized claims it contributed to,
    with the system-assigned TYPE. This lets the convergence engine
    compute TYPE distribution and other metrics correctly.
    """
    model_names = [p.model for p in parsed]
    model_claims = {name: [] for name in model_names}

    for sc in synthesized:
        claim = Claim(
            statement=sc.statement,
            type=sc.type,
            confidence=sc.confidence,
            mechanism=sc.mechanism,
            falsifiable_by=sc.falsifiable_by,
        )
        for model in sc.models:
            if model in model_claims:
                model_claims[model].append(claim)

    return [model_claims[name] for name in model_names]
