"""
Stage Orchestrator — S1 → S2 → S3

S1 (Formulation): PULSE fires, responses parsed into claims.
S2 (Contribution Synthesis): Pure Python overlap analysis on claim tuples.
    Models never see each other's outputs. The SYSTEM finds convergence
    by counting how many independent mirrors produced matching claims.
    TYPE assigned by overlap count: 5/5→T0, 4/5→T0, 3/5→T1, 2/5→T2, 1/5→T3.
    Zero API calls.
S3 (Stable Attractor): Strictest convergence gate.
    Cosine > 0.85, TYPE 0/1 >= domain-adaptive threshold, Jaccard floor 0.10.
    FAIL routes to human review with the divergence map.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.parser import parse_response, ParsedResponse
from src.convergence.convergence import compute, delta, ConvergenceSnapshot
from src.stages.synthesis import run_s2_synthesis
from src.pulse.pulse import fire


# S3 gate thresholds
S3_CONVERGENCE_THRESHOLD = 0.85  # Primary: cosine (semantic), with Jaccard floor
S3_JACCARD_FLOOR = 0.10          # Minimum Jaccard — lexical sanity check
S3_JACCARD_THRESHOLD = 0.85      # Legacy, used in reporting
S3_TYPE01_THRESHOLD = 0.90


async def run_s1(compiled: dict) -> dict:
    """S1 — Formulation. Fire PULSE, parse claims.

    Returns:
        Dict with parsed responses, convergence snapshot, and PULSE metadata.
    """
    # Fire PULSE
    pulse_result = await fire(compiled)

    # Parse all responses
    parsed = []
    for resp in pulse_result["responses"]:
        if resp["status"] == "ok":
            p = parse_response(resp["content"], model=resp["model"])
            parsed.append(p)

    # Compute initial convergence
    claims_per_model = [p.claims for p in parsed]
    snapshot = compute(claims_per_model, round_num=0)

    return {
        "stage": "S1",
        "parsed": parsed,
        "snapshot": snapshot,
        "pulse_meta": pulse_result["meta"],
        "total_calls": pulse_result["meta"]["models_dispatched"],
    }


def run_s3_gate(s2_result: dict, compiled: dict = None) -> dict:
    """S3 — Stable Attractor Gate. Strictest convergence check.

    PASS requires:
    - Cosine > 0.85 (semantic similarity) with Jaccard floor 0.10
    - TYPE 0/1 >= domain-adaptive threshold

    Cosine similarity on claim embeddings captures semantic convergence
    that tuple-based Jaccard misses when models express the same science
    in structurally different ways.

    FAIL routes to human review with the full divergence map.
    No retry. Failure is data.
    """
    final_snapshot = s2_result["snapshots"][-1]

    # Domain-adaptive TYPE threshold: the compiler sets this based on
    # domain maturity. Established pharmacology = 0.90, frontier
    # cross-domain (bioelectric + pharmacology) = 0.80.
    type01_threshold = S3_TYPE01_THRESHOLD  # default
    if compiled and "s3_type01_threshold" in compiled:
        type01_threshold = compiled["s3_type01_threshold"]

    # Convergence check: cosine (semantic similarity on claim embeddings)
    # is the primary metric. Jaccard floor prevents pathological cases where
    # models agree semantically but share zero vocabulary (shouldn't happen
    # with real scientific text, but guards against degenerate inputs).
    convergence_score = final_snapshot.cosine
    jaccard_floor_pass = final_snapshot.jaccard >= S3_JACCARD_FLOOR
    convergence_pass = (convergence_score > S3_CONVERGENCE_THRESHOLD) and jaccard_floor_pass
    type_pass = final_snapshot.type_01_ratio >= type01_threshold

    # Legacy: still report individual Jaccard pass for diagnostics
    jaccard_pass = final_snapshot.jaccard > S3_JACCARD_THRESHOLD

    passed = convergence_pass and type_pass

    gate_result = {
        "stage": "S3",
        "passed": passed,
        "convergence_score": round(convergence_score, 4),
        "convergence_threshold": S3_CONVERGENCE_THRESHOLD,
        "convergence_pass": convergence_pass,
        "jaccard": round(final_snapshot.jaccard, 4),
        "jaccard_threshold": S3_JACCARD_THRESHOLD,
        "jaccard_pass": jaccard_pass,
        "type_01_ratio": round(final_snapshot.type_01_ratio, 4),
        "type_01_threshold": type01_threshold,
        "type_pass": type_pass,
        "cosine": round(final_snapshot.cosine, 4),
        "jsd": round(final_snapshot.jsd, 4),
        "kappa": round(final_snapshot.kappa, 4),
        "type_distribution": final_snapshot.type_distribution,
    }

    if not passed:
        # Build divergence map for human review
        gate_result["divergence_map"] = _build_divergence_map(s2_result)
        gate_result["recommendation"] = (
            "S3 gate FAILED. The system did not converge to stable consensus. "
            "This indicates genuine scientific disagreement worth investigating. "
            "Review the divergence map below."
        )

    return gate_result


def _build_divergence_map(s2_result: dict) -> dict:
    """Build a divergence map showing where models disagree.

    Useful for human review when S3 fails — tells you WHERE
    the disagreement is, not just that it exists.
    """
    parsed = s2_result["parsed"]

    # Collect all unique claim statements
    all_claims = {}
    for p in parsed:
        for c in p.claims:
            key = c.statement[:80]  # Truncate for grouping
            if key not in all_claims:
                all_claims[key] = {
                    "statement": c.statement,
                    "models_supporting": [],
                    "types_assigned": [],
                    "confidences": [],
                }
            all_claims[key]["models_supporting"].append(p.model)
            all_claims[key]["types_assigned"].append(c.type)
            all_claims[key]["confidences"].append(c.confidence)

    # Sort by disagreement (high variance in TYPE = more disagreement)
    import numpy as np
    divergent = []
    for key, info in all_claims.items():
        type_var = float(np.var(info["types_assigned"])) if len(info["types_assigned"]) > 1 else 0
        info["type_variance"] = round(type_var, 4)
        info["n_supporters"] = len(info["models_supporting"])
        divergent.append(info)

    divergent.sort(key=lambda x: x["type_variance"], reverse=True)

    return {
        "total_unique_claims": len(all_claims),
        "most_divergent": divergent[:10],
        "models_present": [p.model for p in parsed],
    }


async def run_pipeline(
    compiled: dict,
    session_seed: Optional[int] = None,
) -> dict:
    """Run the full S1 → S2 → S3 pipeline.

    Returns:
        Complete pipeline result with all stages, convergence history,
        and S3 gate outcome.
    """
    # S1
    s1_result = await run_s1(compiled)

    # S2 — Contribution Synthesis (0 API calls)
    s2_result = run_s2_synthesis(s1_result)

    # S3
    s3_result = run_s3_gate(s2_result, compiled=compiled)

    total_calls = s1_result["total_calls"] + s2_result["total_calls"]

    return {
        "session_id": compiled["session_id"],
        "question": compiled["question"],
        "s1": {
            "snapshot": asdict(s1_result["snapshot"]),
            "n_models_responded": len(s1_result["parsed"]),
            "calls": s1_result["total_calls"],
        },
        "s2": {
            "total_rounds": s2_result["total_rounds"],
            "early_stopped": s2_result["early_stopped"],
            "rounds": s2_result["rounds"],
            "calls": s2_result["total_calls"],
        },
        "s3": s3_result,
        "total_calls": total_calls,
        "final_claims": _extract_final_claims(s2_result["parsed"]),
    }


def _extract_final_claims(parsed: list[ParsedResponse]) -> list[dict]:
    """Extract and deduplicate final claims across all models."""
    claims = []
    for p in parsed:
        for c in p.claims:
            claims.append({
                "statement": c.statement,
                "type": c.type,
                "confidence": c.confidence,
                "mechanism": c.mechanism,
                "falsifiable_by": c.falsifiable_by,
                "model": p.model,
            })
    return claims


def build_recirculation_context(s2_result: dict, s3_result: dict) -> str:
    """Build a recirculation block from synthesized claims for the next cycle.

    Uses synthesized_claims from S2 synthesis when available (new path),
    falls back to parsing s2_result["parsed"] claims (legacy path).

    Converged claims (TYPE 0/1) become established priors.
    Singular claims (TYPE 3) become investigation threads.
    """
    from src.convergence.claim_tuples import extract_tuples, group_relation
    from src.parser import Claim

    # New path: use synthesized_claims from synthesis stage
    synthesized = s2_result.get("synthesized_claims")
    if synthesized:
        return _build_recirculation_from_synthesis(synthesized, s3_result)

    # Legacy path: extract from parsed responses
    return _build_recirculation_from_parsed(s2_result, s3_result)


def _build_recirculation_from_synthesis(synthesized, s3_result: dict) -> str:
    """Build recirculation context from SynthesizedClaim objects."""
    converged = [s for s in synthesized if s.type in (0, 1)]
    singulars = [s for s in synthesized if s.type == 3]

    if not converged and not singulars:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════",
        "INDEPENDENT CONSENSUS — FROM PREVIOUS CYCLE",
        "═══════════════════════════════════════════════════════",
        "",
        "The following claims showed independent convergence across multiple",
        "mirrors in the previous cycle. No mirror saw another's output —",
        "this convergence emerged independently. Go deeper on these threads:",
        "explore mechanisms, quantitative predictions, and falsification conditions.",
        "",
    ]

    # Converged claims (TYPE 0/1)
    for i, sc in enumerate(converged[:12], 1):
        label = "ESTABLISHED" if sc.type == 0 else "REPLICATED"
        lines.append(f"  [{label}] {sc.statement}")
        if sc.mechanism:
            lines.append(f"    MECHANISM: {sc.mechanism}")
        lines.append(f"    OVERLAP: {sc.overlap_count}/5 mirrors | CONFIDENCE: {sc.confidence}")
        lines.append("")

    # Singular claims (TYPE 3) — potential novel insights
    if singulars:
        lines.append("")
        lines.append("SINGULAR THREADS (1 mirror only — potential novel insights):")
        for sc in singulars[:6]:  # Cap at 6
            lines.append(f"  [SINGULAR] {sc.statement}")
            source = sc.models[0] if sc.models else "unknown"
            lines.append(f"    SOURCE: {source} | CONFIDENCE: {sc.confidence}")
        lines.append("")
        lines.append(
            "One mirror saw these — investigate whether they're real. "
            "If you find supporting evidence, include them."
        )

    lines.append("")
    type_01_ratio = s3_result.get("type_01_ratio", 0)
    lines.append(
        f"NOTE: Previous cycle achieved {type_01_ratio:.0%} TYPE 0/1 overlap. "
        f"Explore mechanisms and quantitative predictions for converged claims. "
        f"Investigate singular threads for potential novel discoveries."
    )
    lines.append("═══════════════════════════════════════════════════════")

    return "\n".join(lines)


def _build_recirculation_from_parsed(s2_result: dict, s3_result: dict) -> str:
    """Legacy path: build recirculation from parsed claims with tuple-based dedup."""
    from src.convergence.claim_tuples import extract_tuples, group_relation
    from src.parser import Claim

    parsed = s2_result["parsed"]

    # Collect TYPE 0/1 claims
    converged_claims = []
    for p in parsed:
        for c in p.claims:
            if c.type in (0, 1) and c.confidence >= 0.7:
                converged_claims.append({
                    "statement": c.statement,
                    "type": c.type,
                    "confidence": c.confidence,
                    "mechanism": c.mechanism,
                })

    # Deduplicate using claim tuples
    seen_tuples: set = set()
    unique_claims = []
    for claim in converged_claims:
        claim_obj = Claim(
            statement=claim["statement"],
            type=claim["type"],
            confidence=claim["confidence"],
            mechanism=claim["mechanism"],
        )
        tuples = extract_tuples(claim_obj)
        grouped = frozenset(
            (t.subject, group_relation(t.relation), t.object) for t in tuples
        )

        if grouped and grouped & seen_tuples:
            for existing in unique_claims:
                existing_obj = Claim(
                    statement=existing["statement"],
                    type=existing["type"],
                    confidence=existing["confidence"],
                    mechanism=existing["mechanism"],
                )
                existing_tuples = extract_tuples(existing_obj)
                existing_grouped = frozenset(
                    (t.subject, group_relation(t.relation), t.object) for t in existing_tuples
                )
                if existing_grouped & grouped:
                    existing["confidence"] = max(existing["confidence"], claim["confidence"])
                    break
            continue

        seen_tuples |= grouped
        unique_claims.append(claim)

    if not unique_claims:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════",
        "INDEPENDENT CONSENSUS — FROM PREVIOUS CYCLE",
        "═══════════════════════════════════════════════════════",
        "",
        "The following claims showed independent convergence across multiple",
        "mirrors in the previous cycle. Go deeper on these threads.",
        "",
    ]

    for i, claim in enumerate(unique_claims[:12], 1):
        type_label = "ESTABLISHED" if claim["type"] == 0 else "REPLICATED"
        lines.append(f"  [{type_label}] {claim['statement']}")
        if claim["mechanism"]:
            lines.append(f"    MECHANISM: {claim['mechanism']}")
        lines.append(f"    CONFIDENCE: {claim['confidence']}")
        lines.append("")

    type_01_ratio = s3_result.get("type_01_ratio", 0)
    lines.append(
        f"NOTE: Previous cycle achieved {type_01_ratio:.0%} TYPE 0/1 overlap. "
        f"Explore mechanisms and quantitative predictions for converged claims."
    )
    lines.append("═══════════════════════════════════════════════════════")

    return "\n".join(lines)


def enrich_compiled_for_recirculation(
    compiled: dict,
    recirculation_context: str,
    cycle_num: int,
) -> dict:
    """Inject recirculation context into the compiled prompt for another S1→S2→S3 cycle.

    The original priors stay. The recirculation context is appended before the
    scaffold sections, giving models both the quantitative priors AND the
    accumulated consensus from previous cycles.
    """
    if not recirculation_context:
        return compiled

    original_prompt = compiled["prompt"]

    # Inject the recirculation context before the SECTION 1 marker
    marker = "SECTION 1: DECOMPOSITION"
    if marker in original_prompt:
        parts = original_prompt.split(marker, 1)
        enriched_prompt = (
            parts[0]
            + f"\n{recirculation_context}\n\n"
            + marker
            + parts[1]
        )
    else:
        # Fallback: append to the end
        enriched_prompt = original_prompt + f"\n\n{recirculation_context}"

    # Update session_id to track cycles
    session_id = compiled["session_id"]
    if "_cycle" not in session_id:
        session_id = f"{session_id}_cycle{cycle_num}"
    else:
        # Replace existing cycle marker
        session_id = session_id.rsplit("_cycle", 1)[0] + f"_cycle{cycle_num}"

    return {
        **compiled,
        "prompt": enriched_prompt,
        "session_id": session_id,
    }


def run_pipeline_sync(compiled: dict, session_seed: Optional[int] = None) -> dict:
    """Synchronous wrapper for run_pipeline."""
    return asyncio.run(run_pipeline(compiled, session_seed))
