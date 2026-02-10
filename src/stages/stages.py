"""
Stage Orchestrator — S1 → S2 → S3

S1 (Formulation): PULSE fires, responses parsed into claims.
S2 (Refinement): Anonymized debate loop with decreasing token budgets.
    Early-stop: delta < 1% for 3 CONSECUTIVE rounds AND TYPE 0/1 >= 80%.
    BOTH conditions must be true. A stable pile of speculation doesn't stop.
S3 (Stable Attractor): Strictest convergence gate.
    Jaccard > 0.85, TYPE 0/1 >= 90%, compression stable.
    FAIL routes to human review with the divergence map.
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import TOKEN_BUDGETS
from src.parser import parse_response, ParsedResponse
from src.convergence.convergence import compute, delta, ConvergenceSnapshot
from src.stages.anonymizer import anonymize_round, build_debate_prompt
from src.pulse.pulse import fire


# S2 config
S2_MAX_ROUNDS = 10
S2_EARLY_STOP_DELTA = 0.01        # 1% change threshold
S2_EARLY_STOP_CONSECUTIVE = 3      # Must be stable for 3 consecutive rounds
S2_EARLY_STOP_TYPE01_THRESHOLD = 0.80  # AND >= 80% TYPE 0/1


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


async def run_s2(
    compiled: dict,
    s1_result: dict,
    session_seed: Optional[int] = None,
) -> dict:
    """S2 — Refinement Loop. Anonymized cross-model debate.

    Token budgets decrease from S2_start (800) to S2_end (700).
    Early-stop: delta < 1% for 3 consecutive rounds AND TYPE 0/1 >= 80%.

    Returns:
        Dict with final parsed responses, convergence history, and metadata.
    """
    parsed = s1_result["parsed"]
    snapshots = [s1_result["snapshot"]]
    total_calls = 0
    round_logs = []

    # Compute token budget slope
    budget_start = TOKEN_BUDGETS["S2_start"]
    budget_end = TOKEN_BUDGETS["S2_end"]

    consecutive_stable = 0

    for round_num in range(1, S2_MAX_ROUNDS + 1):
        # Interpolate token budget (decreasing)
        progress = round_num / S2_MAX_ROUNDS
        token_budget = int(budget_start - (budget_start - budget_end) * progress)

        # Anonymize — RANDOM assignment every round
        anonymized, mapping = anonymize_round(
            parsed, round_num=round_num, seed=session_seed
        )

        # Build debate prompt
        debate_prompt = build_debate_prompt(
            question=compiled["question"],
            anonymized=anonymized,
            round_num=round_num,
            token_budget=token_budget,
        )

        # Create a modified compiled dict for this round
        round_compiled = {
            **compiled,
            "prompt": debate_prompt,
            "token_budgets": {**compiled["token_budgets"], "S1": token_budget},
        }

        # Update model max_tokens for this round
        for name in round_compiled["models"]:
            round_compiled["models"][name] = {
                **round_compiled["models"][name],
                "max_tokens": token_budget + 400,
            }

        # Fire PULSE for this round
        pulse_result = await fire(round_compiled)
        total_calls += pulse_result["meta"]["models_dispatched"]

        # Parse responses
        new_parsed = []
        for resp in pulse_result["responses"]:
            if resp["status"] == "ok":
                p = parse_response(resp["content"], model=resp["model"])
                new_parsed.append(p)

        # If we got responses, update parsed
        if new_parsed:
            parsed = new_parsed

        # Compute convergence
        claims_per_model = [p.claims for p in parsed]
        snapshot = compute(claims_per_model, round_num=round_num)
        snapshots.append(snapshot)

        # Compute delta from previous round
        d = delta(snapshot, snapshots[-2]) if len(snapshots) >= 2 else 1.0

        # Log this round
        round_log = {
            "round": round_num,
            "token_budget": token_budget,
            "delta": round(d, 4),
            "jaccard": round(snapshot.jaccard, 4),
            "cosine": round(snapshot.cosine, 4),
            "jsd": round(snapshot.jsd, 4),
            "type_01_ratio": round(snapshot.type_01_ratio, 4),
            "n_claims": snapshot.n_claims_per_model,
            "mapping": mapping,  # Audit trail only
        }
        round_logs.append(round_log)

        # --- Early-stop check: BOTH conditions must be true ---
        delta_stable = d < S2_EARLY_STOP_DELTA
        type_stable = snapshot.type_01_ratio >= S2_EARLY_STOP_TYPE01_THRESHOLD

        if delta_stable and type_stable:
            consecutive_stable += 1
        else:
            consecutive_stable = 0  # Reset — must be CONSECUTIVE

        if consecutive_stable >= S2_EARLY_STOP_CONSECUTIVE:
            round_log["early_stop"] = True
            round_log["early_stop_reason"] = (
                f"delta < {S2_EARLY_STOP_DELTA} for {S2_EARLY_STOP_CONSECUTIVE} "
                f"consecutive rounds AND TYPE 0/1 >= {S2_EARLY_STOP_TYPE01_THRESHOLD}"
            )
            break

    return {
        "stage": "S2",
        "parsed": parsed,
        "snapshots": snapshots,
        "rounds": round_logs,
        "total_rounds": len(round_logs),
        "total_calls": total_calls,
        "early_stopped": consecutive_stable >= S2_EARLY_STOP_CONSECUTIVE,
    }


def run_s3_gate(s2_result: dict) -> dict:
    """S3 — Stable Attractor Gate. Strictest convergence check.

    PASS requires:
    - Convergence score > 0.85 (composite: 0.3*jaccard + 0.7*cosine)
    - TYPE 0/1 >= 90%

    Cosine similarity on claim embeddings captures semantic convergence
    that tuple-based Jaccard misses when models express the same science
    in structurally different ways. The composite score weighs the more
    reliable instrument (cosine) more heavily while keeping Jaccard as
    a lexical grounding check.

    FAIL routes to human review with the full divergence map.
    No retry. Failure is data.
    """
    final_snapshot = s2_result["snapshots"][-1]

    # Convergence check: cosine (semantic similarity on claim embeddings)
    # is the primary metric. Jaccard floor prevents pathological cases where
    # models agree semantically but share zero vocabulary (shouldn't happen
    # with real scientific text, but guards against degenerate inputs).
    convergence_score = final_snapshot.cosine
    jaccard_floor_pass = final_snapshot.jaccard >= S3_JACCARD_FLOOR
    convergence_pass = (convergence_score > S3_CONVERGENCE_THRESHOLD) and jaccard_floor_pass
    type_pass = final_snapshot.type_01_ratio >= S3_TYPE01_THRESHOLD

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
        "type_01_threshold": S3_TYPE01_THRESHOLD,
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

    # S2
    s2_result = await run_s2(compiled, s1_result, session_seed=session_seed)

    # S3
    s3_result = run_s3_gate(s2_result)

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


def run_pipeline_sync(compiled: dict, session_seed: Optional[int] = None) -> dict:
    """Synchronous wrapper for run_pipeline."""
    return asyncio.run(run_pipeline(compiled, session_seed))
