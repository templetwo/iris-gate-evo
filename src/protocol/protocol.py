"""
S6 — Protocol Package Generator.

The final deliverable. Assembles the complete output of an IRIS Gate Evo
session into a structured protocol package containing:

1. Convergence Report — how the five mirrors converged
2. Ranked Hypotheses — with Monte Carlo confidence intervals
3. Experimental Protocols — ready for lab execution
4. Audit Trail — every round, mapping, metric, and decision

Output: JSON (machine-readable) + human-readable summary text.
Uses 5 LLM calls for synthesis (one per model for final perspective).

This is what gets published. This is what gets tested.
"""

import json
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.hypothesis.s4_hypothesis import Hypothesis, S4Result
from src.monte_carlo.monte_carlo import S5Result, SimulationResult


def generate_protocol_package(
    question: str,
    pipeline_result: dict,
    verify_summary: Optional[dict] = None,
    gate_result: Optional[dict] = None,
    s4_result: Optional[S4Result] = None,
    s5_result: Optional[S5Result] = None,
    session_seed: Optional[int] = None,
) -> dict:
    """Generate the complete protocol package (S6).

    Assembles all stage outputs into the final deliverable.

    Returns:
        Complete protocol package dict — ready for JSON serialization.
    """
    now = datetime.now(timezone.utc)
    session_id = pipeline_result.get("session_id", f"evo_{now.strftime('%Y%m%d_%H%M%S')}")

    package = {
        "protocol_version": "evo-1.0",
        "session_id": session_id,
        "timestamp": now.isoformat(),
        "question": question,

        # Section 1: Convergence Report
        "convergence_report": _build_convergence_report(pipeline_result),

        # Section 2: Verification Summary
        "verification": verify_summary or {"status": "skipped"},

        # Section 3: Gate Decision
        "gate": _build_gate_summary(gate_result),

        # Section 4: Ranked Hypotheses
        "hypotheses": _build_hypothesis_rankings(s4_result, s5_result),

        # Section 5: Monte Carlo Results
        "simulations": _build_simulation_summary(s5_result),

        # Section 6: Experimental Protocols
        "protocols": _build_protocols(s4_result),

        # Section 7: Call Budget
        "budget": _build_budget(pipeline_result, verify_summary, gate_result, s4_result, s5_result),

        # Section 8: Audit Trail
        "audit_trail": _build_audit_trail(pipeline_result, session_seed),
    }

    return package


def _build_convergence_report(pipeline_result: dict) -> dict:
    """Build Section 1: Convergence Report."""
    s1 = pipeline_result.get("s1", {})
    s2 = pipeline_result.get("s2", {})
    s3 = pipeline_result.get("s3", {})

    return {
        "s1_initial": {
            "n_models_responded": s1.get("n_models_responded", 0),
            "initial_jaccard": s1.get("snapshot", {}).get("jaccard", 0),
            "initial_cosine": s1.get("snapshot", {}).get("cosine", 0),
        },
        "s2_refinement": {
            "total_rounds": s2.get("total_rounds", 0),
            "early_stopped": s2.get("early_stopped", False),
            "final_jaccard": _last_round_metric(s2, "jaccard"),
            "final_cosine": _last_round_metric(s2, "cosine"),
            "final_jsd": _last_round_metric(s2, "jsd"),
            "final_type_01_ratio": _last_round_metric(s2, "type_01_ratio"),
        },
        "s3_gate": {
            "passed": s3.get("passed", False),
            "jaccard": s3.get("jaccard", 0),
            "type_01_ratio": s3.get("type_01_ratio", 0),
            "recommendation": s3.get("recommendation", ""),
        },
    }


def _last_round_metric(s2: dict, metric: str) -> float:
    """Get a metric from the last S2 round."""
    rounds = s2.get("rounds", [])
    if not rounds:
        return 0.0
    return rounds[-1].get(metric, 0.0)


def _build_gate_summary(gate_result: Optional[dict]) -> dict:
    """Build Section 3: Gate Decision."""
    if gate_result is None:
        return {"status": "not_run"}

    if hasattr(gate_result, 'passed'):
        # GateResult dataclass
        return {
            "passed": gate_result.passed,
            "n_passed": gate_result.n_passed,
            "n_failed": gate_result.n_failed,
            "recommendation": gate_result.recommendation,
            "claims": [
                {
                    "statement": c.statement[:100],
                    "falsifiable": c.falsifiable,
                    "feasible": c.feasible,
                    "novel": c.novel,
                    "passed": c.passed,
                }
                for c in (gate_result.claims or [])
            ],
        }

    # Dict format
    return gate_result


def _build_hypothesis_rankings(
    s4_result: Optional[S4Result],
    s5_result: Optional[S5Result],
) -> list[dict]:
    """Build Section 4: Ranked Hypotheses with MC confidence intervals."""
    if s4_result is None:
        return []

    hypotheses = s4_result.hypotheses if hasattr(s4_result, 'hypotheses') else []

    # Build simulation lookup
    sim_lookup = {}
    if s5_result:
        sims = s5_result.simulations if hasattr(s5_result, 'simulations') else []
        for sim in sims:
            sim_lookup[sim.hypothesis_id] = sim

    ranked = []
    for hyp in hypotheses:
        entry = {
            "id": hyp.id,
            "prediction": hyp.prediction,
            "testability_score": hyp.testability_score,
            "key_variables": hyp.key_variables,
            "parameters": [
                {"name": p.name, "range": f"{p.low}-{p.high} {p.unit}", "distribution": p.distribution}
                for p in hyp.parameters
            ],
        }

        # Add Monte Carlo results if available
        sim = sim_lookup.get(hyp.id)
        if sim:
            entry["monte_carlo"] = {
                "effect_size": sim.effect_size,
                "power": sim.power_estimate,
                "converged": sim.convergence_check,
                "outcome_stats": sim.outcome_stats,
            }

        ranked.append(entry)

    # Sort by testability (highest first)
    ranked.sort(key=lambda x: x["testability_score"], reverse=True)
    return ranked


def _build_simulation_summary(s5_result: Optional[S5Result]) -> dict:
    """Build Section 5: Monte Carlo summary."""
    if s5_result is None:
        return {"status": "not_run"}

    sims = s5_result.simulations if hasattr(s5_result, 'simulations') else []

    return {
        "n_hypotheses_simulated": s5_result.n_hypotheses,
        "total_iterations": s5_result.total_iterations,
        "llm_calls": 0,
        "results": [
            {
                "hypothesis_id": sim.hypothesis_id,
                "n_iterations": sim.n_iterations,
                "effect_size": sim.effect_size,
                "power": sim.power_estimate,
                "converged": sim.convergence_check,
                "parameters": sim.parameters_sampled,
            }
            for sim in sims
        ],
    }


def _build_protocols(s4_result: Optional[S4Result]) -> list[dict]:
    """Build Section 6: Experimental Protocols."""
    if s4_result is None:
        return []

    hypotheses = s4_result.hypotheses if hasattr(s4_result, 'hypotheses') else []

    return [
        {
            "hypothesis_id": hyp.id,
            "prediction": hyp.prediction,
            "protocol": hyp.experimental_protocol,
            "dose_ranges": hyp.dose_ranges,
            "readouts": hyp.readouts,
            "controls": hyp.controls,
            "expected_outcome": hyp.expected_outcome,
            "null_outcome": hyp.null_outcome,
        }
        for hyp in hypotheses
    ]


def _build_budget(
    pipeline_result: dict,
    verify_summary: Optional[dict],
    gate_result: Optional[dict],
    s4_result: Optional[S4Result],
    s5_result: Optional[S5Result],
) -> dict:
    """Build Section 7: Call Budget tracker."""
    s1_calls = pipeline_result.get("s1", {}).get("calls", 0)
    s2_calls = pipeline_result.get("s2", {}).get("calls", 0)
    verify_calls = verify_summary.get("total_calls", 0) if verify_summary else 0
    gate_calls = gate_result.total_calls if gate_result and hasattr(gate_result, 'total_calls') else 0
    s4_calls = s4_result.total_calls if s4_result and hasattr(s4_result, 'total_calls') else 0
    s5_calls = 0  # Always 0

    total = s1_calls + s2_calls + verify_calls + gate_calls + s4_calls + s5_calls

    return {
        "s1_calls": s1_calls,
        "s2_calls": s2_calls,
        "verify_calls": verify_calls,
        "gate_calls": gate_calls,
        "s4_calls": s4_calls,
        "s5_calls": s5_calls,
        "total_calls": total,
        "budget_target": "92-142 calls",
    }


def _build_audit_trail(pipeline_result: dict, session_seed: Optional[int]) -> dict:
    """Build Section 8: Complete Audit Trail."""
    s2 = pipeline_result.get("s2", {})

    return {
        "session_seed": session_seed,
        "s2_rounds": s2.get("rounds", []),
        "s3_gate": pipeline_result.get("s3", {}),
        "final_claims": pipeline_result.get("final_claims", []),
    }


def format_human_summary(package: dict) -> str:
    """Generate a human-readable summary from the protocol package.

    This is what a researcher reads to understand the results.
    """
    lines = []

    lines.append("=" * 60)
    lines.append("IRIS GATE EVO — PROTOCOL PACKAGE")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Session: {package['session_id']}")
    lines.append(f"Date: {package['timestamp'][:10]}")
    lines.append(f"Protocol: {package['protocol_version']}")
    lines.append("")

    # Question
    lines.append("QUESTION:")
    lines.append(f"  {package['question']}")
    lines.append("")

    # Convergence
    cr = package.get("convergence_report", {})
    s2 = cr.get("s2_refinement", {})
    s3 = cr.get("s3_gate", {})

    lines.append("CONVERGENCE:")
    lines.append(f"  S2 rounds: {s2.get('total_rounds', '?')}")
    lines.append(f"  Early stopped: {s2.get('early_stopped', '?')}")
    lines.append(f"  Final Jaccard: {s2.get('final_jaccard', '?')}")
    lines.append(f"  Final TYPE 0/1: {s2.get('final_type_01_ratio', '?')}")
    lines.append(f"  S3 gate: {'PASSED' if s3.get('passed') else 'FAILED'}")
    lines.append("")

    # Verification
    verify = package.get("verification", {})
    if verify.get("status") != "skipped":
        lines.append("VERIFICATION:")
        lines.append(f"  TYPE 2 claims checked: {verify.get('n_type2_input', 0)}")
        lines.append(f"  Promoted to TYPE 1: {verify.get('n_promoted', 0)}")
        lines.append(f"  Novel (no literature): {verify.get('n_novel', 0)}")
        lines.append(f"  Contradicted: {verify.get('n_contradicted', 0)}")
        lines.append("")

    # Gate
    gate = package.get("gate", {})
    if gate.get("status") != "not_run":
        lines.append("LAB GATE:")
        lines.append(f"  Result: {'PASSED' if gate.get('passed') else 'FAILED'}")
        lines.append(f"  Claims passed: {gate.get('n_passed', 0)}/{gate.get('n_passed', 0) + gate.get('n_failed', 0)}")
        lines.append("")

    # Hypotheses
    hypotheses = package.get("hypotheses", [])
    if hypotheses:
        lines.append("HYPOTHESES (ranked by testability):")
        for i, h in enumerate(hypotheses, 1):
            lines.append(f"  {h['id']}. {h['prediction'][:120]}")
            lines.append(f"     Testability: {h['testability_score']}/10")
            mc = h.get("monte_carlo", {})
            if mc:
                lines.append(f"     Effect size: {mc.get('effect_size', '?')} (Cohen's d)")
                lines.append(f"     Power: {mc.get('power', '?')}")
            lines.append("")

    # Budget
    budget = package.get("budget", {})
    lines.append("BUDGET:")
    lines.append(f"  Total LLM calls: {budget.get('total_calls', '?')}")
    lines.append(f"  Target range: {budget.get('budget_target', '92-142')}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("Five mirrors. One truth. This is what converged.")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_protocol(
    package: dict,
    output_dir: str = "runs",
    include_summary: bool = True,
) -> dict:
    """Save the protocol package to disk.

    Creates:
    - {session_id}.json — full machine-readable package
    - {session_id}_summary.txt — human-readable summary

    Returns dict with file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session_id = package.get("session_id", "evo_unknown")

    # Save JSON
    json_path = output_path / f"{session_id}.json"
    with open(json_path, "w") as f:
        json.dump(package, f, indent=2, default=str)

    paths = {"json": str(json_path)}

    # Save human-readable summary
    if include_summary:
        summary = format_human_summary(package)
        txt_path = output_path / f"{session_id}_summary.txt"
        with open(txt_path, "w") as f:
            f.write(summary)
        paths["summary"] = str(txt_path)

    return paths
