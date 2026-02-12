#!/usr/bin/env python3
"""
IRIS Gate Evo — Main Entry Point

Question in. Five mirrors converge. Protocol package out.

Usage:
    python main.py "Your research question here"
    python main.py --compile-only "Your question"
    python main.py --stage s1 "Your question"       # Stop after S1
    python main.py --stage s3 "Your question"       # Stop after S3 gate
    python main.py --offline "Your question"         # No API calls (offline mode)
    python main.py                                   # Default CBD test question
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment before anything touches APIs
load_dotenv()

from src.compiler import compile
from src.pulse import fire_sync
from src.models import TOKEN_BUDGETS
from src.preflight import run_preflight, format_preflight
from src.dashboard import Dashboard


# The test question from the spec — validates the full pipeline
DEFAULT_QUESTION = (
    "What are the mechanisms by which CBD induces selective cytotoxicity "
    "in cancer cells while sparing healthy cells, with specific reference "
    "to VDAC1-mediated mitochondrial membrane potential disruption?"
)


def display_compiled(compiled: dict) -> None:
    """Display compiler output summary."""
    print("\n" + "=" * 60)
    print("C0 — COMPILER OUTPUT")
    print("=" * 60)
    print(f"Session:      {compiled['session_id']}")
    print(f"Domains:      {', '.join(compiled['domains'])}")
    print(f"Cross-domain: {compiled['cross_domain_flag']}")
    print(f"Priors:       {len(compiled['priors'])} injected")

    for p in compiled["priors"]:
        value = p["value"]
        if isinstance(value, list):
            value = f"[{value[0]}, {value[1]}]"
        print(f"  TYPE {p['type']} | {p['param']} = {value} {p['unit']}")

    print(f"Models:       {', '.join(compiled['models'].keys())}")
    print(f"Token budget: S1={compiled['token_budgets']['S1']}")
    print("=" * 60)


def display_convergence(snapshot: dict, label: str = "S1") -> None:
    """Display convergence metrics."""
    print(f"\n  [{label}] Jaccard: {snapshot.get('jaccard', 0):.4f} | "
          f"Cosine: {snapshot.get('cosine', 0):.4f} | "
          f"JSD: {snapshot.get('jsd', 0):.4f} | "
          f"TYPE 0/1: {snapshot.get('type_01_ratio', 0):.2%}")


def display_s2_synthesis(s2_result: dict) -> None:
    """Display S2 contribution synthesis results."""
    synthesized = s2_result.get("synthesized_claims", [])
    conflicts = s2_result.get("conflicts", [])

    print("\n" + "-" * 60)
    print("S2 — CONTRIBUTION SYNTHESIS (0 API calls)")
    print("-" * 60)

    if not synthesized:
        print("  No claims synthesized.")
        return

    # Overlap distribution
    from collections import Counter
    overlap_counts = Counter(s.overlap_count for s in synthesized)
    type_labels = {0: "TYPE 0", 1: "TYPE 0", 2: "TYPE 2", 3: "TYPE 3"}

    for n in sorted(overlap_counts.keys(), reverse=True):
        count = overlap_counts[n]
        from src.stages.synthesis import OVERLAP_TYPE_MAP, TYPE_LABELS
        t = OVERLAP_TYPE_MAP.get(n, 3)
        label = TYPE_LABELS.get(t, "?")
        print(f"  {n}/5 overlap: {count} claims (TYPE {t} — {label})")

    # Singulars highlighted
    singulars = [s for s in synthesized if s.type == 3]
    if singulars:
        print(f"\n  SINGULAR — potential novel insights:")
        for s in singulars[:5]:
            source = s.models[0] if s.models else "?"
            print(f"    [{source}] {s.statement[:80]}")

    # Conflicts
    if conflicts:
        print(f"\n  Conflicts: {len(conflicts)}")
        for c in conflicts[:3]:
            vals = ", ".join(f"{m}: {v}" for m, v in c.values.items())
            print(f"    {c.subject} {c.relation} {c.object}: {vals}")

    # TYPE 0/1 ratio
    t01 = sum(1 for s in synthesized if s.type in (0, 1))
    total = len(synthesized)
    ratio = t01 / total if total else 0
    print(f"\n  TYPE 0/1: {ratio:.2%} ({t01}/{total} claims)")


def display_s3_gate(s3_result: dict) -> None:
    """Display S3 gate decision."""
    passed = s3_result.get("passed", False)
    status = "PASSED" if passed else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"S3 GATE: {status}")
    print(f"  Cosine (semantic): {s3_result.get('convergence_score', 0):.4f} "
          f"(threshold: {s3_result.get('convergence_threshold', 0.85)})")
    print(f"  Jaccard (lexical): {s3_result.get('jaccard', 0):.4f} "
          f"(floor: 0.10)")
    print(f"  TYPE 0/1: {s3_result.get('type_01_ratio', 0):.2%} "
          f"(threshold: {s3_result.get('type_01_threshold', 0.90):.0%})")
    if not passed:
        print(f"\n  {s3_result.get('recommendation', '')}")
    print("=" * 60)


def display_verify(verify_summary: dict) -> None:
    """Display VERIFY stage results."""
    print(f"\n{'─' * 60}")
    print("VERIFY — TYPE 2 CLAIM CHECKING")
    print(f"{'─' * 60}")
    print(f"  Claims checked:  {verify_summary.get('n_type2_input', 0)}")
    print(f"  PROMOTED → T1:   {verify_summary.get('n_promoted', 0)}")
    print(f"  HELD (T2):       {verify_summary.get('n_held', 0)}")
    print(f"  NOVEL:           {verify_summary.get('n_novel', 0)}")
    print(f"  CONTRADICTED:    {verify_summary.get('n_contradicted', 0)}")


def display_gate(gate_result) -> None:
    """Display Lab Gate results."""
    passed = gate_result.passed if hasattr(gate_result, 'passed') else False
    status = "PASSED" if passed else "FAILED"
    print(f"\n{'─' * 60}")
    print(f"LAB GATE: {status}")
    print(f"{'─' * 60}")
    n_p = gate_result.n_passed if hasattr(gate_result, 'n_passed') else 0
    n_f = gate_result.n_failed if hasattr(gate_result, 'n_failed') else 0
    print(f"  Passed: {n_p} | Failed: {n_f}")
    rec = gate_result.recommendation if hasattr(gate_result, 'recommendation') else ""
    if rec:
        print(f"  {rec}")


def display_s4(s4_result) -> None:
    """Display S4 hypothesis results."""
    print(f"\n{'─' * 60}")
    print("S4 — HYPOTHESES OPERATIONALIZED")
    print(f"{'─' * 60}")
    for h in s4_result.hypotheses:
        print(f"\n  {h.id}: {h.prediction[:100]}")
        print(f"    Testability: {h.testability_score}/10 | Params: {len(h.parameters)}")


def display_s5(s5_result) -> None:
    """Display S5 Monte Carlo results."""
    print(f"\n{'─' * 60}")
    print("S5 — MONTE CARLO (0 LLM calls)")
    print(f"{'─' * 60}")
    print(f"  Total iterations: {s5_result.total_iterations}")
    for sim in s5_result.simulations:
        stats = sim.outcome_stats.get("effect_magnitude", {})
        print(f"\n  {sim.hypothesis_id}: effect_size={sim.effect_size:.3f} "
              f"power={sim.power_estimate:.3f} "
              f"converged={sim.convergence_check}")
        print(f"    Outcome: mean={stats.get('mean', 0):.4f} "
              f"CI=[{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}]")


async def run_full_pipeline(compiled: dict, args) -> dict:
    """Run the complete IRIS Gate Evo pipeline."""
    from src.stages.stages import (
        run_s1, run_s3_gate,
        build_recirculation_context, enrich_compiled_for_recirculation,
    )
    from src.stages.synthesis import run_s2_synthesis
    from src.verify.verify import run_verify, apply_verdicts
    from src.gate.gate import run_gate
    from src.hypothesis.s4_hypothesis import run_s4
    from src.monte_carlo.monte_carlo import run_s5
    from src.protocol.protocol import generate_protocol_package, save_protocol, format_human_summary
    from dataclasses import asdict

    MAX_RECIRCULATIONS = 2  # Max 3 total cycles (1 initial + 2 recirculations)

    session_seed = int(time.time()) % 2**31
    total_calls = 0
    stop_stage = getattr(args, 'stage', 'full')
    offline = getattr(args, 'offline', False)
    use_dashboard = getattr(args, 'dashboard', True)

    # Initialize dashboard
    dash = Dashboard(enabled=use_dashboard and sys.stdout.isatty())
    type01_threshold = compiled.get("s3_type01_threshold", 0.90)
    dash.set_thresholds(cosine=0.85, type01=type01_threshold, jaccard_floor=0.10)

    current_compiled = compiled
    s1_result = None
    s2_result = None
    s3_result = None
    cycle = 0

    while cycle <= MAX_RECIRCULATIONS:
        cycle_label = f"CYCLE {cycle + 1}" if cycle > 0 else ""

        # ── S1: Formulation ──
        dash.update_cycle(cycle)

        if cycle > 0:
            print(f"\n{'=' * 60}")
            print(f"RECIRCULATION {cycle_label} — feeding converged claims back")
            print(f"{'=' * 60}")

        dash.update_stage(f"S1 — PULSE{f' ({cycle_label})' if cycle_label else ''}")
        print(f"\nFiring S1 — PULSE...{f' ({cycle_label})' if cycle_label else ''}")
        s1_result = await run_s1(current_compiled)
        total_calls += s1_result["total_calls"]
        n_ok = len(s1_result["parsed"])
        print(f"  {n_ok} mirrors responded")

        dash.update_models(n_ok)
        dash.update_calls(total_calls)
        dash.update_metrics(s1_result["snapshot"])
        dash.update_stage(f"S1 complete{f' ({cycle_label})' if cycle_label else ''}")
        dash.render()

        display_convergence(s1_result["snapshot"].__dict__ if hasattr(s1_result["snapshot"], '__dict__') else s1_result["snapshot"], "S1")

        if stop_stage == "s1":
            dash.finalize()
            print(f"\n[--stage s1] Stopping. {total_calls} calls used.")
            return {"s1": s1_result, "total_calls": total_calls}

        # ── S2: Contribution Synthesis (0 API calls) ──
        dash.update_stage(f"S2 — Synthesis{f' ({cycle_label})' if cycle_label else ''}")
        print(f"\nRunning S2 — Contribution Synthesis (0 API calls)...{f' ({cycle_label})' if cycle_label else ''}")
        s2_result = run_s2_synthesis(s1_result)
        # total_calls unchanged — synthesis uses 0 API calls

        # Update dashboard with synthesis snapshot
        if s2_result["snapshots"]:
            dash.update_metrics(s2_result["snapshots"][-1])
        dash.update_calls(total_calls)
        dash.render()

        display_s2_synthesis(s2_result)

        if stop_stage == "s2":
            dash.finalize()
            print(f"\n[--stage s2] Stopping. {total_calls} calls used.")
            return {"s1": s1_result, "s2": s2_result, "total_calls": total_calls}

        # ── S3: Convergence Gate (domain-adaptive threshold) ──
        dash.update_stage(f"S3 — Gate{f' ({cycle_label})' if cycle_label else ''}")
        s3_result = run_s3_gate(s2_result, compiled=compiled)
        dash.set_gate_status(s3_result["passed"])
        dash.render()
        display_s3_gate(s3_result)

        if stop_stage == "s3":
            dash.finalize()
            print(f"\n{total_calls} calls used.")
            return {"s1": s1_result, "s2": s2_result, "s3": s3_result, "total_calls": total_calls}

        if s3_result["passed"]:
            break  # Gate passed — continue to VERIFY and beyond

        # ── S3 FAILED — Recirculate or route to human review ──
        if cycle < MAX_RECIRCULATIONS:
            recirculation_context = build_recirculation_context(s2_result, s3_result)
            if recirculation_context:
                current_compiled = enrich_compiled_for_recirculation(
                    compiled,  # Always enrich from ORIGINAL compiled, not stacked
                    recirculation_context,
                    cycle_num=cycle + 2,
                )
                print(f"\nRecirculating — injecting {len(recirculation_context)} chars of independent consensus")
                cycle += 1
                continue
            else:
                print("\nNo claims to recirculate. Routing to human review.")
                break
        else:
            print(f"\nMax recirculations ({MAX_RECIRCULATIONS}) reached. Routing to human review.")
            break

    # If S3 never passed, save what we have and stop
    if not s3_result["passed"]:
        dash.finalize()
        print(f"\nS3 gate failed after {cycle + 1} cycle(s). Human review required.")
        print(f"\n{total_calls} calls used.")

        # Save partial results to structured folder
        from dataclasses import asdict
        partial_stage_data = {}
        if s1_result:
            partial_stage_data["s1_formulations"] = {
                "parsed": [
                    {"model": p.model, "raw": p.raw, "claims": [asdict(c) for c in p.claims]}
                    for p in s1_result.get("parsed", [])
                ],
                "snapshot": asdict(s1_result["snapshot"]) if hasattr(s1_result.get("snapshot"), '__dict__') else s1_result.get("snapshot"),
                "total_calls": s1_result.get("total_calls", 0),
            }
        if s2_result:
            partial_stage_data["s2_synthesis"] = {
                "synthesized_claims": [
                    asdict(c) if hasattr(c, '__dict__') else c
                    for c in s2_result.get("synthesized_claims", [])
                ],
                "conflicts": [
                    asdict(c) if hasattr(c, '__dict__') else c
                    for c in s2_result.get("conflicts", [])
                ],
                "total_rounds": s2_result.get("total_rounds", 0),
                "snapshots": [
                    asdict(s) if hasattr(s, '__dict__') else s
                    for s in s2_result.get("snapshots", [])
                ],
            }
        if s3_result:
            partial_stage_data["s3_convergence"] = s3_result

        from datetime import datetime, timezone
        partial_package = {
            "protocol_version": "evo-1.0",
            "session_id": compiled["session_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": compiled["question"],
            "outcome": "S3_FAILED",
            "cycles": cycle + 1,
            "total_calls": total_calls,
            "convergence_report": s3_result,
        }
        output_dir = Path(__file__).parent / "runs"
        paths = save_protocol(partial_package, output_dir=str(output_dir), stage_data=partial_stage_data)
        print(f"\nSaved to: {paths.get('dir', '')}/")
        n_files = len([k for k in paths if k != 'dir'])
        print(f"  {n_files} files ({', '.join(k for k in sorted(paths) if k != 'dir')})")

        return {"s1": s1_result, "s2": s2_result, "s3": s3_result, "total_calls": total_calls, "cycles": cycle + 1}

    # Build pipeline result for downstream stages
    pipeline_result = {
        "session_id": compiled["session_id"],
        "question": compiled["question"],
        "s1": {
            "n_models_responded": len(s1_result["parsed"]),
            "snapshot": asdict(s1_result["snapshot"]),
            "calls": s1_result["total_calls"],
        },
        "s2": {
            "total_rounds": s2_result["total_rounds"],
            "early_stopped": s2_result["early_stopped"],
            "rounds": s2_result["rounds"],
            "calls": s2_result["total_calls"],
        },
        "s3": s3_result,
        "final_claims": _extract_final_claims(s2_result["parsed"]),
    }

    # ── VERIFY: TYPE 2 Claim Checking ──
    verify_summary = None
    if not offline:
        print("\nRunning VERIFY — checking TYPE 2 claims...")
        verify_result = await run_verify(pipeline_result)
        total_calls += verify_result.total_calls
        pipeline_result = apply_verdicts(pipeline_result, verify_result)
        verify_summary = pipeline_result.get("verify_summary")
        if verify_summary:
            display_verify(verify_summary)
    else:
        print("\n[offline] Skipping VERIFY stage.")

    # ── LAB GATE: Falsifiability, Feasibility, Novelty ──
    print("\nRunning Lab Gate...")
    gate_result = await run_gate(pipeline_result, use_offline=offline)
    total_calls += gate_result.total_calls
    display_gate(gate_result)

    if not gate_result.passed:
        print(f"\nLab Gate failed. {total_calls} calls used. Route to human review.")
        return {**pipeline_result, "gate": gate_result, "total_calls": total_calls}

    # ── S4: Hypothesis Operationalization ──
    print("\nRunning S4 — Operationalizing hypotheses...")
    s4_result = await run_s4(pipeline_result, gate_result=gate_result, use_offline=offline)
    total_calls += s4_result.total_calls
    display_s4(s4_result)

    # ── S5: Monte Carlo ──
    print("\nRunning S5 — Monte Carlo simulation...")
    s5_result = run_s5(s4_result, n_iterations=300)
    display_s5(s5_result)

    # ── S6: Protocol Package ──
    print("\nGenerating S6 — Protocol Package...")
    package = generate_protocol_package(
        question=compiled["question"],
        pipeline_result=pipeline_result,
        verify_summary=verify_summary,
        gate_result=gate_result,
        s4_result=s4_result,
        s5_result=s5_result,
        session_seed=session_seed,
    )

    # Collect per-stage outputs for structured folder
    stage_data = {}

    # S1: formulations
    if s1_result:
        stage_data["s1_formulations"] = {
            "parsed": [
                {
                    "model": p.model,
                    "raw": p.raw,
                    "claims": [asdict(c) for c in p.claims],
                }
                for p in s1_result.get("parsed", [])
            ],
            "snapshot": asdict(s1_result["snapshot"]) if hasattr(s1_result.get("snapshot"), '__dict__') else s1_result.get("snapshot"),
            "total_calls": s1_result.get("total_calls", 0),
        }

    # S2: synthesis (serialize SynthesizedClaim objects as dicts, not repr strings)
    if s2_result:
        stage_data["s2_synthesis"] = {
            "synthesized_claims": [
                asdict(c) if hasattr(c, '__dict__') else c
                for c in s2_result.get("synthesized_claims", [])
            ],
            "conflicts": [
                asdict(c) if hasattr(c, '__dict__') else c
                for c in s2_result.get("conflicts", [])
            ],
            "total_rounds": s2_result.get("total_rounds", 0),
            "early_stopped": s2_result.get("early_stopped", False),
            "snapshots": [
                asdict(s) if hasattr(s, '__dict__') else s
                for s in s2_result.get("snapshots", [])
            ],
        }

    # S3: convergence gate
    if s3_result:
        stage_data["s3_convergence"] = s3_result

    # VERIFY
    if verify_summary:
        stage_data["verify"] = verify_summary

    # Lab Gate
    if gate_result:
        stage_data["gate"] = asdict(gate_result) if hasattr(gate_result, '__dict__') else gate_result

    # S4: hypotheses
    if s4_result:
        stage_data["s4_hypotheses"] = asdict(s4_result) if hasattr(s4_result, '__dict__') else s4_result

    # S5: Monte Carlo
    if s5_result:
        stage_data["s5_monte_carlo"] = asdict(s5_result) if hasattr(s5_result, '__dict__') else s5_result

    # Save
    output_dir = Path(__file__).parent / "runs"
    paths = save_protocol(package, output_dir=str(output_dir), stage_data=stage_data)

    # Display summary
    dash.update_stage("S6 — COMPLETE")
    dash.finalize()
    summary = format_human_summary(package)
    print(f"\n{summary}")
    print(f"\nSaved to: {paths.get('dir', '')}/")
    n_files = len([k for k in paths if k != 'dir'])
    print(f"  {n_files} files ({', '.join(k for k in sorted(paths) if k != 'dir')})")
    print(f"\nTotal LLM calls: {total_calls}")

    # Optional: cross-run comparison against all existing runs
    if getattr(args, 'cross_run', False):
        from src.cross_run import find_runs, cross_match as xmatch
        print("\nRunning cross-run analysis...")
        all_runs = find_runs([str(output_dir)])
        if len(all_runs) >= 2:
            xresult = xmatch(all_runs)
            if xresult.matches:
                print(f"  {len(xresult.matches)} cross-run matches found:")
                for m in xresult.matches[:5]:
                    print(f"    [{m.classification}] cos={m.cosine} | {m.run_a} ↔ {m.run_b}")
            else:
                print("  No cross-run matches above threshold.")
        else:
            print("  Need 2+ runs for cross-run analysis.")

    return package


def _extract_final_claims(parsed) -> list:
    """Extract claims from parsed responses."""
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


def main():
    parser = argparse.ArgumentParser(
        description="IRIS Gate Evo — Multi-LLM Convergence Protocol"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="The research question to investigate",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Force a specific domain (pharmacology, bioelectric, etc.)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to use (default: all five)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only run the compiler, don't dispatch to models",
    )
    parser.add_argument(
        "--stage",
        default="full",
        choices=["s1", "s2", "s3", "full"],
        help="Stop after this stage (default: full pipeline)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run without API calls (offline heuristic mode)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save session to runs/ (default: True)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip API key preflight check",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable live dashboard (plain text output)",
    )
    parser.add_argument(
        "--cross-run",
        action="store_true",
        help="After pipeline, compare this session against all existing runs",
    )

    args = parser.parse_args()
    args.dashboard = not args.no_dashboard

    print("\nIRIS Gate Evo")
    print(f"Question: {args.question[:80]}{'...' if len(args.question) > 80 else ''}")

    # C0 — Compile
    compiled = compile(args.question, domain_override=args.domain)
    display_compiled(compiled)

    if args.compile_only:
        print("\n[--compile-only] Stopping after C0.")
        print("\nGenerated prompt:")
        print(compiled["prompt"])
        return

    # Pre-flight — test all API keys before burning budget
    if not args.offline and not args.skip_preflight:
        model_list = args.models.split(",") if args.models else None
        include_verify = (args.stage == "full")
        print()
        preflight = asyncio.run(run_preflight(
            models=model_list,
            include_verify=include_verify,
        ))
        print(format_preflight(preflight))

        if not preflight.all_passed:
            print("\nAborting — fix the failed keys and retry.")
            print("Use --skip-preflight to bypass this check.")
            sys.exit(1)

        print()

    # Run the full async pipeline
    asyncio.run(run_full_pipeline(compiled, args))


if __name__ == "__main__":
    main()
