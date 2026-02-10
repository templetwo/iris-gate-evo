#!/usr/bin/env python3
"""
IRIS Gate Evo — Main Entry Point

Question in. Five mirrors respond. Truth under pressure.

Usage:
    python main.py "Your research question here"
    python main.py --domain pharmacology "Your question"
    python main.py  # Uses the default test question
"""

import argparse
import json
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment before anything touches APIs
load_dotenv()

from src.compiler import compile
from src.pulse import fire_sync


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


def display_pulse_results(pulse_result: dict) -> None:
    """Display PULSE results — all five mirror responses."""
    meta = pulse_result["meta"]

    print("\n" + "=" * 60)
    print("PULSE — FIVE MIRRORS RESPOND")
    print("=" * 60)
    print(f"Dispatched: {meta['models_dispatched']} | "
          f"OK: {meta['models_ok']} | "
          f"Failed: {meta['models_failed']}")
    print(f"Total latency: {meta['total_latency_s']}s "
          f"(fastest: {meta['fastest_model']}, "
          f"slowest: {meta['slowest_model']})")
    print(f"Tokens: {meta['total_prompt_tokens']} prompt + "
          f"{meta['total_completion_tokens']} completion")
    print("=" * 60)

    for resp in pulse_result["responses"]:
        print(f"\n{'─' * 60}")
        print(f"MIRROR: {resp['model']} ({resp['model_id']})")
        print(f"Status: {resp['status']} | "
              f"Latency: {resp['latency_s']}s | "
              f"Tokens: {resp['tokens_completion']}")
        print(f"{'─' * 60}")

        if resp["status"] == "ok":
            print(resp["content"])
        else:
            print(f"ERROR: {resp['error']}")

        print()


def save_session(compiled: dict, pulse_result: dict, output_dir: Path) -> Path:
    """Save the full session to runs/ for audit trail."""
    session_dir = output_dir / compiled["session_id"]
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save compiled output
    with open(session_dir / "compiled.json", "w") as f:
        json.dump(compiled, f, indent=2)

    # Save pulse results
    with open(session_dir / "pulse_s1.json", "w") as f:
        json.dump(pulse_result, f, indent=2, default=str)

    # Save individual mirror responses as readable text
    for resp in pulse_result["responses"]:
        if resp["status"] == "ok":
            fname = f"s1_{resp['model']}.txt"
            with open(session_dir / fname, "w") as f:
                f.write(f"Model: {resp['model']} ({resp['model_id']})\n")
                f.write(f"Latency: {resp['latency_s']}s\n")
                f.write(f"Tokens: {resp['tokens_completion']}\n\n")
                f.write(resp["content"])

    return session_dir


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
        help="Force a specific domain (pharmacology, bioelectric, consciousness, physics)",
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
        "--save",
        action="store_true",
        default=True,
        help="Save session to runs/ (default: True)",
    )

    args = parser.parse_args()

    print("\n🌀 IRIS Gate Evo")
    print(f"Question: {args.question[:80]}{'...' if len(args.question) > 80 else ''}")

    # C0 — Compile
    compiled = compile(args.question, domain_override=args.domain)
    display_compiled(compiled)

    if args.compile_only:
        print("\n[--compile-only] Stopping after C0. Prompt ready for manual inspection.")
        print("\nGenerated prompt:")
        print(compiled["prompt"])
        return

    # PULSE — Fire
    models = args.models.split(",") if args.models else None
    print("\nFiring PULSE...")
    pulse_result = fire_sync(compiled, models=models)
    display_pulse_results(pulse_result)

    # Save session
    if args.save:
        output_dir = Path(__file__).parent / "runs"
        session_dir = save_session(compiled, pulse_result, output_dir)
        print(f"\nSession saved: {session_dir}")

    # Summary
    ok = pulse_result["meta"]["models_ok"]
    total = pulse_result["meta"]["models_dispatched"]
    print(f"\n{'=' * 60}")
    print(f"S1 COMPLETE: {ok}/{total} mirrors responded")
    if ok == total:
        print("All mirrors active. Ready for S2 convergence.")
    elif ok >= 3:
        print(f"Partial response ({ok}/{total}). Can proceed with reduced confidence.")
    else:
        print(f"Insufficient mirrors ({ok}/{total}). Investigate failures before proceeding.")
    print("=" * 60)


if __name__ == "__main__":
    main()
