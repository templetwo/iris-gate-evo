#!/usr/bin/env python3
"""
Cross-Run Convergence Detection — finds buried gold across IRIS runs.

Embeds claims from multiple runs, finds semantic matches across runs,
reclassifies TYPE based on cross-run evidence, and reports buried gold.

Usage:
    python cross_run.py runs/evo_*024747* runs/evo_*024750*
    python cross_run.py --all
    python cross_run.py --all --dirs runs/ ~/iris-evo-findings/runs/
    python cross_run.py --threshold 0.80
    python cross_run.py --output results/thc_cross/
"""

import argparse
import sys
from pathlib import Path

from src.cross_run.loader import load_run, find_runs
from src.cross_run.matcher import cross_match
from src.cross_run.report import save_report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Run Convergence Detection for IRIS Gate Evo"
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run directories to compare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all runs in --dirs directories",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["runs/"],
        help="Directories to scan for runs (default: runs/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for cross-matching (default: 0.75)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: results/cross_run/)",
    )

    args = parser.parse_args()

    # Load runs
    if args.all:
        print(f"Scanning: {', '.join(args.dirs)}")
        runs = find_runs(args.dirs)
    elif args.runs:
        runs = []
        for run_path in args.runs:
            p = Path(run_path)
            if p.is_dir() and (p / "s2_synthesis.json").exists():
                runs.append(load_run(p))
            else:
                # Try glob expansion
                for match in sorted(Path(".").glob(run_path)):
                    if match.is_dir() and (match / "s2_synthesis.json").exists():
                        runs.append(load_run(match))
    else:
        print("Error: provide run directories or use --all")
        parser.print_help()
        sys.exit(1)

    if len(runs) < 2:
        print(f"Need at least 2 runs, found {len(runs)}.")
        sys.exit(1)

    print(f"\nLoaded {len(runs)} runs:")
    for r in runs:
        status = "PASS" if r.s3_passed else "FAIL"
        print(f"  {r.session_id} [{r.domain}] S3={status} claims={len(r.claims)}")

    # Cross-match
    print(f"\nRunning cross-match (threshold={args.threshold})...")
    result = cross_match(runs, threshold=args.threshold)

    # Summary
    print(f"\nResults:")
    print(f"  Cross-matches: {len(result.matches)}")
    for m in result.matches:
        print(f"    [{m.classification}] cos={m.cosine} | {m.run_a} ↔ {m.run_b}")
        print(f"      A: {m.claim_a.get('statement', '')[:80]}")
        print(f"      B: {m.claim_b.get('statement', '')[:80]}")

    if result.structural_patterns:
        print(f"\n  Structural patterns:")
        for pattern, run_ids in result.structural_patterns.items():
            print(f"    {pattern}: {', '.join(run_ids)}")

    # Save report
    output_dir = Path(args.output) if args.output else Path("results/cross_run")
    paths = save_report(result, output_dir)
    print(f"\nSaved to: {paths['dir']}/")
    print(f"  {paths['json']}")
    print(f"  {paths['markdown']}")


if __name__ == "__main__":
    main()
