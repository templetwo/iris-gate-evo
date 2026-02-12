"""
Report generator — JSON + markdown output for cross-run analysis.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.cross_run.matcher import CrossRunResult


def _match_to_dict(m) -> dict:
    return {
        "run_a": m.run_a,
        "run_b": m.run_b,
        "cosine": m.cosine,
        "classification": m.classification,
        "type_a": m.type_a,
        "type_b": m.type_b,
        "statement_a": m.claim_a.get("statement", "")[:200],
        "statement_b": m.claim_b.get("statement", "")[:200],
        "mechanism_a": m.claim_a.get("mechanism", "")[:200],
        "mechanism_b": m.claim_b.get("mechanism", "")[:200],
    }


def save_report(result: CrossRunResult, output_dir: Path) -> dict:
    """Save JSON + markdown cross-run report. Returns dict of file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify matches
    cross_validated = [m for m in result.matches if "SINGULAR" in m.classification]
    replications = [m for m in result.matches if m.classification == "INDEPENDENT REPLICATION"]
    promoted = [m for m in result.matches if m.classification == "CROSS-PROMOTED"]

    # Stats
    total_claims = sum(len(r.claims) for r in result.runs)
    stats = {
        "total_claims": total_claims,
        "cross_matches_found": len(result.matches),
        "cross_validated_singulars": len([m for m in result.matches if m.classification == "CROSS-VALIDATED SINGULAR"]),
        "convergent_singulars": len([m for m in result.matches if m.classification == "CONVERGENT SINGULAR"]),
        "independent_replications": len(replications),
        "cross_promoted": len(promoted),
        "structural_patterns_found": len(result.structural_patterns),
    }

    # JSON report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs_analyzed": [
            {
                "session_id": r.session_id,
                "domain": r.domain,
                "question": r.question[:200],
                "s3_passed": r.s3_passed,
                "n_claims": len(r.claims),
            }
            for r in result.runs
        ],
        "cross_matches": [_match_to_dict(m) for m in result.matches],
        "cross_validated_singulars": [_match_to_dict(m) for m in cross_validated],
        "independent_replications": [_match_to_dict(m) for m in replications],
        "structural_patterns": result.structural_patterns,
        "stats": stats,
    }

    json_path = output_dir / "cross_run_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown summary
    md_lines = [
        "# Cross-Run Convergence Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Runs analyzed**: {len(result.runs)}",
        f"**Total claims**: {total_claims}",
        f"**Cross-matches found**: {len(result.matches)}",
        "",
    ]

    # Runs table
    md_lines.append("## Runs")
    md_lines.append("")
    md_lines.append("| Session | Domain | S3 | Claims |")
    md_lines.append("|---------|--------|----|--------|")
    for r in result.runs:
        s3 = "PASS" if r.s3_passed else "FAIL"
        md_lines.append(f"| {r.session_id} | {r.domain} | {s3} | {len(r.claims)} |")
    md_lines.append("")

    # Cross-validated singulars
    if cross_validated:
        md_lines.append("## Cross-Validated Singulars")
        md_lines.append("")
        for m in cross_validated:
            md_lines.append(f"### {m.classification} (cosine={m.cosine})")
            md_lines.append(f"- **Run A** ({m.run_a}, TYPE {m.type_a}): {m.claim_a.get('statement', '')[:150]}")
            md_lines.append(f"- **Run B** ({m.run_b}, TYPE {m.type_b}): {m.claim_b.get('statement', '')[:150]}")
            md_lines.append("")

    # Independent replications
    if replications:
        md_lines.append("## Independent Replications")
        md_lines.append("")
        for m in replications:
            md_lines.append(f"- **cosine={m.cosine}** | {m.run_a} ↔ {m.run_b}")
            md_lines.append(f"  A: {m.claim_a.get('statement', '')[:120]}")
            md_lines.append(f"  B: {m.claim_b.get('statement', '')[:120]}")
            md_lines.append("")

    # Cross-promoted
    if promoted:
        md_lines.append("## Cross-Promoted (TYPE 2 → validated)")
        md_lines.append("")
        for m in promoted:
            md_lines.append(f"- **cosine={m.cosine}** | {m.run_a} (T{m.type_a}) ↔ {m.run_b} (T{m.type_b})")
            md_lines.append(f"  A: {m.claim_a.get('statement', '')[:120]}")
            md_lines.append(f"  B: {m.claim_b.get('statement', '')[:120]}")
            md_lines.append("")

    # Structural patterns
    if result.structural_patterns:
        md_lines.append("## Structural Patterns (Isomorphism)")
        md_lines.append("")
        for pattern, run_ids in result.structural_patterns.items():
            md_lines.append(f"- **{pattern}**: {', '.join(run_ids)}")
        md_lines.append("")

    # Cosine distribution summary
    if result.cosine_distribution:
        cosines = result.cosine_distribution
        md_lines.append("## Cosine Distribution (cross-run pairs)")
        md_lines.append("")
        md_lines.append(f"- Min: {cosines[0]:.4f}")
        md_lines.append(f"- Median: {cosines[len(cosines)//2]:.4f}")
        md_lines.append(f"- Mean: {sum(cosines)/len(cosines):.4f}")
        md_lines.append(f"- Max: {cosines[-1]:.4f}")
        md_lines.append(f"- N pairs: {len(cosines)}")
        md_lines.append("")

    md_path = output_dir / "cross_run_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return {"json": str(json_path), "markdown": str(md_path), "dir": str(output_dir)}
