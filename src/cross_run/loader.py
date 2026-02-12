"""
Claim loader — parses s2_synthesis.json files from run directories.

Handles both the legacy repr() string format and future dict format.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunData:
    """All relevant data from a single IRIS run."""
    session_id: str
    run_dir: Path
    question: str
    domain: str
    s3_passed: bool
    claims: list[dict]


def _parse_claim_repr(repr_str: str) -> dict:
    """Parse a SynthesizedClaim repr() string into a dict.

    These look like:
        SynthesizedClaim(statement='...', mechanism='...', falsifiable_by='...',
        type=1, confidence=0.743, overlap_count=3, models=['gemini', 'grok', 'mistral'],
        tuples=frozenset({...}), chunk_text='...')

    Edge cases: nested quotes, Unicode (Greek letters, arrows), markdown bold/italic.
    """
    result = {}

    # Extract string fields using non-greedy match between field= and next field=
    # The repr format is: field='value', next_field=...
    # We need to handle escaped quotes and nested content carefully.

    # String fields: statement, mechanism, falsifiable_by, chunk_text
    for field in ("statement", "mechanism", "falsifiable_by", "chunk_text"):
        # Match field='...' allowing for escaped quotes and any content
        # Use a lookahead for the next known field or end of string
        pattern = rf"{field}='(.*?)(?:',\s*(?:statement|mechanism|falsifiable_by|type|confidence|overlap_count|models|tuples|chunk_text)=|'\)$)"
        match = re.search(pattern, repr_str, re.DOTALL)
        if match:
            result[field] = match.group(1)
        else:
            # Fallback: try double quotes
            pattern = rf'{field}="(.*?)"(?:,\s*(?:statement|mechanism|falsifiable_by|type|confidence|overlap_count|models|tuples|chunk_text)=|\)$)'
            match = re.search(pattern, repr_str, re.DOTALL)
            result[field] = match.group(1) if match else ""

    # Numeric fields
    type_match = re.search(r'\btype=(\d+)', repr_str)
    result["type"] = int(type_match.group(1)) if type_match else 3

    conf_match = re.search(r'confidence=([\d.]+)', repr_str)
    result["confidence"] = float(conf_match.group(1)) if conf_match else 0.0

    overlap_match = re.search(r'overlap_count=(\d+)', repr_str)
    result["overlap_count"] = int(overlap_match.group(1)) if overlap_match else 1

    # Models list
    models_match = re.search(r"models=\[(.*?)\]", repr_str)
    if models_match:
        raw = models_match.group(1)
        result["models"] = [m.strip().strip("'\"") for m in raw.split(",") if m.strip()]
    else:
        result["models"] = []

    return result


def _parse_claim(raw) -> dict:
    """Parse a claim from s2_synthesis.json — handles both dict and repr string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.startswith("SynthesizedClaim("):
        return _parse_claim_repr(raw)
    if isinstance(raw, str):
        # Try parsing as JSON dict string
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {"statement": str(raw), "mechanism": "", "falsifiable_by": "",
            "type": 3, "confidence": 0.0, "overlap_count": 1, "models": []}


def load_run(run_dir: Path) -> RunData:
    """Load a single run from its directory."""
    run_dir = Path(run_dir)
    session_id = run_dir.name

    # Load claims from s2_synthesis.json
    s2_path = run_dir / "s2_synthesis.json"
    claims = []
    if s2_path.exists():
        with open(s2_path) as f:
            s2_data = json.load(f)
        for raw_claim in s2_data.get("synthesized_claims", []):
            claims.append(_parse_claim(raw_claim))

    # Load metadata from full_package.json or s3_convergence.json
    question = ""
    domain = ""
    s3_passed = False

    full_path = run_dir / "full_package.json"
    if full_path.exists():
        with open(full_path) as f:
            pkg = json.load(f)
        question = pkg.get("question", "")
        outcome = pkg.get("outcome", "")
        s3_passed = outcome != "S3_FAILED" and pkg.get("convergence_report", {}).get("s3_gate", {}).get("passed", False)

    s3_path = run_dir / "s3_convergence.json"
    if s3_path.exists():
        with open(s3_path) as f:
            s3_data = json.load(f)
        s3_passed = s3_data.get("passed", False)

    # Extract domain from session_id (e.g. evo_20260211_024747_pharmacology)
    parts = session_id.split("_")
    if len(parts) >= 4:
        domain = "_".join(parts[3:])  # handles "pharmacology+bioelectric"

    return RunData(
        session_id=session_id,
        run_dir=run_dir,
        question=question,
        domain=domain,
        s3_passed=s3_passed,
        claims=claims,
    )


def find_runs(directories: list[str]) -> list[RunData]:
    """Scan directories for run folders containing s2_synthesis.json."""
    runs = []
    for dir_str in directories:
        base = Path(dir_str)
        if not base.exists():
            continue
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and (sub / "s2_synthesis.json").exists():
                try:
                    runs.append(load_run(sub))
                except Exception as e:
                    print(f"  Warning: skipping {sub.name}: {e}")
    # Deduplicate by session_id (same run may appear in multiple directories)
    seen = set()
    unique = []
    for r in runs:
        if r.session_id not in seen:
            seen.add(r.session_id)
            unique.append(r)
    # Sort by session_id (which includes timestamp)
    unique.sort(key=lambda r: r.session_id)
    return unique
