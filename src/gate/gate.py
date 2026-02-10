"""
Lab Gate — Falsifiability, Feasibility, Novelty.

The final checkpoint before S4. Every surviving claim must pass
three criteria or be routed to human review. This is where bad
hypotheses die before they waste 20 calls in simulation.

Three criteria, all required for PASS:
  1. FALSIFIABLE — Does it have a concrete disproof condition?
  2. FEASIBLE    — Can it be tested with standard methods?
  3. NOVEL       — Does it add to knowledge, not restate textbooks?

One LLM call per gate invocation. The gate model sees all verified
claims at once and evaluates them as a batch.

PASS → S4 (hypothesis operationalization)
FAIL → Human review with failure report
"""

import asyncio
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import litellm

from src.models import MODELS


@dataclass
class ClaimGateResult:
    """Gate result for a single claim."""
    statement: str
    falsifiable: bool
    falsifiable_reason: str
    feasible: bool
    feasible_reason: str
    novel: bool
    novel_reason: str
    passed: bool = False

    def __post_init__(self):
        self.passed = self.falsifiable and self.feasible and self.novel


@dataclass
class GateResult:
    """Complete Lab Gate output."""
    passed: bool                     # Overall: all claims passed?
    claims: list[ClaimGateResult]    # Per-claim results
    n_passed: int = 0
    n_failed: int = 0
    total_calls: int = 1             # Always 1 call (batch evaluation)
    latency_s: float = 0.0
    recommendation: str = ""


# ── Gate prompt ──

GATE_PROMPT_TEMPLATE = """\
You are a scientific review panel evaluating hypotheses for lab testing.

RESEARCH QUESTION:
"{question}"

CLAIMS TO EVALUATE:
{claims_block}

For EACH claim above, evaluate three criteria:

1. **FALSIFIABLE** — Does this claim have a specific, concrete condition that would disprove it? \
"More research needed" is not falsifiable. "If VDAC1 knockout cells show identical CBD toxicity, this is disproved" IS falsifiable.

2. **FEASIBLE** — Can this be tested with standard laboratory equipment and methods within a \
reasonable budget (<$50K)? Requires standard cell lines, available reagents, established protocols. \
"Build a particle accelerator" is not feasible. "JC-1 staining to measure mitochondrial membrane potential" IS feasible.

3. **NOVEL** — Does this add to existing knowledge? A claim that merely restates established \
textbook facts is not novel. A claim that proposes a new mechanism, quantitative relationship, \
or unexpected interaction IS novel.

Respond in EXACTLY this format for each claim (numbered to match):

CLAIM 1:
FALSIFIABLE: [YES|NO] — [one sentence reason]
FEASIBLE: [YES|NO] — [one sentence reason]
NOVEL: [YES|NO] — [one sentence reason]

CLAIM 2:
FALSIFIABLE: [YES|NO] — [one sentence reason]
FEASIBLE: [YES|NO] — [one sentence reason]
NOVEL: [YES|NO] — [one sentence reason]

(continue for all claims)

Be strict. A weak hypothesis that passes gate wastes experimental resources."""


def build_claims_block(claims: list[dict]) -> str:
    """Format claims into a numbered block for the gate prompt."""
    lines = []
    for i, c in enumerate(claims, 1):
        stmt = c.get("statement", "")
        ctype = c.get("type", "?")
        conf = c.get("confidence", "?")
        mechanism = c.get("mechanism", "")
        falsifiable = c.get("falsifiable_by", "")

        block = f"CLAIM {i}: {stmt}\n  TYPE: {ctype} | CONFIDENCE: {conf}"
        if mechanism:
            block += f"\n  MECHANISM: {mechanism}"
        if falsifiable:
            block += f"\n  FALSIFIABLE BY: {falsifiable}"
        lines.append(block)

    return "\n\n".join(lines)


def build_gate_prompt(question: str, claims: list[dict]) -> str:
    """Build the complete Lab Gate prompt."""
    claims_block = build_claims_block(claims)
    return GATE_PROMPT_TEMPLATE.format(
        question=question,
        claims_block=claims_block,
    )


def parse_gate_response(
    response_text: str,
    claims: list[dict],
) -> list[ClaimGateResult]:
    """Parse gate model response into per-claim results.

    Handles varied formatting from different models.
    """
    results = []

    for i, claim in enumerate(claims, 1):
        # Find this claim's block in the response
        # Look for "CLAIM N:" or just parse sequentially
        pattern = re.compile(
            rf'CLAIM\s*{i}\s*:.*?'
            rf'FALSIFIABLE\s*:\s*(YES|NO)\s*[-—]?\s*(.*?)\n'
            rf'.*?FEASIBLE\s*:\s*(YES|NO)\s*[-—]?\s*(.*?)\n'
            rf'.*?NOVEL\s*:\s*(YES|NO)\s*[-—]?\s*(.*?)(?=\nCLAIM\s*\d|\Z)',
            re.IGNORECASE | re.DOTALL,
        )

        match = pattern.search(response_text)

        if match:
            falsifiable = match.group(1).upper() == "YES"
            falsifiable_reason = match.group(2).strip()
            feasible = match.group(3).upper() == "YES"
            feasible_reason = match.group(4).strip()
            novel = match.group(5).upper() == "YES"
            novel_reason = match.group(6).strip()
        else:
            # If parsing fails, try a looser pattern for this claim
            result = _parse_claim_loose(response_text, i)
            if result:
                results.append(ClaimGateResult(
                    statement=claim["statement"],
                    **result,
                ))
                continue

            # Ultimate fallback: conservative — FAIL
            falsifiable = False
            falsifiable_reason = "Could not parse gate response"
            feasible = False
            feasible_reason = "Could not parse gate response"
            novel = False
            novel_reason = "Could not parse gate response"

        results.append(ClaimGateResult(
            statement=claim["statement"],
            falsifiable=falsifiable,
            falsifiable_reason=falsifiable_reason,
            feasible=feasible,
            feasible_reason=feasible_reason,
            novel=novel,
            novel_reason=novel_reason,
        ))

    return results


def _parse_claim_loose(text: str, claim_num: int) -> Optional[dict]:
    """Loose parsing fallback for a single claim's gate evaluation."""
    # Find the section for this claim
    start_pattern = re.compile(rf'CLAIM\s*{claim_num}\b', re.IGNORECASE)
    start_match = start_pattern.search(text)
    if not start_match:
        return None

    # Find the end (next CLAIM or end of text)
    end_pattern = re.compile(rf'CLAIM\s*{claim_num + 1}\b', re.IGNORECASE)
    end_match = end_pattern.search(text, start_match.end())
    section = text[start_match.start():end_match.start() if end_match else len(text)]

    def _extract_bool(field_name: str) -> tuple[bool, str]:
        m = re.search(
            rf'{field_name}\s*:\s*(YES|NO)\s*[-—]?\s*(.+?)(?=\n|$)',
            section, re.IGNORECASE,
        )
        if m:
            return m.group(1).upper() == "YES", m.group(2).strip()
        return False, "Not found in response"

    f_val, f_reason = _extract_bool("FALSIFIABLE")
    fe_val, fe_reason = _extract_bool("FEASIBLE")
    n_val, n_reason = _extract_bool("NOVEL")

    return {
        "falsifiable": f_val,
        "falsifiable_reason": f_reason,
        "feasible": fe_val,
        "feasible_reason": fe_reason,
        "novel": n_val,
        "novel_reason": n_reason,
    }


def evaluate_claims_offline(claims: list[dict]) -> list[ClaimGateResult]:
    """Evaluate claims using heuristic rules — no LLM call.

    Used for testing and as a fallback when no API is available.
    Checks:
    - falsifiable_by field is non-empty and specific
    - mechanism field exists (proxy for feasibility)
    - type < 3 (speculation is not novel, it's ungrounded)
    """
    results = []
    for claim in claims:
        stmt = claim.get("statement", "")
        falsifiable_by = claim.get("falsifiable_by", "")
        mechanism = claim.get("mechanism", "")
        ctype = claim.get("type", 3)

        # Falsifiable: needs a non-trivial falsifiable_by field
        has_falsifiable = (
            len(falsifiable_by) > 20
            and any(word in falsifiable_by.lower() for word in [
                "if", "would", "should", "disprove", "knockout",
                "inhibit", "block", "abolish", "reduce", "increase",
            ])
        )

        # Feasible: has mechanism (proxy) and isn't purely speculative
        has_feasible = len(mechanism) > 10 and ctype <= 2

        # Novel: not just restating priors (heuristic: TYPE 0 are established,
        # TYPE 1 is literature-backed — TYPE 2 claims that survived S3 are likely novel)
        is_novel = ctype >= 1 and len(stmt) > 30

        results.append(ClaimGateResult(
            statement=stmt,
            falsifiable=has_falsifiable,
            falsifiable_reason="Has specific disproof condition" if has_falsifiable else "Missing concrete falsification criterion",
            feasible=has_feasible,
            feasible_reason="Mechanism described, testable" if has_feasible else "Insufficient mechanism detail for lab protocol",
            novel=is_novel,
            novel_reason="Extends beyond established knowledge" if is_novel else "Restates known information",
        ))

    return results


async def run_gate(
    pipeline_result: dict,
    model_name: str = "claude",
    use_offline: bool = False,
) -> GateResult:
    """Run the Lab Gate on verified claims.

    Args:
        pipeline_result: Output from apply_verdicts() or run_pipeline().
            Needs 'verified_claims' or 'final_claims' and 'question'.
        model_name: Which model to use for gate evaluation.
        use_offline: Use heuristic evaluation instead of LLM call.

    Returns:
        GateResult with per-claim evaluations and overall PASS/FAIL.
    """
    question = pipeline_result.get("question", "")

    # Use verified_claims if available, otherwise final_claims
    claims = pipeline_result.get("verified_claims",
             pipeline_result.get("final_claims", []))

    # Filter: only evaluate claims that weren't contradicted
    gate_claims = [
        c for c in claims
        if not c.get("contradicted", False)
        and c.get("type", 3) <= 2  # Don't gate TYPE 3 speculation
    ]

    if not gate_claims:
        return GateResult(
            passed=False,
            claims=[],
            recommendation="No claims survived for gate evaluation.",
        )

    t_start = time.monotonic()

    if use_offline:
        claim_results = evaluate_claims_offline(gate_claims)
        total_calls = 0
    else:
        # Build gate prompt and dispatch
        prompt = build_gate_prompt(question, gate_claims)

        model = MODELS.get(model_name, MODELS["claude"])
        prefix = {
            "anthropic": "anthropic/",
            "openai": "openai/",
            "xai": "openai/",
            "google": "gemini/",
            "deepseek": "openai/",
        }.get(model["provider"], "openai/")

        kwargs = {
            "model": f"{prefix}{model['id']}",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1500,
        }

        if model["provider"] in ("xai", "deepseek"):
            kwargs["api_base"] = model["base_url"]

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            claim_results = parse_gate_response(content, gate_claims)
        except Exception as e:
            # Gate failure → conservative: offline evaluation
            claim_results = evaluate_claims_offline(gate_claims)

        total_calls = 1

    t_elapsed = time.monotonic() - t_start

    n_passed = sum(1 for r in claim_results if r.passed)
    n_failed = len(claim_results) - n_passed

    # Overall gate passes only if ALL claims pass
    overall_pass = n_failed == 0 and n_passed > 0

    recommendation = ""
    if overall_pass:
        recommendation = (
            f"Lab Gate PASSED. {n_passed} claims cleared all three criteria. "
            f"Proceeding to S4 hypothesis operationalization."
        )
    else:
        failed_claims = [r for r in claim_results if not r.passed]
        reasons = []
        for r in failed_claims:
            if not r.falsifiable:
                reasons.append(f"'{r.statement[:50]}...' lacks falsifiability")
            if not r.feasible:
                reasons.append(f"'{r.statement[:50]}...' not feasible")
            if not r.novel:
                reasons.append(f"'{r.statement[:50]}...' lacks novelty")

        recommendation = (
            f"Lab Gate FAILED. {n_failed}/{len(claim_results)} claims failed. "
            f"Route to human review. Issues: {'; '.join(reasons[:5])}"
        )

    return GateResult(
        passed=overall_pass,
        claims=claim_results,
        n_passed=n_passed,
        n_failed=n_failed,
        total_calls=total_calls,
        latency_s=round(t_elapsed, 2),
        recommendation=recommendation,
    )


def run_gate_sync(
    pipeline_result: dict,
    model_name: str = "claude",
    use_offline: bool = False,
) -> GateResult:
    """Synchronous wrapper for run_gate."""
    return asyncio.run(run_gate(pipeline_result, model_name, use_offline))
