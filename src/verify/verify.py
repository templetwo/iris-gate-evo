"""
VERIFY — Perplexity integration for TYPE 2 claim verification.

TYPE 2 claims are grounded but unverified in context. This module
sends them to Perplexity (or any search-grounded LLM) and returns
a verdict:

  PROMOTED  → Evidence confirms. Reclassify as TYPE 1.
  HELD      → Insufficient evidence. Remains TYPE 2.
  NOVEL     → No prior literature. Potentially original contribution.
  CONTRADICTED → Evidence actively disputes this claim.

Each verdict includes an evidence summary and confidence score.
The system uses 5-15 API calls depending on how many TYPE 2 claims
survived S3. This is the last checkpoint before Lab Gate.
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import litellm

from src.parser import Claim


# Verdicts — what VERIFY can assign to a TYPE 2 claim
PROMOTED = "PROMOTED"       # → TYPE 1
HELD = "HELD"               # → stays TYPE 2
NOVEL = "NOVEL"             # → no literature found, potential discovery
CONTRADICTED = "CONTRADICTED"  # → flagged for human review


@dataclass
class VerifyResult:
    """Verification result for a single TYPE 2 claim."""
    claim: str
    verdict: str                  # PROMOTED, HELD, NOVEL, CONTRADICTED
    confidence: float             # 0.0 - 1.0 from the verifier
    evidence_summary: str         # One paragraph from Perplexity
    citations: list[str] = field(default_factory=list)  # URLs or references
    original_type: int = 2
    new_type: int = 2             # Updated based on verdict

    def __post_init__(self):
        if self.verdict == PROMOTED:
            self.new_type = 1
        elif self.verdict == CONTRADICTED:
            self.new_type = 2  # Stays 2 but flagged
        elif self.verdict == NOVEL:
            self.new_type = 2  # Stays 2 — novel doesn't mean validated
        else:
            self.new_type = 2


@dataclass
class VerifyStageResult:
    """Complete VERIFY stage output."""
    results: list[VerifyResult]
    n_type2_input: int
    n_promoted: int = 0
    n_held: int = 0
    n_novel: int = 0
    n_contradicted: int = 0
    total_calls: int = 0
    latency_s: float = 0.0


# ── Verification prompt ──

VERIFY_PROMPT_TEMPLATE = """\
You are a scientific fact-checker with access to current literature.

CLAIM TO VERIFY:
"{claim}"

CONTEXT: This claim emerged from a multi-model scientific convergence protocol. \
It was classified as TYPE 2 (grounded but unverified in this specific context). \
Your job is to check it against current evidence.

Respond in EXACTLY this format:

VERDICT: [PROMOTED|HELD|NOVEL|CONTRADICTED]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [One paragraph summarizing the evidence for or against this claim. \
Cite specific studies, authors, or datasets where possible.]
CITATIONS: [Comma-separated list of key references or URLs]

Decision criteria:
- PROMOTED: Multiple peer-reviewed sources support this claim in this context
- HELD: Some evidence exists but is insufficient or tangential
- NOVEL: No relevant literature found — this may be an original finding
- CONTRADICTED: Evidence actively disputes or disproves this claim

Be rigorous. PROMOTED requires strong evidence, not just plausibility."""


def extract_type2_claims(claims: list[dict]) -> list[dict]:
    """Extract TYPE 2 claims from the final claims list.

    Args:
        claims: List of claim dicts from pipeline output (final_claims).

    Returns:
        List of claim dicts where type == 2.
    """
    return [c for c in claims if c.get("type") == 2]


def build_verify_prompt(claim_statement: str) -> str:
    """Build the verification prompt for a single claim."""
    return VERIFY_PROMPT_TEMPLATE.format(claim=claim_statement)


def parse_verify_response(claim_statement: str, response_text: str) -> VerifyResult:
    """Parse a verification response into a VerifyResult.

    Handles messy formatting gracefully — models vary.
    """
    import re

    # Extract verdict
    verdict_match = re.search(
        r'VERDICT\s*:\s*(PROMOTED|HELD|NOVEL|CONTRADICTED)',
        response_text, re.IGNORECASE
    )
    verdict = verdict_match.group(1).upper() if verdict_match else HELD

    # Extract confidence
    conf_match = re.search(
        r'CONFIDENCE\s*:\s*([\d.]+)',
        response_text, re.IGNORECASE
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    confidence = max(0.0, min(1.0, confidence))

    # Extract evidence summary
    evidence_match = re.search(
        r'EVIDENCE\s*:\s*(.+?)(?=CITATIONS|$)',
        response_text, re.IGNORECASE | re.DOTALL
    )
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    evidence = re.sub(r'\s+', ' ', evidence).strip()

    # Extract citations
    citations_match = re.search(
        r'CITATIONS\s*:\s*(.+?)$',
        response_text, re.IGNORECASE | re.DOTALL
    )
    citations = []
    if citations_match:
        raw = citations_match.group(1).strip()
        # Split on commas or newlines
        parts = re.split(r'[,\n]', raw)
        citations = [p.strip() for p in parts if p.strip()]

    return VerifyResult(
        claim=claim_statement,
        verdict=verdict,
        confidence=confidence,
        evidence_summary=evidence,
        citations=citations,
    )


async def _verify_single_claim(
    claim_statement: str,
    model_id: str = "perplexity/sonar-pro",
    api_base: Optional[str] = None,
) -> VerifyResult:
    """Verify a single TYPE 2 claim against current literature.

    Uses Perplexity (or compatible) as the search-grounded verifier.
    Never raises — returns HELD verdict on error.
    """
    prompt = build_verify_prompt(claim_statement)

    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Low temp for factual checking
        "max_tokens": 600,
    }
    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = await litellm.acompletion(**kwargs)
        content = response.choices[0].message.content
        return parse_verify_response(claim_statement, content)

    except Exception as e:
        # Verification failure is not fatal — claim stays TYPE 2
        return VerifyResult(
            claim=claim_statement,
            verdict=HELD,
            confidence=0.0,
            evidence_summary=f"Verification failed: {type(e).__name__}: {e}",
            citations=[],
        )


async def run_verify(
    pipeline_result: dict,
    model_id: str = "perplexity/sonar-pro",
    api_base: Optional[str] = None,
    max_claims: int = 10,
) -> VerifyStageResult:
    """Run the full VERIFY stage on TYPE 2 claims from the pipeline.

    Args:
        pipeline_result: Output from run_pipeline() — needs 'final_claims'.
        model_id: LiteLLM model string for the verifier.
        api_base: Optional base URL override.
        max_claims: Max TYPE 2 claims to verify (budget protection).

    Returns:
        VerifyStageResult with all verdicts and summary counts.
    """
    final_claims = pipeline_result.get("final_claims", [])
    type2_claims = extract_type2_claims(final_claims)

    # Budget protection — don't verify more than max_claims
    type2_claims = type2_claims[:max_claims]

    if not type2_claims:
        return VerifyStageResult(
            results=[],
            n_type2_input=0,
            total_calls=0,
        )

    t_start = time.monotonic()

    # Verify all TYPE 2 claims in parallel
    tasks = [
        _verify_single_claim(c["statement"], model_id=model_id, api_base=api_base)
        for c in type2_claims
    ]
    results = await asyncio.gather(*tasks)

    t_elapsed = time.monotonic() - t_start

    # Count verdicts
    n_promoted = sum(1 for r in results if r.verdict == PROMOTED)
    n_held = sum(1 for r in results if r.verdict == HELD)
    n_novel = sum(1 for r in results if r.verdict == NOVEL)
    n_contradicted = sum(1 for r in results if r.verdict == CONTRADICTED)

    return VerifyStageResult(
        results=list(results),
        n_type2_input=len(type2_claims),
        n_promoted=n_promoted,
        n_held=n_held,
        n_novel=n_novel,
        n_contradicted=n_contradicted,
        total_calls=len(type2_claims),
        latency_s=round(t_elapsed, 2),
    )


def run_verify_sync(
    pipeline_result: dict,
    model_id: str = "perplexity/sonar-pro",
    api_base: Optional[str] = None,
) -> VerifyStageResult:
    """Synchronous wrapper for run_verify."""
    return asyncio.run(run_verify(pipeline_result, model_id=model_id, api_base=api_base))


def apply_verdicts(pipeline_result: dict, verify_result: VerifyStageResult) -> dict:
    """Apply verification verdicts back to the pipeline result.

    Updates TYPE classifications based on VERIFY outcomes:
    - PROMOTED claims get reclassified from TYPE 2 → TYPE 1
    - CONTRADICTED claims get flagged
    - NOVEL claims get tagged for special attention

    Returns updated pipeline_result with 'verified_claims' added.
    """
    verified_claims = []
    verdict_lookup = {r.claim: r for r in verify_result.results}

    for claim in pipeline_result.get("final_claims", []):
        updated = dict(claim)

        if claim.get("type") == 2 and claim["statement"] in verdict_lookup:
            result = verdict_lookup[claim["statement"]]
            updated["verify_verdict"] = result.verdict
            updated["verify_confidence"] = result.confidence
            updated["verify_evidence"] = result.evidence_summary
            updated["verify_citations"] = result.citations

            if result.verdict == PROMOTED:
                updated["type"] = 1  # Reclassify
                updated["type_note"] = "Promoted from TYPE 2 by VERIFY"
            elif result.verdict == CONTRADICTED:
                updated["contradicted"] = True
                updated["type_note"] = "CONTRADICTED by VERIFY — review required"
            elif result.verdict == NOVEL:
                updated["novel"] = True
                updated["type_note"] = "NOVEL — no prior literature found"

        verified_claims.append(updated)

    result = dict(pipeline_result)
    result["verified_claims"] = verified_claims
    result["verify_summary"] = {
        "n_type2_input": verify_result.n_type2_input,
        "n_promoted": verify_result.n_promoted,
        "n_held": verify_result.n_held,
        "n_novel": verify_result.n_novel,
        "n_contradicted": verify_result.n_contradicted,
        "total_calls": verify_result.total_calls,
    }

    return result
