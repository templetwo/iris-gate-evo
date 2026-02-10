"""
Anonymizer — Strip model identity from cross-model debate.

In S2 debate rounds, each model sees all 5 responses labeled
[Mirror A] through [Mirror E]. The letter assignment is RANDOMIZED
EVERY ROUND. Mirror A in round 1 might be Claude; Mirror A in
round 2 might be DeepSeek.

This prevents:
- Sycophancy (models deferring to perceived authority)
- Anchoring bias (models locking onto a familiar voice)
- Self-recognition (models identifying their own prior response)
"""

import random
from typing import Optional

from src.parser import ParsedResponse

MIRROR_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def anonymize_round(
    responses: list[ParsedResponse],
    round_num: int,
    seed: Optional[int] = None,
) -> tuple[list[dict], dict]:
    """Anonymize a set of model responses for one debate round.

    Returns:
        anonymized: List of dicts with mirror labels and content (no model names)
        mapping: Dict of {mirror_label: model_name} for audit trail only
                 (NEVER shown to models)

    The mapping changes every round. Mirror A in round N is NOT
    the same model as Mirror A in round N+1.
    """
    # Seed with round number for reproducibility in the audit trail,
    # but different every round
    rng = random.Random(seed + round_num if seed is not None else None)

    # Shuffle the order — this IS the randomization
    indices = list(range(len(responses)))
    rng.shuffle(indices)

    anonymized = []
    mapping = {}

    for new_idx, original_idx in enumerate(indices):
        resp = responses[original_idx]
        label = MIRROR_LABELS[new_idx]

        mapping[label] = resp.model

        anonymized.append({
            "label": label,
            "content": resp.raw,
            "claims": [
                {
                    "statement": c.statement,
                    "type": c.type,
                    "confidence": c.confidence,
                    "mechanism": c.mechanism,
                    "falsifiable_by": c.falsifiable_by,
                }
                for c in resp.claims
            ],
        })

    return anonymized, mapping


def build_debate_prompt(
    question: str,
    anonymized: list[dict],
    round_num: int,
    token_budget: int,
) -> str:
    """Build the S2 debate prompt with anonymized mirror responses.

    Each model sees all responses (including its own, but it can't
    tell which is its own because the labels are randomized).
    """
    mirror_blocks = []
    for mirror in anonymized:
        label = mirror["label"]
        # Format claims compactly for the debate prompt
        claims_text = _format_claims_for_debate(mirror["claims"])
        mirror_blocks.append(
            f"[Mirror {label}]\n{claims_text}"
        )

    mirrors_section = "\n\n".join(mirror_blocks)

    return f"""\
═══════════════════════════════════════════════════════
IRIS GATE EVO — REFINEMENT ROUND {round_num}
═══════════════════════════════════════════════════════

ROLE: You are one of five independent scientific minds. \
You have seen the first-round responses from all five mirrors \
(including your own, but you cannot identify which is yours). \
Your job: refine your position in light of the evidence.

QUESTION:
{question}

MIRROR RESPONSES FROM PREVIOUS ROUND:
{mirrors_section}

YOUR TASK — respond in exactly this format:

─── SECTION 2: CLAIMS (max {token_budget - 200} tokens) ───
State your REFINED claims. For EACH claim:

CLAIM: {{statement}}
TYPE: {{0|1|2|3}}
CONFIDENCE: {{0.0-1.0}}
MECHANISM: {{one sentence}}
FALSIFIABLE BY: {{what evidence would disprove this}}

RULES:
- If multiple mirrors agree on a claim, raise your confidence
- If you see strong counter-evidence, lower your confidence or change TYPE
- If a mirror introduced a claim you hadn't considered, engage with it
- You may add new claims or drop claims you no longer support
- Do NOT defer to mirrors just because they agree — evaluate the REASONING

─── SECTION 3: UNKNOWNS (max 100 tokens) ───
What remains uncertain after seeing all mirrors.

═══════════════════════════════════════════════════════
TOKEN BUDGET: {token_budget} total. Compression is signal.
═══════════════════════════════════════════════════════"""


def _format_claims_for_debate(claims: list[dict]) -> str:
    """Format claims compactly for inclusion in debate prompt."""
    if not claims:
        return "(No structured claims extracted)"

    lines = []
    for c in claims:
        lines.append(f"CLAIM: {c['statement']}")
        lines.append(f"TYPE: {c['type']} | CONFIDENCE: {c['confidence']}")
        if c.get("mechanism"):
            lines.append(f"MECHANISM: {c['mechanism']}")
        lines.append("")

    return "\n".join(lines).rstrip()
