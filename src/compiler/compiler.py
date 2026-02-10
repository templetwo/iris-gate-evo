"""
C0 — The Compiler

Transforms a raw research question into a compiled prompt seeded with
quantitative priors. This is the innovation: without prior injection,
you have five chatbots. With it, you have five constrained scientific
minds pushing against real numbers.

C0 fires once per session. Then PULSE carries the signal.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import MODELS, TOKEN_BUDGETS

# Domain detection keywords — lowercase, matched against question
DOMAIN_TRIGGERS = {
    "pharmacology": [
        "receptor", "binding", "kd", "ic50", "dose-response", "dose response",
        "agonist", "antagonist", "cbd", "thc", "drug", "therapeutic",
        "pharmacokinetic", "bioavailability", "cytotoxicity", "apoptosis",
        "vdac", "trpv", "selectivity",
    ],
    "bioelectric": [
        "membrane potential", "gap junction", "v_mem", "vmem",
        "depolarization", "hyperpolarization", "ion channel", "bioelectric",
        "connexin", "calcium wave", "ros", "mitochondrial membrane",
        "voltage-gated", "regeneration",
    ],
    "consciousness": [
        "kuramoto", "oscillator", "coherence", "r-value", "entropy",
        "lantern", "phase synchronization", "consciousness", "awareness",
        "phi", "integrated information", "binding problem",
    ],
    "physics": [
        "field", "coupling", "hamiltonian", "lagrangian", "energy landscape",
        "phase transition", "critical point", "renormalization",
        "symmetry breaking", "order parameter",
    ],
}

PRIORS_DIR = Path(__file__).parent.parent.parent / "priors"

# The scaffold — identical for all five models
SCAFFOLD_TEMPLATE = """\
═══════════════════════════════════════════════════════
IRIS GATE EVO — PULSE PROMPT
═══════════════════════════════════════════════════════

ROLE: You are one of five independent scientific minds \
examining the same question. You do not know which \
model you are. You do not know what the others will say. \
Your job is honest reasoning, not consensus.

QUESTION:
{question}

QUANTITATIVE PRIORS (validated, use as constraints):
{priors_block}

YOUR TASK — respond in exactly this format:

─── SECTION 1: DECOMPOSITION (max 200 tokens) ───
Break the question into 2-4 sub-questions.
Identify the key variables and their relationships.

─── SECTION 2: CLAIMS (max 400 tokens) ───
State your claims. For EACH claim, provide:

CLAIM: {{statement}}
TYPE: {{0|1|2|3}}
CONFIDENCE: {{0.0-1.0}}
MECHANISM: {{one sentence}}
FALSIFIABLE BY: {{what evidence would disprove this}}

─── SECTION 3: UNKNOWNS (max 150 tokens) ───
What you don't know. What would change your answer.
Be specific — name the missing data, not vague gestures.

─── SECTION 4: NEXT STEP (max 50 tokens) ───
The single most valuable experiment to run next.

═══════════════════════════════════════════════════════
TOKEN BUDGET: {token_budget} total. Quality over quantity.
EPISTEMIC HONESTY: TYPE 3 claims are welcome. \
Speculation clearly marked is more valuable than \
false certainty.
═══════════════════════════════════════════════════════"""


def detect_domains(question: str) -> list[str]:
    """Classify the question into one or more domains.

    Returns a list of matched domain names, or ["general"] if none match.
    """
    q_lower = question.lower()
    matched = []

    for domain, triggers in DOMAIN_TRIGGERS.items():
        score = sum(1 for t in triggers if t in q_lower)
        if score >= 2:
            matched.append((domain, score))

    if not matched:
        return ["general"]

    # Sort by match strength, return domain names
    matched.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in matched]


def load_priors(domains: list[str]) -> list[dict]:
    """Load and merge priors from domain JSON files.

    Returns the merged list of prior objects. Skips domains
    that don't have a priors file (logs a note, doesn't crash).
    """
    all_priors = []
    seen_params = set()

    for domain in domains:
        priors_file = PRIORS_DIR / f"{domain}.json"
        if not priors_file.exists():
            continue

        with open(priors_file) as f:
            data = json.load(f)

        for prior in data.get("priors", []):
            # Deduplicate by parameter name
            if prior["param"] not in seen_params:
                all_priors.append(prior)
                seen_params.add(prior["param"])

    return all_priors


def select_relevant_priors(priors: list[dict], question: str, max_priors: int = 8) -> list[dict]:
    """Select only priors relevant to the question. 3-8, not 30.

    Scores each prior by keyword overlap with the question,
    then takes the top `max_priors`. Never injects TYPE 3.
    """
    q_lower = question.lower()
    scored = []

    for prior in priors:
        # Never inject speculative priors — models generate those
        if prior.get("type", 3) >= 3:
            continue

        # Score by relevance: param name keywords in question
        param_words = set(re.split(r'[_\s]+', prior["param"].lower()))
        source_words = set(prior.get("source", "").lower().split())

        score = 0
        for word in param_words:
            if len(word) > 2 and word in q_lower:
                score += 2
        for word in source_words:
            if len(word) > 3 and word in q_lower:
                score += 1

        # TYPE 0 priors get a boost — they're the strongest constraints
        if prior.get("type") == 0:
            score += 1

        if score > 0:
            scored.append((prior, score))

    # Sort by relevance, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in scored[:max_priors]]

    # If nothing matched by keyword, include all TYPE 0 priors as baseline
    if not selected:
        selected = [p for p in priors if p.get("type") == 0][:max_priors]

    return selected


def format_priors_block(priors: list[dict]) -> str:
    """Format priors into the text block for prompt injection."""
    if not priors:
        return "(No domain-specific quantitative priors available. Reason from first principles.)"

    lines = []
    for p in priors:
        value = p["value"]
        if isinstance(value, list):
            value = f"[{value[0]}, {value[1]}]"

        lines.append(f"PRIOR: {p['param']} = {value} {p['unit']}")
        lines.append(f"SOURCE: {p.get('source', 'unspecified')}")
        lines.append(f"TYPE: {p.get('type', 2)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_scaffold(question: str, priors_block: str, token_budget: int = 800) -> str:
    """Build the TMK scaffold prompt from template."""
    return SCAFFOLD_TEMPLATE.format(
        question=question,
        priors_block=priors_block,
        token_budget=token_budget,
    )


def compile(question: str, domain_override: Optional[str] = None) -> dict:
    """Run the full C0 compilation pipeline.

    Takes a raw research question. Returns the compiled session object
    ready for PULSE dispatch.

    Args:
        question: The raw research question from the user.
        domain_override: Force a specific domain instead of auto-detection.

    Returns:
        Compiled session dict matching the spec in compiler-template.md.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Step 1 — Domain detection
    if domain_override:
        domains = [domain_override]
    else:
        domains = detect_domains(question)

    cross_domain = len(domains) > 1

    # Step 2 — Prior loading and selection
    all_priors = load_priors(domains)
    selected = select_relevant_priors(all_priors, question)

    # Step 3 — Scaffold construction
    priors_block = format_priors_block(selected)
    prompt = build_scaffold(question, priors_block, token_budget=TOKEN_BUDGETS["S1"])

    # Step 4 — Model configs (identical prompt, API-level params only)
    models_config = {}
    for name, model in MODELS.items():
        models_config[name] = {
            "id": model["id"],
            "temperature": model["temperature"],
            "max_tokens": model["max_tokens"],
        }

    # Step 5 — Compiler output
    domain_label = "+".join(domains)
    session_id = f"evo_{timestamp}_{domain_label}"

    return {
        "session_id": session_id,
        "question": question,
        "domains": domains,
        "date": now.isoformat(),
        "priors": selected,
        "cross_domain_flag": cross_domain,
        "prompt": prompt,
        "models": models_config,
        "token_budgets": TOKEN_BUDGETS,
        "spm_target": "maximize coherence quality × goal attainment / calls consumed",
    }
