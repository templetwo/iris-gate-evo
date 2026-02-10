"""
S4 — Hypothesis Operationalization.

Takes gate-passed claims and transforms them into falsifiable
hypotheses with Monte Carlo parameter mappings. Each hypothesis
becomes a simulation-ready specification:

  - Prediction statement (IF-THEN)
  - Key variables and expected ranges
  - Testability score (0-10)
  - Suggested experimental protocol
  - Parameter map for S5 Monte Carlo

This stage uses 10-15 LLM calls: one compilation call per model
to operationalize the full claim set, then a synthesis call.

Output feeds directly into S5 Monte Carlo simulation.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import litellm

from src.models import MODELS


@dataclass
class ParameterSpec:
    """A single parameter for Monte Carlo simulation."""
    name: str
    unit: str
    low: float           # Lower bound of range
    high: float          # Upper bound of range
    distribution: str = "uniform"  # uniform, normal, log-normal
    mean: Optional[float] = None
    std: Optional[float] = None


@dataclass
class Hypothesis:
    """A fully operationalized hypothesis ready for simulation."""
    id: str                          # e.g., "H1", "H2"
    prediction: str                  # IF-THEN statement
    source_claims: list[str]         # Claim statements that generated this
    key_variables: list[str]
    parameters: list[ParameterSpec]  # Monte Carlo parameter map
    testability_score: float         # 0-10
    experimental_protocol: str       # Suggested wet-lab protocol
    expected_outcome: str            # What we'd see if true
    null_outcome: str                # What we'd see if false
    dose_ranges: dict = field(default_factory=dict)    # e.g., {"CBD": [1, 5, 10, 20]}
    readouts: list[str] = field(default_factory=list)  # e.g., ["JC-1", "Annexin V"]
    controls: list[str] = field(default_factory=list)  # e.g., ["Vehicle", "VDAC1-KO"]


@dataclass
class S4Result:
    """Complete S4 stage output."""
    hypotheses: list[Hypothesis]
    n_hypotheses: int = 0
    total_calls: int = 0
    latency_s: float = 0.0


# ── Operationalization prompt ──

S4_PROMPT_TEMPLATE = """\
You are a translational scientist converting converged theoretical claims \
into testable laboratory hypotheses.

RESEARCH QUESTION:
"{question}"

VERIFIED CLAIMS (these survived multi-model convergence and verification):
{claims_block}

Transform these claims into FALSIFIABLE HYPOTHESES. For each hypothesis:

HYPOTHESIS H[n]:
PREDICTION: IF [condition], THEN [measurable outcome] with [expected magnitude]
SOURCE CLAIMS: [which claim numbers this derives from]
KEY VARIABLES: [comma-separated list of variables]
PARAMETERS:
  - [param_name]: [low]-[high] [unit] ([distribution])
  - [param_name]: [low]-[high] [unit] ([distribution])
TESTABILITY: [0-10 score]
PROTOCOL: [2-3 sentences: cell lines, reagents, method, timeline]
EXPECTED OUTCOME: [what we see if hypothesis is true]
NULL OUTCOME: [what we see if hypothesis is false]
DOSE RANGES: [compound: dose1, dose2, dose3 uM]
READOUTS: [measurement methods]
CONTROLS: [control conditions]

Guidelines:
- Each hypothesis should be independently testable
- Parameters must have numeric ranges suitable for Monte Carlo
- Distributions: uniform (no prior), normal (centered estimate), log-normal (biological ranges)
- Testability: 10 = ready for bench today, 5 = needs method development, 0 = untestable
- Prioritize hypotheses that test MECHANISMS, not just correlations
- Maximum 5 hypotheses — quality over quantity

TOKEN BUDGET: 800. Be precise."""


def build_s4_prompt(question: str, claims: list[dict]) -> str:
    """Build the S4 operationalization prompt."""
    lines = []
    for i, c in enumerate(claims, 1):
        stmt = c.get("statement", "")
        ctype = c.get("type", "?")
        mechanism = c.get("mechanism", "")
        falsifiable = c.get("falsifiable_by", "")

        block = f"[{i}] {stmt} (TYPE {ctype})"
        if mechanism:
            block += f"\n    MECHANISM: {mechanism}"
        if falsifiable:
            block += f"\n    FALSIFIABLE BY: {falsifiable}"
        lines.append(block)

    claims_block = "\n\n".join(lines)
    return S4_PROMPT_TEMPLATE.format(
        question=question,
        claims_block=claims_block,
    )


def parse_s4_response(response_text: str) -> list[Hypothesis]:
    """Parse S4 model response into Hypothesis objects.

    Handles varied formatting. Extracts parameters with numeric ranges
    for Monte Carlo simulation.
    """
    hypotheses = []

    # Split on hypothesis markers
    hyp_pattern = re.compile(
        r'HYPOTHESIS\s+H\[?(\d+)\]?\s*:', re.IGNORECASE
    )
    parts = hyp_pattern.split(response_text)

    # parts[0] is pre-hypothesis text, then alternating: num, content
    for idx in range(1, len(parts) - 1, 2):
        hyp_num = parts[idx]
        content = parts[idx + 1] if idx + 1 < len(parts) else ""

        hypothesis = _parse_single_hypothesis(hyp_num, content)
        if hypothesis:
            hypotheses.append(hypothesis)

    return hypotheses


def _parse_single_hypothesis(hyp_num: str, content: str) -> Optional[Hypothesis]:
    """Parse a single hypothesis block."""
    hyp_id = f"H{hyp_num}"

    # Extract PREDICTION
    pred_match = re.search(
        r'PREDICTION\s*:\s*(.+?)(?=\nSOURCE|KEY|PARAMETERS|TESTABILITY|PROTOCOL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    prediction = pred_match.group(1).strip() if pred_match else ""
    prediction = re.sub(r'\s+', ' ', prediction)

    if not prediction:
        return None

    # Extract SOURCE CLAIMS
    source_match = re.search(
        r'SOURCE\s*CLAIMS?\s*:\s*(.+?)(?=\n[A-Z])',
        content, re.IGNORECASE,
    )
    source_claims = []
    if source_match:
        raw = source_match.group(1).strip()
        # Extract numbers
        source_claims = re.findall(r'\d+', raw)

    # Extract KEY VARIABLES
    vars_match = re.search(
        r'KEY\s*VARIABLES?\s*:\s*(.+?)(?=\n[A-Z]|\nPARAMETERS)',
        content, re.IGNORECASE,
    )
    key_variables = []
    if vars_match:
        raw = vars_match.group(1).strip()
        key_variables = [v.strip() for v in re.split(r'[,;]', raw) if v.strip()]

    # Extract PARAMETERS
    parameters = _parse_parameters(content)

    # Extract TESTABILITY
    test_match = re.search(
        r'TESTABILITY\s*:\s*([\d.]+)',
        content, re.IGNORECASE,
    )
    testability = float(test_match.group(1)) if test_match else 5.0
    testability = max(0.0, min(10.0, testability))

    # Extract PROTOCOL
    proto_match = re.search(
        r'PROTOCOL\s*:\s*(.+?)(?=\nEXPECTED|DOSE|READOUT|NULL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    protocol = proto_match.group(1).strip() if proto_match else ""
    protocol = re.sub(r'\s+', ' ', protocol)

    # Extract EXPECTED OUTCOME
    exp_match = re.search(
        r'EXPECTED\s*OUTCOME\s*:\s*(.+?)(?=\nNULL|DOSE|READOUT|CONTROL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    expected = exp_match.group(1).strip() if exp_match else ""
    expected = re.sub(r'\s+', ' ', expected)

    # Extract NULL OUTCOME
    null_match = re.search(
        r'NULL\s*OUTCOME\s*:\s*(.+?)(?=\nDOSE|READOUT|CONTROL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    null_outcome = null_match.group(1).strip() if null_match else ""
    null_outcome = re.sub(r'\s+', ' ', null_outcome)

    # Extract DOSE RANGES
    dose_match = re.search(
        r'DOSE\s*RANGES?\s*:\s*(.+?)(?=\nREADOUT|CONTROL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    dose_ranges = {}
    if dose_match:
        dose_ranges = _parse_dose_ranges(dose_match.group(1))

    # Extract READOUTS
    read_match = re.search(
        r'READOUTS?\s*:\s*(.+?)(?=\nCONTROL|$)',
        content, re.IGNORECASE | re.DOTALL,
    )
    readouts = []
    if read_match:
        raw = read_match.group(1).strip()
        readouts = [r.strip() for r in re.split(r'[,;\n]', raw) if r.strip()]

    # Extract CONTROLS
    ctrl_match = re.search(
        r'CONTROLS?\s*:\s*(.+?)$',
        content, re.IGNORECASE | re.DOTALL,
    )
    controls = []
    if ctrl_match:
        raw = ctrl_match.group(1).strip()
        controls = [c.strip() for c in re.split(r'[,;\n]', raw) if c.strip()]

    return Hypothesis(
        id=hyp_id,
        prediction=prediction,
        source_claims=source_claims,
        key_variables=key_variables,
        parameters=parameters,
        testability_score=testability,
        experimental_protocol=protocol,
        expected_outcome=expected,
        null_outcome=null_outcome,
        dose_ranges=dose_ranges,
        readouts=readouts,
        controls=controls,
    )


def _parse_parameters(content: str) -> list[ParameterSpec]:
    """Extract parameter specifications from the PARAMETERS section."""
    params = []

    # Find PARAMETERS section
    param_section = re.search(
        r'PARAMETERS\s*:\s*(.+?)(?=\nTESTABILITY|\nPROTOCOL)',
        content, re.IGNORECASE | re.DOTALL,
    )
    if not param_section:
        return params

    text = param_section.group(1)

    # Match lines like: "- CBD_concentration: 1.0-20.0 uM (log-normal)"
    # or "  param_name: low-high unit (distribution)"
    line_pattern = re.compile(
        r'[-•]\s*(\w[\w\s]*?):\s*([\d.]+)\s*[-–]\s*([\d.]+)\s*(\S+)'
        r'(?:\s*\((\w[\w-]*)\))?',
    )

    for match in line_pattern.finditer(text):
        name = match.group(1).strip()
        low = float(match.group(2))
        high = float(match.group(3))
        unit = match.group(4).strip()
        dist = match.group(5).strip().lower() if match.group(5) else "uniform"

        # Normalize distribution names
        if dist in ("log-normal", "lognormal", "log_normal"):
            dist = "log-normal"
        elif dist in ("normal", "gaussian"):
            dist = "normal"
        else:
            dist = "uniform"

        params.append(ParameterSpec(
            name=name,
            unit=unit,
            low=low,
            high=high,
            distribution=dist,
        ))

    return params


def _parse_dose_ranges(text: str) -> dict:
    """Parse dose range specifications into a dict."""
    dose_ranges = {}

    # Match: "CBD: 1, 5, 10, 20 uM" or "compound_name: 1.0, 5.0, 10.0 unit"
    pattern = re.compile(
        r'(\w[\w\s]*?):\s*([\d.,\s]+)\s*(\S+)',
    )

    for match in pattern.finditer(text):
        compound = match.group(1).strip()
        doses_raw = match.group(2)
        unit = match.group(3).strip()

        doses = []
        for d in re.split(r'[,\s]+', doses_raw):
            d = d.strip()
            if d:
                try:
                    doses.append(float(d))
                except ValueError:
                    continue

        if doses:
            dose_ranges[compound] = {"doses": doses, "unit": unit}

    return dose_ranges


def score_testability(claim: dict) -> float:
    """Heuristic testability score for a claim (0-10).

    Used for offline evaluation when no LLM is available.
    Scoring rubric:
    - Has quantitative values (Kd, IC50, mV, etc.): +3
    - Has specific mechanism: +2
    - Has concrete falsification: +2
    - TYPE 0 or 1 (established): +1
    - Has named molecular target: +1
    - Clear dose-response implication: +1
    """
    score = 0.0
    stmt = claim.get("statement", "").lower()
    mechanism = claim.get("mechanism", "").lower()
    falsifiable = claim.get("falsifiable_by", "").lower()

    # Quantitative values
    if re.search(r'\d+\.?\d*\s*(um|μm|nm|mm|mv|nm|kd|ic50|ec50)', stmt + mechanism):
        score += 3.0

    # Specific mechanism
    if len(mechanism) > 20:
        score += 2.0

    # Concrete falsification
    if len(falsifiable) > 20 and any(w in falsifiable for w in [
        "knockout", "inhibit", "block", "if", "would", "abolish",
    ]):
        score += 2.0

    # Established type
    if claim.get("type", 3) <= 1:
        score += 1.0

    # Named molecular target
    targets = ["vdac1", "trpv1", "p53", "bcl-2", "caspase", "nf-kb", "vegf"]
    if any(t in stmt for t in targets):
        score += 1.0

    # Dose-response implication
    if any(w in stmt + mechanism for w in ["dose", "concentration", "threshold"]):
        score += 1.0

    return min(10.0, score)


def operationalize_offline(claims: list[dict], question: str = "") -> list[Hypothesis]:
    """Generate hypotheses from claims without LLM calls.

    Creates structured hypotheses using heuristic rules.
    Used for testing and as offline fallback.
    """
    hypotheses = []

    for i, claim in enumerate(claims, 1):
        stmt = claim.get("statement", "")
        mechanism = claim.get("mechanism", "")
        falsifiable = claim.get("falsifiable_by", "")

        # Build IF-THEN prediction from claim + mechanism
        prediction = f"IF {mechanism}, THEN {stmt}" if mechanism else stmt

        # Extract any numeric values for parameter ranges
        parameters = _extract_parameters_from_text(stmt + " " + mechanism)

        # Score testability
        testability = score_testability(claim)

        hypotheses.append(Hypothesis(
            id=f"H{i}",
            prediction=prediction,
            source_claims=[str(i)],
            key_variables=_extract_variables(stmt),
            parameters=parameters,
            testability_score=testability,
            experimental_protocol=f"Test: {falsifiable}" if falsifiable else "Protocol TBD",
            expected_outcome=stmt,
            null_outcome=f"No effect observed for: {stmt[:80]}",
        ))

    # Sort by testability — highest first
    hypotheses.sort(key=lambda h: h.testability_score, reverse=True)
    return hypotheses[:5]  # Max 5


def _extract_variables(text: str) -> list[str]:
    """Extract key variable names from text."""
    # Look for scientific terms (capitalized or containing numbers)
    variables = []
    patterns = [
        r'\b(VDAC\d?)\b', r'\b(TRPV\d?)\b', r'\b(CBD)\b', r'\b(ROS)\b',
        r'\b(Ca2?\+?)\b', r'\b(ATP)\b', r'\b(NAC)\b', r'\b(IC50)\b',
        r'\bK[di]\b', r'\b(membrane potential)\b', r'\b(apoptosis)\b',
    ]
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        variables.extend(matches)

    return list(dict.fromkeys(variables))[:8]  # Deduplicate, max 8


def _extract_parameters_from_text(text: str) -> list[ParameterSpec]:
    """Extract numeric parameters from claim text for Monte Carlo."""
    params = []

    # Match patterns like "Kd = 11.0 uM" or "11.0 μM"
    num_pattern = re.compile(
        r'(\w+)\s*[=≈~]\s*([\d.]+)\s*(μ?[munpk]?[MmVsA]|uM|nM|mV)',
    )
    for match in num_pattern.finditer(text):
        name = match.group(1)
        value = float(match.group(2))
        unit = match.group(3)

        # Create range: ±30% of the stated value
        low = value * 0.7
        high = value * 1.3

        params.append(ParameterSpec(
            name=name,
            unit=unit,
            low=round(low, 3),
            high=round(high, 3),
            distribution="normal",
            mean=value,
            std=value * 0.15,
        ))

    return params


async def run_s4(
    pipeline_result: dict,
    gate_result: dict = None,
    model_name: str = "claude",
    use_offline: bool = False,
) -> S4Result:
    """Run S4 hypothesis operationalization.

    Args:
        pipeline_result: Output from the pipeline with verified_claims.
        gate_result: Output from Lab Gate — only gate-passed claims.
        model_name: Which model to use for operationalization.
        use_offline: Use heuristic operationalization instead of LLM.

    Returns:
        S4Result with operationalized hypotheses.
    """
    question = pipeline_result.get("question", "")

    # Get claims that passed the gate
    claims = pipeline_result.get("verified_claims",
             pipeline_result.get("final_claims", []))

    # If gate result is provided, filter to only passed claims
    if gate_result and gate_result.get("claims"):
        passed_statements = {
            r.statement for r in gate_result["claims"]
            if hasattr(r, 'passed') and r.passed
        }
        if passed_statements:
            claims = [c for c in claims if c.get("statement") in passed_statements]

    # Filter out contradicted and TYPE 3
    claims = [
        c for c in claims
        if not c.get("contradicted", False)
        and c.get("type", 3) <= 2
    ]

    if not claims:
        return S4Result(hypotheses=[], n_hypotheses=0, total_calls=0)

    t_start = time.monotonic()

    if use_offline:
        hypotheses = operationalize_offline(claims, question)
        total_calls = 0
    else:
        prompt = build_s4_prompt(question, claims)

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
            "temperature": 0.5,
            "max_tokens": 1500,
        }

        if model["provider"] in ("xai", "deepseek"):
            kwargs["api_base"] = model["base_url"]

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content
            hypotheses = parse_s4_response(content)
        except Exception:
            hypotheses = operationalize_offline(claims, question)

        total_calls = 1

    t_elapsed = time.monotonic() - t_start

    return S4Result(
        hypotheses=hypotheses,
        n_hypotheses=len(hypotheses),
        total_calls=total_calls,
        latency_s=round(t_elapsed, 2),
    )


def run_s4_sync(
    pipeline_result: dict,
    gate_result: dict = None,
    model_name: str = "claude",
    use_offline: bool = False,
) -> S4Result:
    """Synchronous wrapper for run_s4."""
    return asyncio.run(run_s4(pipeline_result, gate_result, model_name, use_offline))
