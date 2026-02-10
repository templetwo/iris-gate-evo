"""
Claim Parser — Extract structured claims from model responses.

Models respond with Section 2 containing typed claims. This parser
extracts them into structured objects that convergence metrics can
operate on. Handles imperfect formatting gracefully — models are
creative with whitespace and punctuation.

The CLAIMS array is the unit of convergence. Not raw text.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Claim:
    """A single structured claim extracted from a model response."""
    statement: str
    type: int  # 0, 1, 2, or 3
    confidence: float  # 0.0 - 1.0
    mechanism: str = ""
    falsifiable_by: str = ""


@dataclass
class ParsedResponse:
    """Fully parsed model response with all four sections."""
    model: str
    decomposition: str = ""
    claims: list = field(default_factory=list)
    unknowns: str = ""
    next_step: str = ""
    raw: str = ""


def parse_claims(text: str) -> list[Claim]:
    """Extract structured claims from Section 2 of a response.

    Handles variations in formatting:
    - CLAIM: / Claim: / **CLAIM:**
    - TYPE: 1 / Type: 1 / TYPE 1
    - CONFIDENCE: 0.8 / Confidence: 0.80
    - MECHANISM: / Mechanism:
    - FALSIFIABLE BY: / Falsifiable by: / FALSIFIABLE:

    Returns list of Claim objects. Empty list if no claims found.
    """
    claims = []

    # Split on CLAIM markers — each chunk is one claim block
    # Match "CLAIM:" with optional formatting around it
    claim_pattern = re.compile(
        r'(?:^|\n)\s*\**\s*CLAIM\s*:?\s*\**\s*',
        re.IGNORECASE
    )

    parts = claim_pattern.split(text)

    # First part is pre-claim text, skip it
    for part in parts[1:]:
        if not part.strip():
            continue

        claim = _parse_single_claim(part)
        if claim and claim.statement:
            claims.append(claim)

    return claims


def _parse_single_claim(block: str) -> Claim:
    """Parse a single claim block into a Claim object."""

    # Extract statement — everything before the first field marker
    field_pattern = re.compile(
        r'\n\s*\**\s*(?:TYPE|CONFIDENCE|MECHANISM|FALSIFIABLE(?:\s+BY)?)\s*:',
        re.IGNORECASE
    )

    field_match = field_pattern.search(block)
    if field_match:
        statement = block[:field_match.start()].strip()
    else:
        # No fields found — entire block is the statement
        statement = block.strip()

    # Clean statement
    statement = re.sub(r'\s+', ' ', statement).strip()
    # Remove trailing punctuation artifacts
    statement = statement.rstrip('*').strip()

    # Extract TYPE
    type_match = re.search(
        r'\**\s*TYPE\s*:?\s*\**\s*(\d)',
        block, re.IGNORECASE
    )
    claim_type = int(type_match.group(1)) if type_match else 2  # Default TYPE 2

    # Extract CONFIDENCE
    conf_match = re.search(
        r'\**\s*CONFIDENCE\s*:?\s*\**\s*([\d.]+)',
        block, re.IGNORECASE
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5  # Default 0.5

    # Clamp confidence to [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    # Extract MECHANISM
    mech_match = re.search(
        r'\**\s*MECHANISM\s*:?\s*\**\s*(.+?)(?=\n\s*\**\s*(?:FALSIFIABLE|TYPE|CONFIDENCE|CLAIM)|$)',
        block, re.IGNORECASE | re.DOTALL
    )
    mechanism = mech_match.group(1).strip() if mech_match else ""
    mechanism = re.sub(r'\s+', ' ', mechanism).strip()

    # Extract FALSIFIABLE BY
    fals_match = re.search(
        r'\**\s*FALSIFIABLE(?:\s+BY)?\s*:?\s*\**\s*(.+?)(?=\n\s*\**\s*(?:CLAIM|TYPE|CONFIDENCE|MECHANISM)|$)',
        block, re.IGNORECASE | re.DOTALL
    )
    falsifiable_by = fals_match.group(1).strip() if fals_match else ""
    falsifiable_by = re.sub(r'\s+', ' ', falsifiable_by).strip()

    return Claim(
        statement=statement,
        type=claim_type,
        confidence=confidence,
        mechanism=mechanism,
        falsifiable_by=falsifiable_by,
    )


def parse_response(text: str, model: str = "unknown") -> ParsedResponse:
    """Parse a full model response into structured sections.

    Extracts:
    - Section 1: DECOMPOSITION
    - Section 2: CLAIMS (parsed into Claim objects)
    - Section 3: UNKNOWNS
    - Section 4: NEXT STEP
    """
    parsed = ParsedResponse(model=model, raw=text)

    # Section markers — flexible matching
    sec1 = re.search(
        r'SECTION\s*1\s*:?\s*DECOMPOSITION.*?\n(.*?)(?=SECTION\s*2|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    sec2 = re.search(
        r'SECTION\s*2\s*:?\s*CLAIMS.*?\n(.*?)(?=SECTION\s*3|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    sec3 = re.search(
        r'SECTION\s*3\s*:?\s*UNKNOWNS.*?\n(.*?)(?=SECTION\s*4|$)',
        text, re.IGNORECASE | re.DOTALL
    )
    sec4 = re.search(
        r'SECTION\s*4\s*:?\s*NEXT\s*STEP.*?\n(.*?)(?=═|$)',
        text, re.IGNORECASE | re.DOTALL
    )

    if sec1:
        parsed.decomposition = _clean_section(sec1.group(1))
    if sec2:
        claims_text = sec2.group(1)
        parsed.claims = parse_claims(claims_text)
    if sec3:
        parsed.unknowns = _clean_section(sec3.group(1))
    if sec4:
        parsed.next_step = _clean_section(sec4.group(1))

    # If section markers weren't found, try parsing claims from full text
    if not parsed.claims:
        parsed.claims = parse_claims(text)

    return parsed


def _clean_section(text: str) -> str:
    """Clean extracted section text."""
    # Remove separator lines
    text = re.sub(r'─+', '', text)
    text = re.sub(r'═+', '', text)
    return text.strip()
