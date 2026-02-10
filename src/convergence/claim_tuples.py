"""
Claim Tuple Extraction — Normalized semantic units from claim text.

Extracts (subject, relation, object, value) tuples from claim statements
and mechanisms. Two models saying the same thing in different words produce
identical tuples, enabling meaningful Jaccard computation.

Pure Python. No LLM calls. No NLP libraries. Regex + dictionaries.
"""

import re
from dataclasses import dataclass
from src.parser import Claim


@dataclass(frozen=True)
class ClaimTuple:
    """Normalized semantic unit. Frozen for set membership (hashable)."""
    subject: str
    relation: str
    object: str
    value: str = ""


# ── Entity Synonyms ──
# Multi-word phrases first (longest match), then single tokens.
# Maps surface forms → canonical keys.

ENTITY_SYNONYMS: dict[str, str] = {
    # Cannabinoids
    "cannabidiol": "cbd",
    "delta-9-thc": "thc",
    "tetrahydrocannabinol": "thc",
    "n-acetylcysteine": "nac",
    # Ion channels & receptors
    "voltage-dependent anion channel 1": "vdac1",
    "voltage dependent anion channel": "vdac1",
    "voltage-dependent anion channel": "vdac1",
    "transient receptor potential vanilloid 1": "trpv1",
    "transient receptor potential vanilloid 2": "trpv2",
    "vanilloid receptor": "trpv1",
    "peroxisome proliferator-activated receptor gamma": "ppargamma",
    "ppar gamma": "ppargamma",
    "ppar-gamma": "ppargamma",
    "ppargamma": "ppargamma",
    # Cellular components
    "mitochondrial membrane potential": "membrane_potential",
    "mitochondrial membrane permeability": "membrane_permeability",
    "membrane permeability": "membrane_permeability",
    "membrane potential": "membrane_potential",
    "reactive oxygen species": "ros",
    "calcium influx": "calcium_influx",
    "calcium signaling": "calcium_signaling",
    "apoptotic cascade": "apoptosis",
    "programmed cell death": "apoptosis",
    "cell death": "apoptosis",
    "cytochrome c release": "cytochrome_c_release",
    "cytochrome c": "cytochrome_c",
    "caspase activation": "caspase_activation",
    "caspase cascade": "caspase_activation",
    # Cell types
    "cancer cells": "cancer_cell",
    "cancer cell": "cancer_cell",
    "tumor cells": "cancer_cell",
    "tumor cell": "cancer_cell",
    "healthy cells": "healthy_cell",
    "healthy cell": "healthy_cell",
    "normal cells": "healthy_cell",
    "normal cell": "healthy_cell",
    # Measurements
    "dissociation constant": "kd",
    "binding affinity": "kd",
    "half-maximal inhibitory concentration": "ic50",
    "half maximal effective concentration": "ec50",
    "hill coefficient": "hill_n",
    "dose-response": "dose_response",
    "dose response": "dose_response",
    # General biology
    "oxidative stress": "oxidative_stress",
    "endoplasmic reticulum stress": "er_stress",
    "er stress": "er_stress",
    "gap junctions": "gap_junction",
    "gap junction": "gap_junction",
    "action potential": "action_potential",
    "resting potential": "resting_potential",
}

# Sort multi-word entries by length (longest first) for greedy matching
_MULTI_WORD_SYNONYMS = sorted(
    [(k, v) for k, v in ENTITY_SYNONYMS.items() if " " in k or "-" in k],
    key=lambda x: len(x[0]),
    reverse=True,
)

# Single-word synonyms
_SINGLE_WORD_SYNONYMS = {k: v for k, v in ENTITY_SYNONYMS.items() if " " not in k and "-" not in k}

# Pattern for entity-like tokens: abbreviations (2+ uppercase letters, may include digits)
_ENTITY_TOKEN_RE = re.compile(r'\b([A-Z][A-Z0-9]{1,}[a-z]?\d*)\b')

# Garbage tokens to filter out (Roman numerals, section labels, etc.)
_ENTITY_BLACKLIST = {"II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
                     "IF", "OR", "AND", "THE", "FOR", "NOT", "BUT", "MAX", "MIN"}


# ── Relation Patterns ──

RELATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\binteracts?\s+with\b', re.I), "binds"),
    (re.compile(r'\bbind(?:s|ing)?\s+to\b', re.I), "binds"),
    (re.compile(r'\bbind(?:s|ing)?\b', re.I), "binds"),
    (re.compile(r'\bincreas(?:es?|ing)\b', re.I), "increases"),
    (re.compile(r'\belevat(?:es?|ed|ing)\b', re.I), "increases"),
    (re.compile(r'\benhance[sd]?\b', re.I), "increases"),
    (re.compile(r'\bamplifie[sd]?\b', re.I), "increases"),
    (re.compile(r'\bupregulat(?:es?|ing)\b', re.I), "increases"),
    (re.compile(r'\bdecreas(?:es?|ing)\b', re.I), "decreases"),
    (re.compile(r'\breduc(?:es?|ing)\b', re.I), "decreases"),
    (re.compile(r'\bdownregulat(?:es?|ing)\b', re.I), "decreases"),
    (re.compile(r'\blower(?:s|ing)?\b', re.I), "decreases"),
    (re.compile(r'\binhibit(?:s|ing)?\b', re.I), "inhibits"),
    (re.compile(r'\bblock(?:s|ing)?\b', re.I), "inhibits"),
    (re.compile(r'\bsuppress(?:es|ing)?\b', re.I), "inhibits"),
    (re.compile(r'\bactivat(?:es?|ing)\b', re.I), "activates"),
    (re.compile(r'\bmediat(?:es?|ing)\b', re.I), "mediates"),
    (re.compile(r'\btrigger(?:s|ing)?\b', re.I), "triggers"),
    (re.compile(r'\binduc(?:es?|ing)\b', re.I), "induces"),
    (re.compile(r'\bcaus(?:es?|ing)\b', re.I), "causes"),
    (re.compile(r'\bsynergiz(?:es?|ing)\b', re.I), "synergizes"),
    (re.compile(r'\bmodulat(?:es?|ing)\b', re.I), "modulates"),
    (re.compile(r'\bdisrupt(?:s|ing)?\b', re.I), "disrupts"),
    (re.compile(r'\balter(?:s|ing)?\b', re.I), "alters"),
    (re.compile(r'\bpromot(?:es?|ing)\b', re.I), "promotes"),
    (re.compile(r'\bdepolariz(?:es?|ing)\b', re.I), "depolarizes"),
]

# Broad relation groups for Jaccard comparison.
# Detailed relations preserved in ClaimTuple; these groups reduce
# "increases" vs "promotes" vs "activates" into the same bucket.
RELATION_GROUPS: dict[str, str] = {
    "binds": "interacts",
    "associated": "interacts",
    "activates": "upregulates",
    "increases": "upregulates",
    "promotes": "upregulates",
    "triggers": "upregulates",
    "induces": "upregulates",
    "synergizes": "upregulates",
    "causes": "upregulates",
    "decreases": "downregulates",
    "inhibits": "downregulates",
    "modulates": "modulates",
    "mediates": "modulates",
    "disrupts": "modulates",
    "alters": "modulates",
    "depolarizes": "modulates",
}


def group_relation(relation: str) -> str:
    """Map a detailed relation to its broad group for Jaccard comparison."""
    return RELATION_GROUPS.get(relation, relation)


# ── Value Extraction ──

# Parameter name = value unit
_VALUE_RE = re.compile(
    r'(?:\b(Kd|Ki|Km|IC50|EC50|ED50|TD50|psi|Vm)|([ψΨ]|Δ[ψΨ]m?))\s*'
    r'[=≈~:]*\s*'
    r'([~≈]?\s*-?\d+\.?\d*)\s*'
    r'([uμµ]?[Mm]|nM|mM|mV|V|%)?',
    re.IGNORECASE,
)

# Standalone numeric with unit (e.g., "-120mV", "0.45 relative")
_NUMERIC_UNIT_RE = re.compile(
    r'(-?\d+\.?\d*)\s*'
    r'(uM|[uμµ]M|nM|mM|mV|V|%)',
    re.IGNORECASE,
)

UNIT_TO_BASE: dict[str, float] = {
    "nm": 1e-9, "um": 1e-6, "μm": 1e-6, "µm": 1e-6,  # μ=U+03BC, µ=U+00B5
    "mm": 1e-3, "m": 1.0,
    "mv": 1e-3, "v": 1.0,
}

PARAM_CANONICAL: dict[str, str] = {
    "kd": "kd", "ki": "ki", "km": "km",
    "ic50": "ic50", "ec50": "ec50",
    "ed50": "ed50", "td50": "td50",
    "psi": "psi", "vm": "psi",
    "ψ": "psi", "Ψ": "psi", "δψm": "psi", "δψ": "psi", "ΔΨm": "psi", "ΔΨ": "psi",
}


def _normalize_value(param: str, value_str: str, unit: str) -> str:
    """Normalize a parameter=value+unit to canonical string."""
    param = PARAM_CANONICAL.get(param.lower(), param.lower())
    try:
        val = float(value_str.replace("~", "").replace("≈", "").strip())
    except ValueError:
        return ""
    if unit:
        multiplier = UNIT_TO_BASE.get(unit.lower(), 1.0)
        val *= multiplier
    return f"{param}={val:.2e}"


# ── Entity Extraction ──

def extract_entities(text: str) -> list[tuple[str, int]]:
    """Extract canonical entities with their positions in text.

    Returns list of (canonical_entity, start_position).
    """
    text_lower = text.lower()
    entities = []
    used_spans = set()

    # Phase 1: Multi-word synonym matches (longest first)
    for surface, canonical in _MULTI_WORD_SYNONYMS:
        start = 0
        while True:
            idx = text_lower.find(surface.lower(), start)
            if idx == -1:
                break
            end = idx + len(surface)
            # Check no overlap with already-matched spans
            span = range(idx, end)
            if not any(p in used_spans for p in span):
                entities.append((canonical, idx))
                used_spans.update(span)
            start = end

    # Phase 2: Single-word synonyms
    for surface, canonical in _SINGLE_WORD_SYNONYMS.items():
        pattern = re.compile(r'\b' + re.escape(surface) + r'\b', re.I)
        for m in pattern.finditer(text):
            if not any(p in used_spans for p in range(m.start(), m.end())):
                entities.append((canonical, m.start()))
                used_spans.update(range(m.start(), m.end()))

    # Phase 3: Abbreviation-like tokens (e.g., CBD, VDAC1, TRPV1, ROS, CYP3A4)
    for m in _ENTITY_TOKEN_RE.finditer(text):
        token = m.group(1)
        if token in _ENTITY_BLACKLIST:
            continue
        if not any(p in used_spans for p in range(m.start(), m.end())):
            canonical = token.lower()
            # Check single-word synonyms for the lowercased form
            canonical = _SINGLE_WORD_SYNONYMS.get(canonical, canonical)
            entities.append((canonical, m.start()))
            used_spans.update(range(m.start(), m.end()))

    return sorted(entities, key=lambda x: x[1])


def extract_relations(text: str) -> list[tuple[str, int]]:
    """Extract canonical relations with their positions in text.

    Returns list of (canonical_relation, start_position).
    """
    relations = []
    used_positions = set()

    for pattern, canonical in RELATION_PATTERNS:
        for m in pattern.finditer(text):
            if m.start() not in used_positions:
                relations.append((canonical, m.start()))
                used_positions.add(m.start())

    return sorted(relations, key=lambda x: x[1])


def extract_values(text: str) -> list[tuple[str, int]]:
    """Extract normalized parameter=value strings with positions.

    Returns list of (normalized_value_string, start_position).
    """
    values = []

    # Named parameters (Kd = 11 uM, ΔΨm = -120 mV)
    for m in _VALUE_RE.finditer(text):
        param = m.group(1) or m.group(2)  # group 1 = ASCII params, group 2 = Unicode ψ/Ψ/ΔΨ
        val_str, unit = m.group(3), m.group(4) or ""
        normalized = _normalize_value(param, val_str, unit)
        if normalized:
            values.append((normalized, m.start()))

    return sorted(values, key=lambda x: x[1])


# ── Tuple Assembly ──

def extract_tuples(claim: Claim) -> set[ClaimTuple]:
    """Extract normalized claim tuples from a Claim object.

    Uses both claim.statement and claim.mechanism for richer extraction.
    Returns a set of ClaimTuple (frozen dataclass, hashable).
    """
    # Combine statement and mechanism
    text = claim.statement
    if claim.mechanism:
        text = text + " " + claim.mechanism

    if not text.strip():
        return set()

    entities = extract_entities(text)
    relations = extract_relations(text)
    values = extract_values(text)

    tuples = set()

    if relations and entities:
        # For each relation, find nearest preceding entity (subject)
        # and nearest following entity (object)
        for rel_name, rel_pos in relations:
            # Subject: nearest entity BEFORE the relation
            subject = None
            for ent_name, ent_pos in reversed(entities):
                if ent_pos < rel_pos:
                    subject = ent_name
                    break

            # Object: nearest entity AFTER the relation
            obj = None
            for ent_name, ent_pos in entities:
                if ent_pos > rel_pos:
                    obj = ent_name
                    break

            if subject and obj and subject != obj:
                # Find nearest value to this relation
                val = ""
                best_dist = float("inf")
                for v_str, v_pos in values:
                    dist = abs(v_pos - rel_pos)
                    if dist < best_dist:
                        best_dist = dist
                        val = v_str
                # Sort subject/object alphabetically to make tuples
                # directionally invariant: "CBD increases ROS" and
                # "ROS elevated by CBD" produce the same tuple.
                a, b = sorted([subject, obj])
                tuples.add(ClaimTuple(a, rel_name, b, val))

    # Fallback: if no relation-based tuples but we have 2+ entities,
    # emit co-occurrence tuples (adjacent pairs only, not all-pairs)
    if not tuples and len(entities) >= 2:
        unique_ents = list(dict.fromkeys(e[0] for e in entities))  # preserve order, dedup
        for i in range(len(unique_ents) - 1):
            tuples.add(ClaimTuple(unique_ents[i], "associated", unique_ents[i + 1], ""))

    # Also emit entity-only tuples for single-entity claims with values
    if not tuples and entities and values:
        ent = entities[0][0]
        for v_str, _ in values:
            tuples.add(ClaimTuple(ent, "has", "value", v_str))

    return tuples
