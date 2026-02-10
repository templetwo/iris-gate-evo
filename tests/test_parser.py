"""
Tests for the Claim Parser.

Validates extraction of structured claims from model responses,
including imperfect formatting, missing fields, and edge cases.
"""

import pytest
from src.parser import parse_claims, parse_response, Claim


# Realistic model response (abbreviated)
SAMPLE_RESPONSE = """\
─── SECTION 1: DECOMPOSITION (max 200 tokens) ───
1. How does CBD interact with VDAC1 at the molecular level?
2. What makes cancer cell mitochondrial membranes selectively vulnerable?
3. How does ROS amplification drive selective apoptosis?

─── SECTION 2: CLAIMS (max 400 tokens) ───

CLAIM: CBD binds VDAC1 with Kd = 11 uM, causing conformational change that increases mitochondrial membrane permeability
TYPE: 1
CONFIDENCE: 0.85
MECHANISM: CBD interaction with VDAC1 alters its gating state, collapsing the mitochondrial membrane potential
FALSIFIABLE BY: VDAC1 knockout cells showing identical CBD cytotoxicity would disprove this as the primary mechanism

CLAIM: Cancer cells are selectively vulnerable due to depolarized mitochondrial membrane potential (-120mV vs -180mV healthy)
TYPE: 0
CONFIDENCE: 0.90
MECHANISM: The 60mV difference means cancer mitochondria are closer to the permeability transition threshold
FALSIFIABLE BY: Artificially depolarizing healthy cell mitochondria to -120mV should increase CBD sensitivity

CLAIM: CBD-induced ROS amplification exceeds the already-elevated cancer baseline, triggering apoptotic cascade
TYPE: 1
CONFIDENCE: 0.75
MECHANISM: Cancer cells at 0.45 ROS baseline have less antioxidant headroom than healthy cells at 0.08
FALSIFIABLE BY: Pre-treatment with NAC (ROS scavenger) should abolish CBD selectivity

CLAIM: TRPV1-mediated calcium influx synergizes with VDAC1-mediated depolarization
TYPE: 2
CONFIDENCE: 0.60
MECHANISM: TRPV1 activation at Kd 2.0 uM provides additional calcium overload signal
FALSIFIABLE BY: Capsazepine (TRPV1 antagonist) co-treatment reducing CBD efficacy by >50%

─── SECTION 3: UNKNOWNS (max 150 tokens) ───
Missing: CBD concentration kinetics at the mitochondrial membrane vs plasma.
Unknown: Whether VDAC1 conformational change is reversible at sub-Kd concentrations.

─── SECTION 4: NEXT STEP (max 50 tokens) ───
Measure CBD-induced mitochondrial membrane potential change in paired cancer/healthy cell lines with and without VDAC1 knockdown.
"""


class TestClaimParsing:
    def test_extracts_four_claims(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        assert len(claims) == 4

    def test_claim_statements_not_empty(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        for c in claims:
            assert len(c.statement) > 10

    def test_type_extraction(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        types = [c.type for c in claims]
        assert types == [1, 0, 1, 2]

    def test_confidence_extraction(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        confidences = [c.confidence for c in claims]
        assert confidences == [0.85, 0.90, 0.75, 0.60]

    def test_mechanism_extraction(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        assert "gating state" in claims[0].mechanism
        assert "permeability transition" in claims[1].mechanism

    def test_falsifiable_extraction(self):
        claims = parse_claims(SAMPLE_RESPONSE)
        assert "knockout" in claims[0].falsifiable_by.lower()

    def test_confidence_clamped_to_01(self):
        text = "CLAIM: test\nTYPE: 1\nCONFIDENCE: 5.0"
        claims = parse_claims(text)
        assert claims[0].confidence == 1.0

    def test_missing_type_defaults_to_2(self):
        text = "CLAIM: some statement without type\nCONFIDENCE: 0.5"
        claims = parse_claims(text)
        assert claims[0].type == 2

    def test_missing_confidence_defaults_to_05(self):
        text = "CLAIM: some statement\nTYPE: 1"
        claims = parse_claims(text)
        assert claims[0].confidence == 0.5

    def test_empty_text_returns_empty(self):
        assert parse_claims("") == []
        assert parse_claims("no claims here, just prose") == []

    def test_handles_bold_formatting(self):
        text = "**CLAIM:** Bold statement\n**TYPE:** 1\n**CONFIDENCE:** 0.9"
        claims = parse_claims(text)
        assert len(claims) == 1
        assert claims[0].type == 1


class TestResponseParsing:
    def test_full_response_parse(self):
        parsed = parse_response(SAMPLE_RESPONSE, model="test")
        assert parsed.model == "test"
        assert len(parsed.claims) == 4
        assert "VDAC1" in parsed.decomposition or len(parsed.decomposition) > 0
        assert len(parsed.unknowns) > 0
        assert len(parsed.next_step) > 0

    def test_raw_preserved(self):
        parsed = parse_response(SAMPLE_RESPONSE, model="test")
        assert parsed.raw == SAMPLE_RESPONSE

    def test_partial_response_still_parses(self):
        """A response with only claims should still work."""
        text = "CLAIM: only claim\nTYPE: 1\nCONFIDENCE: 0.8"
        parsed = parse_response(text)
        assert len(parsed.claims) == 1
