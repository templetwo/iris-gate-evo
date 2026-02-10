"""
Tests for the VERIFY stage.

Validates:
- TYPE 2 claim extraction
- Verification prompt construction
- Response parsing (PROMOTED, HELD, NOVEL, CONTRADICTED)
- Verdict application back to pipeline
- Edge cases: empty claims, malformed responses
"""

import pytest
from src.verify.verify import (
    extract_type2_claims,
    build_verify_prompt,
    parse_verify_response,
    apply_verdicts,
    VerifyResult,
    VerifyStageResult,
    PROMOTED,
    HELD,
    NOVEL,
    CONTRADICTED,
)


# ── Fixtures ──

@pytest.fixture
def mixed_claims():
    """Claims with mixed types, as they come from the pipeline."""
    return [
        {"statement": "CBD binds VDAC1 with Kd = 11 uM", "type": 0, "confidence": 0.9},
        {"statement": "Cancer cells are depolarized at -120mV", "type": 1, "confidence": 0.85},
        {"statement": "TRPV1-VDAC1 synergy drives selective apoptosis", "type": 2, "confidence": 0.6},
        {"statement": "ROS amplification exceeds cancer baseline", "type": 1, "confidence": 0.75},
        {"statement": "CBD induces novel mitochondrial fission pathway", "type": 2, "confidence": 0.5},
        {"statement": "Quantum effects may contribute to selectivity", "type": 3, "confidence": 0.2},
    ]


@pytest.fixture
def pipeline_result(mixed_claims):
    return {
        "question": "How does CBD selectively kill cancer cells?",
        "final_claims": mixed_claims,
    }


# ── TYPE 2 Extraction ──

class TestType2Extraction:
    def test_extracts_only_type2(self, mixed_claims):
        type2 = extract_type2_claims(mixed_claims)
        assert len(type2) == 2
        assert all(c["type"] == 2 for c in type2)

    def test_preserves_statement(self, mixed_claims):
        type2 = extract_type2_claims(mixed_claims)
        assert "TRPV1-VDAC1 synergy" in type2[0]["statement"]

    def test_empty_input(self):
        assert extract_type2_claims([]) == []

    def test_no_type2_claims(self):
        claims = [
            {"statement": "a", "type": 0},
            {"statement": "b", "type": 1},
            {"statement": "c", "type": 3},
        ]
        assert extract_type2_claims(claims) == []


# ── Prompt Construction ──

class TestVerifyPrompt:
    def test_contains_claim(self):
        prompt = build_verify_prompt("CBD binds VDAC1")
        assert "CBD binds VDAC1" in prompt

    def test_contains_verdict_options(self):
        prompt = build_verify_prompt("test claim")
        assert "PROMOTED" in prompt
        assert "HELD" in prompt
        assert "NOVEL" in prompt
        assert "CONTRADICTED" in prompt

    def test_contains_format_instructions(self):
        prompt = build_verify_prompt("test")
        assert "VERDICT:" in prompt
        assert "CONFIDENCE:" in prompt
        assert "EVIDENCE:" in prompt


# ── Response Parsing ──

class TestVerifyParsing:
    def test_parse_promoted(self):
        response = """\
VERDICT: PROMOTED
CONFIDENCE: 0.85
EVIDENCE: Multiple studies confirm TRPV1-VDAC1 crosstalk in mitochondrial calcium signaling. \
Smith et al. 2024 demonstrated direct interaction in HEK293 cells.
CITATIONS: Smith et al. 2024, PMID:39281234, doi:10.1038/nature.2024.12345"""

        result = parse_verify_response("test claim", response)
        assert result.verdict == PROMOTED
        assert result.confidence == 0.85
        assert "Smith" in result.evidence_summary
        assert len(result.citations) >= 1
        assert result.new_type == 1  # Promoted to TYPE 1

    def test_parse_held(self):
        response = """\
VERDICT: HELD
CONFIDENCE: 0.4
EVIDENCE: Limited evidence. One preprint from 2025 suggests possible interaction but not replicated.
CITATIONS: Zhang preprint 2025"""

        result = parse_verify_response("test", response)
        assert result.verdict == HELD
        assert result.confidence == 0.4
        assert result.new_type == 2

    def test_parse_novel(self):
        response = """\
VERDICT: NOVEL
CONFIDENCE: 0.3
EVIDENCE: No published literature found on this specific mechanism in this context.
CITATIONS: None"""

        result = parse_verify_response("test", response)
        assert result.verdict == NOVEL
        assert result.new_type == 2  # Novel stays TYPE 2

    def test_parse_contradicted(self):
        response = """\
VERDICT: CONTRADICTED
CONFIDENCE: 0.9
EVIDENCE: Direct evidence against this claim. Lee et al. 2025 showed VDAC1 knockout \
had no effect on CBD cytotoxicity in 3 cancer cell lines.
CITATIONS: Lee et al. 2025 Nature Cell Biology"""

        result = parse_verify_response("test", response)
        assert result.verdict == CONTRADICTED
        assert result.confidence == 0.9
        assert result.new_type == 2

    def test_parse_malformed_defaults_to_held(self):
        result = parse_verify_response("test", "This is just gibberish text")
        assert result.verdict == HELD
        assert result.confidence == 0.5

    def test_parse_empty_response(self):
        result = parse_verify_response("test", "")
        assert result.verdict == HELD

    def test_confidence_clamped(self):
        response = "VERDICT: PROMOTED\nCONFIDENCE: 5.0\nEVIDENCE: Strong."
        result = parse_verify_response("test", response)
        assert result.confidence == 1.0

    def test_confidence_zero_floor(self):
        """Negative confidence is unparseable — defaults to 0.5."""
        response = "VERDICT: HELD\nCONFIDENCE: -0.5\nEVIDENCE: Weak."
        result = parse_verify_response("test", response)
        # Regex doesn't capture negative sign, so falls to default 0.5
        assert result.confidence == 0.5

    def test_confidence_above_1_clamped(self):
        response = "VERDICT: PROMOTED\nCONFIDENCE: 1.5\nEVIDENCE: Very strong."
        result = parse_verify_response("test", response)
        assert result.confidence == 1.0


# ── Verdict Application ──

class TestApplyVerdicts:
    def test_promoted_becomes_type1(self, pipeline_result):
        verify_result = VerifyStageResult(
            results=[
                VerifyResult(
                    claim="TRPV1-VDAC1 synergy drives selective apoptosis",
                    verdict=PROMOTED,
                    confidence=0.85,
                    evidence_summary="Confirmed",
                    citations=["Study A"],
                ),
            ],
            n_type2_input=1,
            n_promoted=1,
        )

        updated = apply_verdicts(pipeline_result, verify_result)
        verified = updated["verified_claims"]

        # Find the promoted claim
        promoted = [c for c in verified if c.get("verify_verdict") == PROMOTED]
        assert len(promoted) == 1
        assert promoted[0]["type"] == 1  # Reclassified

    def test_contradicted_gets_flagged(self, pipeline_result):
        verify_result = VerifyStageResult(
            results=[
                VerifyResult(
                    claim="CBD induces novel mitochondrial fission pathway",
                    verdict=CONTRADICTED,
                    confidence=0.9,
                    evidence_summary="Disproved",
                ),
            ],
            n_type2_input=1,
            n_contradicted=1,
        )

        updated = apply_verdicts(pipeline_result, verify_result)
        contradicted = [c for c in updated["verified_claims"] if c.get("contradicted")]
        assert len(contradicted) == 1

    def test_non_type2_claims_unchanged(self, pipeline_result):
        verify_result = VerifyStageResult(results=[], n_type2_input=0)
        updated = apply_verdicts(pipeline_result, verify_result)

        # TYPE 0 claim should be untouched
        type0 = [c for c in updated["verified_claims"] if c["type"] == 0]
        assert len(type0) == 1
        assert "verify_verdict" not in type0[0]

    def test_summary_counts(self, pipeline_result):
        verify_result = VerifyStageResult(
            results=[],
            n_type2_input=2,
            n_promoted=1,
            n_held=1,
            total_calls=2,
        )

        updated = apply_verdicts(pipeline_result, verify_result)
        summary = updated["verify_summary"]
        assert summary["n_type2_input"] == 2
        assert summary["n_promoted"] == 1
        assert summary["total_calls"] == 2

    def test_empty_verify_preserves_claims(self, pipeline_result):
        verify_result = VerifyStageResult(results=[], n_type2_input=0)
        updated = apply_verdicts(pipeline_result, verify_result)
        assert len(updated["verified_claims"]) == len(pipeline_result["final_claims"])
