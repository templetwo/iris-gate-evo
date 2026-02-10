"""
Tests for Lab Gate.

Validates:
- Three-criteria evaluation (falsifiable, feasible, novel)
- Offline heuristic evaluation
- Gate prompt construction
- Response parsing
- Overall PASS/FAIL logic
- Edge cases: empty claims, all-fail, all-pass
"""

import pytest
from src.gate.gate import (
    ClaimGateResult,
    GateResult,
    build_claims_block,
    build_gate_prompt,
    parse_gate_response,
    evaluate_claims_offline,
    run_gate_sync,
)


# ── Fixtures ──

@pytest.fixture
def strong_claims():
    """Claims that should pass all three criteria."""
    return [
        {
            "statement": "CBD binds VDAC1 with Kd = 11 uM causing conformational change that increases mitochondrial membrane permeability",
            "type": 1,
            "confidence": 0.85,
            "mechanism": "CBD interaction with VDAC1 alters its gating state, collapsing the mitochondrial membrane potential",
            "falsifiable_by": "VDAC1 knockout cells showing identical CBD cytotoxicity would disprove this as the primary mechanism",
        },
        {
            "statement": "Cancer cells are selectively vulnerable due to depolarized mitochondrial membrane potential",
            "type": 0,
            "confidence": 0.90,
            "mechanism": "The 60mV difference means cancer mitochondria are closer to the permeability transition threshold",
            "falsifiable_by": "Artificially depolarizing healthy cell mitochondria to -120mV should increase CBD sensitivity",
        },
    ]


@pytest.fixture
def weak_claims():
    """Claims that should fail gate criteria."""
    return [
        {
            "statement": "Something happens",
            "type": 3,
            "confidence": 0.2,
            "mechanism": "",
            "falsifiable_by": "",
        },
        {
            "statement": "CBD does things to cells maybe through some process",
            "type": 2,
            "confidence": 0.3,
            "mechanism": "unclear",
            "falsifiable_by": "more research needed",
        },
    ]


# ── Claims Block ──

class TestClaimsBlock:
    def test_numbered_output(self, strong_claims):
        block = build_claims_block(strong_claims)
        assert "CLAIM 1:" in block
        assert "CLAIM 2:" in block

    def test_contains_statement(self, strong_claims):
        block = build_claims_block(strong_claims)
        assert "VDAC1" in block

    def test_includes_mechanism(self, strong_claims):
        block = build_claims_block(strong_claims)
        assert "MECHANISM:" in block

    def test_includes_falsifiable(self, strong_claims):
        block = build_claims_block(strong_claims)
        assert "FALSIFIABLE BY:" in block


# ── Gate Prompt ──

class TestGatePrompt:
    def test_contains_question(self, strong_claims):
        prompt = build_gate_prompt("How does CBD kill cancer?", strong_claims)
        assert "How does CBD kill cancer?" in prompt

    def test_contains_three_criteria(self, strong_claims):
        prompt = build_gate_prompt("test?", strong_claims)
        assert "FALSIFIABLE" in prompt
        assert "FEASIBLE" in prompt
        assert "NOVEL" in prompt

    def test_contains_claims(self, strong_claims):
        prompt = build_gate_prompt("test?", strong_claims)
        assert "VDAC1" in prompt


# ── Response Parsing ──

class TestGateParsing:
    def test_parse_all_pass(self, strong_claims):
        response = """\
CLAIM 1:
FALSIFIABLE: YES — VDAC1 knockout experiment provides clear disproof condition
FEASIBLE: YES — Standard cell culture with commercially available knockout lines
NOVEL: YES — Specific Kd quantification for VDAC1-CBD interaction is new

CLAIM 2:
FALSIFIABLE: YES — Artificial depolarization experiment is testable
FEASIBLE: YES — JC-1 staining is standard mitochondrial membrane potential assay
NOVEL: YES — Quantitative 60mV threshold model is a new contribution"""

        results = parse_gate_response(response, strong_claims)
        assert len(results) == 2
        assert all(r.passed for r in results)
        assert all(r.falsifiable for r in results)
        assert all(r.feasible for r in results)
        assert all(r.novel for r in results)

    def test_parse_partial_fail(self, strong_claims):
        response = """\
CLAIM 1:
FALSIFIABLE: YES — Clear knockout test
FEASIBLE: YES — Standard assay
NOVEL: NO — Already well-established in literature

CLAIM 2:
FALSIFIABLE: YES — Depolarization test exists
FEASIBLE: YES — Standard equipment
NOVEL: YES — New quantitative model"""

        results = parse_gate_response(response, strong_claims)
        assert not results[0].passed  # Failed novelty
        assert not results[0].novel
        assert results[0].falsifiable
        assert results[1].passed

    def test_parse_all_fail(self, strong_claims):
        response = """\
CLAIM 1:
FALSIFIABLE: NO — Too vague
FEASIBLE: NO — Requires unavailable equipment
NOVEL: NO — Textbook knowledge

CLAIM 2:
FALSIFIABLE: NO — No disproof condition
FEASIBLE: NO — Impractical
NOVEL: NO — Well established"""

        results = parse_gate_response(response, strong_claims)
        assert not any(r.passed for r in results)

    def test_malformed_response_fails_conservatively(self, strong_claims):
        results = parse_gate_response("Just some random text", strong_claims)
        assert len(results) == 2
        # Conservative: parse failure → all False → FAIL
        assert not any(r.passed for r in results)


# ── Offline Evaluation ──

class TestOfflineEvaluation:
    """Offline evaluator checks metadata presence, not content quality.
    Content evaluation is the model's job, not ours."""

    def test_strong_claims_pass(self, strong_claims):
        results = evaluate_claims_offline(strong_claims)
        assert len(results) == 2
        for r in results:
            assert r.falsifiable  # Both have falsifiable_by fields
            assert r.feasible     # Both have mechanisms
            assert r.novel        # Offline always passes novelty to model

    def test_empty_fields_fail(self):
        claims = [{
            "statement": "Something happens",
            "type": 3,
            "mechanism": "",
            "falsifiable_by": "",
        }]
        results = evaluate_claims_offline(claims)
        assert not results[0].falsifiable  # Empty falsifiable_by
        assert not results[0].feasible     # Empty mechanism

    def test_populated_fields_pass(self):
        claims = [{
            "statement": "CBD does things",
            "type": 2,
            "mechanism": "unclear",
            "falsifiable_by": "more research needed",
        }]
        results = evaluate_claims_offline(claims)
        assert results[0].falsifiable  # Field is populated (quality is model's job)
        assert results[0].feasible     # Field is populated

    def test_novelty_always_passes_offline(self):
        """Offline mode never judges novelty — that requires a model."""
        claims = [{
            "statement": "textbook fact",
            "type": 0,
            "mechanism": "well known",
            "falsifiable_by": "standard test",
        }]
        results = evaluate_claims_offline(claims)
        assert results[0].novel  # Offline doesn't pretend to judge novelty


# ── ClaimGateResult ──

class TestClaimGateResult:
    def test_all_true_passes(self):
        r = ClaimGateResult(
            statement="test",
            falsifiable=True, falsifiable_reason="good",
            feasible=True, feasible_reason="good",
            novel=True, novel_reason="good",
        )
        assert r.passed is True

    def test_one_false_fails(self):
        r = ClaimGateResult(
            statement="test",
            falsifiable=True, falsifiable_reason="good",
            feasible=True, feasible_reason="good",
            novel=False, novel_reason="bad",
        )
        assert r.passed is False

    def test_all_false_fails(self):
        r = ClaimGateResult(
            statement="test",
            falsifiable=False, falsifiable_reason="bad",
            feasible=False, feasible_reason="bad",
            novel=False, novel_reason="bad",
        )
        assert r.passed is False


# ── Full Gate Run (offline) ──

class TestGateRun:
    def test_offline_gate_with_strong_claims(self, strong_claims):
        pipeline_result = {
            "question": "How does CBD kill cancer?",
            "final_claims": strong_claims,
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        assert isinstance(result, GateResult)
        assert result.n_passed > 0
        assert result.total_calls == 0  # Offline = no calls

    def test_offline_gate_with_no_claims(self):
        pipeline_result = {
            "question": "test?",
            "final_claims": [],
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        assert result.passed is False
        assert "No claims" in result.recommendation

    def test_gate_evaluates_all_types(self):
        """All claim types reach the gate — the model decides, not pre-filtering."""
        pipeline_result = {
            "question": "test?",
            "final_claims": [
                {"statement": "speculative claim", "type": 3, "mechanism": "", "falsifiable_by": ""},
            ],
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        assert len(result.claims) == 1  # Claim reaches the gate
        assert result.passed is False   # But fails (empty fields)

    def test_gate_filters_contradicted(self, strong_claims):
        """Contradicted claims should not reach the gate."""
        strong_claims[0]["contradicted"] = True
        pipeline_result = {
            "question": "test?",
            "final_claims": strong_claims,
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        assert len(result.claims) == 1  # Only the non-contradicted one

    def test_recommendation_on_failure(self, weak_claims):
        pipeline_result = {
            "question": "test?",
            "final_claims": weak_claims,
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        if not result.passed and result.claims:
            assert "FAILED" in result.recommendation
