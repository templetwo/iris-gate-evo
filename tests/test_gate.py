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
    def test_strong_claims_pass(self, strong_claims):
        results = evaluate_claims_offline(strong_claims)
        # Strong claims should mostly pass
        assert len(results) == 2
        for r in results:
            assert r.falsifiable  # Both have good falsifiable_by fields
            assert r.feasible     # Both have mechanisms

    def test_weak_claims_fail(self, weak_claims):
        results = evaluate_claims_offline(weak_claims)
        # TYPE 3 with empty fields should fail
        assert not results[0].falsifiable
        assert not results[0].feasible
        # Short/vague falsifiable_by should fail
        assert not results[1].falsifiable

    def test_falsifiable_needs_specific_keywords(self):
        claims = [{
            "statement": "test claim about something",
            "type": 1,
            "mechanism": "some mechanism here that is long enough",
            "falsifiable_by": "we could maybe look into this further sometime",
        }]
        results = evaluate_claims_offline(claims)
        assert not results[0].falsifiable  # No IF/WOULD/SHOULD keywords

    def test_falsifiable_with_knockout_keyword(self):
        claims = [{
            "statement": "VDAC1 mediates CBD toxicity",
            "type": 1,
            "mechanism": "VDAC1 conformational change opens pore",
            "falsifiable_by": "If VDAC1 knockout cells show identical CBD toxicity, this would be disproved",
        }]
        results = evaluate_claims_offline(claims)
        assert results[0].falsifiable

    def test_feasible_needs_mechanism(self):
        claims = [{
            "statement": "something happens",
            "type": 1,
            "mechanism": "short",
            "falsifiable_by": "If knockout would show no effect",
        }]
        results = evaluate_claims_offline(claims)
        assert not results[0].feasible  # Mechanism too short

    def test_type3_not_feasible(self):
        claims = [{
            "statement": "speculative claim about quantum effects",
            "type": 3,
            "mechanism": "quantum tunneling through mitochondrial membrane",
            "falsifiable_by": "If blocking quantum effects would prevent cytotoxicity",
        }]
        results = evaluate_claims_offline(claims)
        assert not results[0].feasible  # TYPE 3 blocked


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

    def test_gate_filters_type3(self):
        """TYPE 3 speculation should not even reach the gate."""
        pipeline_result = {
            "question": "test?",
            "final_claims": [
                {"statement": "pure speculation", "type": 3, "mechanism": "", "falsifiable_by": ""},
            ],
        }
        result = run_gate_sync(pipeline_result, use_offline=True)
        assert result.passed is False
        assert len(result.claims) == 0

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
