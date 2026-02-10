"""
Tests for S6 Protocol Package Generator.

Validates:
- Package structure completeness
- Human-readable summary generation
- Budget tracking
- Audit trail integrity
- File save/load
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.hypothesis.s4_hypothesis import Hypothesis, ParameterSpec, S4Result
from src.monte_carlo.monte_carlo import S5Result, SimulationResult
from src.protocol.protocol import (
    generate_protocol_package,
    format_human_summary,
    save_protocol,
)


# ── Fixtures ──

@pytest.fixture
def mock_pipeline_result():
    """Minimal pipeline result for testing."""
    return {
        "session_id": "evo_20260209_test",
        "question": "How does CBD selectively kill cancer cells?",
        "s1": {
            "n_models_responded": 5,
            "snapshot": {"jaccard": 0.45, "cosine": 0.50},
            "calls": 5,
        },
        "s2": {
            "total_rounds": 6,
            "early_stopped": True,
            "rounds": [
                {"round": 1, "jaccard": 0.55, "cosine": 0.60, "jsd": 0.20, "type_01_ratio": 0.60},
                {"round": 2, "jaccard": 0.65, "cosine": 0.70, "jsd": 0.15, "type_01_ratio": 0.70},
                {"round": 3, "jaccard": 0.75, "cosine": 0.78, "jsd": 0.10, "type_01_ratio": 0.78},
                {"round": 4, "jaccard": 0.82, "cosine": 0.83, "jsd": 0.06, "type_01_ratio": 0.85},
                {"round": 5, "jaccard": 0.88, "cosine": 0.87, "jsd": 0.04, "type_01_ratio": 0.88},
                {"round": 6, "jaccard": 0.91, "cosine": 0.90, "jsd": 0.03, "type_01_ratio": 0.92, "early_stop": True},
            ],
            "calls": 30,
        },
        "s3": {
            "passed": True,
            "jaccard": 0.91,
            "type_01_ratio": 0.92,
        },
        "final_claims": [
            {"statement": "CBD binds VDAC1", "type": 1, "confidence": 0.85},
            {"statement": "Cancer depolarized", "type": 0, "confidence": 0.90},
        ],
    }


@pytest.fixture
def mock_verify_summary():
    return {
        "n_type2_input": 2,
        "n_promoted": 1,
        "n_held": 1,
        "n_novel": 0,
        "n_contradicted": 0,
        "total_calls": 2,
    }


@pytest.fixture
def mock_s4_result():
    return S4Result(
        hypotheses=[
            Hypothesis(
                id="H1",
                prediction="IF CBD at Kd, THEN depolarization > 40mV",
                source_claims=["1", "2"],
                key_variables=["CBD", "VDAC1"],
                parameters=[
                    ParameterSpec("CBD_conc", "uM", 1.0, 20.0, "uniform"),
                ],
                testability_score=8.0,
                experimental_protocol="JC-1 staining on MCF-7 vs MCF-10A",
                expected_outcome="Dose-dependent depolarization in cancer",
                null_outcome="No differential effect",
                readouts=["JC-1", "Annexin V"],
                controls=["Vehicle", "VDAC1 siRNA"],
            ),
        ],
        n_hypotheses=1,
        total_calls=1,
    )


@pytest.fixture
def mock_s5_result():
    return S5Result(
        simulations=[
            SimulationResult(
                hypothesis_id="H1",
                prediction="IF CBD at Kd, THEN depolarization > 40mV",
                n_iterations=300,
                parameters_sampled={
                    "CBD_conc": {"mean": 10.5, "std": 5.5, "ci_lower": 1.2, "ci_upper": 19.8},
                },
                outcome_stats={
                    "effect_magnitude": {"mean": 0.5, "std": 0.28, "ci_lower": 0.05, "ci_upper": 0.95},
                },
                effect_size=0.65,
                power_estimate=0.88,
                convergence_check=True,
                seed=42,
            ),
        ],
        n_hypotheses=1,
        total_iterations=300,
        total_calls=0,
    )


# ── Package Structure ──

class TestPackageStructure:
    def test_has_all_sections(self, mock_pipeline_result):
        pkg = generate_protocol_package(
            question="How does CBD kill cancer?",
            pipeline_result=mock_pipeline_result,
        )
        assert "protocol_version" in pkg
        assert "session_id" in pkg
        assert "question" in pkg
        assert "convergence_report" in pkg
        assert "verification" in pkg
        assert "gate" in pkg
        assert "hypotheses" in pkg
        assert "simulations" in pkg
        assert "protocols" in pkg
        assert "budget" in pkg
        assert "audit_trail" in pkg

    def test_session_id_preserved(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        assert pkg["session_id"] == "evo_20260209_test"

    def test_question_preserved(self, mock_pipeline_result):
        pkg = generate_protocol_package("How does CBD kill cancer?", mock_pipeline_result)
        assert pkg["question"] == "How does CBD kill cancer?"

    def test_json_serializable(self, mock_pipeline_result, mock_verify_summary, mock_s4_result, mock_s5_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            verify_summary=mock_verify_summary,
            s4_result=mock_s4_result,
            s5_result=mock_s5_result,
        )
        # Should not raise
        json_str = json.dumps(pkg, default=str)
        assert len(json_str) > 100


# ── Convergence Report ──

class TestConvergenceReport:
    def test_s2_metrics(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        cr = pkg["convergence_report"]
        assert cr["s2_refinement"]["total_rounds"] == 6
        assert cr["s2_refinement"]["early_stopped"] is True
        assert cr["s2_refinement"]["final_jaccard"] == 0.91

    def test_s3_gate(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        assert pkg["convergence_report"]["s3_gate"]["passed"] is True


# ── Hypothesis Rankings ──

class TestHypothesisRankings:
    def test_hypothesis_included(self, mock_pipeline_result, mock_s4_result, mock_s5_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            s4_result=mock_s4_result,
            s5_result=mock_s5_result,
        )
        assert len(pkg["hypotheses"]) == 1
        h = pkg["hypotheses"][0]
        assert h["id"] == "H1"
        assert h["testability_score"] == 8.0

    def test_monte_carlo_attached(self, mock_pipeline_result, mock_s4_result, mock_s5_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            s4_result=mock_s4_result,
            s5_result=mock_s5_result,
        )
        h = pkg["hypotheses"][0]
        assert "monte_carlo" in h
        assert h["monte_carlo"]["effect_size"] == 0.65
        assert h["monte_carlo"]["power"] == 0.88

    def test_no_s4_empty_hypotheses(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        assert pkg["hypotheses"] == []


# ── Budget ──

class TestBudget:
    def test_budget_tracks_calls(self, mock_pipeline_result, mock_verify_summary, mock_s4_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            verify_summary=mock_verify_summary,
            s4_result=mock_s4_result,
        )
        budget = pkg["budget"]
        assert budget["s1_calls"] == 5
        assert budget["s2_calls"] == 30
        assert budget["verify_calls"] == 2
        assert budget["s4_calls"] == 1
        assert budget["s5_calls"] == 0  # Always 0
        assert budget["total_calls"] == 38

    def test_budget_target_present(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        assert "92-142" in pkg["budget"]["budget_target"]


# ── Audit Trail ──

class TestAuditTrail:
    def test_s2_rounds_preserved(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result, session_seed=42)
        trail = pkg["audit_trail"]
        assert len(trail["s2_rounds"]) == 6
        assert trail["session_seed"] == 42

    def test_final_claims_preserved(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        assert len(pkg["audit_trail"]["final_claims"]) == 2


# ── Human Summary ──

class TestHumanSummary:
    def test_contains_question(self, mock_pipeline_result):
        pkg = generate_protocol_package("How does CBD kill cancer?", mock_pipeline_result)
        summary = format_human_summary(pkg)
        assert "How does CBD kill cancer?" in summary

    def test_contains_convergence(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        summary = format_human_summary(pkg)
        assert "CONVERGENCE" in summary
        assert "PASSED" in summary

    def test_contains_budget(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        summary = format_human_summary(pkg)
        assert "BUDGET" in summary

    def test_contains_session_info(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        summary = format_human_summary(pkg)
        assert "evo_20260209_test" in summary

    def test_contains_hypotheses_when_present(self, mock_pipeline_result, mock_s4_result, mock_s5_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            s4_result=mock_s4_result,
            s5_result=mock_s5_result,
        )
        summary = format_human_summary(pkg)
        assert "HYPOTHESES" in summary
        assert "H1" in summary


# ── File Save ──

class TestSaveProtocol:
    def test_saves_json(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_protocol(pkg, output_dir=tmpdir)
            assert "json" in paths
            assert Path(paths["json"]).exists()

            # Verify JSON is valid
            with open(paths["json"]) as f:
                loaded = json.load(f)
            assert loaded["session_id"] == "evo_20260209_test"

    def test_saves_summary(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_protocol(pkg, output_dir=tmpdir, include_summary=True)
            assert "summary" in paths
            assert Path(paths["summary"]).exists()

    def test_creates_output_dir(self, mock_pipeline_result):
        pkg = generate_protocol_package("test?", mock_pipeline_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "dir"
            paths = save_protocol(pkg, output_dir=str(nested))
            assert nested.exists()


# ── Protocols Section ──

class TestProtocols:
    def test_protocols_from_s4(self, mock_pipeline_result, mock_s4_result):
        pkg = generate_protocol_package(
            "test?", mock_pipeline_result,
            s4_result=mock_s4_result,
        )
        protocols = pkg["protocols"]
        assert len(protocols) == 1
        assert protocols[0]["hypothesis_id"] == "H1"
        assert "JC-1" in protocols[0]["protocol"]
        assert len(protocols[0]["readouts"]) >= 1
        assert len(protocols[0]["controls"]) >= 1
