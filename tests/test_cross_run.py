"""Tests for cross-run convergence detection."""

import json
import tempfile
from pathlib import Path

import pytest

from src.cross_run.loader import _parse_claim_repr, _parse_claim, load_run, find_runs, RunData
from src.cross_run.matcher import cross_match, _classify_cross_match, _detect_structural_patterns, CrossRunResult
from src.cross_run.report import save_report


# ── Repr Parser Tests ──

# Actual claim strings from THC pharmacology run
THC_CLAIM_REPR = (
    "SynthesizedClaim(statement='2:** The therapeutic index (TD50/ED50) for wellbeing "
    "vs. tolerance exceeds 10, with tolerance emerging at ~25 mg/day due to CB1 "
    "internalization exceeding recycling/synthesis rates (k_int > k_rec).', "
    "mechanism='Tolerance onset is governed by a first-order kinetic crossover point "
    "where THC-driven internalization outpaces receptor recovery, reducing surface CB1 "
    "density.', falsifiable_by='If tolerance ED50 <25 mg/day (e.g., in PET studies "
    "showing >50% CB1 loss at lower doses), or if recycling/synthesis rates (k_rec) "
    "are faster than predicted.', type=1, confidence=0.743, overlap_count=3, "
    "models=['gemini', 'grok', 'mistral'], tuples=frozenset({('cb1', 'downregulates', "
    "'thc')}), chunk_text='2:** The therapeutic index (TD50/ED50) for wellbeing vs. "
    "tolerance exceeds 10, with tolerance emerging at ~25 mg/day due to CB1 "
    "internalization exceeding recycling/synthesis rates (k_int > k_rec).')"
)

UNICODE_CLAIM_REPR = (
    "SynthesizedClaim(statement='1:** Chronic low-dose THC (≤2.5 mg/day) improves "
    "wellbeing via *biphasic CB1 signaling*: G-protein activation (EC50 ~10 nM) "
    "dominates at 10–30% occupancy, while β-arrestin recruitment (EC50 >100 nM) is "
    "avoided, preserving cAMP/PKA tone.', mechanism='Partial agonism at CB1 maintains "
    "receptor sensitivity by favoring G-protein coupling over β-arrestin, preventing "
    "desensitization while enhancing tonic endocannabinoid signaling.', "
    "falsifiable_by='If β-arrestin recruitment occurs at ≤2.5 mg/day (e.g., via "
    "BRET assays in vivo), or if cAMP/PKA signaling is suppressed at this dose.', "
    "type=2, confidence=0.707, overlap_count=2, models=['claude', 'mistral'], "
    "tuples=frozenset({('cb1', 'interacts', 'ec50'), ('ec50', 'interacts', 'pka'), "
    "('thc', 'interacts', 'cb1')}), chunk_text='1:** Chronic low-dose THC (≤2.5 "
    "mg/day) improves wellbeing via *biphasic CB1 signaling*.')"
)

SINGULAR_CLAIM_REPR = (
    "SynthesizedClaim(statement='Sustained wellbeing is mediated by a lasting "
    "potentiation of endocannabinoid-dependent long-term depression (eCB-LTD) at "
    "excitatory synapses onto CRH-releasing neurons in the PVN.', mechanism=\"Chronic "
    "sub-saturating CB1 occupancy lowers the threshold for synaptically-released 2-AG "
    "to trigger presynaptic inhibition, effectively 'stamping in' a less anxious "
    "state.\", falsifiable_by='Rodent slice electrophysiology showing that chronic "
    "low-dose THC *ex vivo* enhances LTD in amygdalar slices.', type=3, "
    "confidence=0.75, overlap_count=1, models=['gemini'], "
    "tuples=frozenset({('ag', 'downregulates', 'cb1')}), chunk_text='Sustained "
    "wellbeing is mediated by eCB-LTD.')"
)


class TestReprParser:
    def test_basic_fields(self):
        result = _parse_claim_repr(THC_CLAIM_REPR)
        assert result["type"] == 1
        assert result["confidence"] == 0.743
        assert result["overlap_count"] == 3
        assert result["models"] == ["gemini", "grok", "mistral"]
        assert "therapeutic index" in result["statement"]
        assert "first-order kinetic" in result["mechanism"]
        assert "tolerance ED50" in result["falsifiable_by"]

    def test_unicode_chars(self):
        result = _parse_claim_repr(UNICODE_CLAIM_REPR)
        assert result["type"] == 2
        assert result["confidence"] == 0.707
        assert "β-arrestin" in result["statement"]
        assert result["models"] == ["claude", "mistral"]

    def test_singular_with_double_quotes_in_mechanism(self):
        result = _parse_claim_repr(SINGULAR_CLAIM_REPR)
        assert result["type"] == 3
        assert result["confidence"] == 0.75
        assert result["overlap_count"] == 1
        assert result["models"] == ["gemini"]
        assert "eCB-LTD" in result["statement"]

    def test_dict_passthrough(self):
        d = {"statement": "test", "type": 1, "confidence": 0.8}
        result = _parse_claim(d)
        assert result == d

    def test_repr_via_parse_claim(self):
        result = _parse_claim(THC_CLAIM_REPR)
        assert result["type"] == 1
        assert result["models"] == ["gemini", "grok", "mistral"]


# ── TYPE Reclassification Tests ──

class TestClassification:
    def test_convergent_singular(self):
        assert _classify_cross_match(3, 3) == "CONVERGENT SINGULAR"

    def test_cross_validated_singular(self):
        assert _classify_cross_match(3, 0) == "CROSS-VALIDATED SINGULAR"
        assert _classify_cross_match(3, 1) == "CROSS-VALIDATED SINGULAR"
        assert _classify_cross_match(3, 2) == "CROSS-VALIDATED SINGULAR"
        assert _classify_cross_match(0, 3) == "CROSS-VALIDATED SINGULAR"

    def test_independent_replication(self):
        assert _classify_cross_match(1, 1) == "INDEPENDENT REPLICATION"
        assert _classify_cross_match(0, 0) == "INDEPENDENT REPLICATION"
        assert _classify_cross_match(0, 1) == "INDEPENDENT REPLICATION"

    def test_cross_promoted(self):
        assert _classify_cross_match(2, 0) == "CROSS-PROMOTED"
        assert _classify_cross_match(2, 1) == "CROSS-PROMOTED"
        assert _classify_cross_match(1, 2) == "CROSS-PROMOTED"


# ── Structural Pattern Detection Tests ──

class TestStructuralPatterns:
    def test_two_pathway_detected(self):
        runs = [
            RunData("run_a", Path("."), "", "pharmacology", False, [
                {"statement": "biphasic response", "mechanism": "dual pathway"},
                {"statement": "dose-dependent switch", "mechanism": "two-pathway model"},
            ]),
            RunData("run_b", Path("."), "", "pharmacology", False, [
                {"statement": "biphasic kinetics", "mechanism": "dual mechanism"},
                {"statement": "another biphasic claim", "mechanism": ""},
            ]),
        ]
        patterns = _detect_structural_patterns(runs)
        assert "two_pathway" in patterns
        assert "run_a" in patterns["two_pathway"]
        assert "run_b" in patterns["two_pathway"]

    def test_single_mention_not_flagged(self):
        runs = [
            RunData("run_a", Path("."), "", "pharmacology", False, [
                {"statement": "biphasic response", "mechanism": ""},
            ]),
            RunData("run_b", Path("."), "", "pharmacology", False, [
                {"statement": "biphasic kinetics", "mechanism": ""},
            ]),
        ]
        patterns = _detect_structural_patterns(runs)
        assert "two_pathway" not in patterns  # only 1 mention per run


# ── Cross-Matcher Integration Tests ──

class TestCrossMatcher:
    def test_matching_claims_across_runs(self):
        """Two runs with semantically similar claims should match."""
        runs = [
            RunData("run_a", Path("."), "THC question", "pharmacology", False, [
                {"statement": "CB1 receptor occupancy below 30% favors G-protein signaling",
                 "mechanism": "Low occupancy biased agonism at CB1",
                 "type": 1, "confidence": 0.7, "overlap_count": 3, "models": ["a", "b", "c"]},
            ]),
            RunData("run_b", Path("."), "THC question 2", "pharmacology", False, [
                {"statement": "CB1 occupancy under 30 percent biases toward G-protein pathway",
                 "mechanism": "Biased signaling at low CB1 occupancy",
                 "type": 3, "confidence": 0.7, "overlap_count": 1, "models": ["d"]},
            ]),
        ]
        result = cross_match(runs, threshold=0.60)
        assert len(result.matches) >= 1
        # The TYPE 3 + TYPE 1 match should be classified
        match = result.matches[0]
        assert match.classification == "CROSS-VALIDATED SINGULAR"

    def test_no_within_run_matches(self):
        """Claims from the same run should never match each other."""
        runs = [
            RunData("run_a", Path("."), "", "pharmacology", False, [
                {"statement": "identical claim", "mechanism": "", "type": 1,
                 "confidence": 0.9, "overlap_count": 3, "models": ["a"]},
                {"statement": "identical claim", "mechanism": "", "type": 1,
                 "confidence": 0.9, "overlap_count": 3, "models": ["b"]},
            ]),
            RunData("run_b", Path("."), "", "pharmacology", False, [
                {"statement": "completely different topic", "mechanism": "", "type": 1,
                 "confidence": 0.5, "overlap_count": 1, "models": ["c"]},
            ]),
        ]
        result = cross_match(runs, threshold=0.95)
        # The two identical claims in run_a should NOT appear as matches
        for m in result.matches:
            assert not (m.run_a == m.run_b)

    def test_single_run_returns_empty(self):
        runs = [RunData("run_a", Path("."), "", "pharm", False, [
            {"statement": "test", "mechanism": "", "type": 1}
        ])]
        result = cross_match(runs)
        assert len(result.matches) == 0


# ── Loader Integration Tests ──

class TestLoader:
    def test_load_run_from_disk(self, tmp_path):
        """Test loading a run with repr-format claims."""
        run_dir = tmp_path / "evo_20260211_test_pharmacology"
        run_dir.mkdir()

        s2_data = {
            "synthesized_claims": [THC_CLAIM_REPR],
            "conflicts": [],
            "total_rounds": 0,
            "snapshots": [],
        }
        (run_dir / "s2_synthesis.json").write_text(json.dumps(s2_data))

        s3_data = {"passed": False, "convergence_score": 0.81}
        (run_dir / "s3_convergence.json").write_text(json.dumps(s3_data))

        pkg = {"question": "THC wellbeing?", "outcome": "S3_FAILED", "convergence_report": {"s3_gate": {"passed": False}}}
        (run_dir / "full_package.json").write_text(json.dumps(pkg))

        run = load_run(run_dir)
        assert run.session_id == "evo_20260211_test_pharmacology"
        assert run.domain == "pharmacology"
        assert run.s3_passed is False
        assert len(run.claims) == 1
        assert run.claims[0]["type"] == 1
        assert "therapeutic index" in run.claims[0]["statement"]

    def test_load_run_dict_format(self, tmp_path):
        """Test loading a run with dict-format claims (future format)."""
        run_dir = tmp_path / "evo_20260212_test_pharmacology"
        run_dir.mkdir()

        s2_data = {
            "synthesized_claims": [
                {"statement": "Test claim", "mechanism": "Test mech", "type": 0,
                 "confidence": 0.9, "overlap_count": 5, "models": ["a", "b"]}
            ],
        }
        (run_dir / "s2_synthesis.json").write_text(json.dumps(s2_data))
        (run_dir / "s3_convergence.json").write_text(json.dumps({"passed": True}))
        (run_dir / "full_package.json").write_text(json.dumps({"question": "Q?"}))

        run = load_run(run_dir)
        assert len(run.claims) == 1
        assert run.claims[0]["type"] == 0
        assert run.s3_passed is True

    def test_find_runs(self, tmp_path):
        """Test scanning directories for runs."""
        for name in ["evo_a", "evo_b", "not_a_run"]:
            d = tmp_path / name
            d.mkdir()
            if name != "not_a_run":
                (d / "s2_synthesis.json").write_text(json.dumps({"synthesized_claims": []}))
                (d / "full_package.json").write_text(json.dumps({"question": "Q?"}))

        runs = find_runs([str(tmp_path)])
        assert len(runs) == 2


# ── Report Tests ──

class TestReport:
    def test_save_report(self, tmp_path):
        result = CrossRunResult(
            runs=[
                RunData("run_a", Path("."), "Q1", "pharm", False, [{"statement": "A", "mechanism": ""}]),
                RunData("run_b", Path("."), "Q2", "neuro", False, [{"statement": "B", "mechanism": ""}]),
            ],
            matches=[],
            structural_patterns={},
            cosine_distribution=[0.3, 0.5, 0.7],
        )
        paths = save_report(result, tmp_path / "output")
        assert Path(paths["json"]).exists()
        assert Path(paths["markdown"]).exists()

        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["stats"]["total_claims"] == 2
        assert len(data["runs_analyzed"]) == 2
