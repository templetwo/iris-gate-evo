"""
Tests for Convergence Engine.

Validates metrics are computed on parsed CLAIMS, not raw text.
Tests known inputs with expected outputs for each metric.
Tests the early-stop logic: AND not OR.
"""

import pytest
import numpy as np
from src.parser import Claim
from src.convergence.convergence import (
    compute,
    delta,
    ConvergenceSnapshot,
    _jaccard_similarity,
    _tokenize_claim,
    _model_type_distribution,
    _compute_type_distribution,
)


def _make_claim(statement: str, type: int = 1, confidence: float = 0.8) -> Claim:
    return Claim(statement=statement, type=type, confidence=confidence)


# ── Jaccard ──

class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        j = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(j - 0.5) < 0.01  # 2/4

    def test_empty_sets(self):
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_tokenize_claim(self):
        c = _make_claim("CBD binds to VDAC1 receptor")
        tokens = _tokenize_claim(c)
        assert "cbd" in tokens
        assert "vdac1" in tokens
        assert "to" not in tokens  # Too short


# ── TYPE Distribution ──

class TestTypeDistribution:
    def test_all_type_1(self):
        claims = [_make_claim("a", type=1), _make_claim("b", type=1)]
        dist = _model_type_distribution(claims)
        assert dist[1] == 1.0
        assert dist[0] == 0.0

    def test_mixed_types(self):
        claims = [
            _make_claim("a", type=0),
            _make_claim("b", type=1),
            _make_claim("c", type=2),
            _make_claim("d", type=3),
        ]
        dist = _model_type_distribution(claims)
        assert all(abs(d - 0.25) < 0.01 for d in dist)

    def test_empty_claims_uniform(self):
        dist = _model_type_distribution([])
        assert all(abs(d - 0.25) < 0.01 for d in dist)

    def test_overall_distribution(self):
        claims = [_make_claim("a", type=0), _make_claim("b", type=1)]
        dist = _compute_type_distribution(claims)
        assert dist[0] == 0.5
        assert dist[1] == 0.5
        assert dist[2] == 0.0
        assert dist[3] == 0.0


# ── ConvergenceSnapshot ──

class TestConvergenceCompute:
    def test_identical_claims_high_jaccard(self):
        """All models making the same claims should give high Jaccard."""
        claims = [_make_claim("CBD binds VDAC1 causing membrane permeability", type=1)]
        claims_per_model = [claims] * 5  # All same
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.jaccard == 1.0

    def test_different_claims_low_jaccard(self):
        """Completely different claims should give low Jaccard."""
        claims_per_model = [
            [_make_claim("alpha beta gamma")],
            [_make_claim("delta epsilon zeta")],
            [_make_claim("eta theta iota")],
            [_make_claim("kappa lambda mu")],
            [_make_claim("nu xi omicron")],
        ]
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.jaccard < 0.1

    def test_type_01_ratio_correct(self):
        claims_per_model = [
            [_make_claim("a", type=0), _make_claim("b", type=1)],
            [_make_claim("c", type=1), _make_claim("d", type=2)],
        ]
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        # 3 out of 4 are type 0 or 1
        assert abs(snap.type_01_ratio - 0.75) < 0.01

    def test_all_type_0_gives_ratio_1(self):
        claims_per_model = [
            [_make_claim("a", type=0)] * 3,
            [_make_claim("b", type=0)] * 3,
        ]
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.type_01_ratio == 1.0

    def test_all_type_3_gives_ratio_0(self):
        claims_per_model = [
            [_make_claim("a", type=3)] * 3,
            [_make_claim("b", type=3)] * 3,
        ]
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.type_01_ratio == 0.0

    def test_jsd_identical_distributions(self):
        """Same TYPE distribution across all models = JSD near 0."""
        claims = [_make_claim("a", type=1), _make_claim("b", type=1)]
        claims_per_model = [claims] * 5
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.jsd < 0.01

    def test_n_claims_tracked(self):
        claims_per_model = [
            [_make_claim("a")] * 3,
            [_make_claim("b")] * 2,
            [_make_claim("c")] * 4,
        ]
        snap = compute(claims_per_model, round_num=0, use_embeddings=False)
        assert snap.n_claims_per_model == [3, 2, 4]


# ── Delta ──

class TestDelta:
    def test_identical_snapshots_zero_delta(self):
        snap = ConvergenceSnapshot(
            round_num=0, jaccard=0.8, cosine=0.7, jsd=0.1,
            kappa=0.6, type_distribution={}, type_01_ratio=0.8,
            n_claims_per_model=[3, 3, 3],
        )
        assert delta(snap, snap) == 0.0

    def test_large_change_large_delta(self):
        snap1 = ConvergenceSnapshot(
            round_num=0, jaccard=0.3, cosine=0.3, jsd=0.5,
            kappa=0.2, type_distribution={}, type_01_ratio=0.4,
            n_claims_per_model=[3, 3, 3],
        )
        snap2 = ConvergenceSnapshot(
            round_num=1, jaccard=0.9, cosine=0.9, jsd=0.1,
            kappa=0.8, type_distribution={}, type_01_ratio=0.9,
            n_claims_per_model=[3, 3, 3],
        )
        d = delta(snap2, snap1)
        assert d > 0.3  # Should be a large delta

    def test_small_change_small_delta(self):
        snap1 = ConvergenceSnapshot(
            round_num=0, jaccard=0.80, cosine=0.75, jsd=0.10,
            kappa=0.6, type_distribution={}, type_01_ratio=0.8,
            n_claims_per_model=[3, 3, 3],
        )
        snap2 = ConvergenceSnapshot(
            round_num=1, jaccard=0.805, cosine=0.755, jsd=0.095,
            kappa=0.6, type_distribution={}, type_01_ratio=0.8,
            n_claims_per_model=[3, 3, 3],
        )
        d = delta(snap2, snap1)
        assert d < 0.01


# ── Early-Stop Logic (AND, not OR) ──

class TestEarlyStopLogic:
    """These test the LOGIC of early-stop conditions,
    not the full stage runner (which needs API calls)."""

    def test_stable_delta_but_low_type01_should_NOT_stop(self):
        """Delta < 1% but TYPE 0/1 < 80% — should NOT trigger early stop.
        System is stable on garbage (all TYPE 3)."""
        # This is the key test: AND, not OR
        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.9, cosine=0.9, jsd=0.01,
            kappa=0.9, type_distribution={0: 0.1, 1: 0.1, 2: 0.3, 3: 0.5},
            type_01_ratio=0.20,  # Only 20% TYPE 0/1 — bad
            n_claims_per_model=[3, 3, 3],
        )

        delta_stable = True  # Pretend delta < 1%
        type_stable = snap.type_01_ratio >= 0.80

        # AND gate
        should_stop = delta_stable and type_stable
        assert should_stop is False, "Should NOT stop — stable on garbage"

    def test_high_type01_but_unstable_delta_should_NOT_stop(self):
        """TYPE 0/1 >= 80% but delta > 1% — should NOT trigger."""
        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.5, cosine=0.5, jsd=0.3,
            kappa=0.3, type_distribution={0: 0.4, 1: 0.45, 2: 0.1, 3: 0.05},
            type_01_ratio=0.85,  # Good TYPE ratio
            n_claims_per_model=[3, 3, 3],
        )

        delta_stable = False  # Delta still moving
        type_stable = snap.type_01_ratio >= 0.80

        should_stop = delta_stable and type_stable
        assert should_stop is False, "Should NOT stop — still converging"

    def test_both_conditions_met_should_stop(self):
        """Delta < 1% AND TYPE 0/1 >= 80% — should trigger."""
        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.9, cosine=0.9, jsd=0.02,
            kappa=0.8, type_distribution={0: 0.4, 1: 0.45, 2: 0.1, 3: 0.05},
            type_01_ratio=0.85,
            n_claims_per_model=[3, 3, 3],
        )

        delta_stable = True
        type_stable = snap.type_01_ratio >= 0.80

        should_stop = delta_stable and type_stable
        assert should_stop is True

    def test_consecutive_requirement(self):
        """Must be stable for 3 CONSECUTIVE rounds, not just 3 total."""
        # Simulate: stable, stable, unstable, stable, stable, stable
        stable_rounds = [True, True, False, True, True, True]

        consecutive = 0
        triggered_at = None
        for i, is_stable in enumerate(stable_rounds):
            if is_stable:
                consecutive += 1
            else:
                consecutive = 0  # RESET

            if consecutive >= 3:
                triggered_at = i
                break

        # Should trigger at index 5 (the 6th round), not index 2
        assert triggered_at == 5


# ── S3 Gate ──

class TestS3Gate:
    def test_pass_conditions(self):
        """Cosine > 0.85 and TYPE 0/1 >= 90% needed to pass."""
        from src.stages.stages import S3_CONVERGENCE_THRESHOLD, S3_TYPE01_THRESHOLD

        assert 0.90 > S3_CONVERGENCE_THRESHOLD
        assert 0.92 >= S3_TYPE01_THRESHOLD

    def test_low_cosine_blocks_gate(self):
        """High TYPE ratio but low cosine should fail S3."""
        from src.stages.stages import run_s3_gate

        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.70, cosine=0.80, jsd=0.02,
            kappa=0.8, type_distribution={0: 0.5, 1: 0.45, 2: 0.05, 3: 0.0},
            type_01_ratio=0.95,
            n_claims_per_model=[3, 3, 3],
        )

        s2_result = {
            "snapshots": [snap],
            "parsed": [],
        }

        gate = run_s3_gate(s2_result)
        assert gate["passed"] is False
        assert gate["convergence_pass"] is False
        assert gate["type_pass"] is True

    def test_low_jaccard_floor_blocks_gate(self):
        """High cosine but Jaccard below floor (0.10) should fail."""
        from src.stages.stages import run_s3_gate

        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.05, cosine=0.95, jsd=0.02,
            kappa=0.8, type_distribution={0: 0.5, 1: 0.45, 2: 0.05, 3: 0.0},
            type_01_ratio=0.95,
            n_claims_per_model=[3, 3, 3],
        )

        s2_result = {
            "snapshots": [snap],
            "parsed": [],
        }

        gate = run_s3_gate(s2_result)
        assert gate["passed"] is False
        assert gate["convergence_pass"] is False

    def test_type_fail_blocks_gate(self):
        """High convergence (cosine > 0.85) but low TYPE ratio should fail S3."""
        from src.stages.stages import run_s3_gate

        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.90, cosine=0.90, jsd=0.02,
            kappa=0.8, type_distribution={0: 0.3, 1: 0.3, 2: 0.2, 3: 0.2},
            type_01_ratio=0.60,
            n_claims_per_model=[3, 3, 3],
        )

        s2_result = {
            "snapshots": [snap],
            "parsed": [],
        }

        gate = run_s3_gate(s2_result)
        assert gate["passed"] is False
        assert gate["convergence_pass"] is True
        assert gate["type_pass"] is False

    def test_both_pass(self):
        from src.stages.stages import run_s3_gate

        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.90, cosine=0.9, jsd=0.02,
            kappa=0.8, type_distribution={0: 0.5, 1: 0.45, 2: 0.05, 3: 0.0},
            type_01_ratio=0.95,
            n_claims_per_model=[3, 3, 3],
        )

        s2_result = {
            "snapshots": [snap],
            "parsed": [],
        }

        gate = run_s3_gate(s2_result)
        assert gate["passed"] is True

    def test_failure_includes_recommendation(self):
        from src.stages.stages import run_s3_gate

        snap = ConvergenceSnapshot(
            round_num=5, jaccard=0.50, cosine=0.5, jsd=0.3,
            kappa=0.3, type_distribution={0: 0.2, 1: 0.2, 2: 0.3, 3: 0.3},
            type_01_ratio=0.40,
            n_claims_per_model=[3, 3, 3],
        )

        s2_result = {
            "snapshots": [snap],
            "parsed": [],
        }

        gate = run_s3_gate(s2_result)
        assert gate["passed"] is False
        assert "FAILED" in gate["recommendation"]
        assert "disagreement" in gate["recommendation"].lower()


# ── Tuple-Based Jaccard ──

class TestTupleBasedJaccard:
    """Verify that tuple extraction sharpens Jaccard for scientific claims."""

    def test_synonymous_claims_high_jaccard(self):
        """Same science, different words → high Jaccard via tuples."""
        model_a = [_make_claim("CBD binds VDAC1 with Kd = 11 uM")]
        model_b = [_make_claim("cannabidiol interacts with VDAC1, Kd ~11 μM")]
        snap = compute([model_a, model_b], round_num=0, use_embeddings=False)
        # With token bags this was ~0.2. With tuples it should be much higher.
        assert snap.jaccard > 0.5

    def test_identical_claims_still_perfect(self):
        """Backward compat: identical claims still give 1.0."""
        claims = [_make_claim("CBD binds VDAC1 with Kd = 11 uM")]
        snap = compute([claims, claims], round_num=0, use_embeddings=False)
        assert snap.jaccard == 1.0

    def test_unrelated_claims_still_low(self):
        """Completely different topics → low Jaccard even with tuples."""
        model_a = [_make_claim("CBD binds VDAC1 receptor")]
        model_b = [_make_claim("p53 activates BAX in apoptosis")]
        snap = compute([model_a, model_b], round_num=0, use_embeddings=False)
        assert snap.jaccard < 0.5

    def test_non_scientific_text_falls_back(self):
        """Non-scientific text with no entities → token fallback."""
        model_a = [_make_claim("alpha beta gamma delta")]
        model_b = [_make_claim("epsilon zeta eta theta")]
        snap = compute([model_a, model_b], round_num=0, use_embeddings=False)
        # Falls back to token-based → 0.0 (disjoint)
        assert snap.jaccard == 0.0
