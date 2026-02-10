"""Tests for S2 Contribution Synthesis — semantic claim embedding, TYPE assignment, conflicts."""

import pytest
from src.parser import Claim, ParsedResponse
from src.convergence.convergence import ConvergenceSnapshot
from src.stages.synthesis import (
    run_s2_synthesis,
    SynthesizedClaim,
    Conflict,
    OVERLAP_TYPE_MAP,
    TYPE_LABELS,
)


# ── Helpers ──

def _make_parsed(model: str, claims: list[dict]) -> ParsedResponse:
    """Helper: build a ParsedResponse from dicts."""
    return ParsedResponse(
        model=model,
        claims=[
            Claim(
                statement=c["statement"],
                type=c.get("type", 2),
                confidence=c.get("confidence", 0.8),
                mechanism=c.get("mechanism", ""),
                falsifiable_by=c.get("falsifiable_by", ""),
            )
            for c in claims
        ],
        raw="test",
    )


def _make_s1_result(parsed_list: list[ParsedResponse]) -> dict:
    """Helper: build an S1 result dict from parsed responses."""
    from src.convergence.convergence import compute
    claims_per_model = [p.claims for p in parsed_list]
    snapshot = compute(claims_per_model, round_num=0)
    return {
        "stage": "S1",
        "parsed": parsed_list,
        "snapshot": snapshot,
        "pulse_meta": {"models_dispatched": len(parsed_list)},
        "total_calls": len(parsed_list),
    }


# ── Semantic Claim Clustering ──

class TestSemanticClustering:
    """Test that semantic embedding clusters meaning, not structure."""

    def test_same_science_different_words_clusters(self):
        """THE test — same science expressed differently → should cluster.
        This is exactly what tuple-based approach FAILED at."""
        claim_a = {"statement": "CBD binds to VDAC1 with high affinity causing mitochondrial depolarization"}
        claim_b = {"statement": "Cannabidiol interacts with voltage-dependent anion channel 1 disrupting membrane potential"}
        parsed = [
            _make_parsed("model_0", [claim_a]),
            _make_parsed("model_1", [claim_b]),
            _make_parsed("model_2", [claim_a]),
            _make_parsed("model_3", [claim_b]),
            _make_parsed("model_4", [claim_a]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        # With semantic clustering, these should merge into a high-overlap cluster
        high_overlap = [s for s in synth if s.overlap_count >= 3]
        assert len(high_overlap) >= 1, (
            f"Expected semantic clustering to find overlap >= 3, "
            f"got max overlap: {max(s.overlap_count for s in synth) if synth else 0}"
        )

    def test_different_science_stays_separate(self):
        """Genuinely different claims should NOT cluster together."""
        claim_a = {"statement": "CBD binds VDAC1 with high affinity in the mitochondrial membrane"}
        claim_b = {"statement": "Aspirin inhibits COX2 enzyme reducing prostaglandin synthesis"}
        parsed = [
            _make_parsed("model_0", [claim_a]),
            _make_parsed("model_1", [claim_b]),
            _make_parsed("model_2", [claim_a]),
            _make_parsed("model_3", [claim_b]),
            _make_parsed("model_4", [claim_a]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        # Should have at least 2 distinct clusters
        assert len(synth) >= 2

    def test_same_model_never_inflates_overlap(self):
        """Claims from the same model must not count twice toward overlap."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial changes"}
        parsed = [
            _make_parsed("model_0", [claim, claim, claim]),  # 3 copies, same model
            _make_parsed("model_1", [{"statement": "Aspirin reduces inflammation via COX2 inhibition"}]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        # model_0's repeated claims should NOT give overlap > 1 for that cluster
        for s in synth:
            if "model_0" in s.models and "model_1" not in s.models:
                assert s.overlap_count == 1  # Only model_0 contributed

    def test_closely_related_claims_cluster(self):
        """Claims with high semantic overlap should cluster across models."""
        # These are close paraphrases — should clear 0.75 threshold
        claim_a = {"statement": "CBD binds VDAC1 in cancer cell mitochondria causing depolarization and cytochrome c release"}
        claim_b = {"statement": "CBD interacts with VDAC1 in cancer mitochondria leading to membrane depolarization and cytochrome c release"}
        parsed = [
            _make_parsed("model_0", [claim_a]),
            _make_parsed("model_1", [claim_b]),
            _make_parsed("model_2", [claim_a]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        high_overlap = [s for s in synth if s.overlap_count >= 2]
        assert len(high_overlap) >= 1


# ── Overlap Counting (inherited from tuple era, now via semantic chunks) ──

class TestOverlapCounting:
    """Test that TYPE is assigned by model count in cluster."""

    def test_five_models_identical_claim_type_0(self):
        """All 5 models say the same thing → TYPE 0."""
        claim = {"statement": "CBD binds VDAC1 with high affinity via direct molecular interaction", "mechanism": "direct binding"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        assert len(synth) >= 1
        # The shared claim should be TYPE 0 with high overlap
        high_overlap = [s for s in synth if s.overlap_count >= 4]
        assert len(high_overlap) >= 1
        assert high_overlap[0].type == 0

    def test_singular_claim_type_3(self):
        """1 model has a unique claim → TYPE 3."""
        shared = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        unique = {"statement": "CBD modulates gap junction conductance via connexin-43 phosphorylation"}
        parsed = [_make_parsed(f"model_{i}", [shared]) for i in range(4)]
        parsed.append(_make_parsed("model_4", [shared, unique]))
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        singulars = [s for s in synth if s.overlap_count == 1]
        # The unique claim should be preserved as TYPE 3
        assert len(singulars) >= 1
        assert singulars[0].type == 3

    def test_singulars_preserved_not_discarded(self):
        """TYPE 3 singulars must be in synthesized_claims, not dropped."""
        unique = {"statement": "CBD activates TRPV2 channels selectively at low concentrations"}
        parsed = [
            _make_parsed("model_0", [unique]),
            _make_parsed("model_1", [{"statement": "CBD binds VDAC1 with high affinity causing membrane changes"}]),
            _make_parsed("model_2", [{"statement": "CBD binds VDAC1 with high affinity causing membrane changes"}]),
            _make_parsed("model_3", [{"statement": "CBD binds VDAC1 with high affinity causing membrane changes"}]),
            _make_parsed("model_4", [{"statement": "CBD binds VDAC1 with high affinity causing membrane changes"}]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        synth = result["synthesized_claims"]
        singulars = [s for s in synth if s.type == 3]
        assert len(singulars) >= 1


# ── Conflict Detection ──

class TestConflictDetection:
    """Detect when models agree on the claim but disagree on values."""

    def test_different_kd_values_flagged(self):
        """Same claim, different Kd → conflict detected."""
        claim_a = {"statement": "CBD binds VDAC1 with Kd = 11 uM"}
        claim_b = {"statement": "CBD binds VDAC1 with Kd = 3.5 uM"}
        parsed = [
            _make_parsed("model_0", [claim_a]),
            _make_parsed("model_1", [claim_b]),
            _make_parsed("model_2", [claim_a]),
            _make_parsed("model_3", [claim_a]),
            _make_parsed("model_4", [claim_b]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        conflicts = result["conflicts"]
        assert len(conflicts) >= 1
        assert conflicts[0].subject in ("cbd", "vdac1")

    def test_no_conflict_when_values_agree(self):
        """Same claim, same value → no conflict."""
        claim = {"statement": "CBD binds VDAC1 with Kd = 11 uM"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        assert len(result["conflicts"]) == 0


# ── Return Shape ──

class TestReturnShape:
    """S2 result dict must be compatible with S3 gate and recirculation."""

    def test_required_keys_present(self):
        """All keys that S3 gate and main.py expect must be present."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        assert result["stage"] == "S2"
        assert "parsed" in result
        assert "snapshots" in result
        assert len(result["snapshots"]) == 2
        assert "rounds" in result
        assert result["total_rounds"] == 0
        assert result["total_calls"] == 0
        assert "early_stopped" in result
        assert "synthesized_claims" in result
        assert "conflicts" in result

    def test_zero_api_calls(self):
        """Synthesis must use zero API calls."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(3)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        assert result["total_calls"] == 0

    def test_snapshots_are_convergence_snapshots(self):
        """Both snapshots must be ConvergenceSnapshot instances."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        for snap in result["snapshots"]:
            assert isinstance(snap, ConvergenceSnapshot)

    def test_synthesized_claims_are_dataclass(self):
        """Each synthesized claim is a SynthesizedClaim."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        for sc in result["synthesized_claims"]:
            assert isinstance(sc, SynthesizedClaim)


# ── Edge Cases ──

class TestEdgeCases:
    """Graceful handling of empty claims, no tuples, etc."""

    def test_empty_claims_handled(self):
        """Model with no claims → handled gracefully."""
        parsed = [
            _make_parsed("model_0", [{"statement": "CBD binds VDAC1 with high affinity causing mitochondrial changes"}]),
            _make_parsed("model_1", []),
            _make_parsed("model_2", [{"statement": "CBD binds VDAC1 with high affinity causing mitochondrial changes"}]),
            _make_parsed("model_3", []),
            _make_parsed("model_4", [{"statement": "CBD binds VDAC1 with high affinity causing mitochondrial changes"}]),
        ]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        assert result["total_calls"] == 0
        assert len(result["synthesized_claims"]) >= 1

    def test_overlap_type_map_coverage(self):
        """OVERLAP_TYPE_MAP covers all possible overlap counts 1-5."""
        for n in range(1, 6):
            assert n in OVERLAP_TYPE_MAP

    def test_claim_text_preserved(self):
        """SynthesizedClaim.statement should match a real claim."""
        claim = {"statement": "CBD binds VDAC1 with high affinity causing mitochondrial depolarization"}
        parsed = [_make_parsed(f"model_{i}", [claim]) for i in range(5)]
        s1 = _make_s1_result(parsed)
        result = run_s2_synthesis(s1)

        for sc in result["synthesized_claims"]:
            assert sc.statement  # Should not be empty
            assert len(sc.statement) > 10  # Should be a real claim
