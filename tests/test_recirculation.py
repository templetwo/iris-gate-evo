"""
Tests for S3 Recirculation — feeding converged claims back through S1→S2→S3.

Validates:
- Recirculation context construction from converged claims
- Compiled prompt enrichment for subsequent cycles
- Edge cases (no TYPE 0/1 claims, empty parsed, deduplication)
"""

from dataclasses import dataclass, field

import pytest

from src.stages.stages import (
    build_recirculation_context,
    enrich_compiled_for_recirculation,
)
from src.parser import Claim, ParsedResponse


# ── Fixtures ──

def _make_parsed(claims_data: list[tuple]) -> list[ParsedResponse]:
    """Build ParsedResponse list from (statement, type, confidence, mechanism) tuples."""
    parsed = []
    for i, model_claims in enumerate(claims_data):
        claims = [
            Claim(
                statement=c[0],
                type=c[1],
                confidence=c[2],
                mechanism=c[3] if len(c) > 3 else "",
            )
            for c in model_claims
        ]
        parsed.append(ParsedResponse(
            model=f"model_{i}",
            claims=claims,
            raw=f"raw response from model_{i}",
        ))
    return parsed


def _make_s2_result(parsed):
    """Build minimal s2_result dict."""
    from src.convergence.convergence import ConvergenceSnapshot
    snapshot = ConvergenceSnapshot(
        round_num=10,
        jaccard=0.25,
        cosine=0.88,
        jsd=0.12,
        kappa=0.65,
        type_01_ratio=0.66,
        type_distribution={0: 10, 1: 15, 2: 8, 3: 2},
        n_claims_per_model=[5, 5, 5, 5, 5],
    )
    return {
        "stage": "S2",
        "parsed": parsed,
        "snapshots": [snapshot],
        "rounds": [],
        "total_rounds": 10,
        "total_calls": 50,
        "early_stopped": False,
    }


def _make_s3_result(passed=False, type_01_ratio=0.66):
    return {
        "stage": "S3",
        "passed": passed,
        "type_01_ratio": type_01_ratio,
        "type_distribution": {0: 10, 1: 15, 2: 8, 3: 2},
        "convergence_score": 0.88,
    }


SAMPLE_COMPILED = {
    "session_id": "evo_pharmacology_20260209",
    "question": "How does CBD affect VDAC1?",
    "prompt": (
        "Some prior content here.\n\n"
        "─── SECTION 1: DECOMPOSITION ───\n"
        "Decompose the question.\n\n"
        "─── SECTION 2: CLAIMS ───\n"
        "State your claims."
    ),
    "domains": ["pharmacology"],
    "priors": [],
    "models": {},
    "token_budgets": {"S1": 800, "S2_start": 800, "S2_end": 700, "S3": 600},
    "cross_domain_flag": False,
}


# ── build_recirculation_context ──

class TestBuildRecirculationContext:
    def test_extracts_type_01_claims(self):
        parsed = _make_parsed([
            [
                ("CBD binds VDAC1 with Kd=11uM", 0, 0.95, "Direct binding"),
                ("CBD induces ROS in cancer cells", 1, 0.85, "Oxidative stress"),
                ("CBD may affect p53 signaling", 2, 0.5, "Speculative"),
            ],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)

        assert "PRIOR CONSENSUS" in context
        assert "CBD binds VDAC1" in context
        assert "CBD induces ROS" in context
        assert "p53" not in context  # TYPE 2 excluded

    def test_excludes_low_confidence(self):
        parsed = _make_parsed([
            [
                ("Weak claim about something", 1, 0.5, "Low confidence"),
                ("Strong claim about VDAC1", 1, 0.9, "High confidence"),
            ],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)

        assert "Strong claim" in context
        assert "Weak claim" not in context  # Below 0.7 threshold

    def test_deduplicates_identical_claims(self):
        parsed = _make_parsed([
            [("CBD binds VDAC1 with high affinity", 0, 0.9, "")],
            [("CBD binds VDAC1 with high affinity", 0, 0.85, "")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)
        assert context.count("CBD binds VDAC1") == 1

    def test_deduplicates_synonymous_claims(self):
        """THE test — two claims saying the same thing in different words should dedup."""
        parsed = _make_parsed([
            [("CBD binds VDAC1 with high affinity", 0, 0.9, "Direct binding interaction")],
            [("Cannabidiol interacts with voltage-dependent anion channel 1", 0, 0.85, "Binding")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)

        # Both express (cbd, binds, vdac1) — only one should appear
        lines = [l for l in context.split("\n") if l.strip().startswith("[ESTABLISHED]") or l.strip().startswith("[REPLICATED]")]
        assert len(lines) == 1, f"Expected 1 claim after dedup, got {len(lines)}: {lines}"

    def test_dedup_keeps_highest_confidence(self):
        parsed = _make_parsed([
            [("CBD binds VDAC1", 0, 0.75, "")],
            [("Cannabidiol interacts with voltage-dependent anion channel 1", 0, 0.95, "")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)
        # Should keep the first claim but upgrade confidence to 0.95
        assert "0.95" in context

    def test_empty_when_no_type_01(self):
        parsed = _make_parsed([
            [("Speculative claim", 2, 0.8, "Guess")],
            [("Another guess", 3, 0.6, "Wild")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)
        assert context == ""

    def test_includes_type_distribution_note(self):
        parsed = _make_parsed([
            [("Solid claim", 0, 0.95, "Evidence-backed")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result(type_01_ratio=0.66)

        context = build_recirculation_context(s2, s3)
        assert "66%" in context
        assert "90%" in context

    def test_caps_at_12_claims(self):
        claims = [(f"Claim number {i}", 0, 0.95, f"Mechanism {i}") for i in range(20)]
        parsed = _make_parsed([claims])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)

        # Should have at most 12 [ESTABLISHED] entries
        assert context.count("[ESTABLISHED]") <= 12

    def test_mechanism_included_when_present(self):
        parsed = _make_parsed([
            [("CBD depolarizes mitochondria", 1, 0.9, "Via VDAC1 channel opening")],
        ])
        s2 = _make_s2_result(parsed)
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)
        assert "Via VDAC1 channel opening" in context

    def test_empty_parsed_returns_empty(self):
        s2 = _make_s2_result([])
        s3 = _make_s3_result()

        context = build_recirculation_context(s2, s3)
        assert context == ""


# ── enrich_compiled_for_recirculation ──

class TestEnrichCompiled:
    def test_injects_before_section_1(self):
        context = "PRIOR CONSENSUS\nSome claims here"
        enriched = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=2)

        prompt = enriched["prompt"]
        consensus_pos = prompt.index("PRIOR CONSENSUS")
        section1_pos = prompt.index("SECTION 1: DECOMPOSITION")
        assert consensus_pos < section1_pos

    def test_preserves_original_priors(self):
        context = "PRIOR CONSENSUS\nClaims"
        enriched = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=2)

        assert "Some prior content here" in enriched["prompt"]

    def test_session_id_tracks_cycle(self):
        context = "PRIOR CONSENSUS"
        enriched = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=2)
        assert "cycle2" in enriched["session_id"]

    def test_cycle_number_updates(self):
        context = "PRIOR CONSENSUS"
        enriched1 = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=2)
        enriched2 = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=3)

        assert "cycle2" in enriched1["session_id"]
        assert "cycle3" in enriched2["session_id"]

    def test_empty_context_returns_original(self):
        enriched = enrich_compiled_for_recirculation(SAMPLE_COMPILED, "", cycle_num=2)
        assert enriched is SAMPLE_COMPILED

    def test_enriches_from_original_not_stacked(self):
        """Each recirculation should enrich from ORIGINAL compiled, not stack."""
        context1 = "CYCLE 1 CONSENSUS"
        context2 = "CYCLE 2 CONSENSUS"

        enriched1 = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context1, cycle_num=2)
        enriched2 = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context2, cycle_num=3)

        # enriched2 should NOT contain context1
        assert "CYCLE 1 CONSENSUS" not in enriched2["prompt"]
        assert "CYCLE 2 CONSENSUS" in enriched2["prompt"]

    def test_fallback_when_no_section_marker(self):
        compiled_no_marker = {**SAMPLE_COMPILED, "prompt": "Just a plain prompt"}
        context = "PRIOR CONSENSUS"
        enriched = enrich_compiled_for_recirculation(compiled_no_marker, context, cycle_num=2)

        assert "PRIOR CONSENSUS" in enriched["prompt"]
        assert "Just a plain prompt" in enriched["prompt"]

    def test_other_fields_preserved(self):
        context = "PRIOR CONSENSUS"
        enriched = enrich_compiled_for_recirculation(SAMPLE_COMPILED, context, cycle_num=2)

        assert enriched["question"] == SAMPLE_COMPILED["question"]
        assert enriched["domains"] == SAMPLE_COMPILED["domains"]
        assert enriched["token_budgets"] == SAMPLE_COMPILED["token_budgets"]
