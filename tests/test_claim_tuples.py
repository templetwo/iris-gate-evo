"""
Tests for Claim Tuple Extraction.

Validates:
- Entity canonicalization (synonyms, abbreviations)
- Relation extraction (verb patterns)
- Value normalization (units, parameters)
- Full tuple extraction from claims
- Synonymous claims produce overlapping tuples (THE KEY TEST)
"""

import pytest
from src.parser import Claim
from src.convergence.claim_tuples import (
    ClaimTuple,
    extract_entities,
    extract_relations,
    extract_values,
    extract_tuples,
    _normalize_value,
)


# ── Entity Extraction ──

class TestEntityExtraction:
    def test_abbreviation(self):
        entities = extract_entities("CBD binds VDAC1")
        names = [e[0] for e in entities]
        assert "cbd" in names
        assert "vdac1" in names

    def test_synonym_resolution(self):
        entities = extract_entities("cannabidiol interacts with voltage-dependent anion channel 1")
        names = [e[0] for e in entities]
        assert "cbd" in names
        assert "vdac1" in names

    def test_multi_word_phrase(self):
        entities = extract_entities("reactive oxygen species increase in cancer cells")
        names = [e[0] for e in entities]
        assert "ros" in names
        assert "cancer_cell" in names

    def test_membrane_potential(self):
        entities = extract_entities("mitochondrial membrane potential is depolarized")
        names = [e[0] for e in entities]
        assert "membrane_potential" in names

    def test_unknown_entity_passthrough(self):
        entities = extract_entities("FOXO3 regulates something")
        names = [e[0] for e in entities]
        assert "foxo3" in names

    def test_position_ordering(self):
        entities = extract_entities("CBD activates TRPV1 and TRPV2")
        assert entities[0][1] < entities[1][1] < entities[2][1]

    def test_empty_text(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        assert extract_entities("the quick brown fox") == []


# ── Relation Extraction ──

class TestRelationExtraction:
    def test_binds(self):
        rels = extract_relations("CBD binds to VDAC1")
        names = [r[0] for r in rels]
        assert "binds" in names

    def test_interacts_with(self):
        rels = extract_relations("cannabidiol interacts with the receptor")
        names = [r[0] for r in rels]
        assert "binds" in names  # "interacts with" → "binds"

    def test_increases(self):
        rels = extract_relations("CBD increases membrane permeability")
        names = [r[0] for r in rels]
        assert "increases" in names

    def test_verb_forms(self):
        for text in ["increasing ROS", "increases ROS", "elevated ROS"]:
            rels = extract_relations(text)
            names = [r[0] for r in rels]
            assert "increases" in names, f"Failed for: {text}"

    def test_inhibits_synonyms(self):
        for text in ["inhibits growth", "blocks channel", "suppresses expression"]:
            rels = extract_relations(text)
            names = [r[0] for r in rels]
            assert "inhibits" in names, f"Failed for: {text}"

    def test_synergizes(self):
        rels = extract_relations("calcium influx synergizes with depolarization")
        names = [r[0] for r in rels]
        assert "synergizes" in names

    def test_no_relations(self):
        assert extract_relations("the mitochondria") == []


# ── Value Normalization ──

class TestValueNormalization:
    def test_um_normalization(self):
        v1 = _normalize_value("Kd", "11", "uM")
        v2 = _normalize_value("Kd", "11.0", "μM")   # Greek mu U+03BC
        v3 = _normalize_value("Kd", "~11", "uM")
        v4 = _normalize_value("Kd", "11", "µM")      # Micro sign U+00B5
        assert v1 == v2 == v3 == v4 == "kd=1.10e-05"

    def test_mv_normalization(self):
        v = _normalize_value("psi", "-120", "mV")
        assert "psi" in v
        assert "-1.20e-01" in v

    def test_nm_conversion(self):
        v = _normalize_value("Kd", "500", "nM")
        assert "kd=5.00e-07" == v

    def test_invalid_value(self):
        assert _normalize_value("Kd", "abc", "uM") == ""

    def test_extract_from_text(self):
        values = extract_values("CBD has Kd = 11 uM at the receptor")
        assert len(values) >= 1
        assert "kd=1.10e-05" == values[0][0]

    def test_extract_micro_sign(self):
        """Micro sign (U+00B5) must be handled same as Greek mu (U+03BC)."""
        v1 = extract_values("Kd ~11 µM")   # micro sign
        v2 = extract_values("Kd ~11 μM")   # Greek mu
        v3 = extract_values("Kd ~11 uM")   # ASCII u
        assert len(v1) >= 1
        assert v1[0][0] == v2[0][0] == v3[0][0] == "kd=1.10e-05"

    def test_extract_psi(self):
        values = extract_values("psi = -120 mV in cancer cells")
        assert len(values) >= 1
        assert "psi" in values[0][0]


# ── Full Tuple Extraction ──

class TestTupleExtraction:
    def test_cbd_vdac1_binding(self):
        claim = Claim(
            statement="CBD binds VDAC1 with Kd = 11 uM",
            type=1, confidence=0.85,
        )
        tuples = extract_tuples(claim)
        assert len(tuples) >= 1
        # Should have a binding tuple with cbd and vdac1
        subjects = {t.subject for t in tuples}
        objects = {t.object for t in tuples}
        assert "cbd" in subjects
        assert "vdac1" in objects

    def test_synonymous_claims_overlap(self):
        """THE KEY TEST: same science, different words → overlapping tuples."""
        c1 = Claim(statement="CBD binds VDAC1 with Kd = 11 uM", type=1, confidence=0.85)
        c2 = Claim(statement="cannabidiol interacts with VDAC1, Kd ~11 μM", type=1, confidence=0.85)
        t1 = extract_tuples(c1)
        t2 = extract_tuples(c2)
        overlap = t1 & t2
        assert len(overlap) >= 1, f"No overlap: {t1} vs {t2}"

    def test_ros_claim(self):
        claim = Claim(
            statement="CBD-induced ROS amplification exceeds the cancer baseline",
            type=1, confidence=0.7,
            mechanism="CBD increases reactive oxygen species production",
        )
        tuples = extract_tuples(claim)
        assert len(tuples) >= 1

    def test_mechanism_enriches(self):
        c_no_mech = Claim(statement="CBD binds VDAC1", type=1, confidence=0.8)
        c_with_mech = Claim(
            statement="CBD binds VDAC1", type=1, confidence=0.8,
            mechanism="alters gating state, increasing membrane permeability",
        )
        t1 = extract_tuples(c_no_mech)
        t2 = extract_tuples(c_with_mech)
        assert len(t2) >= len(t1)

    def test_empty_claim(self):
        claim = Claim(statement="", type=1, confidence=0.5)
        assert extract_tuples(claim) == set()

    def test_cooccurrence_fallback(self):
        """Claims with entities but no relations → co-occurrence tuples."""
        claim = Claim(statement="CBD and VDAC1 in cancer cells", type=2, confidence=0.5)
        tuples = extract_tuples(claim)
        assert len(tuples) >= 1
        assert any(t.relation == "associated" for t in tuples)

    def test_membrane_potential_claim(self):
        claim = Claim(
            statement="Cancer cells have depolarized mitochondrial membrane potential (-120mV vs -180mV healthy)",
            type=0, confidence=0.90,
        )
        tuples = extract_tuples(claim)
        assert len(tuples) >= 1

    def test_trpv1_synergy(self):
        claim = Claim(
            statement="TRPV1-mediated calcium influx synergizes with VDAC1-mediated depolarization",
            type=2, confidence=0.6,
        )
        tuples = extract_tuples(claim)
        assert len(tuples) >= 1

    def test_tuple_hashable(self):
        """ClaimTuples must be hashable for set operations."""
        t = ClaimTuple("cbd", "binds", "vdac1", "kd=1.10e-05")
        s = {t}
        assert t in s

    def test_identical_text_identical_tuples(self):
        c1 = Claim(statement="CBD binds VDAC1 with Kd = 11 uM", type=1, confidence=0.8)
        c2 = Claim(statement="CBD binds VDAC1 with Kd = 11 uM", type=1, confidence=0.9)
        assert extract_tuples(c1) == extract_tuples(c2)
