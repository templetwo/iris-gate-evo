"""
Tests for C0 — The Compiler

All offline. No API calls. Validates domain detection, prior loading,
prior selection, scaffold construction, and full compilation.
"""

import json
from pathlib import Path

import pytest

from src.compiler.compiler import (
    compile,
    detect_domains,
    format_priors_block,
    load_priors,
    select_relevant_priors,
    build_scaffold,
    PRIORS_DIR,
)


# The test question from the spec
CBD_QUESTION = (
    "What are the mechanisms by which CBD induces selective cytotoxicity "
    "in cancer cells while sparing healthy cells, with specific reference "
    "to VDAC1-mediated mitochondrial membrane potential disruption?"
)

KURAMOTO_QUESTION = (
    "How does Kuramoto oscillator coupling strength K affect phase "
    "synchronization and entropy in neural populations?"
)

VAGUE_QUESTION = "What causes cancer?"


class TestDomainDetection:
    def test_cbd_detects_pharmacology_and_bioelectric(self):
        domains = detect_domains(CBD_QUESTION)
        assert "pharmacology" in domains
        assert "bioelectric" in domains

    def test_kuramoto_detects_consciousness(self):
        domains = detect_domains(KURAMOTO_QUESTION)
        assert "consciousness" in domains

    def test_vague_question_falls_back_to_general(self):
        domains = detect_domains(VAGUE_QUESTION)
        assert domains == ["general"]

    def test_cross_domain_detection(self):
        q = "How does gap junction coupling affect Kuramoto oscillator coherence in bioelectric networks?"
        domains = detect_domains(q)
        assert len(domains) >= 2

    def test_returns_list(self):
        domains = detect_domains(CBD_QUESTION)
        assert isinstance(domains, list)
        assert all(isinstance(d, str) for d in domains)


class TestPriorLoading:
    def test_priors_dir_exists(self):
        assert PRIORS_DIR.exists()

    def test_pharmacology_priors_load(self):
        priors = load_priors(["pharmacology"])
        assert len(priors) > 0
        assert all("param" in p for p in priors)
        assert all("value" in p for p in priors)
        assert all("unit" in p for p in priors)

    def test_bioelectric_priors_load(self):
        priors = load_priors(["bioelectric"])
        assert len(priors) > 0

    def test_general_priors_are_empty(self):
        priors = load_priors(["general"])
        assert priors == []

    def test_multi_domain_merge(self):
        pharma = load_priors(["pharmacology"])
        bio = load_priors(["bioelectric"])
        merged = load_priors(["pharmacology", "bioelectric"])
        assert len(merged) == len(pharma) + len(bio)

    def test_deduplication(self):
        # Loading same domain twice shouldn't duplicate
        priors = load_priors(["pharmacology", "pharmacology"])
        params = [p["param"] for p in priors]
        assert len(params) == len(set(params))

    def test_missing_domain_doesnt_crash(self):
        priors = load_priors(["nonexistent_domain"])
        assert priors == []

    def test_no_type_3_in_stored_priors(self):
        for domain_file in PRIORS_DIR.glob("*.json"):
            with open(domain_file) as f:
                data = json.load(f)
            for p in data.get("priors", []):
                assert p.get("type", 0) < 3, \
                    f"TYPE 3 prior found in {domain_file.name}: {p['param']}"


class TestPriorSelection:
    def test_cbd_selects_vdac_and_trpv(self):
        priors = load_priors(["pharmacology", "bioelectric"])
        selected = select_relevant_priors(priors, CBD_QUESTION)
        params = [p["param"] for p in selected]
        assert any("VDAC1" in p for p in params), f"Expected VDAC1 prior, got: {params}"
        assert any("TRPV" in p for p in params), f"Expected TRPV prior, got: {params}"

    def test_max_8_priors(self):
        priors = load_priors(["pharmacology", "bioelectric"])
        selected = select_relevant_priors(priors, CBD_QUESTION, max_priors=8)
        assert len(selected) <= 8

    def test_respects_max_priors_param(self):
        priors = load_priors(["pharmacology", "bioelectric"])
        selected = select_relevant_priors(priors, CBD_QUESTION, max_priors=3)
        assert len(selected) <= 3

    def test_cancer_question_gets_ros_and_psi(self):
        priors = load_priors(["pharmacology", "bioelectric"])
        selected = select_relevant_priors(priors, CBD_QUESTION)
        params = [p["param"] for p in selected]
        # Should get cancer-related bioelectric priors
        has_cancer_prior = any("cancer" in p.lower() for p in params)
        assert has_cancer_prior, f"Expected cancer-related prior, got: {params}"


class TestScaffold:
    def test_scaffold_contains_question(self):
        scaffold = build_scaffold("Test question?", "No priors.", 800)
        assert "Test question?" in scaffold

    def test_scaffold_contains_priors(self):
        scaffold = build_scaffold("Q?", "PRIOR: Kd = 2.0 uM", 800)
        assert "PRIOR: Kd = 2.0 uM" in scaffold

    def test_scaffold_contains_token_budget(self):
        scaffold = build_scaffold("Q?", "priors", 600)
        assert "600" in scaffold

    def test_scaffold_has_all_sections(self):
        scaffold = build_scaffold("Q?", "priors", 800)
        assert "SECTION 1: DECOMPOSITION" in scaffold
        assert "SECTION 2: CLAIMS" in scaffold
        assert "SECTION 3: UNKNOWNS" in scaffold
        assert "SECTION 4: NEXT STEP" in scaffold

    def test_scaffold_has_type_system(self):
        scaffold = build_scaffold("Q?", "priors", 800)
        assert "TYPE:" in scaffold
        assert "CONFIDENCE:" in scaffold
        assert "FALSIFIABLE BY:" in scaffold

    def test_scaffold_has_role(self):
        scaffold = build_scaffold("Q?", "priors", 800)
        assert "five independent scientific minds" in scaffold
        assert "You do not know which model you are" in scaffold


class TestFormatPriors:
    def test_formats_simple_prior(self):
        priors = [{"param": "Kd", "value": 2.0, "unit": "uM", "type": 1, "source": "Test"}]
        block = format_priors_block(priors)
        assert "PRIOR: Kd = 2.0 uM" in block
        assert "SOURCE: Test" in block
        assert "TYPE: 1" in block

    def test_formats_range_prior(self):
        priors = [{"param": "range", "value": [1.0, 10.0], "unit": "uM", "type": 1, "source": "Test"}]
        block = format_priors_block(priors)
        assert "[1.0, 10.0]" in block

    def test_empty_priors(self):
        block = format_priors_block([])
        assert "first principles" in block.lower()


class TestFullCompilation:
    def test_cbd_compiles(self):
        result = compile(CBD_QUESTION)
        assert result["session_id"].startswith("evo_")
        assert "pharmacology" in result["domains"]
        assert result["cross_domain_flag"] is True
        assert len(result["priors"]) > 0
        assert "prompt" in result
        assert len(result["models"]) == 5

    def test_all_models_present(self):
        result = compile(CBD_QUESTION)
        expected = {"claude", "gpt", "grok", "gemini", "deepseek"}
        assert set(result["models"].keys()) == expected

    def test_domain_override(self):
        result = compile("anything", domain_override="consciousness")
        assert result["domains"] == ["consciousness"]

    def test_token_budgets_present(self):
        result = compile(CBD_QUESTION)
        assert "S1" in result["token_budgets"]
        assert "S2_start" in result["token_budgets"]
        assert "S2_end" in result["token_budgets"]
        assert "S3" in result["token_budgets"]

    def test_budgets_decrease(self):
        result = compile(CBD_QUESTION)
        budgets = result["token_budgets"]
        assert budgets["S1"] >= budgets["S2_start"]
        assert budgets["S2_start"] >= budgets["S2_end"]
        assert budgets["S2_end"] >= budgets["S3"]

    def test_session_id_contains_domain(self):
        result = compile(CBD_QUESTION)
        assert "pharmacology" in result["session_id"]

    def test_prompt_contains_injected_priors(self):
        result = compile(CBD_QUESTION)
        # The compiled prompt should contain actual prior values
        assert "PRIOR:" in result["prompt"]
        assert "SOURCE:" in result["prompt"]
