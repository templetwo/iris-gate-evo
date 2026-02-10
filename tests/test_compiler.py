"""
Tests for C0 — The Compiler

All offline. No API calls. Validates domain detection (keyword + embedding),
prior loading, prior selection, scaffold construction, and full compilation
across all 11 scientific domains.
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
    compute_type01_threshold,
    PRIORS_DIR,
    DOMAIN_TRIGGERS,
    DOMAIN_DESCRIPTIONS,
    DOMAIN_MATURITY,
    MATURITY_TYPE01_THRESHOLD,
)


# ── Test questions per domain ──

CBD_QUESTION = (
    "What are the mechanisms by which CBD induces selective cytotoxicity "
    "in cancer cells while sparing healthy cells, with specific reference "
    "to VDAC1-mediated mitochondrial membrane potential disruption?"
)

KURAMOTO_QUESTION = (
    "How does Kuramoto oscillator coupling strength K affect phase "
    "synchronization and entropy in neural populations?"
)

NEURO_QUESTION = (
    "What role does BDNF play in hippocampal synaptic plasticity "
    "and how does serotonin modulate this pathway?"
)

IMMUNE_QUESTION = (
    "How do PD-1/PD-L1 checkpoint inhibitors modulate T cell "
    "exhaustion and cytokine release in the tumor microenvironment?"
)

GENETICS_QUESTION = (
    "What is the efficiency of CRISPR-Cas9 gene editing at TP53 "
    "loci and how does CpG methylation affect off-target rates?"
)

ONCOLOGY_QUESTION = (
    "How does the Warburg effect drive tumor glycolysis and what "
    "role does angiogenesis play in metastasis progression?"
)

CHEMISTRY_QUESTION = (
    "What determines the activation energy of enzyme-catalyzed "
    "reactions and how does pH affect Michaelis-Menten kinetics?"
)

ECOLOGY_QUESTION = (
    "How does coral bleaching from climate change affect marine "
    "biodiversity and trophic cascade dynamics in reef ecosystems?"
)

MATERIALS_QUESTION = (
    "What are the band gap engineering strategies for perovskite "
    "solar cells and how does nanoparticle size affect efficiency?"
)

PHYSICS_QUESTION = (
    "How does symmetry breaking at the critical point of the Ising "
    "model relate to phase transition order parameters?"
)

VAGUE_QUESTION = "What causes cancer?"

# This should miss keywords but hit embeddings
EMBEDDING_FALLBACK_QUESTION = (
    "Describe how dendritic arborization patterns influence "
    "computational capacity in cortical microcircuits"
)


# ── Domain Detection ──

class TestDomainDetection:
    def test_cbd_detects_pharmacology_and_bioelectric(self):
        domains = detect_domains(CBD_QUESTION, use_embeddings=False)
        assert "pharmacology" in domains
        assert "bioelectric" in domains

    def test_kuramoto_detects_consciousness(self):
        domains = detect_domains(KURAMOTO_QUESTION, use_embeddings=False)
        assert "consciousness" in domains

    def test_neuroscience_detection(self):
        domains = detect_domains(NEURO_QUESTION, use_embeddings=False)
        assert "neuroscience" in domains

    def test_immunology_detection(self):
        domains = detect_domains(IMMUNE_QUESTION, use_embeddings=False)
        assert "immunology" in domains

    def test_genetics_detection(self):
        domains = detect_domains(GENETICS_QUESTION, use_embeddings=False)
        assert "genetics" in domains

    def test_oncology_detection(self):
        domains = detect_domains(ONCOLOGY_QUESTION, use_embeddings=False)
        assert "oncology" in domains

    def test_chemistry_detection(self):
        domains = detect_domains(CHEMISTRY_QUESTION, use_embeddings=False)
        assert "chemistry" in domains

    def test_ecology_detection(self):
        domains = detect_domains(ECOLOGY_QUESTION, use_embeddings=False)
        assert "ecology" in domains

    def test_materials_detection(self):
        domains = detect_domains(MATERIALS_QUESTION, use_embeddings=False)
        assert "materials" in domains

    def test_physics_detection(self):
        domains = detect_domains(PHYSICS_QUESTION, use_embeddings=False)
        assert "physics" in domains

    def test_vague_question_falls_back_to_general_without_embeddings(self):
        domains = detect_domains(VAGUE_QUESTION, use_embeddings=False)
        assert domains == ["general"]

    def test_cross_domain_detection(self):
        q = "How does gap junction coupling affect Kuramoto oscillator coherence in bioelectric networks?"
        domains = detect_domains(q, use_embeddings=False)
        assert len(domains) >= 2

    def test_returns_list(self):
        domains = detect_domains(CBD_QUESTION)
        assert isinstance(domains, list)
        assert all(isinstance(d, str) for d in domains)

    def test_all_domains_have_descriptions(self):
        """Every keyword domain must have a matching embedding description."""
        for domain in DOMAIN_TRIGGERS:
            assert domain in DOMAIN_DESCRIPTIONS, \
                f"Domain '{domain}' has keywords but no description for embedding fallback"

    def test_immune_oncology_cross_domain(self):
        """Immunotherapy questions should hit both immunology and oncology."""
        q = "How does anti-PD-1 immunotherapy affect T cell function in the tumor microenvironment?"
        domains = detect_domains(q, use_embeddings=False)
        assert "immunology" in domains
        assert "oncology" in domains


class TestEmbeddingFallback:
    """Tests for the hybrid embedding detection path.

    These tests require sentence-transformers. Skip gracefully if not installed.
    """

    @pytest.fixture(autouse=True)
    def check_sentence_transformers(self):
        try:
            import sentence_transformers
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_embedding_catches_missed_keywords(self):
        """A question with no keyword hits should still detect via embeddings."""
        domains = detect_domains(EMBEDDING_FALLBACK_QUESTION, use_embeddings=True)
        assert domains != ["general"], \
            f"Embedding fallback should have caught this, got: {domains}"
        # Should detect neuroscience
        assert "neuroscience" in domains

    def test_embedding_returns_max_2_domains(self):
        domains = detect_domains(EMBEDDING_FALLBACK_QUESTION, use_embeddings=True)
        assert len(domains) <= 2

    def test_keywords_take_priority_over_embeddings(self):
        """When keywords match, embeddings should not be consulted."""
        # CBD question has strong keyword matches — result should be same
        kw_domains = detect_domains(CBD_QUESTION, use_embeddings=False)
        hybrid_domains = detect_domains(CBD_QUESTION, use_embeddings=True)
        assert kw_domains == hybrid_domains


# ── Prior Loading ──

class TestPriorLoading:
    def test_priors_dir_exists(self):
        assert PRIORS_DIR.exists()

    def test_all_domain_files_exist(self):
        """Every domain with keywords should have a priors JSON file."""
        for domain in DOMAIN_TRIGGERS:
            priors_file = PRIORS_DIR / f"{domain}.json"
            assert priors_file.exists(), f"Missing priors file: {priors_file}"

    def test_pharmacology_priors_load(self):
        priors = load_priors(["pharmacology"])
        assert len(priors) > 0
        assert all("param" in p for p in priors)
        assert all("value" in p for p in priors)
        assert all("unit" in p for p in priors)

    def test_all_domains_load_without_error(self):
        for domain in DOMAIN_TRIGGERS:
            priors = load_priors([domain])
            assert isinstance(priors, list), f"{domain} failed to load"

    def test_all_domains_have_priors(self):
        """Every domain (except general) should have at least 5 priors."""
        for domain in DOMAIN_TRIGGERS:
            priors = load_priors([domain])
            assert len(priors) >= 5, \
                f"{domain} has only {len(priors)} priors, need at least 5"

    def test_general_priors_are_empty(self):
        priors = load_priors(["general"])
        assert priors == []

    def test_multi_domain_merge(self):
        pharma = load_priors(["pharmacology"])
        bio = load_priors(["bioelectric"])
        merged = load_priors(["pharmacology", "bioelectric"])
        assert len(merged) == len(pharma) + len(bio)

    def test_deduplication(self):
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

    def test_all_priors_have_sources(self):
        """Every prior should cite its source."""
        for domain_file in PRIORS_DIR.glob("*.json"):
            with open(domain_file) as f:
                data = json.load(f)
            for p in data.get("priors", []):
                assert p.get("source"), \
                    f"Missing source in {domain_file.name}: {p['param']}"


# ── Prior Selection ──

class TestPriorSelection:
    def test_cbd_selects_vdac_and_trpv(self):
        priors = load_priors(["pharmacology", "bioelectric"])
        selected = select_relevant_priors(priors, CBD_QUESTION)
        params = [p["param"] for p in selected]
        assert any("VDAC1" in p for p in params)
        assert any("TRPV" in p for p in params)

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
        has_cancer_prior = any("cancer" in p.lower() for p in params)
        assert has_cancer_prior

    def test_neuroscience_selects_relevant(self):
        priors = load_priors(["neuroscience"])
        selected = select_relevant_priors(priors, NEURO_QUESTION)
        params = [p["param"] for p in selected]
        assert any("bdnf" in p.lower() or "serotonin" in p.lower() for p in params)

    def test_crispr_selects_relevant(self):
        priors = load_priors(["genetics"])
        selected = select_relevant_priors(priors, GENETICS_QUESTION)
        params = [p["param"] for p in selected]
        assert any("crispr" in p.lower() or "cas9" in p.lower() for p in params)


# ── Scaffold ──

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


# ── Format Priors ──

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


# ── Full Compilation ──

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
        expected = {"claude", "mistral", "grok", "gemini", "deepseek"}
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
        assert "PRIOR:" in result["prompt"]
        assert "SOURCE:" in result["prompt"]

    def test_each_domain_compiles(self):
        """Every domain should compile without error when overridden."""
        for domain in DOMAIN_TRIGGERS:
            result = compile("test question", domain_override=domain)
            assert result["domains"] == [domain]
            assert len(result["priors"]) >= 0  # general may be empty


# ── Domain-Adaptive TYPE Threshold ──

class TestDomainAdaptiveThreshold:
    def test_all_domains_have_maturity(self):
        """Every keyword domain must have a maturity tier."""
        for domain in DOMAIN_TRIGGERS:
            assert domain in DOMAIN_MATURITY, \
                f"Domain '{domain}' has no maturity tier"

    def test_established_domain_gets_90(self):
        threshold = compute_type01_threshold(["pharmacology"], cross_domain=False)
        assert threshold == 0.90

    def test_frontier_domain_gets_80(self):
        threshold = compute_type01_threshold(["consciousness"], cross_domain=False)
        assert threshold == 0.80

    def test_moderate_domain_gets_85(self):
        threshold = compute_type01_threshold(["ecology"], cross_domain=False)
        assert threshold == 0.85

    def test_cross_domain_uses_lowest_tier(self):
        """pharmacology (established) + bioelectric (frontier) → frontier (0.80)."""
        threshold = compute_type01_threshold(
            ["pharmacology", "bioelectric"], cross_domain=True
        )
        assert threshold == 0.80

    def test_cross_domain_established_only(self):
        """Two established domains still use established threshold."""
        threshold = compute_type01_threshold(
            ["pharmacology", "neuroscience"], cross_domain=True
        )
        # Cross-domain with both established — worst tier is established
        assert threshold == 0.90

    def test_cbd_question_gets_frontier_threshold(self):
        """The CBD/VDAC1 question is pharmacology+bioelectric → 0.80."""
        result = compile(CBD_QUESTION)
        assert result["s3_type01_threshold"] == 0.80

    def test_single_established_question(self):
        result = compile("What is aspirin's mechanism?", domain_override="pharmacology")
        assert result["s3_type01_threshold"] == 0.90

    def test_threshold_present_in_compiled(self):
        result = compile("test question")
        assert "s3_type01_threshold" in result

    def test_unknown_domain_defaults_moderate(self):
        threshold = compute_type01_threshold(["nonexistent"], cross_domain=False)
        assert threshold == 0.85


# ── Prior Count Summary ──

class TestPriorCoverage:
    def test_total_prior_count(self):
        """Sanity check: we should have 80+ priors across all domains."""
        total = 0
        for domain_file in PRIORS_DIR.glob("*.json"):
            with open(domain_file) as f:
                data = json.load(f)
            total += len(data.get("priors", []))
        assert total >= 80, f"Expected 80+ priors total, got {total}"
