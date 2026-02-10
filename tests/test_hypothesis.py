"""
Tests for S4 Hypothesis Operationalization.

Validates:
- Hypothesis parsing from model responses
- Parameter extraction for Monte Carlo
- Testability scoring
- Offline operationalization
- Dose range parsing
- Edge cases
"""

import pytest
from src.hypothesis.s4_hypothesis import (
    Hypothesis,
    ParameterSpec,
    S4Result,
    build_s4_prompt,
    parse_s4_response,
    score_testability,
    operationalize_offline,
    _parse_parameters,
    _parse_dose_ranges,
    _extract_variables,
    _extract_parameters_from_text,
    _strip_markdown,
)


# ── Fixtures ──

@pytest.fixture
def verified_claims():
    """Claims that passed VERIFY and gate."""
    return [
        {
            "statement": "CBD binds VDAC1 with Kd = 11 uM causing conformational change",
            "type": 1,
            "confidence": 0.85,
            "mechanism": "CBD interaction with VDAC1 alters gating state at Kd = 11.0 uM",
            "falsifiable_by": "VDAC1 knockout cells showing identical CBD cytotoxicity would disprove this",
        },
        {
            "statement": "Cancer cells depolarized at -120mV vs healthy at -180mV creates selectivity window",
            "type": 0,
            "confidence": 0.90,
            "mechanism": "60mV difference means cancer mitochondria are closer to permeability transition threshold",
            "falsifiable_by": "Artificially depolarizing healthy cells to -120mV should increase CBD sensitivity",
        },
        {
            "statement": "TRPV1-mediated calcium influx at Kd = 2.0 uM synergizes with VDAC1 depolarization",
            "type": 2,
            "confidence": 0.6,
            "mechanism": "TRPV1 activation provides additional calcium overload signal",
            "falsifiable_by": "Capsazepine co-treatment reducing CBD efficacy by >50% would confirm",
        },
    ]


@pytest.fixture
def sample_s4_response():
    """Simulated model response to S4 prompt."""
    return """\
HYPOTHESIS H[1]:
PREDICTION: IF CBD is applied at concentrations near VDAC1 Kd (11 uM), THEN mitochondrial membrane potential will decrease by >40mV in cancer cells (baseline -120mV) but <10mV in healthy cells (baseline -180mV)
SOURCE CLAIMS: 1, 2
KEY VARIABLES: CBD concentration, VDAC1 binding, membrane potential, cell type
PARAMETERS:
  - CBD_concentration: 1.0-20.0 uM (log-normal)
  - VDAC1_Kd: 8.0-14.0 uM (normal)
  - cancer_psi: -130.0--110.0 mV (normal)
  - healthy_psi: -190.0--170.0 mV (normal)
TESTABILITY: 8.5
PROTOCOL: Use paired cancer/healthy cell lines (e.g., MCF-7 vs MCF-10A). Apply CBD at 1, 5, 10, 15, 20 uM. Measure mitochondrial membrane potential via JC-1 staining at 4h and 24h timepoints.
EXPECTED OUTCOME: Dose-dependent depolarization in cancer cells with EC50 near 11 uM, minimal effect in healthy cells below 15 uM
NULL OUTCOME: No differential effect between cancer and healthy cells, or effect independent of VDAC1 expression
DOSE RANGES: CBD: 1, 5, 10, 15, 20 uM
READOUTS: JC-1 fluorescence ratio, Annexin V/PI staining, TMRE assay
CONTROLS: Vehicle (DMSO), VDAC1 siRNA knockdown, DIDS (VDAC inhibitor)

HYPOTHESIS H[2]:
PREDICTION: IF TRPV1 is pharmacologically blocked with capsazepine, THEN CBD cytotoxicity in cancer cells will decrease by >50% at concentrations where TRPV1 Kd (2 uM) is exceeded
SOURCE CLAIMS: 3
KEY VARIABLES: TRPV1 activity, capsazepine concentration, CBD efficacy, calcium influx
PARAMETERS:
  - TRPV1_Kd: 1.5-3.0 uM (normal)
  - capsazepine_dose: 5.0-20.0 uM (uniform)
  - CBD_dose: 2.0-20.0 uM (log-normal)
TESTABILITY: 7.0
PROTOCOL: Pre-treat cancer cells with capsazepine (10 uM, 30 min), then apply CBD dose curve. Compare viability (MTT) at 48h against CBD-only controls.
EXPECTED OUTCOME: >50% reduction in CBD-induced cell death when TRPV1 is blocked, especially at CBD concentrations 2-10 uM
NULL OUTCOME: Capsazepine has no effect on CBD cytotoxicity, suggesting TRPV1 is not a significant contributor
DOSE RANGES: CBD: 2, 5, 10, 20 uM
READOUTS: MTT viability, Calcium imaging (Fluo-4), Caspase-3 activity
CONTROLS: Vehicle, Capsazepine alone, CBD alone, AMG9810 (alternative TRPV1 antagonist)"""


# ── Prompt Construction ──

class TestS4Prompt:
    def test_contains_question(self, verified_claims):
        prompt = build_s4_prompt("How does CBD kill cancer?", verified_claims)
        assert "How does CBD kill cancer?" in prompt

    def test_contains_claims(self, verified_claims):
        prompt = build_s4_prompt("test?", verified_claims)
        assert "VDAC1" in prompt
        assert "TRPV1" in prompt

    def test_contains_format_instructions(self, verified_claims):
        prompt = build_s4_prompt("test?", verified_claims)
        assert "PREDICTION:" in prompt
        assert "PARAMETERS:" in prompt
        assert "PROTOCOL:" in prompt


# ── Response Parsing ──

class TestS4Parsing:
    def test_extracts_two_hypotheses(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert len(hyps) == 2

    def test_hypothesis_ids(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert hyps[0].id == "H1"
        assert hyps[1].id == "H2"

    def test_prediction_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert "IF CBD" in hyps[0].prediction
        assert "mitochondrial membrane potential" in hyps[0].prediction

    def test_parameters_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        params = hyps[0].parameters
        assert len(params) >= 2
        # Check first parameter
        cbd_param = [p for p in params if "CBD" in p.name or "cbd" in p.name.lower()]
        assert len(cbd_param) >= 1
        assert cbd_param[0].low == 1.0
        assert cbd_param[0].high == 20.0

    def test_testability_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert hyps[0].testability_score == 8.5
        assert hyps[1].testability_score == 7.0

    def test_protocol_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert "MCF-7" in hyps[0].experimental_protocol or "JC-1" in hyps[0].experimental_protocol

    def test_readouts_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert len(hyps[0].readouts) >= 1

    def test_controls_extracted(self, sample_s4_response):
        hyps = parse_s4_response(sample_s4_response)
        assert len(hyps[0].controls) >= 1

    def test_empty_response(self):
        hyps = parse_s4_response("")
        assert len(hyps) == 0

    def test_malformed_response(self):
        hyps = parse_s4_response("Just some random text without hypothesis markers")
        assert len(hyps) == 0


# ── Parameter Parsing ──

class TestParameterParsing:
    def test_parse_uniform(self):
        text = "PARAMETERS:\n  - dose: 1.0-20.0 uM (uniform)\nTESTABILITY: 5"
        params = _parse_parameters(text)
        assert len(params) == 1
        assert params[0].name == "dose"
        assert params[0].low == 1.0
        assert params[0].high == 20.0
        assert params[0].distribution == "uniform"

    def test_parse_normal(self):
        text = "PARAMETERS:\n  - Kd: 8.0-14.0 uM (normal)\nTESTABILITY: 5"
        params = _parse_parameters(text)
        assert params[0].distribution == "normal"

    def test_parse_lognormal_variants(self):
        text = "PARAMETERS:\n  - conc: 1.0-100.0 nM (log-normal)\nTESTABILITY: 5"
        params = _parse_parameters(text)
        assert params[0].distribution == "log-normal"

    def test_parse_multiple_params(self):
        text = """\
PARAMETERS:
  - param_a: 1.0-10.0 uM (uniform)
  - param_b: 5.0-15.0 mV (normal)
  - param_c: 0.1-1.0 nM (log-normal)
TESTABILITY: 7"""
        params = _parse_parameters(text)
        assert len(params) == 3

    def test_no_params_section(self):
        params = _parse_parameters("no parameters here")
        assert len(params) == 0


# ── Dose Range Parsing ──

class TestDoseRangeParsing:
    def test_basic_dose_range(self):
        text = "CBD: 1, 5, 10, 20 uM"
        ranges = _parse_dose_ranges(text)
        assert "CBD" in ranges
        assert ranges["CBD"]["doses"] == [1.0, 5.0, 10.0, 20.0]
        assert ranges["CBD"]["unit"] == "uM"

    def test_multiple_compounds(self):
        text = "CBD: 1, 5, 10 uM\nCapsazepine: 5, 10, 20 uM"
        ranges = _parse_dose_ranges(text)
        assert len(ranges) >= 1  # At least CBD


# ── Testability Scoring ──

class TestTestability:
    def test_high_testability_claim(self):
        claim = {
            "statement": "CBD binds VDAC1 with Kd = 11.0 uM causing pore opening",
            "mechanism": "CBD interaction with VDAC1 at concentration-dependent manner alters gating state",
            "falsifiable_by": "If VDAC1 knockout cells show identical CBD cytotoxicity, this would be disproved",
            "type": 1,
        }
        score = score_testability(claim)
        assert score >= 7.0  # Quantitative + mechanism + falsifiable + type1 + target

    def test_low_testability_claim(self):
        claim = {
            "statement": "something affects cells",
            "mechanism": "",
            "falsifiable_by": "",
            "type": 3,
        }
        score = score_testability(claim)
        assert score <= 2.0

    def test_medium_testability(self):
        claim = {
            "statement": "ROS levels increase in cancer cells after treatment",
            "mechanism": "oxidative stress pathway activated by mitochondrial disruption cascade",
            "falsifiable_by": "NAC pre-treatment should abolish effect",
            "type": 1,
        }
        score = score_testability(claim)
        assert 3.0 <= score <= 7.0

    def test_score_capped_at_10(self):
        claim = {
            "statement": "CBD binds VDAC1 with Kd = 11.0 uM at dose threshold causing concentration-dependent apoptosis",
            "mechanism": "VDAC1 conformational change opens permeability transition pore at threshold concentration",
            "falsifiable_by": "If VDAC1 knockout cells would show no effect, this inhibit mechanism would be disproved",
            "type": 0,
        }
        score = score_testability(claim)
        assert score <= 10.0


# ── Variable Extraction ──

class TestVariableExtraction:
    def test_extracts_molecular_targets(self):
        variables = _extract_variables("CBD binds VDAC1 and activates TRPV1")
        assert "VDAC1" in variables or "TRPV1" in variables

    def test_extracts_cbd(self):
        variables = _extract_variables("CBD concentration affects cell viability")
        assert "CBD" in variables

    def test_max_8_variables(self):
        text = "VDAC1 TRPV1 CBD ROS Ca2+ ATP NAC IC50 p53 BCL-2 caspase extra"
        variables = _extract_variables(text)
        assert len(variables) <= 8


# ── Parameter Extraction from Text ──

class TestParameterFromText:
    def test_extracts_kd(self):
        params = _extract_parameters_from_text("Kd = 11.0 uM for VDAC1 binding")
        assert len(params) >= 1
        kd = params[0]
        assert kd.name == "Kd"
        assert abs(kd.low - 7.7) < 0.1   # 11.0 * 0.7
        assert abs(kd.high - 14.3) < 0.1  # 11.0 * 1.3

    def test_creates_normal_distribution(self):
        params = _extract_parameters_from_text("Kd = 11.0 uM")
        assert params[0].distribution == "normal"
        assert params[0].mean == 11.0

    def test_empty_text(self):
        params = _extract_parameters_from_text("")
        assert len(params) == 0


# ── Offline Operationalization ──

class TestOfflineOperationalization:
    def test_generates_hypotheses(self, verified_claims):
        hyps = operationalize_offline(verified_claims, "How does CBD kill cancer?")
        assert len(hyps) >= 1
        assert len(hyps) <= 5  # Max 5

    def test_sorted_by_testability(self, verified_claims):
        hyps = operationalize_offline(verified_claims)
        if len(hyps) >= 2:
            assert hyps[0].testability_score >= hyps[1].testability_score

    def test_hypothesis_has_prediction(self, verified_claims):
        hyps = operationalize_offline(verified_claims)
        for h in hyps:
            assert len(h.prediction) > 10

    def test_hypothesis_has_id(self, verified_claims):
        hyps = operationalize_offline(verified_claims)
        ids = [h.id for h in hyps]
        assert all(id.startswith("H") for id in ids)

    def test_extracts_parameters_from_claims(self, verified_claims):
        hyps = operationalize_offline(verified_claims)
        # The first claim has "Kd = 11" — should extract a parameter
        all_params = [p for h in hyps for p in h.parameters]
        assert len(all_params) >= 1

    def test_empty_claims(self):
        hyps = operationalize_offline([])
        assert len(hyps) == 0

    def test_protocol_from_falsifiable_by(self, verified_claims):
        hyps = operationalize_offline(verified_claims)
        # At least one should have a protocol derived from falsifiable_by
        has_protocol = any(len(h.experimental_protocol) > 5 for h in hyps)
        assert has_protocol


# ── Markdown Stripping ──

class TestMarkdownStripping:
    def test_strips_bold(self):
        text = "**PREDICTION:** IF CBD is applied"
        assert _strip_markdown(text) == "PREDICTION: IF CBD is applied"

    def test_strips_nested_bold(self):
        text = "**HYPOTHESIS H[1]:**\n**PREDICTION:** test"
        result = _strip_markdown(text)
        assert "**" not in result
        assert "HYPOTHESIS H[1]:" in result

    def test_preserves_bullet_points(self):
        text = "* bullet item\n  * nested"
        result = _strip_markdown(text)
        # Bullet points should not be stripped
        assert "bullet item" in result

    def test_parse_markdown_response(self):
        """Models often return markdown — parser should handle it."""
        response = """\
**HYPOTHESIS H[1]:**
**PREDICTION:** IF CBD is applied at 11 uM, THEN VDAC1 opens in cancer cells
**SOURCE CLAIMS:** 1, 2
**KEY VARIABLES:** CBD, VDAC1, membrane potential
**PARAMETERS:**
  - CBD_dose: 1.0-20.0 uM (log-normal)
  - VDAC1_Kd: 8.0-14.0 uM (normal)
**TESTABILITY:** 8.0
**PROTOCOL:** Use MCF-7 vs MCF-10A cell lines with JC-1 staining
**EXPECTED OUTCOME:** Selective depolarization in cancer cells
**NULL OUTCOME:** No differential effect
**DOSE RANGES:** CBD: 1, 5, 10, 20 uM
**READOUTS:** JC-1, Annexin V
**CONTROLS:** Vehicle, VDAC1-KO"""

        hyps = parse_s4_response(response)
        assert len(hyps) == 1
        assert "IF CBD" in hyps[0].prediction
        assert len(hyps[0].parameters) >= 2
        assert hyps[0].testability_score == 8.0
        assert len(hyps[0].key_variables) >= 2
