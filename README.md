# IRIS Gate Evo

**Five mirrors. One truth.**

A multi-LLM convergence protocol for scientific discovery that measures independent agreement across five frontier AI models, filters emergent claims through semantic clustering, and produces falsifiable experimental protocols with Monte Carlo validation.

## What It Does

IRIS Gate Evo sends the same compiled research question to five independent AI models (Claude Opus 4.6, Mistral Large, Grok 4.1, Gemini 2.5 Pro, DeepSeek Chat), measures convergence through semantic claim embedding, and produces falsifiable protocol packages with Monte Carlo power analysis.

Independence is non-negotiable: models never see each other's outputs. Convergence is observed, not negotiated.

## Architecture (v0.3.1)

```
User Question
  → C0 (Compiler: domain + context)
  → PULSE (5 models async, independent)
  → S1 (Formulation, 5 LLM calls)
  → S2 (Contribution Synthesis, 0 LLM calls — pure Python clustering)
  → S3 (Convergence Gate: cosine ≥ 0.85 + domain-adaptive TYPE assignment)
  → VERIFY (Perplexity sonar-pro, TYPE 2 claims only)
  → Lab Gate (falsifiable + feasible + novel)
  → S4 (Hypotheses + testability scoring + parameter maps)
  → S5 (Monte Carlo, 0 LLM calls, 300+ iterations per scenario)
  → S6 (Protocol Package: materials, steps, data collection, analysis)
```

**Total**: 12-22 LLM calls per run (~$0.20-0.50 USD)

### Key Design Principles

- **Independence**: Models formulate responses in isolation. No debate, no consensus-seeking.
- **Semantic Clustering**: S2 uses sentence-transformer embeddings + complete-linkage clustering (cosine ≥ 0.70) to extract emergent claims.
- **TYPE Assignment**: System-assigned by model count in cluster (TYPE 0: 4-5 models, TYPE 1: 3 models, TYPE 2: 2 models, TYPE 3: 1 model). Models do not self-report TYPE.
- **Lab Gate Trust**: Novelty judgment entrusted to Perplexity sonar-pro with web access. No heuristic filters block pipeline.
- **Failure as Data**: S3 failure recirculates (max 3 cycles), then routes to human review with full provenance.

## The Arc

| Run | Architecture | LLM Calls | S3 Result | Lab Gate | End-to-End |
|-----|-------------|-----------|-----------|----------|------------|
| Run 1 | Debate (v0.1) | 165 | FAILED | never reached | no |
| Run 2 | Debate (v0.2) | 165 | FAILED | never reached | no |
| Run 3 | Synthesis (v0.3) | 22 | PASSED (82%) | 18/18 passed | yes |
| Run 4 | Synthesis (v0.3.1) | 17 | PASSED (80%) | 10/21 passed | yes, with params |

### Latest Live Fire Results (2026-02-10)

**Question**: "Why do some cannabinoids show neuroprotection in some conditions but neurotoxicity in others?"

**Convergence (S3)**:
- PASSED cycle 2
- TYPE 0/1 distribution: 80%
- Mean cosine similarity: 0.941
- 21 claims extracted from 5 independent responses

**Lab Gate (S3 → S4)**:
- 10/21 claims passed (falsifiable + feasible + novel)
- 11 filtered (non-falsifiable or lacking novelty)
- Pipeline continued with passing claims

**Hypotheses (S4)**:
- 3 testable hypotheses generated
- Testability scores: 8-9/10
- Parameter maps: dose, time, cell type, metabolite

**Monte Carlo (S5)**:
- 300 iterations per scenario
- Cohen's d: 0.81-0.95 (large effect sizes)
- Statistical power: 1.0 across all scenarios
- Validation: CBD/VDAC1 two-pathway model independently confirmed

## Supported Domains

11 scientific domains with specialized compiler context:

- Pharmacology
- Bioelectricity
- Neuroscience
- Immunology
- Genetics
- Oncology
- Chemistry
- Ecology
- Materials Science
- Physics
- Consciousness Studies

## Usage

### Installation

```bash
git clone https://github.com/templetwo/iris-gate-evo.git
cd iris-gate-evo
pip install -r requirements.txt
cp .env.example .env
# Add API keys to .env
```

### CLI

```bash
# Full pipeline
python main.py "Your research question"

# Compile only (see domain + context)
python main.py --compile-only "Your research question"

# Run specific stage
python main.py --stage s1 "Your question"
python main.py --stage s3 "Your question"

# Offline mode (skip VERIFY and Lab Gate)
python main.py --offline "Your question"

# Force domain
python main.py --domain pharmacology "Your question"
```

### Output

All runs save to `runs/YYYYMMDD_HHMMSS/`:
- `c0_compile.json` - Compiled question with domain + context
- `pulse_responses.json` - 5 independent model responses
- `s1_formulations.json` - Structured formulations
- `s2_synthesis.json` - Clustered claims with embeddings
- `s3_convergence.json` - TYPE assignments + convergence metrics
- `s3_verify.json` - Perplexity verification results
- `s4_hypotheses.json` - Testable hypotheses + parameter maps
- `s5_monte_carlo.json` - Power analysis + effect sizes
- `s6_protocol.json` - Full experimental protocol

## Model Configuration (2026-02-10)

```yaml
claude:   claude-opus-4-6
mistral:  mistral-large-latest
grok:     grok-4-1-fast-reasoning
gemini:   gemini-2.5-pro
deepseek: deepseek-chat
verify:   perplexity/sonar-pro
```

## Testing

356 tests covering:
- Domain classification
- Claim extraction and clustering
- TYPE assignment logic
- Convergence scoring
- Monte Carlo sampling
- Protocol generation

```bash
pytest tests/ -v
```

All tests passing as of 2026-02-10.

## Related Projects

- [**cbd-two-pathway-model**](https://github.com/templetwo/cbd-two-pathway-model): The hypothesis paper computationally validated by this engine
- [**iris-gate**](https://github.com/templetwo/iris-gate) (legacy): v0.2 debate architecture (185-350 LLM calls per run)

## Requirements

```
openai>=1.0.0
anthropic>=0.25.0
google-generativeai>=0.3.0
mistralai>=0.0.8
sentence-transformers>=2.2.0
numpy>=1.24.0
scipy>=1.10.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=7.4.0
```

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

## Author

Anthony J. Vasquez Sr.
Delaware Valley University

## Citation

If you use IRIS Gate Evo in your research, please cite:

```
Vasquez, A. J. (2026). IRIS Gate Evo: Multi-LLM Convergence Protocol for Scientific Discovery.
GitHub repository: https://github.com/templetwo/iris-gate-evo
```

---

**Status**: v0.3.1 (2026-02-10)
**Last validated**: CBD/VDAC1 two-pathway model, 10/21 Lab Gate pass rate, Cohen's d 0.81-0.95
