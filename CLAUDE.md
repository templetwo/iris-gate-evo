# IRIS Gate Evo — Project Context

> **READ THIS FIRST** every session. Then check MEMORY_LEDGER.md if it exists.

## What This Is

A multi-LLM convergence protocol for scientific discovery. Five independent models receive the same compiled prompt, debate anonymously, and converge on truth through quantitative metrics. The output is a falsifiable protocol package — not an opinion.

## Current State (2026-02-09)

**Phase 1 COMPLETE. C0 compiles. PULSE ready to fire. 43 tests passing.**

- Compiler auto-detects domains, loads/merges priors, builds TMK scaffold
- PULSE dispatches to 5 models in parallel via LiteLLM async
- main.py wires C0 → PULSE with CLI flags (--compile-only, --domain, --models)
- Test question validated: 8 priors injected (pharmacology+bioelectric cross-domain)
- Next: Phase 2 — convergence metrics and S1→S2→S3 stage orchestration

## Build Order

| Phase | Files | Status |
|-------|-------|--------|
| **1 — Skeleton** | `compiler.py`, `pulse.py`, `models.py`, `main.py` | DONE |
| **2 — Convergence** | `convergence.py`, `stages.py`, `anonymizer.py` | NEXT |
| **3 — Verification** | `verify.py`, `gate.py`, `s4_hypothesis.py` | PENDING |
| **4 — Simulation** | `engines/`, `monte_carlo.py`, `protocol.py` | PENDING |

## Architecture

```
User Question → C0 (Compiler) → PULSE (5 models async)
  → S1 (Formulation) → S2 (Anonymized Debate, early-stop)
  → S3 (Convergence Gate: Jaccard > 0.85)
  → VERIFY (Perplexity, TYPE 2 claims)
  → Lab Gate (PASS/FAIL)
  → S4 (Hypotheses) → S5 (Monte Carlo, 0 LLM calls)
  → S6 (Protocol Package)
```

Total budget: 92-142 calls (~$1.50-4.00/run)

## Critical Rules

1. **Anonymization is non-negotiable** — Models see `[Mirror A-E]`, never model names
2. **Token budgets decrease** — S1: 800, S2: 800→700, S3: 600
3. **Convergence is server-side** — Jaccard, Cosine, JSD, Kappa. Never self-reported.
4. **Early-stop saves budget** — S2 exits when delta < 1% for 3 rounds AND TYPE 0/1 >= 80%
5. **Failure is data** — Jaccard < 0.85 at S3 routes to human review, not silent retry

## Model Registry (2026-02-09)

```python
claude:   claude-opus-4-6
gpt:      gpt-5.2
grok:     grok-4-1-fast-reasoning
gemini:   gemini-2.5-pro
deepseek: deepseek-chat
verify:   perplexity
```

Do NOT use older strings from iris-gate v0.2.

## Key Files

| File | Purpose |
|------|---------|
| `src/models.py` | Model registry + token budgets |
| `src/compiler/compiler.py` | C0 — domain detection, prior injection, scaffold |
| `src/pulse/pulse.py` | PULSE — async 5-model dispatch via LiteLLM |
| `main.py` | CLI entry point: C0 → PULSE → display |
| `docs/compiler-template.md` | C0 specification |
| `docs/AI_COBUILDER_README.md` | Full architecture reference |
| `priors/*.json` | Quantitative priors per domain (pharma, bioelectric, consciousness) |
| `templates/` | Output formats (plan, prereg) |
| `tests/test_compiler.py` | 33 offline tests for C0 |
| `tests/test_models.py` | 10 tests for model registry integrity |

## Lineage

- **iris-gate v0.2** (2025) — 8-stage, 185-350 calls. Legacy at `~/iris-gate/`
- **IRIS Gate Evo** (2026) — 9-stage, 92-142 calls. This repo.
- Legacy engines ported from `~/iris-gate/sandbox/engines/`
- Legacy priors ported from `~/iris-gate/sandbox/states/`

## Test Question (Pipeline Validation)

> "What are the mechanisms by which CBD induces selective cytotoxicity in cancer cells while sparing healthy cells, with specific reference to VDAC1-mediated mitochondrial membrane potential disruption?"

Expected compiler priors: VDAC1 Kd = 11.0 uM, TRPV1 Kd = 2.0 uM, cancer psi = -120mV, healthy psi = -180mV, ROS baseline 0.45 vs 0.08.
