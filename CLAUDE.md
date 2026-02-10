# IRIS Gate Evo — Project Context

> **READ THIS FIRST** every session. Then check MEMORY_LEDGER.md if it exists.

## What This Is

A multi-LLM convergence protocol for scientific discovery. Five independent models receive the same compiled prompt, debate anonymously, and converge on truth through quantitative metrics. The output is a falsifiable protocol package — not an opinion.

## Current State (2026-02-09)

**ALL PHASES COMPLETE. Full pipeline: C0 → S1 → S2 → S3 → VERIFY → GATE → S4 → S5 → S6. 255 tests passing.**

- C0 compiles across 11 domains with hybrid keyword+embedding detection
- PULSE dispatches to 5 models in parallel via LiteLLM async
- Parser extracts structured claims (TYPE/CONFIDENCE/MECHANISM/FALSIFIABLE BY)
- Anonymizer randomizes mirror labels every round (seed + round_num)
- Convergence engine: Jaccard, Cosine, JSD, Kappa — computed on CLAIMS, not raw text
- S2 debate loop with AND early-stop (delta < 1% for 3 consecutive AND TYPE 0/1 >= 80%)
- S3 gate: Jaccard > 0.85 AND TYPE 0/1 >= 90%, failure → human review with divergence map
- VERIFY: Perplexity-backed TYPE 2 checking → PROMOTED/HELD/NOVEL/CONTRADICTED
- Lab Gate: Three-criteria filter (falsifiable, feasible, novel), offline fallback
- S4: Hypothesis operationalization with Monte Carlo parameter maps
- S5: Pure Python simulation, 300+ iterations, 95% CIs, effect sizes, power estimates
- S6: Protocol package generator with JSON + human-readable summary
- main.py wired for full pipeline with --stage, --offline, --compile-only flags

## Build Order

| Phase | Files | Status |
|-------|-------|--------|
| **1 — Skeleton** | `compiler.py`, `pulse.py`, `models.py`, `main.py` | DONE |
| **2 — Convergence** | `parser.py`, `convergence.py`, `stages.py`, `anonymizer.py` | DONE |
| **3 — Verification** | `verify.py`, `gate.py`, `s4_hypothesis.py` | DONE |
| **4 — Simulation** | `monte_carlo.py`, `protocol.py` | DONE |

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

## CLI Usage

```bash
python main.py "Your research question"              # Full pipeline
python main.py --compile-only "Your question"         # C0 only
python main.py --stage s1 "Your question"             # Stop after S1
python main.py --stage s3 "Your question"             # Stop after S3 gate
python main.py --offline "Your question"              # No API calls
python main.py --domain pharmacology "Your question"  # Force domain
python main.py                                        # Default CBD test question
```

## Critical Rules

1. **Anonymization is non-negotiable** — Models see `[Mirror A-E]`, never model names
2. **Token budgets decrease** — S1: 800, S2: 800→700, S3: 600
3. **Convergence is server-side** — Jaccard, Cosine, JSD, Kappa. Never self-reported.
4. **Early-stop saves budget** — S2 exits when delta < 1% for 3 rounds AND TYPE 0/1 >= 80%
5. **Failure is data** — Jaccard < 0.85 at S3 routes to human review, not silent retry
6. **Lab Gate kills bad hypotheses** — Falsifiable AND feasible AND novel, all three required
7. **S5 is pure Python** — Zero LLM calls. Monte Carlo only.

## Model Registry (2026-02-09)

```python
claude:   claude-opus-4-6
mistral:  mistral-large-latest
grok:     grok-4-1-fast-reasoning
gemini:   gemini-2.5-pro
deepseek: deepseek-chat
verify:   perplexity/sonar-pro
```

Do NOT use older strings from iris-gate v0.2.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point: full pipeline C0→S6 |
| `src/models.py` | Model registry + token budgets |
| `src/compiler/compiler.py` | C0 — domain detection, prior injection, scaffold |
| `src/pulse/pulse.py` | PULSE — async 5-model dispatch via LiteLLM |
| `src/parser.py` | Claim parser — structured extraction from responses |
| `src/stages/anonymizer.py` | Per-round random mirror label assignment |
| `src/convergence/convergence.py` | Jaccard, Cosine, JSD, Kappa on parsed claims |
| `src/stages/stages.py` | S1→S2→S3 orchestration with AND early-stop |
| `src/verify/verify.py` | VERIFY — Perplexity TYPE 2 claim verification |
| `src/gate/gate.py` | Lab Gate — falsifiability, feasibility, novelty |
| `src/hypothesis/s4_hypothesis.py` | S4 — hypothesis operationalization + parameter maps |
| `src/monte_carlo/monte_carlo.py` | S5 — pure Python Monte Carlo, 300+ iterations, 95% CIs |
| `src/protocol/protocol.py` | S6 — protocol package generator (JSON + summary) |
| `priors/*.json` | Quantitative priors across 11 scientific domains |
| `tests/` | 255 tests across 9 test files |

## Lineage

- **iris-gate v0.2** (2025) — 8-stage, 185-350 calls. Legacy at `~/iris-gate/`
- **IRIS Gate Evo** (2026) — 9-stage, 92-142 calls. This repo.
- Legacy engines ported from `~/iris-gate/sandbox/engines/`
- Legacy priors ported from `~/iris-gate/sandbox/states/`

## Test Question (Pipeline Validation)

> "What are the mechanisms by which CBD induces selective cytotoxicity in cancer cells while sparing healthy cells, with specific reference to VDAC1-mediated mitochondrial membrane potential disruption?"

Expected compiler priors: VDAC1 Kd = 11.0 uM, TRPV1 Kd = 2.0 uM, cancer psi = -120mV, healthy psi = -180mV, ROS baseline 0.45 vs 0.08.
