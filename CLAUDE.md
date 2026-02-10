# IRIS Gate Evo — Project Context

> **READ THIS FIRST** every session. Then check MEMORY_LEDGER.md if it exists.

## What This Is

A multi-LLM convergence protocol for scientific discovery. Five independent models receive the same compiled prompt, respond independently, and the SYSTEM finds convergence through claim tuple overlap counting. The output is a falsifiable protocol package — not an opinion.

## Current State (2026-02-10)

**ALL PHASES COMPLETE. Full pipeline: C0 → S1 → S2 → S3 → VERIFY → GATE → S4 → S5 → S6.**

- C0 compiles across 11 domains with hybrid keyword+embedding detection
- C0 computes domain-adaptive TYPE thresholds (established=90%, moderate=85%, frontier=80%)
- PULSE dispatches to 5 models in parallel via LiteLLM async
- Parser extracts structured claims (TYPE/CONFIDENCE/MECHANISM/FALSIFIABLE BY)
- S2 Contribution Synthesis: semantic claim embedding + complete-linkage clustering (0 API calls)
  - Parsed claim statements embedded directly (not raw text) via all-MiniLM-L6-v2
  - K-SSM lesson: "Any path that doesn't go through R gives the gradient an escape route"
    — raw text scaffolding (headers, formatting) is noise; embed the science (claims) directly
  - Synonym normalization before embedding (CBD=Cannabidiol, VDAC1=voltage-dependent anion channel 1)
  - Complete-linkage clustering (cosine ≥0.70) — prevents transitive chaining that union-find caused
  - TYPE assigned by model count in cluster: 5/5→T0, 4/5→T0, 3/5→T1, 2/5→T2, 1/5→T3
  - Models never see each other's outputs
  - Singulars (TYPE 3) preserved as potential novel insights
  - Conflict detection: same claim, different values across models
  - Claim tuples extracted downstream for S4→S6 structured data, NOT for convergence
- Convergence engine: Jaccard (tuple-based), Cosine, JSD, Kappa — computed on CLAIMS, not raw text
- Claim tuple extraction: synonym resolution, relation grouping, value normalization, bidirectional sorting
- S3 gate: Cosine > 0.85 (primary) + Jaccard floor 0.10 + domain-adaptive TYPE threshold
- S3 recirculation: on failure, converged claims + singulars fed back as independent consensus (max 3 cycles)
- Recirculation dedup uses claim tuple overlap, not string matching
- VERIFY: Perplexity-backed TYPE 2 checking → PROMOTED/HELD/NOVEL/CONTRADICTED
- Lab Gate: Three-criteria filter (falsifiable, feasible, novel), offline fallback
- S4: Hypothesis operationalization with Monte Carlo parameter maps
- S5: Pure Python simulation, 300+ iterations, 95% CIs, effect sizes, power estimates
- S6: Protocol package generator with JSON + human-readable summary
- Live terminal dashboard with real-time convergence metrics
- main.py wired for full pipeline with --stage, --offline, --compile-only, --no-dashboard flags

## Build Order

| Phase | Files | Status |
|-------|-------|--------|
| **1 — Skeleton** | `compiler.py`, `pulse.py`, `models.py`, `main.py` | DONE |
| **2 — Convergence** | `parser.py`, `convergence.py`, `stages.py`, `synthesis.py` | DONE |
| **3 — Verification** | `verify.py`, `gate.py`, `s4_hypothesis.py` | DONE |
| **4 — Simulation** | `monte_carlo.py`, `protocol.py` | DONE |

## Architecture

```
User Question → C0 (Compiler) → PULSE (5 models async)
  → S1 (Formulation, 5 calls)
  → S2 (Contribution Synthesis, 0 calls — pure Python overlap)
  → S3 (Convergence Gate: Cosine > 0.85 + domain-adaptive TYPE)
  → VERIFY (Perplexity, TYPE 2 claims)
  → Lab Gate (PASS/FAIL)
  → S4 (Hypotheses) → S5 (Monte Carlo, 0 LLM calls)
  → S6 (Protocol Package)
```

Total budget: 12-42 calls (~$0.20-1.00/run)

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

1. **Independence is non-negotiable** — Models never see each other's outputs
2. **TYPE is system-assigned** — By overlap count, not model self-report
3. **Convergence is server-side** — Jaccard, Cosine, JSD, Kappa. Never self-reported.
4. **S2 is pure Python** — Zero API calls. Overlap counting on claim tuples.
5. **Failure is data** — S3 failure recirculates (max 3 cycles), then routes to human review
6. **Singulars are signal** — TYPE 3 claims preserved, surfaced to human reviewer
7. **Lab Gate kills bad hypotheses** — Falsifiable AND feasible AND novel, all three required
8. **S5 is pure Python** — Zero LLM calls. Monte Carlo only.

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
| `src/stages/synthesis.py` | S2 — Contribution Synthesis (0 API calls, overlap counting) |
| `src/convergence/convergence.py` | Jaccard, Cosine, JSD, Kappa on parsed claims |
| `src/convergence/claim_tuples.py` | ClaimTuple extraction — synonyms, relations, values |
| `src/stages/stages.py` | S1→S2→S3 orchestration + recirculation |
| `src/verify/verify.py` | VERIFY — Perplexity TYPE 2 claim verification |
| `src/gate/gate.py` | Lab Gate — falsifiability, feasibility, novelty |
| `src/hypothesis/s4_hypothesis.py` | S4 — hypothesis operationalization + parameter maps |
| `src/monte_carlo/monte_carlo.py` | S5 — pure Python Monte Carlo, 300+ iterations, 95% CIs |
| `src/protocol/protocol.py` | S6 — protocol package generator (JSON + summary) |
| `priors/*.json` | Quantitative priors across 11 scientific domains |
| `src/dashboard.py` | Live terminal dashboard — ANSI real-time metrics |
| `tests/` | Tests across 11 test files |

## Lineage

- **iris-gate v0.2** (2025) — 8-stage, 185-350 calls. Legacy at `~/iris-gate/`
- **IRIS Gate Evo** (2026) — 9-stage, 12-42 calls. This repo.
- Legacy engines ported from `~/iris-gate/sandbox/engines/`
- Legacy priors ported from `~/iris-gate/sandbox/states/`

## Test Question (Pipeline Validation)

> "What are the mechanisms by which CBD induces selective cytotoxicity in cancer cells while sparing healthy cells, with specific reference to VDAC1-mediated mitochondrial membrane potential disruption?"

Expected compiler priors: VDAC1 Kd = 11.0 uM, TRPV1 Kd = 2.0 uM, cancer psi = -120mV, healthy psi = -180mV, ROS baseline 0.45 vs 0.08.
