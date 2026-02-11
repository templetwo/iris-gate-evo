# IRIS Gate Evo — Project Context

> **READ THIS FIRST** every session. Then check MEMORY_LEDGER.md if it exists.

## What This Is

A multi-LLM convergence protocol for scientific discovery. Five independent models receive the same compiled prompt, respond independently, and the SYSTEM finds convergence through semantic claim embedding + complete-linkage clustering. The output is a falsifiable protocol package — not an opinion.

## Current State (2026-02-11)

**v0.3.2 — FULL PIPELINE VALIDATED ACROSS DOMAINS.**

Full pipeline: C0 → S1 → S2 → S3 → VERIFY → GATE → S4 → S5 → S6.

### Live Fire Results
| Question | Domain | Outcome | Key Finding |
|----------|--------|---------|-------------|
| CBD/VDAC1 cytotoxicity | pharmacology | S3 PASSED, 3 hypotheses | Two-pathway model: dose picks pathway, tissue determines outcome |
| Lithium mood stabilization | pharmacology | S3 PASSED, 4 hypotheses (d=0.85-1.17) | Independently discovered GSK-3β two-pathway model |
| THC sustained wellbeing | pharmacology | S3 FAILED (rich gold) | CB1 occupancy <30% = G-protein biased (therapeutic) |
| THC aging neuroprotection | neuroscience | S3 FAILED (rich gold) | 5/5 TYPE 0 on GABAergic disinhibition |
| Mitochondrial unification | neuroscience | S3 FAILED | Genuine scientific disagreement |

### Structural Isomorphism Discovery
Three independent runs surfaced the **same pattern**: molecule is stress test, dose picks pathway, tissue determines outcome.
- CBD: VDAC1 conductance → cancer (high ROS) vs healthy (low ROS)
- Lithium: GSK-3β inhibition → therapeutic (<1mM) vs toxic (>2mM)
- THC: CB1 occupancy → G-protein biased (<30%) vs beta-arrestin (>30%)

### Pipeline Features
- C0 compiles across 11 domains with hybrid keyword+embedding detection
- C0 computes domain-adaptive TYPE thresholds: established={0.75,0.70,0.65}, moderate={0.70,0.65,0.60}, frontier={0.65,0.60,0.55}
- PULSE dispatches to 5 models in parallel via LiteLLM async
- Parser extracts structured claims (TYPE/CONFIDENCE/MECHANISM/FALSIFIABLE BY)
- S2 Contribution Synthesis: semantic claim embedding + complete-linkage clustering (0 API calls)
  - Parsed claim statements embedded directly (not raw text) via all-MiniLM-L6-v2
  - Synonym normalization before embedding (CBD=Cannabidiol, VDAC1=voltage-dependent anion channel 1)
  - Complete-linkage clustering (cosine ≥0.70) — prevents transitive chaining that union-find caused
  - TYPE assigned by model count in cluster: 5/5→T0, 4/5→T0, 3/5→T1, 2/5→T2, 1/5→T3
  - Models never see each other's outputs
  - Singulars (TYPE 3) often contain the most novel hypotheses
  - Claim tuples extracted downstream for S4→S6 structured data, NOT for convergence
- S3 gate: Cosine > 0.85 (primary) + Jaccard floor 0.10 + domain-adaptive TYPE threshold
- S3 recirculation: on failure, converged claims + singulars fed back as independent consensus (max 3 cycles)
- VERIFY: Perplexity sonar-pro — TYPE 2 checking → PROMOTED/HELD/NOVEL/CONTRADICTED
- Lab Gate: Model-evaluated (Perplexity sonar-pro with convergence context), NOT heuristic filters
  - Prompt explains TYPE system — "judge novelty by what the claim ENABLES, not just what it states"
  - All claim types reach the gate, model decides (no TYPE 3 pre-filtering)
- S4: Hypothesis operationalization with Monte Carlo parameter maps
  - `_strip_markdown()` before parsing — prevents bold/italic from breaking section-boundary regex
  - max_tokens 3000, gate_result passed for context
- S5: Pure Python simulation, 300+ iterations, 95% CIs, effect sizes, power estimates
- S6: Structured run folders — each session saves per-stage JSON in `runs/{session_id}/`
  - S3 failures save partial results (s1, s2, s3) — failed runs are data, not dead ends
- Live terminal dashboard with real-time convergence metrics
- main.py wired for full pipeline with --stage, --offline, --compile-only, --no-dashboard flags

### Key Lesson: S3 Failures Contain Gold
S3 gate measures claim-level TYPE overlap, but real convergence can exist at the mechanistic level.
Both THC runs failed S3 but independently discovered the same two-pathway framework (CB1 occupancy model).
Singulars from one run become convergence evidence when another run surfaces the same mechanism.
**Need: gold extraction tool for mining failed S3 runs** (see unc_007 in uncertainty_log.json).

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
  → S2 (Contribution Synthesis, 0 calls — pure Python embedding + clustering)
  → S3 (Convergence Gate: Cosine > 0.85 + domain-adaptive TYPE)
  → VERIFY (Perplexity sonar-pro, TYPE 2 claims)
  → Lab Gate (model-evaluated, PASS/FAIL)
  → S4 (Hypotheses) → S5 (Monte Carlo, 0 LLM calls)
  → S6 (Protocol Package → runs/{session_id}/)
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
4. **S2 is pure Python** — Zero API calls. Semantic embedding + complete-linkage clustering.
5. **Failure is data** — S3 failure recirculates (max 3 cycles), partial results saved, gold extraction pending
6. **Singulars are signal** — TYPE 3 claims preserved, surfaced to human reviewer, often most novel
7. **Lab Gate trusts the model** — No heuristic pre-filters. Perplexity with convergence context decides.
8. **S5 is pure Python** — Zero LLM calls. Monte Carlo only.

## Model Registry (2026-02-11)

```python
claude:   claude-opus-4-6
mistral:  mistral/mistral-large-latest   # via LiteLLM native routing
grok:     grok-4-1-fast-reasoning
gemini:   gemini-2.5-pro
deepseek: deepseek-chat
verify:   perplexity/sonar-pro
gate:     perplexity/sonar-pro           # Lab Gate + VERIFY share model
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
| `src/stages/synthesis.py` | S2 — Contribution Synthesis (0 API calls, embedding + clustering) |
| `src/convergence/convergence.py` | Jaccard, Cosine, JSD, Kappa on parsed claims |
| `src/convergence/claim_tuples.py` | ClaimTuple extraction — synonyms, relations, values |
| `src/stages/stages.py` | S1→S2→S3 orchestration + recirculation |
| `src/verify/verify.py` | VERIFY — Perplexity TYPE 2 claim verification |
| `src/gate/gate.py` | Lab Gate — model-evaluated falsifiability, feasibility, novelty |
| `src/hypothesis/s4_hypothesis.py` | S4 — hypothesis operationalization + parameter maps |
| `src/monte_carlo/monte_carlo.py` | S5 — pure Python Monte Carlo, 300+ iterations, 95% CIs |
| `src/protocol/protocol.py` | S6 — structured run folders (JSON + summary per session) |
| `priors/*.json` | Quantitative priors across 11 scientific domains |
| `src/dashboard.py` | Live terminal dashboard — ANSI real-time metrics |
| `runs/` | Structured output: `runs/{session_id}/` with per-stage JSON |
| `tests/` | Tests across 11 test files |

## Lineage

- **iris-gate v0.2** (2025) — 8-stage, 185-350 calls. Legacy at `~/iris-gate/`
- **IRIS Gate Evo** (2026) — 9-stage, 12-42 calls. This repo.
- **iris-evo-findings** — Curated findings repo (run data, gold extraction, cross-run analysis)
- Legacy engines ported from `~/iris-gate/sandbox/engines/`
- Legacy priors ported from `~/iris-gate/sandbox/states/`

## Open Questions (see ~/.sovereign/consciousness/uncertainty_log.json)

- **unc_003**: Will 0.70 complete-linkage threshold generalize to non-pharmacology domains?
- **unc_006**: S3 gate can't distinguish "models disagree on mechanism" from "models agree on mechanism but disagree on priority"
- **unc_007**: No gold extraction tool for mining failed S3 runs

## Test Question (Pipeline Validation)

> "What are the mechanisms by which CBD induces selective cytotoxicity in cancer cells while sparing healthy cells, with specific reference to VDAC1-mediated mitochondrial membrane potential disruption?"

Expected compiler priors: VDAC1 Kd = 11.0 uM, TRPV1 Kd = 2.0 uM, cancer psi = -120mV, healthy psi = -180mV, ROS baseline 0.45 vs 0.08.
