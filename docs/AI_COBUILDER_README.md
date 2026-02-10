# IRIS Gate Evo — AI Co-Builder README

> *This document is written for you, Claude Code.*
> *Read it before touching anything.*

---

## What This Is

IRIS Gate Evo is a **multi-LLM convergence protocol for scientific discovery**. It sends the same research question to 5 independent AI models simultaneously, then uses anonymized debate and quantitative convergence metrics to extract truth from noise.

It is the evolution of [iris-gate v0.2](https://github.com/templetwo/iris-gate), rebuilt from scratch to be lean, independent, and measurable.

**You are building the engine. The architecture is complete. Your job is implementation.**

---

## The Architecture (read this carefully)

```
User Question
     │
     ▼
 ┌────────┐
 │   C0   │  Compiler — detects domain, loads priors,
 │Compiler│  builds identical prompt for all 5 models
 └────┬───┘
      │  1 call (compiler model)
      ▼
 ┌──────────────────────────────────────┐
 │            P U L S E                 │
 │  5 models receive compiled prompt    │
 │  simultaneously via LiteLLM async    │
 │                                      │
 │  Claude · GPT · Grok · Gemini · DS   │
 └────────────────┬─────────────────────┘
                  │  5 calls (parallel)
                  ▼
 ┌────────┐
 │   S1   │  Formulation — first contact, single round
 └────┬───┘   5 calls
      ▼
 ┌────────┐
 │   S2   │  Refinement Loop — anonymized cross-model debate
 │  ⟲     │  Early-stop: Δ < 1% for 3 rounds AND ≥80% TYPE 0/1
 └────┬───┘   50-75 calls (adaptive)
      ▼
 ┌────────┐
 │   S3   │  Stable Attractor — strictest convergence gate
 │  ◆     │  Jaccard > 0.85, ≥90% TYPE 0/1, compression stable
 └────┬───┘   15-25 calls
      ▼
 ┌────────┐
 │ VERIFY │  Perplexity — TYPE 2 claims checked against
 │   🔍   │  current literature, reclassified
 └────┬───┘   5-15 calls
      ▼
 ┌────────┐
 │  GATE  │  Lab Gate — falsifiability, feasibility, novelty
 │   ⊘    │  PASS → S4 / FAIL → human review
 └────┬───┘   1 call
      ▼
 ┌────────┐
 │   S4   │  Hypothesis + Parameters — falsifiable predictions
 │        │  with Monte Carlo parameter mappings
 └────┬───┘   10-15 calls
      ▼
 ┌────────┐
 │   S5   │  Monte Carlo — pure Python, ZERO LLM calls
 │  🎲    │  300+ iterations per hypothesis, 95% CIs
 └────┬───┘   0 calls
      ▼
 ┌────────┐
 │   S6   │  Protocol Package — final deliverable
 │   📦   │  Convergence report, ranked hypotheses, audit trail
 └────────┘   5 calls
```

**Total budget: 92-142 calls (~$1.50-4.00/run)**

---

## The Models (as of February 9, 2026)

These are the EXACT model strings to use. Do not substitute.

```python
MODELS = {
    "claude":   {"id": "claude-opus-4-6",            "provider": "anthropic",  "base_url": "https://api.anthropic.com/v1"},
    "gpt":      {"id": "gpt-5.2",                    "provider": "openai",     "base_url": "https://api.openai.com/v1"},
    "grok":     {"id": "grok-4-1-fast-reasoning",    "provider": "xai",        "base_url": "https://api.x.ai/v1"},
    "gemini":   {"id": "gemini-2.5-pro",             "provider": "google",     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
    "deepseek": {"id": "deepseek-chat",              "provider": "deepseek",   "base_url": "https://api.deepseek.com"},
}

# Verification layer (optional)
VERIFY_MODEL = {"id": "perplexity", "provider": "perplexity"}
```

**Do not use older model strings.** The repo you may see at `github.com/templetwo/iris-gate` has stale IDs (`claude-sonnet-4.5`, `gpt-5`, `grok-4`). Those are wrong.

---

## What's Already Here

```
Iris-Gate-Evo/
├── compiler-template.md          # C0 specification — READ THIS FIRST
├── iris-gate-evo.jsx             # Architecture diagram (React component)
├── Spiral-Tu...e Models.pages    # Philosophy: SPM = Quality × Attainment / Energy
├── IRIS Gate_ A...vergence.pdf   # v2.0 reference (superseded by Evo)
├── Latest LLM A...for 2026.pdf  # Model registry with pricing
└── AI_COBUILDER_README.md        # This file
```

**To port from `iris-gate` v0.2:**
- `sandbox/engines/` → `engines/` (V_mem, Ca²⁺, GJ simulators for S5)
- `sandbox/states/` → `priors/` (frozen S4 priors → first domain JSON)
- `templates/plan_template.yaml` → `templates/` (S6 output format)
- `templates/prereg_template.md` → `templates/` (pre-registration format)

---

## Build Order

**Phase 1 — Skeleton (get responses flowing)**
1. `compiler.py` — Implements C0 per `compiler-template.md`
2. `pulse.py` — LiteLLM async dispatch to 5 models
3. `models.py` — Model registry with the exact strings above
4. `main.py` — Wire C0 → PULSE → print 5 responses

Test: Ask the CBD cytotoxicity question. Get 5 structured responses back.

**Phase 2 — Convergence (make it think)**
5. `convergence.py` — Jaccard, Cosine (sentence-transformers), JSD, Kappa
6. `stages.py` — S1 → S2 (with early-stop) → S3 (convergence gate)
7. `anonymizer.py` — Strip model identity from cross-model debate prompts

Test: Run S1→S3. Watch Jaccard climb. Confirm early-stop fires.

**Phase 3 — Verification & Gating**
8. `verify.py` — Perplexity integration for TYPE 2 claims
9. `gate.py` — Lab Gate: falsifiability, feasibility, novelty check
10. `s4_hypothesis.py` — Operationalize converged priors into testable predictions

Test: TYPE 2 claim goes in, gets PROMOTED/HELD/NOVEL/CONTRADICTED.

**Phase 4 — Simulation & Output**
11. Port `engines/` from iris-gate v0.2 (V_mem, Ca²⁺, GJ)
12. `monte_carlo.py` — S5 simulation runner, 300+ iterations, 95% CIs
13. `protocol.py` — S6 package generator (convergence report, audit trail)

Test: End-to-end run. Question in, protocol package out.

---

## Critical Implementation Rules

### 1. Anonymization is non-negotiable
In S2 debate rounds, each model sees all 5 responses labeled `[Mirror A]` through `[Mirror E]`. Never `[Claude]` or `[GPT]`. Randomize the letter assignment each round. This prevents sycophancy and anchoring bias.

### 2. Token budgets decrease, never increase
S1: 800 → S2: 800→700 (decreasing) → S3: 600. This compression forces signal. If a model can't say it in 600 tokens by S3, it wasn't signal.

### 3. Convergence is server-side, never self-reported
Models do not judge their own convergence. The `convergence.py` engine computes:
- **Jaccard similarity** — lexical claim overlap (target > 0.85)
- **Cosine embedding** — semantic similarity via `all-MiniLM-L6-v2`
- **Jensen-Shannon Divergence** — distributional disagreement, JSD → 0 = convergence
- **Fleiss' Kappa** — 5-rater TYPE classification agreement
- **TYPE distribution** — rising T0/T1 ratio = system stabilizing

### 4. Early-stop saves half your budget
S2 exits when: compression delta < 1% for 3 consecutive rounds AND TYPE 0/1 ratio ≥ 80%. Do not run extra rounds "to be safe."

### 5. The compiler is the innovation
Without C0's quantitative prior injection, this is just "five chatbots answering the same question." The priors constrain the search space. They give the models something to push against. That's where convergence comes from.

### 6. Failure is data
If S3 fails convergence (Jaccard < 0.85), that's interesting — it means the question has genuine disagreement worth investigating. Route to human review with the divergence map, don't retry silently.

---

## The Epistemic TYPE System

Every claim in the system carries a TYPE tag:

| TYPE | Meaning | Action |
|------|---------|--------|
| **0** | Crisis/Conditional — high-confidence IF-THEN | TRUST |
| **1** | Established — literature-backed mechanism | TRUST |
| **2** | Novel/Emerging — grounded but unverified | VERIFY (Perplexity) |
| **3** | Speculation — beyond current evidence | OVERRIDE (human) |

TYPE distribution across iterations is a convergence signal. Rising T0/T1, falling T3 = the system is stabilizing on established claims.

---

## Dependencies

```
litellm>=1.40          # Unified LLM API layer
sentence-transformers  # all-MiniLM-L6-v2 for cosine similarity
scipy                  # JSD calculation
numpy                  # Monte Carlo + statistics
pyyaml                 # Config and plan files
python-dotenv          # .env for API keys
```

No database. No Docker. No cloud infra. This runs from a directory.

---

## Environment

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
XAI_API_KEY=xai-...
GOOGLE_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...
PERPLEXITY_API_KEY=pplx-...   # Optional
```

---

## Philosophy

This system is built on the **Spiral-Tuned Performance** principle:

**SPM = (Coherence Quality × Goal Attainment) / Energy Consumed**

Every architectural decision optimizes this ratio. We merged stages to cut calls. We added early-stop to prevent waste. We added Lab Gate to kill bad hypotheses before they consume 20 calls. The system monitors its own efficiency — not just its accuracy.

The models are not tools. They are five independent minds examining the same question from different training distributions. What survives anonymized debate across all five is more trustworthy than any single model's best answer.

---

## Lineage

- **iris-gate v0.2** (2025) — Original 8-stage protocol, 185-350 calls
- **IRIS Gate Evo** (2026) — Lean 9-stage protocol, 92-142 calls, same convergence quality
- **Spiral-Tuned Performance Framework** (June 2025) — The philosophical foundation
- **Threshold Protocols** — Self-governance principles inherited by Lab Gate

---

## First Test Question

Use this to validate the full pipeline:

> "What are the mechanisms by which CBD induces selective cytotoxicity in cancer cells while sparing healthy cells, with specific reference to VDAC1-mediated mitochondrial membrane potential disruption?"

The compiler should detect `pharmacology` + `bioelectric` domains and inject priors including: VDAC1 Kd = 11.0 μM, TRPV1 Kd = 2.0 μM, cancer ψ = -120mV vs healthy ψ = -180mV, ROS baseline 0.45 vs 0.08.

If those priors appear in the compiled prompt, C0 works.
If five different structured responses come back, PULSE works.
If Jaccard climbs across S2 rounds, convergence works.

---

*Five mirrors. One truth. Build it clean.*

🌀†⟡∞
