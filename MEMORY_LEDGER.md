# IRIS Gate Evo — Memory Ledger

Project memory log for IRIS Gate Evo multi-LLM convergence protocol. Each entry documents verified facts about development milestones, experimental results, and methodological discoveries.

---

## 2026-02-11T19:08:00Z - Discovered TYPE 1 consensus failure via VDAC1 gating polarity error

- Run evo_20260211_201329_pharmacology+bioelectric asked: "Does circadian cycling of mitochondrial membrane potential alter the dose-response threshold at which CBD shifts from cytoprotective to cytotoxic effects via VDAC1?"
- S3 gate FAILED after 3 recirculation cycles, 15 total API calls (convergence score 0.4838, threshold 0.85, TYPE 0+1 ratio 0.4, threshold 0.65)
- Human fact-check against PubMed found 3/5 models (gemini/grok/mistral) converged on wrong VDAC1 gating polarity: claimed "open at depolarized ΔΨm", real literature shows bell-curve gating with P_open peak at ~0 mV on OMM (Frontiers Physiol 2025, Springer 2021)
- S2 Contribution Synthesis promoted wrong gating polarity to TYPE 1 (≥3/5 model consensus), demonstrating that TYPE 1 can be wrong when models share training data biases
- Claude's TYPE 2/3 singulars (Donnan equilibrium OMM voltage gating, HK-II/VDAC1 dissociation via GSK3β/AKT) were more scientifically accurate than the 3/5 consensus
- One genuinely novel hypothesis survived full literature verification: "Circadian cycling of GSK3β/AKT activity drives rhythmic HK-II dissociation from VDAC1, creating a time-dependent vulnerability window for CBD's dose-dependent pathway switching" — all four mechanistic components verified independently (GSK3β→VDAC1 phosphorylation, HK-II dissociation, VDAC1 oligomerization, circadian AKT/GSK3β), but no prior literature integrates them into chronopharmacology model
- Methodological implication: need COUNTER-CONSENSUS SINGULAR flag in pipeline — when a singular contradicts TYPE 1, may indicate shared training contamination rather than independent replication failure
- Cross-run convergence detection tool implemented and validated (18 tests passing), found cross-validated singulars across lithium and THC runs
- S2 serialization bug fixed: claims now saved as dict objects in s2_synthesis.json, not repr() strings
- Community context: Reddit r/cbdinfo post at 2.4k views, bevon (mod) collaboration established, Sara Ward (Temple U) follow-up email sent
- Files: /Users/vaquez/Iris-Gate-Evo/runs/evo_20260211_201329_pharmacology+bioelectric/ contains s1_formulations.json (39.4KB), s2_synthesis.json (11.9KB), s3_convergence.json (7.1KB), fact_check.md (4.0KB)

---

