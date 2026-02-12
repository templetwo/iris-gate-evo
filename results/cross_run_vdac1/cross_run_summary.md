# Cross-Run Convergence Report

**Generated**: 2026-02-12 03:30 UTC
**Runs analyzed**: 6
**Total claims**: 55
**Cross-matches found**: 2

## Runs

| Session | Domain | S3 | Claims |
|---------|--------|----|--------|
| evo_20260210_232904_pharmacology | pharmacology | FAIL | 13 |
| evo_20260210_235609_pharmacology | pharmacology | FAIL | 9 |
| evo_20260211_024747_pharmacology | pharmacology | FAIL | 9 |
| evo_20260211_024750_neuroscience | neuroscience | FAIL | 7 |
| evo_20260211_201329_pharmacology+bioelectric | pharmacology+bioelectric | FAIL | 10 |
| evo_20260212_032819_pharmacology | pharmacology | FAIL | 7 |

## Cross-Validated Singulars

### CROSS-VALIDATED SINGULAR (cosine=0.7765)
- **Run A** (evo_20260210_232904_pharmacology, TYPE 3): Lithium neuroprotection at low dose (<0.5 mM) primarily via partial GSK3β inhibition (IC50 1-2 mM), stabilizing β-catenin, upregulating BDNF and autop
- **Run B** (evo_20260210_235609_pharmacology, TYPE 0): 2:** Neuroprotection at low dose is mediated primarily by indirect GSK-3β inhibition via Akt-Ser9 phosphorylation amplification, with effective EC50 ~

## Cross-Promoted (TYPE 2 → validated)

- **cosine=0.7914** | evo_20260211_024747_pharmacology (T2) ↔ evo_20260211_024750_neuroscience (T0)
  A: 1:** Chronic low-dose THC (≤2.5 mg/day) improves wellbeing via *biphasic CB1 signaling*: G-protein activation (EC50 ~10 
  B: 1:** The biphasic (inverted-U) dose-response of THC occurs because low CB1 occupancy (<20%) preferentially engages Gi/o-

## Structural Patterns (Isomorphism)

- **two_pathway**: evo_20260210_232904_pharmacology, evo_20260212_032819_pharmacology
- **threshold_crossover**: evo_20260210_232904_pharmacology, evo_20260210_235609_pharmacology, evo_20260211_024747_pharmacology, evo_20260211_024750_neuroscience, evo_20260212_032819_pharmacology
- **dose_dependent**: evo_20260210_232904_pharmacology, evo_20260211_024747_pharmacology, evo_20260212_032819_pharmacology

## Cosine Distribution (cross-run pairs)

- Min: -0.1937
- Median: 0.3075
- Mean: 0.3168
- Max: 0.7914
- N pairs: 1248
