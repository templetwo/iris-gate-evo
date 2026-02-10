#!/usr/bin/env python3
"""
Diagnostic: Cosine similarity distribution for claim-level embeddings.

Tests the actual embedding model (all-MiniLM-L6-v2) with representative
CBD/VDAC1 claims to understand the cosine distribution and calibrate
CLAIM_COSINE_THRESHOLD. No API calls — pure local embedding.

Usage: python diag_cosine.py
"""

import numpy as np
from src.convergence.convergence import _get_embed_model

# ── Representative claims from 5 models ──
# Modeled after what real models produce for the CBD/VDAC1 question.
# Each model has 6-8 claims at varying granularity.

CLAIMS_BY_MODEL = {
    "claude": [
        "CBD binds to VDAC1 with high affinity causing mitochondrial membrane depolarization",
        "CBD selectively induces apoptosis in cancer cells while sparing healthy cells",
        "CBD activates TRPV1 channels leading to calcium influx and downstream signaling",
        "Cancer cells have a more depolarized mitochondrial membrane potential around -120mV compared to healthy cells at -180mV",
        "CBD increases reactive oxygen species production preferentially in cancer cells",
        "CBD inhibits the anti-apoptotic protein Bcl-2 leading to cytochrome c release",
        "The VDAC1-CBD interaction has a dissociation constant of approximately 11 uM",
    ],
    "mistral": [
        "Cannabidiol interacts with voltage-dependent anion channel 1 disrupting mitochondrial membrane potential",
        "CBD demonstrates selective cytotoxicity against cancer cells with minimal toxicity to normal cells",
        "TRPV1 receptor activation by CBD causes intracellular calcium elevation",
        "Mitochondrial membrane potential is more depolarized in cancer cells than in healthy cells",
        "CBD promotes oxidative stress through increased ROS generation in tumor cells",
        "CBD triggers the intrinsic apoptotic pathway via cytochrome c release from mitochondria",
        "VDAC1 mediates CBD-induced mitochondrial permeability transition",
    ],
    "grok": [
        "CBD binds VDAC1 on the outer mitochondrial membrane with Kd of 11 uM",
        "CBD kills cancer cells but not healthy cells through mitochondrial dysfunction",
        "CBD activates TRPV1 causing calcium influx which triggers apoptotic signaling",
        "Cancer cell mitochondria are more vulnerable to CBD due to their depolarized membrane potential",
        "Reactive oxygen species levels increase after CBD treatment in cancer cells",
        "Cytochrome c is released from mitochondria following CBD-VDAC1 interaction",
        "CBD modulates PPARgamma receptor activity contributing to anti-cancer effects",
        "Gap junction communication is altered by CBD treatment affecting bystander cell death",
    ],
    "gemini": [
        "CBD targets VDAC1 in the mitochondrial outer membrane leading to membrane depolarization and apoptosis",
        "The selective cytotoxicity of CBD toward cancer cells is mediated by differences in mitochondrial membrane potential",
        "CBD-mediated TRPV1 activation results in calcium influx and downstream apoptotic signaling",
        "Cancer cells exhibit a baseline mitochondrial membrane potential of approximately -120mV versus -180mV in healthy cells",
        "CBD induces elevated ROS production specifically in cancer cells through mitochondrial disruption",
        "CBD promotes cytochrome c release and caspase cascade activation leading to programmed cell death",
    ],
    "deepseek": [
        "CBD interacts with VDAC1 causing mitochondrial outer membrane permeabilization",
        "CBD shows selective anti-cancer activity by exploiting the depolarized mitochondrial state of tumor cells",
        "TRPV1 channel activation by cannabidiol leads to calcium-dependent cell death signaling",
        "The difference in mitochondrial membrane potential between cancer and healthy cells determines CBD selectivity",
        "CBD treatment increases reactive oxygen species in cancer cells above the apoptotic threshold",
        "CBD-induced VDAC1 opening leads to cytochrome c release and activation of intrinsic apoptosis",
        "Endoplasmic reticulum stress contributes to CBD-mediated apoptosis in cancer cells",
    ],
}


def main():
    print("Loading embedding model...")
    model = _get_embed_model()

    # Flatten all claims with metadata
    all_texts = []
    claim_meta = []
    for model_name, claims in CLAIMS_BY_MODEL.items():
        for claim in claims:
            all_texts.append(claim)
            claim_meta.append({"model": model_name, "text": claim})

    print(f"Embedding {len(all_texts)} claims from {len(CLAIMS_BY_MODEL)} models...")
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    # Compute full cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T
    n = len(all_texts)

    # Collect cross-model similarities
    cross_model_sims = []
    same_model_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if claim_meta[i]["model"] != claim_meta[j]["model"]:
                cross_model_sims.append((sim, i, j))
            else:
                same_model_sims.append((sim, i, j))

    cross_model_sims.sort(reverse=True)

    # ── Distribution analysis ──
    sims_only = [s for s, _, _ in cross_model_sims]
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COSINE SIMILARITY DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Total pairs:  {len(sims_only)}")
    print(f"Mean:         {np.mean(sims_only):.4f}")
    print(f"Median:       {np.median(sims_only):.4f}")
    print(f"Std:          {np.std(sims_only):.4f}")
    print(f"Min:          {np.min(sims_only):.4f}")
    print(f"Max:          {np.max(sims_only):.4f}")

    # Histogram at different thresholds
    print(f"\n{'─'*70}")
    print(f"THRESHOLD ANALYSIS (cross-model pairs above threshold)")
    print(f"{'─'*70}")
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        above = sum(1 for s in sims_only if s >= threshold)
        pct = 100 * above / len(sims_only)
        print(f"  >= {threshold:.2f}:  {above:4d} / {len(sims_only)} ({pct:5.1f}%)")

    # Top 30 cross-model pairs
    print(f"\n{'─'*70}")
    print(f"TOP 30 CROSS-MODEL PAIRS (highest cosine)")
    print(f"{'─'*70}")
    for rank, (sim, i, j) in enumerate(cross_model_sims[:30], 1):
        m_i = claim_meta[i]["model"][:8]
        m_j = claim_meta[j]["model"][:8]
        t_i = claim_meta[i]["text"][:60]
        t_j = claim_meta[j]["text"][:60]
        print(f"\n  #{rank:2d}  cos={sim:.4f}  [{m_i}] vs [{m_j}]")
        print(f"       A: {t_i}...")
        print(f"       B: {t_j}...")

    # ── Cluster simulation: COMPLETE-LINKAGE (new algorithm) ──
    print(f"\n{'='*70}")
    print(f"CLUSTER SIMULATION — COMPLETE-LINKAGE (production algorithm)")
    print(f"{'='*70}")
    from src.stages.synthesis import _cluster_claims
    from src.parser import Claim

    # Build claim_meta in the format _cluster_claims expects
    synth_meta = []
    for model_name, claims_list in CLAIMS_BY_MODEL.items():
        for claim_text in claims_list:
            synth_meta.append({
                "model": model_name,
                "claim": Claim(statement=claim_text, type=2, confidence=0.8, mechanism="", falsifiable_by=""),
                "text": claim_text,
            })

    for threshold in [0.70, 0.75, 0.80, 0.85]:
        clusters = _cluster_claims(embeddings, synth_meta, threshold=threshold)
        clusters.sort(key=lambda c: -len(c))

        from collections import defaultdict
        overlap_dist = defaultdict(int)
        for cluster in clusters:
            models = set(synth_meta[i]["model"] for i in cluster)
            overlap_dist[len(models)] += 1

        singulars = overlap_dist.get(1, 0)
        non_singular = len(clusters) - singulars

        print(f"\n  Threshold = {threshold:.2f}")
        print(f"    Total clusters: {len(clusters)}  (converged: {non_singular}, singulars: {singulars})")
        print(f"    Overlap distribution: ", end="")
        for ov in sorted(overlap_dist.keys(), reverse=True):
            label = f"{ov}/5"
            print(f"{label}={overlap_dist[ov]}  ", end="")
        print()

        # Show all non-singular clusters
        for ci, cluster in enumerate(clusters):
            models = set(synth_meta[i]["model"] for i in cluster)
            if len(models) > 1:
                print(f"    Cluster {ci+1}: {len(cluster)} claims, {len(models)} models")
                for idx in cluster:
                    m = synth_meta[idx]["model"][:8]
                    t = synth_meta[idx]["text"][:70]
                    print(f"      [{m}] {t}...")

    # ── Compare OLD (union-find) vs NEW (complete-linkage) at 0.80 ──
    print(f"\n{'='*70}")
    print(f"COMPARISON: union-find vs complete-linkage at threshold=0.80")
    print(f"{'='*70}")

    # Union-find (old)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if claim_meta[i]["model"] != claim_meta[j]["model"]:
                if sim_matrix[i, j] >= 0.80:
                    union(i, j)
    uf_clusters = defaultdict(list)
    for i in range(n):
        uf_clusters[find(i)].append(i)
    uf_list = sorted(uf_clusters.values(), key=lambda c: -len(c))
    uf_overlap = defaultdict(int)
    for cluster in uf_list:
        models = set(claim_meta[i]["model"] for i in cluster)
        uf_overlap[len(models)] += 1

    # Complete-linkage (new)
    cl_clusters = _cluster_claims(embeddings, synth_meta, threshold=0.80)
    cl_clusters.sort(key=lambda c: -len(c))
    cl_overlap = defaultdict(int)
    for cluster in cl_clusters:
        models = set(synth_meta[i]["model"] for i in cluster)
        cl_overlap[len(models)] += 1

    print(f"  Union-find:      {len(uf_list)} clusters — ", end="")
    for ov in sorted(uf_overlap.keys(), reverse=True):
        print(f"{ov}/5={uf_overlap[ov]}  ", end="")
    print()
    print(f"  Complete-linkage: {len(cl_clusters)} clusters — ", end="")
    for ov in sorted(cl_overlap.keys(), reverse=True):
        print(f"{ov}/5={cl_overlap[ov]}  ", end="")
    print()

    # ── Synonym normalization test ──
    print(f"\n{'='*70}")
    print(f"SYNONYM NORMALIZATION IMPACT")
    print(f"{'='*70}")
    from src.convergence.claim_tuples import ENTITY_SYNONYMS, _MULTI_WORD_SYNONYMS

    def normalize_text(text: str) -> str:
        t = text.lower()
        for surface, canonical in _MULTI_WORD_SYNONYMS:
            t = t.replace(surface, canonical)
        for surface, canonical in ENTITY_SYNONYMS.items():
            if " " not in surface and "-" not in surface:
                t = t.replace(surface.lower(), canonical)
        return t

    norm_texts = [normalize_text(t) for t in all_texts]
    norm_embeddings = model.encode(norm_texts, normalize_embeddings=True)
    norm_sim = norm_embeddings @ norm_embeddings.T

    norm_cross = []
    for i in range(n):
        for j in range(i + 1, n):
            if claim_meta[i]["model"] != claim_meta[j]["model"]:
                norm_cross.append(float(norm_sim[i, j]))

    print(f"  Original  — Mean: {np.mean(sims_only):.4f}, Median: {np.median(sims_only):.4f}")
    print(f"  Normalized— Mean: {np.mean(norm_cross):.4f}, Median: {np.median(norm_cross):.4f}")
    improvement = np.mean(norm_cross) - np.mean(sims_only)
    print(f"  Improvement: {improvement:+.4f}")

    for threshold in [0.55, 0.60, 0.65, 0.70]:
        orig_above = sum(1 for s in sims_only if s >= threshold)
        norm_above = sum(1 for s in norm_cross if s >= threshold)
        print(f"  >= {threshold:.2f}:  Original {orig_above} → Normalized {norm_above} ({norm_above - orig_above:+d})")


if __name__ == "__main__":
    main()
