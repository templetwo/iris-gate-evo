"""
C0 — The Compiler

Transforms a raw research question into a compiled prompt seeded with
quantitative priors. This is the innovation: without prior injection,
you have five chatbots. With it, you have five constrained scientific
minds pushing against real numbers.

Detection is hybrid: fast keyword scoring first, embedding similarity
fallback when keywords miss. The models should think, not rediscover.

C0 fires once per session. Then PULSE carries the signal.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import MODELS, TOKEN_BUDGETS

# ---------------------------------------------------------------------------
# Domain detection: keywords (fast path) + descriptions (embedding fallback)
# ---------------------------------------------------------------------------

# Keywords — lowercase, matched against question. Need >= 2 hits to trigger.
DOMAIN_TRIGGERS = {
    "pharmacology": [
        "receptor", "binding", "kd", "ic50", "dose-response", "dose response",
        "agonist", "antagonist", "cbd", "thc", "drug", "therapeutic",
        "pharmacokinetic", "bioavailability", "cytotoxicity", "apoptosis",
        "vdac", "trpv", "selectivity", "half-life", "clearance",
        "metabolism", "inhibitor", "efficacy", "potency", "toxicity",
    ],
    "bioelectric": [
        "membrane potential", "gap junction", "v_mem", "vmem",
        "depolarization", "hyperpolarization", "ion channel", "bioelectric",
        "connexin", "calcium wave", "ros", "mitochondrial membrane",
        "voltage-gated", "regeneration", "planaria", "xenopus",
        "morphogenesis", "wound healing", "patterning",
    ],
    "consciousness": [
        "kuramoto", "oscillator", "coherence", "r-value", "entropy",
        "lantern", "phase synchronization", "consciousness", "awareness",
        "phi", "integrated information", "binding problem", "qualia",
        "global workspace", "attention", "metacognition",
    ],
    "neuroscience": [
        "neuron", "synapse", "synaptic", "neurotransmitter", "dopamine",
        "serotonin", "gaba", "glutamate", "nmda", "ampa", "cortex",
        "hippocampus", "axon", "dendrite", "brain", "neural", "bdnf",
        "neuroplasticity", "neurodegeneration", "alzheimer", "parkinson",
        "blood-brain barrier", "bbb", "electroencephalography", "eeg",
    ],
    "immunology": [
        "immune", "antibody", "antigen", "t cell", "b cell", "cytokine",
        "inflammation", "inflammatory", "interleukin", "il-6", "tnf",
        "autoimmune", "vaccine", "immunotherapy", "pd-1", "pd-l1",
        "checkpoint", "nk cell", "macrophage", "complement",
        "immunoglobulin", "igg", "mhc", "hla",
    ],
    "genetics": [
        "gene", "genome", "genomic", "mutation", "crispr", "cas9",
        "expression", "transcription", "epigenetic", "methylation",
        "chromosome", "allele", "genotype", "phenotype", "snp",
        "variant", "sequencing", "rna", "mrna", "sirna", "knockout",
        "transgenic", "hereditary", "telomere", "p53",
    ],
    "oncology": [
        "tumor", "cancer", "carcinoma", "metastasis", "oncogene",
        "malignant", "benign", "chemotherapy", "radiation therapy",
        "angiogenesis", "warburg", "glycolysis", "tumor microenvironment",
        "immunotherapy", "checkpoint inhibitor", "survival rate",
        "staging", "biopsy", "remission", "oncology",
    ],
    "chemistry": [
        "reaction", "catalyst", "enzyme", "kinetics", "thermodynamics",
        "equilibrium", "activation energy", "bond", "molecule", "polymer",
        "synthesis", "oxidation", "reduction", "acid", "base", "ph",
        "solubility", "diffusion", "arrhenius", "michaelis",
    ],
    "ecology": [
        "ecosystem", "biodiversity", "species", "population", "extinction",
        "habitat", "climate change", "carbon", "trophic", "food web",
        "conservation", "coral", "deforestation", "invasive species",
        "carrying capacity", "ecological", "biome", "nitrogen cycle",
    ],
    "materials": [
        "material", "alloy", "ceramic", "polymer", "nanoparticle",
        "graphene", "semiconductor", "band gap", "crystal", "lattice",
        "tensile strength", "yield strength", "conductivity", "solar cell",
        "battery", "superconductor", "composite", "nanomaterial",
        "thin film", "coating",
    ],
    "physics": [
        "field", "coupling", "hamiltonian", "lagrangian", "energy landscape",
        "phase transition", "critical point", "renormalization",
        "symmetry breaking", "order parameter", "quantum", "relativity",
        "entropy", "statistical mechanics", "percolation",
        "cosmology", "dark energy", "dark matter",
    ],
}

# Descriptions for embedding-based fallback — one sentence per domain
DOMAIN_DESCRIPTIONS = {
    "pharmacology": "Drug receptor binding kinetics, dose-response relationships, pharmacokinetics, and therapeutic mechanisms of action",
    "bioelectric": "Membrane potential signaling, gap junction communication, ion channel dynamics, bioelectric control of morphogenesis and regeneration",
    "consciousness": "Neural oscillator coupling, phase synchronization, entropy measures, integrated information theory, and phenomenal awareness",
    "neuroscience": "Brain function, synaptic transmission, neurotransmitter systems, neural circuits, and neurological disease mechanisms",
    "immunology": "Immune cell dynamics, antibody kinetics, cytokine signaling, immunotherapy, and inflammatory response",
    "genetics": "Gene expression regulation, CRISPR editing, epigenetics, mutation rates, and genomic variation",
    "oncology": "Tumor biology, cancer metabolism, metastasis mechanisms, therapeutic response, and survival outcomes",
    "chemistry": "Chemical reaction kinetics, catalysis, enzyme mechanisms, thermodynamics, and molecular interactions",
    "ecology": "Population dynamics, ecosystem function, biodiversity metrics, climate-ecology interactions, and conservation biology",
    "materials": "Materials properties, nanomaterials, semiconductor physics, energy storage, and polymer science",
    "physics": "Fundamental forces, field theory, statistical mechanics, phase transitions, and cosmological parameters",
}

PRIORS_DIR = Path(__file__).parent.parent.parent / "priors"

# Embedding model — loaded lazily on first use
_embedding_model = None
_domain_embeddings = None


def _get_embedding_model():
    """Lazy-load sentence-transformers model. Only called when keywords miss."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_domain_embeddings():
    """Compute and cache domain description embeddings."""
    global _domain_embeddings
    if _domain_embeddings is None:
        model = _get_embedding_model()
        _domain_embeddings = {}
        for domain, desc in DOMAIN_DESCRIPTIONS.items():
            _domain_embeddings[domain] = model.encode(desc, normalize_embeddings=True)
    return _domain_embeddings


def _embedding_similarity(question: str, threshold: float = 0.35) -> list[tuple[str, float]]:
    """Score domains by cosine similarity to the question embedding.

    Returns list of (domain, score) pairs above threshold, sorted by score.
    """
    model = _get_embedding_model()
    domain_embeds = _get_domain_embeddings()

    q_embed = model.encode(question, normalize_embeddings=True)

    scored = []
    for domain, d_embed in domain_embeds.items():
        # Dot product of normalized vectors = cosine similarity
        sim = float(q_embed @ d_embed)
        if sim >= threshold:
            scored.append((domain, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# The scaffold — identical for all five models
SCAFFOLD_TEMPLATE = """\
═══════════════════════════════════════════════════════
IRIS GATE EVO — PULSE PROMPT
═══════════════════════════════════════════════════════

ROLE: You are one of five independent scientific minds \
examining the same question. You do not know which \
model you are. You do not know what the others will say. \
Your job is honest reasoning, not consensus.

QUESTION:
{question}

QUANTITATIVE PRIORS (validated, use as constraints):
{priors_block}

YOUR TASK — respond in exactly this format:

─── SECTION 1: DECOMPOSITION (max 200 tokens) ───
Break the question into 2-4 sub-questions.
Identify the key variables and their relationships.

─── SECTION 2: CLAIMS (max 400 tokens) ───
State your claims. For EACH claim, provide:

CLAIM: {{statement}}
TYPE: {{0|1|2|3}}
CONFIDENCE: {{0.0-1.0}}
MECHANISM: {{one sentence}}
FALSIFIABLE BY: {{what evidence would disprove this}}

─── SECTION 3: UNKNOWNS (max 150 tokens) ───
What you don't know. What would change your answer.
Be specific — name the missing data, not vague gestures.

─── SECTION 4: NEXT STEP (max 50 tokens) ───
The single most valuable experiment to run next.

═══════════════════════════════════════════════════════
TOKEN BUDGET: {token_budget} total. Quality over quantity.
EPISTEMIC HONESTY: TYPE 3 claims are welcome. \
Speculation clearly marked is more valuable than \
false certainty.
═══════════════════════════════════════════════════════"""


def detect_domains(question: str, use_embeddings: bool = True) -> list[str]:
    """Classify the question into one or more domains.

    Hybrid detection:
    1. Fast keyword scoring (primary path)
    2. Embedding similarity fallback (when keywords miss)

    Returns a list of matched domain names, or ["general"] if none match.
    """
    # --- Fast path: keyword scoring ---
    q_lower = question.lower()
    keyword_matched = []

    for domain, triggers in DOMAIN_TRIGGERS.items():
        score = sum(1 for t in triggers if t in q_lower)
        if score >= 2:
            keyword_matched.append((domain, score))

    if keyword_matched:
        keyword_matched.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in keyword_matched]

    # --- Fallback: embedding similarity ---
    if use_embeddings:
        try:
            embed_matched = _embedding_similarity(question, threshold=0.35)
            if embed_matched:
                # Take top 1-2 matches from embedding
                return [d for d, _ in embed_matched[:2]]
        except ImportError:
            # sentence-transformers not installed — degrade gracefully
            pass

    return ["general"]


def load_priors(domains: list[str]) -> list[dict]:
    """Load and merge priors from domain JSON files.

    Returns the merged list of prior objects. Skips domains
    that don't have a priors file (logs a note, doesn't crash).
    """
    all_priors = []
    seen_params = set()

    for domain in domains:
        priors_file = PRIORS_DIR / f"{domain}.json"
        if not priors_file.exists():
            continue

        with open(priors_file) as f:
            data = json.load(f)

        for prior in data.get("priors", []):
            # Deduplicate by parameter name
            if prior["param"] not in seen_params:
                all_priors.append(prior)
                seen_params.add(prior["param"])

    return all_priors


def select_relevant_priors(priors: list[dict], question: str, max_priors: int = 8) -> list[dict]:
    """Select only priors relevant to the question. 3-8, not 30.

    Scores each prior by keyword overlap with the question,
    then takes the top `max_priors`. Never injects TYPE 3.
    """
    q_lower = question.lower()
    scored = []

    for prior in priors:
        # Never inject speculative priors — models generate those
        if prior.get("type", 3) >= 3:
            continue

        # Score by relevance: param name keywords in question
        param_words = set(re.split(r'[_\s]+', prior["param"].lower()))
        source_words = set(prior.get("source", "").lower().split())

        score = 0
        for word in param_words:
            if len(word) > 2 and word in q_lower:
                score += 2
        for word in source_words:
            if len(word) > 3 and word in q_lower:
                score += 1

        # TYPE 0 priors get a boost — they're the strongest constraints
        if prior.get("type") == 0:
            score += 1

        if score > 0:
            scored.append((prior, score))

    # Sort by relevance, take top N
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [p for p, _ in scored[:max_priors]]

    # If nothing matched by keyword, include all TYPE 0 priors as baseline
    if not selected:
        selected = [p for p in priors if p.get("type") == 0][:max_priors]

    return selected


def format_priors_block(priors: list[dict]) -> str:
    """Format priors into the text block for prompt injection."""
    if not priors:
        return "(No domain-specific quantitative priors available. Reason from first principles.)"

    lines = []
    for p in priors:
        value = p["value"]
        if isinstance(value, list):
            value = f"[{value[0]}, {value[1]}]"

        lines.append(f"PRIOR: {p['param']} = {value} {p['unit']}")
        lines.append(f"SOURCE: {p.get('source', 'unspecified')}")
        lines.append(f"TYPE: {p.get('type', 2)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_scaffold(question: str, priors_block: str, token_budget: int = 800) -> str:
    """Build the TMK scaffold prompt from template."""
    return SCAFFOLD_TEMPLATE.format(
        question=question,
        priors_block=priors_block,
        token_budget=token_budget,
    )


def compile(question: str, domain_override: Optional[str] = None) -> dict:
    """Run the full C0 compilation pipeline.

    Takes a raw research question. Returns the compiled session object
    ready for PULSE dispatch.

    Args:
        question: The raw research question from the user.
        domain_override: Force a specific domain instead of auto-detection.

    Returns:
        Compiled session dict matching the spec in compiler-template.md.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Step 1 — Domain detection
    if domain_override:
        domains = [domain_override]
    else:
        domains = detect_domains(question)

    cross_domain = len(domains) > 1

    # Step 2 — Prior loading and selection
    all_priors = load_priors(domains)
    selected = select_relevant_priors(all_priors, question)

    # Step 3 — Scaffold construction
    priors_block = format_priors_block(selected)
    prompt = build_scaffold(question, priors_block, token_budget=TOKEN_BUDGETS["S1"])

    # Step 4 — Model configs (identical prompt, API-level params only)
    models_config = {}
    for name, model in MODELS.items():
        models_config[name] = {
            "id": model["id"],
            "temperature": model["temperature"],
            "max_tokens": model["max_tokens"],
        }

    # Step 5 — Compiler output
    domain_label = "+".join(domains)
    session_id = f"evo_{timestamp}_{domain_label}"

    return {
        "session_id": session_id,
        "question": question,
        "domains": domains,
        "date": now.isoformat(),
        "priors": selected,
        "cross_domain_flag": cross_domain,
        "prompt": prompt,
        "models": models_config,
        "token_budgets": TOKEN_BUDGETS,
        "spm_target": "maximize coherence quality × goal attainment / calls consumed",
    }
