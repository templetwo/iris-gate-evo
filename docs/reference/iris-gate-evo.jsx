import { useState } from "react";

const c = {
  bg: "#0a0e1a", card: "#111827", border: "#1e293b",
  blue: "#3b82f6", purple: "#8b5cf6", cyan: "#06b6d4",
  green: "#10b981", amber: "#f59e0b", red: "#ef4444", pink: "#f472b6",
  teal: "#2dd4bf", text: "#e2e8f0", muted: "#94a3b8", dim: "#64748b",
};

const stages = [
  {
    id: "input", ch: null, color: c.blue, icon: "👤",
    title: "User Input", calls: "0",
    purpose: "Raw scientific question. No preprocessing. Domain-agnostic.",
    input: "Natural language question",
    output: "Unmodified question → Compiler",
    example: '"What are the molecular mechanisms of CBD\'s selective cytotoxicity in glioblastoma vs healthy astrocytes?"',
  },
  {
    id: "compiler", ch: "C0", color: c.purple, icon: "⚙️",
    title: "Prompt Compiler", calls: "1",
    purpose: "The core innovation. A single LLM takes the raw question and enriches it with computational priors — real numbers, not vibes. Detects domain, pulls quantitative constraints, builds structured scaffold, generates 5 model-specific prompt variants. This is what separates IRIS Gate from 'five chatbots answering the same question.'",
    input: "Raw question + domain knowledge base",
    output: "5 model-specific structured prompts with embedded priors",
    tasks: [
      "Domain detection (pharmacology, physics, etc.)",
      "Quantitative prior injection (Kd values, Hill coefficients, constants)",
      "TMK scaffold (Task-Method-Knowledge JSON)",
      "Model-specific format optimization",
      "Epistemic TYPE instruction embedding",
    ],
    example: "Injects: Kd_VDAC1=11.0μM, Hill_n=1.5, ψ_healthy=-180mV, ψ_cancer=-120mV",
  },
  {
    id: "pulse", ch: "PULSE", color: c.cyan, icon: "📡",
    title: "5 Mirrors — Parallel", calls: "5 per round",
    purpose: "All 5 models receive compiled prompts simultaneously via LiteLLM async. Fresh context per model — zero cross-contamination. Architectural diversity is the mathematical prerequisite for ensemble superiority (Q-diversity).",
    mirrors: [
      { name: "Claude", note: "Constitutional AI — nuanced reasoning" },
      { name: "GPT", note: "Dense transformer — broad knowledge" },
      { name: "Grok", note: "Real-time reasoning — direct communication" },
      { name: "Gemini", note: "Multimodal — long context window" },
      { name: "DeepSeek", note: "Open MoE — mathematical reasoning" },
    ],
    input: "5 compiled prompts",
    output: "5 independent responses → Convergence Engine",
  },
  {
    id: "s1", ch: "S1", color: c.green, icon: "🔭",
    title: "Formulation", calls: "5 (single round)",
    purpose: "One-shot first contact. Each model independently decomposes the question, identifies variables, proposes initial hypotheses with TYPE tags. Single round — the refinement loop does the real work.",
    input: "Compiled prompt, empty prior state",
    output: "5 formulations → S1 consensus",
    convergence: "Orchestrator synthesizes majority consensus via Jaccard + embedding similarity. Expect low agreement (~0.3–0.5). That's fine — it's raw material.",
    tokenBudget: "800 max",
  },
  {
    id: "s2", ch: "S2", color: c.green, icon: "🔬",
    title: "Refinement Loop", calls: "50–75 (5 × 10–15 rounds)",
    purpose: "The engine room. One adaptive loop (merged v2.0's S2+S3). Anonymized cross-model debate — each model sees all 5 responses without knowing which architecture produced what. Prevents sycophancy and anchoring bias. Early-stop kills wasted iterations.",
    input: "S1 consensus (≤2000 tokens) as prior state",
    output: "Refined, compressed claim set → S2 consensus",
    tasks: [
      "Integrate consensus with own reasoning",
      "Challenge or strengthen anonymous mirror claims",
      "Compress: same ideas, fewer tokens (4–8% per round)",
      "Sharpen TYPE classifications",
      "Eliminate redundant / low-confidence claims",
      "Resolve inter-model disagreements",
    ],
    convergence: "Early stop: Δ < 1% for 3 consecutive rounds AND ≥80% TYPE 0/1.\nConvergence Engine metrics fed back each round.\nAnonymization is non-negotiable.",
    tokenBudget: "800 → 700 (decreasing)",
  },
  {
    id: "s3", ch: "S3", color: c.green, icon: "🎯",
    title: "Stable Attractor", calls: "15–25 (5 × 3–5 rounds)",
    purpose: "Convergence chamber. Strictest token budget forces maximum compression. Searching for a fixed point — claims no further iteration will change. Computational priors crystallize here.",
    input: "S2 consensus with iteration history",
    output: "S3 Computational Priors — the converged state",
    convergence: "CONVERGENCE GATE:\n• Jaccard > 0.85 across models\n• ≥90% claims TYPE 0 or TYPE 1\n• Compression Δ < 1% for 3+ rounds\n• TYPE distribution stable (ratio shift < 5%)\n\nFAIL → loop back to S2 with human guidance.",
    tokenBudget: "600 max",
  },
  {
    id: "verify", ch: "VERIFY", color: c.pink, icon: "🔍",
    title: "Perplexity — Literature Check", calls: "5–15",
    purpose: "TYPE 2 (Novel/Emerging) claims get checked against current literature via Perplexity API. Research moves fast — a claim tagged TYPE 2 in S3 may have been validated or contradicted by a paper published last week. This is the system's connection to real-time ground truth.",
    input: "All TYPE 2 claims from S3 Computational Priors",
    output: "Verification status per claim → enriched priors → Lab Gate",
    tasks: [
      "Extract all TYPE 2 claims from S3 output",
      "Query Perplexity for each: recent papers, preprints, retractions",
      "Check date-sensitivity: has the field moved since training cutoff?",
      "Reclassify claims based on findings",
      "Attach citation links to verified claims",
    ],
    statuses: [
      { tag: "PROMOTED → TYPE 1", desc: "Recent literature confirms — upgrade to established" },
      { tag: "HELD — TYPE 2", desc: "Partially supported, still emerging" },
      { tag: "NOVEL", desc: "No match in literature — genuine hypothesis candidate" },
      { tag: "CONTRADICTED", desc: "Recent evidence conflicts — flag for review or demote to TYPE 3" },
    ],
  },
  {
    id: "labgate", ch: "GATE", color: c.red, icon: "🚪",
    title: "Lab Gate", calls: "1 (compiler)",
    purpose: "Quality gate between observation and operationalization. The compiler evaluates verified priors against three criteria before the system spends another 20+ calls. Kills bad hypotheses early, saves money.",
    input: "S3 Computational Priors (Perplexity-enriched)",
    output: "PASS → S4 | FAIL → human review with specific failure reasons",
    checks: [
      { name: "Falsifiability", desc: "Every prediction must specify what would disprove it" },
      { name: "Feasibility", desc: "Can this be tested with available tools, budget, timeline?" },
      { name: "Novelty", desc: "Does this add to literature or just restate known results?" },
    ],
  },
  {
    id: "s4", ch: "S4", color: c.amber, icon: "💡",
    title: "Hypothesis + Parameters", calls: "10–15 (5 × 2–3 rounds)",
    purpose: "Merged hypothesis generation and parameter mapping. Generate falsifiable hypotheses AND map to simulation parameters in the same pass. Each hypothesis is a structured object ready for Monte Carlo.",
    input: "S3 priors (Lab Gate passed)",
    output: "Ranked hypotheses with simulation-ready parameters",
    format: "Each hypothesis:\n• Prediction — what will happen\n• Mechanism — causal chain (why)\n• Observable — how to measure it\n• Falsification — what disproves it\n• Parameters — {name: [min, mean, max, dist]}\n• TYPE [0–3] · Confidence [0.0–1.0]",
    tokenBudget: "1000 max",
  },
  {
    id: "s5", ch: "S5", color: c.amber, icon: "🎲",
    title: "Monte Carlo", calls: "0 (pure computation)",
    purpose: "No LLM calls. Python engine runs 300+ iterations per hypothesis. Generates confidence intervals and validates predictions against S3 priors.",
    input: "S4 parameter distributions",
    output: "Simulation results + CIs + statistical tests",
    tasks: [
      "N=300+ iterations per hypothesis",
      "95% confidence intervals",
      "Test against S3 prior expectations",
      "Flag hypotheses that fail validation",
    ],
  },
  {
    id: "s6", ch: "S6", color: c.amber, icon: "📋",
    title: "Protocol Package", calls: "5 (single round)",
    purpose: "One-round final deliverable. All 5 models contribute. The science was settled in S3 — this is just packaging.",
    input: "S5 results + S4 hypotheses + S3 priors",
    output: "Research package",
    deliverables: [
      "Convergence report (per-chamber metric traces)",
      "Ranked hypotheses with TYPE classification",
      "Monte Carlo results + confidence intervals",
      "Pre-registration draft",
      "Lab protocol (if applicable)",
      "Full model response audit trail",
    ],
  },
  {
    id: "engine", ch: null, color: c.teal, icon: "📊",
    title: "Convergence Engine", calls: "0 (server-side)",
    purpose: "Runs continuously alongside S1–S3. All metrics computed by the Python orchestrator — never self-reported by models. This is the measurement backbone.",
    metrics: [
      { name: "Jaccard Similarity", desc: "Lexical claim overlap. Fast, every response. Target >0.85." },
      { name: "Cosine Embedding", desc: "Semantic similarity (sentence-transformers). Every 2–5s streaming." },
      { name: "Jensen-Shannon Divergence", desc: "Distributional disagreement [0,1]. JSD→0 = convergence." },
      { name: "Fleiss' Kappa", desc: "5-rater TYPE classification agreement." },
      { name: "Bayesian Optimal Weighting", desc: "w_i = log(p_i/(1-p_i)). Beats majority voting." },
      { name: "TYPE Distribution", desc: "Rising T0/T1 ratio = convergence signal." },
    ],
  },
];

const Card = ({ stage: s, open, onToggle }) => (
  <div onClick={onToggle} style={{ background: c.card, border: `1px solid ${open ? s.color : c.border}`, borderRadius: 10, padding: "12px 14px", cursor: "pointer", transition: "border-color 0.2s" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ width: 32, height: 32, borderRadius: 7, background: s.color + "20", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>{s.icon}</div>
      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {s.ch && <span style={{ background: s.color + "30", color: s.color, fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 3 }}>{s.ch}</span>}
          <span style={{ color: c.text, fontSize: 13, fontWeight: 700 }}>{s.title}</span>
        </div>
        <div style={{ color: c.dim, fontSize: 10, marginTop: 1 }}>{s.calls} calls</div>
      </div>
      <span style={{ color: c.dim, fontSize: 14, transform: open ? "rotate(180deg)" : "none", transition: "0.2s" }}>▾</span>
    </div>
    {open && (
      <div style={{ marginTop: 12, borderTop: `1px solid ${c.border}`, paddingTop: 12 }}>
        <p style={{ color: c.muted, fontSize: 11, lineHeight: 1.7, margin: "0 0 10px" }}>{s.purpose}</p>
        {s.input && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 10 }}>
            <div style={{ background: c.bg, borderRadius: 6, padding: "6px 8px" }}>
              <div style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase", marginBottom: 2 }}>Input</div>
              <div style={{ color: c.muted, fontSize: 10 }}>{s.input}</div>
            </div>
            <div style={{ background: c.bg, borderRadius: 6, padding: "6px 8px" }}>
              <div style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase", marginBottom: 2 }}>Output</div>
              <div style={{ color: c.muted, fontSize: 10 }}>{s.output}</div>
            </div>
          </div>
        )}
        {s.tasks && <div style={{ marginBottom: 10 }}>
          <div style={{ color: s.color, fontSize: 9, fontWeight: 700, textTransform: "uppercase", marginBottom: 4 }}>Tasks</div>
          {s.tasks.map((t, i) => <div key={i} style={{ display: "flex", gap: 5, marginBottom: 3 }}><span style={{ color: s.color, fontSize: 9, marginTop: 1 }}>→</span><span style={{ color: c.muted, fontSize: 11, lineHeight: 1.4 }}>{t}</span></div>)}
        </div>}
        {s.mirrors && <div style={{ marginBottom: 10 }}>
          {s.mirrors.map((m, i) => <div key={i} style={{ background: c.bg, borderRadius: 5, padding: "4px 8px", marginBottom: 3 }}>
            <span style={{ color: s.color, fontSize: 11, fontWeight: 700 }}>{m.name}</span>
            <span style={{ color: c.dim, fontSize: 10 }}> — {m.note}</span>
          </div>)}
        </div>}
        {s.metrics && <div style={{ marginBottom: 10 }}>
          {s.metrics.map((m, i) => <div key={i} style={{ background: c.bg, borderRadius: 5, padding: "6px 8px", marginBottom: 3 }}>
            <div style={{ color: c.text, fontSize: 10, fontWeight: 600 }}>{m.name}</div>
            <div style={{ color: c.muted, fontSize: 10, marginTop: 1 }}>{m.desc}</div>
          </div>)}
        </div>}
        {s.checks && <div style={{ marginBottom: 10 }}>
          {s.checks.map((ck, i) => <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 5 }}>
            <span style={{ background: c.red + "30", color: c.red, fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 3, flexShrink: 0 }}>{ck.name}</span>
            <span style={{ color: c.muted, fontSize: 11 }}>{ck.desc}</span>
          </div>)}
        </div>}
        {s.statuses && <div style={{ marginBottom: 10 }}>
          <div style={{ color: s.color, fontSize: 9, fontWeight: 700, textTransform: "uppercase", marginBottom: 4 }}>Reclassification Outcomes</div>
          {s.statuses.map((st, i) => <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 5 }}>
            <span style={{ background: [c.green,c.amber,c.purple,c.red][i] + "30", color: [c.green,c.amber,c.purple,c.red][i], fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 3, flexShrink: 0, whiteSpace: "nowrap" }}>{st.tag}</span>
            <span style={{ color: c.muted, fontSize: 11 }}>{st.desc}</span>
          </div>)}
        </div>}
        {s.deliverables && <div style={{ marginBottom: 10 }}>
          <div style={{ color: s.color, fontSize: 9, fontWeight: 700, textTransform: "uppercase", marginBottom: 4 }}>Deliverables</div>
          {s.deliverables.map((d, i) => <div key={i} style={{ display: "flex", gap: 5, marginBottom: 2 }}><span style={{ color: s.color, fontSize: 9 }}>📄</span><span style={{ color: c.muted, fontSize: 11 }}>{d}</span></div>)}
        </div>}
        {s.format && <div style={{ background: c.bg, borderRadius: 6, padding: "8px 10px", marginBottom: 10, fontFamily: "monospace" }}>
          <div style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase", marginBottom: 3 }}>Format</div>
          <div style={{ color: c.cyan, fontSize: 10, lineHeight: 1.6, whiteSpace: "pre-line" }}>{s.format}</div>
        </div>}
        {s.convergence && <div style={{ background: c.green + "10", border: `1px solid ${c.green}25`, borderRadius: 6, padding: "8px 10px", marginBottom: 6 }}>
          <div style={{ color: c.green, fontSize: 8, fontWeight: 700, textTransform: "uppercase", marginBottom: 3 }}>Convergence</div>
          <div style={{ color: c.muted, fontSize: 10, lineHeight: 1.6, whiteSpace: "pre-line" }}>{s.convergence}</div>
        </div>}
        {s.tokenBudget && <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase" }}>Token Budget:</span>
          <span style={{ background: s.color + "20", color: s.color, fontSize: 10, fontWeight: 600, padding: "1px 5px", borderRadius: 3 }}>{s.tokenBudget}</span>
        </div>}
        {s.example && <div style={{ marginTop: 6, background: c.bg, borderRadius: 6, padding: "6px 8px" }}>
          <div style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase", marginBottom: 2 }}>Example</div>
          <div style={{ color: c.cyan, fontSize: 10, fontStyle: "italic", lineHeight: 1.5 }}>{s.example}</div>
        </div>}
      </div>
    )}
  </div>
);

export default function IRISGateEvo() {
  const [expanded, setExpanded] = useState(new Set(["compiler"]));
  const [view, setView] = useState("both");
  const toggle = (id) => setExpanded(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });

  const W = 860, H = 1150, cx = W / 2;

  const obsStages = stages.filter(s => ["s1","s2","s3"].includes(s.id));
  const opStages = stages.filter(s => ["s4","s5","s6"].includes(s.id));
  const pipeStages = stages.filter(s => ["input","compiler","pulse"].includes(s.id));

  return (
    <div style={{ background: c.bg, minHeight: "100vh", padding: "16px", fontFamily: "system-ui" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <h1 style={{ color: c.text, fontSize: 22, fontWeight: 800, margin: 0 }}>IRIS Gate Evo</h1>
          <p style={{ color: c.muted, fontSize: 12, margin: "4px 0 0" }}>Lean Multi-Architecture Scientific Convergence Protocol</p>
          <p style={{ color: c.dim, fontSize: 10, margin: "2px 0 10px" }}>Vasquez, A.J. (2026) · Temple of Two</p>
          <div style={{ display: "flex", justifyContent: "center", gap: 6 }}>
            {["both","diagram","legend"].map(v => (
              <button key={v} onClick={() => setView(v)} style={{ background: view === v ? c.purple + "30" : c.card, border: `1px solid ${view === v ? c.purple : c.border}`, color: view === v ? c.purple : c.muted, borderRadius: 6, padding: "5px 12px", fontSize: 10, fontWeight: 600, cursor: "pointer", textTransform: "uppercase" }}>{v}</button>
            ))}
          </div>
        </div>

        {/* Pipeline bar */}
        <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 10, padding: "10px 14px", marginBottom: 16, textAlign: "center" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flexWrap: "wrap", gap: 3 }}>
            {[
              { l: "User", c: c.blue }, { l: "→" }, { l: "C0", c: c.purple }, { l: "→" },
              { l: "PULSE ×5", c: c.cyan }, { l: "→" }, { l: "S1", c: c.green }, { l: "→" },
              { l: "S2", c: c.green }, { l: "⟲", c: c.red }, { l: "S3", c: c.green }, { l: "→" },
              { l: "VERIFY", c: c.pink }, { l: "→" }, { l: "GATE", c: c.red }, { l: "→" }, { l: "S4", c: c.amber }, { l: "→" },
              { l: "S5", c: c.amber }, { l: "→" }, { l: "S6", c: c.amber }, { l: "→" }, { l: "📦", c: c.blue },
            ].map((x, i) => x.c
              ? <span key={i} style={{ background: x.c + "25", color: x.c, fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4 }}>{x.l}</span>
              : <span key={i} style={{ color: c.dim, fontSize: 11 }}>{x.l}</span>
            )}
          </div>
          <div style={{ color: c.dim, fontSize: 10, marginTop: 4 }}>~92–142 API calls · ~$1.50–4.00 per run</div>
        </div>

        {/* Epistemic types */}
        <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 10, padding: "10px 14px", marginBottom: 16 }}>
          <div style={{ color: c.dim, fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: "1px", marginBottom: 6 }}>Epistemic Classification</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>
            {[
              { t: "TYPE 0", d: "Crisis / Conditional — IF-THEN causal chains", col: "#22c55e", act: "TRUST" },
              { t: "TYPE 1", d: "Established — literature-backed mechanisms", col: "#3b82f6", act: "TRUST" },
              { t: "TYPE 2", d: "Novel / Emerging — grounded but unverified", col: "#f59e0b", act: "VERIFY" },
              { t: "TYPE 3", d: "Speculation — beyond current evidence", col: "#ef4444", act: "OVERRIDE" },
            ].map(t => (
              <div key={t.t} style={{ background: t.col + "10", border: `1px solid ${t.col}25`, borderRadius: 6, padding: "6px 8px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: t.col, fontSize: 11, fontWeight: 700 }}>{t.t}</span>
                  <span style={{ background: t.col + "30", color: t.col, fontSize: 8, fontWeight: 700, padding: "1px 5px", borderRadius: 3 }}>{t.act}</span>
                </div>
                <div style={{ color: c.muted, fontSize: 10, marginTop: 2 }}>{t.d}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
          {/* DIAGRAM */}
          {(view === "both" || view === "diagram") && (
            <div style={{ flex: view === "diagram" ? "1" : "0 0 480px" }}>
              <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
                <defs>
                  {[c.dim, c.blue, c.purple, c.cyan, c.green, c.amber, c.red, c.teal, c.pink].map(col => (
                    <marker key={col} id={`a${col.replace('#','')}`} viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto-start-reverse">
                      <path d="M0 0L10 3L0 6z" fill={col}/>
                    </marker>
                  ))}
                </defs>
                <pattern id="g" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0L0 0 0 20" fill="none" stroke={c.border} strokeWidth="0.3" opacity="0.3"/></pattern>
                <rect width={W} height={H} fill="url(#g)" opacity="0.4"/>

                {/* INPUT */}
                <text x={20} y={25} fill={c.blue} fontSize={10} fontWeight={700} fontFamily="system-ui">INPUT</text>
                <rect x={cx-70} y={35} width={140} height={38} rx={10} fill={c.card} stroke={c.blue} strokeWidth={1}/>
                <text x={cx} y={58} textAnchor="middle" fill={c.blue} fontSize={12} fontWeight={700} fontFamily="system-ui">👤 User</text>
                <line x1={cx} y1={73} x2={cx} y2={95} stroke={c.blue} strokeWidth={1.5} markerEnd={`url(#a${c.blue.replace('#','')})`}/>

                {/* COMPILER */}
                <text x={20} y={92} fill={c.purple} fontSize={10} fontWeight={700} fontFamily="system-ui">C0 — COMPILER</text>
                <rect x={cx-110} y={100} width={220} height={42} rx={10} fill={c.purple+"15"} stroke={c.purple} strokeWidth={1.5}/>
                <text x={cx} y={119} textAnchor="middle" fill={c.purple} fontSize={12} fontWeight={700} fontFamily="system-ui">⚙️ Prompt Compiler</text>
                <text x={cx} y={133} textAnchor="middle" fill={c.muted} fontSize={9} fontFamily="system-ui">Domain → Priors → TMK → 5 Prompts</text>
                
                <text x={180} y={110} textAnchor="end" fill={c.dim} fontSize={8} fontFamily="system-ui">priors</text>
                <line x1={185} y1={112} x2={cx-112} y2={118} stroke={c.purple} strokeWidth={1} strokeDasharray="4 3"/>
                <text x={cx+140} y={110} fill={c.dim} fontSize={8} fontFamily="system-ui">quantitative</text>
                <line x1={cx+112} y1={118} x2={cx+135} y2={112} stroke={c.purple} strokeWidth={1} strokeDasharray="4 3"/>

                <line x1={cx} y1={142} x2={cx} y2={165} stroke={c.purple} strokeWidth={1.5} markerEnd={`url(#a${c.purple.replace('#','')})`}/>
                <text x={cx+5} y={158} fill={c.dim} fontSize={8} fontFamily="system-ui">5 compiled prompts</text>

                {/* PULSE */}
                <text x={20} y={172} fill={c.cyan} fontSize={10} fontWeight={700} fontFamily="system-ui">PULSE — PARALLEL</text>
                {[
                  { n: "GPT", x: 90 }, { n: "Grok", x: 230 }, { n: "Claude", x: 370 }, { n: "Gemini", x: 510 }, { n: "DeepSeek", x: 650 },
                ].map((m) => (
                  <g key={m.n}>
                    <line x1={cx} y1={165} x2={m.x+55} y2={185} stroke={c.cyan} strokeWidth={1} markerEnd={`url(#a${c.cyan.replace('#','')})`}/>
                    <rect x={m.x} y={188} width={110} height={32} rx={8} fill={c.card} stroke={c.cyan} strokeWidth={1}/>
                    <text x={m.x+55} y={208} textAnchor="middle" fill={c.cyan} fontSize={11} fontWeight={600} fontFamily="system-ui">{m.n}</text>
                  </g>
                ))}
                {[145, 285, 425, 565, 705].map((x, i) => (
                  <line key={i} x1={x} y1={220} x2={cx} y2={255} stroke={c.cyan} strokeWidth={1} markerEnd={`url(#a${c.cyan.replace('#','')})`}/>
                ))}

                {/* S1 */}
                <text x={20} y={255} fill={c.green} fontSize={10} fontWeight={700} fontFamily="system-ui">OBSERVATION — S1→S3</text>
                <rect x={cx-85} y={265} width={170} height={36} rx={10} fill={c.card} stroke={c.green} strokeWidth={1}/>
                <text x={cx} y={283} textAnchor="middle" fill={c.green} fontSize={11} fontWeight={700} fontFamily="system-ui">🔭 S1 — Formulation</text>
                <text x={cx} y={295} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">5 calls · single round</text>
                <line x1={cx} y1={301} x2={cx} y2={325} stroke={c.green} strokeWidth={1.5} markerEnd={`url(#a${c.green.replace('#','')})`}/>

                {/* S2 */}
                <rect x={cx-95} y={330} width={190} height={42} rx={10} fill={c.card} stroke={c.green} strokeWidth={1.5}/>
                <text x={cx} y={350} textAnchor="middle" fill={c.green} fontSize={11} fontWeight={700} fontFamily="system-ui">🔬 S2 — Refinement Loop</text>
                <text x={cx} y={363} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">50–75 calls · anonymized debate</text>
                
                {/* Loop arrow */}
                <path d={`M${cx-97} 345 Q${cx-135} 345 ${cx-135} 360 Q${cx-135} 375 ${cx-97} 375`} fill="none" stroke={c.green} strokeWidth={1} strokeDasharray="4 3"/>
                <text x={cx-148} y={364} fill={c.dim} fontSize={7} fontFamily="system-ui" transform={`rotate(-90 ${cx-148} 364)`}>iterate</text>

                {/* Convergence Engine box */}
                <rect x={590} y={268} width={230} height={170} rx={8} fill={c.teal+"08"} stroke={c.teal} strokeWidth={1} strokeDasharray="4 3"/>
                <text x={705} y={288} textAnchor="middle" fill={c.teal} fontSize={10} fontWeight={700} fontFamily="system-ui">📊 Convergence Engine</text>
                <text x={705} y={301} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">(server-side, never self-reported)</text>
                {["Jaccard Similarity", "Cosine Embedding", "Jensen-Shannon Div", "Fleiss' Kappa (5)", "Bayesian OW", "TYPE Distribution"].map((m, i) => (
                  <g key={m}><circle cx={605} cy={316+i*18} r={2.5} fill={c.teal}/><text x={614} y={320+i*18} fill={c.muted} fontSize={9} fontFamily="system-ui">{m}</text></g>
                ))}
                
                <line x1={cx+97} y1={340} x2={588} y2={340} stroke={c.teal} strokeWidth={1} strokeDasharray="4 3"/>
                <line x1={588} y1={365} x2={cx+97} y2={365} stroke={c.teal} strokeWidth={1} strokeDasharray="4 3" markerEnd={`url(#a${c.teal.replace('#','')})`}/>
                <text x={540} y={336} fill={c.dim} fontSize={7} fontFamily="system-ui">responses</text>
                <text x={540} y={374} fill={c.dim} fontSize={7} fontFamily="system-ui">metrics</text>

                {/* Early stop badge */}
                <rect x={cx+105} y={383} width={180} height={16} rx={8} fill={c.green+"15"} stroke={c.green} strokeWidth={0.5}/>
                <text x={cx+195} y={394} textAnchor="middle" fill={c.green} fontSize={8} fontWeight={600} fontFamily="system-ui">EARLY STOP: Δ&lt;1% × 3 rounds</text>

                <line x1={cx} y1={372} x2={cx} y2={410} stroke={c.green} strokeWidth={1.5} markerEnd={`url(#a${c.green.replace('#','')})`}/>

                {/* S3 */}
                <rect x={cx-95} y={415} width={190} height={42} rx={10} fill={c.green+"12"} stroke={c.green} strokeWidth={1.5}/>
                <text x={cx} y={435} textAnchor="middle" fill={c.green} fontSize={11} fontWeight={700} fontFamily="system-ui">🎯 S3 — Stable Attractor</text>
                <text x={cx} y={448} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">15–25 calls · 600 token max</text>
                
                {/* Convergence gate */}
                <line x1={cx} y1={457} x2={cx} y2={480} stroke={c.green} strokeWidth={1.5} markerEnd={`url(#a${c.green.replace('#','')})`}/>
                <rect x={cx-45} y={485} width={90} height={26} rx={13} fill={c.green+"20"} stroke={c.green} strokeWidth={1.5}/>
                <text x={cx} y={502} textAnchor="middle" fill={c.green} fontSize={10} fontWeight={700} fontFamily="system-ui">CONVERGED?</text>
                
                <text x={cx+58} y={493} fill={c.red} fontSize={8} fontWeight={600} fontFamily="system-ui">NO</text>
                <path d={`M${cx+45} 498 Q${cx+110} 498 ${cx+110} 445 Q${cx+110} 365 ${cx+97} 355`} fill="none" stroke={c.red} strokeWidth={1} strokeDasharray="4 3" markerEnd={`url(#a${c.red.replace('#','')})`}/>
                
                <text x={cx-8} y={525} fill={c.green} fontSize={8} fontWeight={600} fontFamily="system-ui">YES</text>
                <line x1={cx} y1={511} x2={cx} y2={540} stroke={c.green} strokeWidth={1.5} markerEnd={`url(#a${c.green.replace('#','')})`}/>

                {/* VERIFY — Perplexity */}
                <rect x={cx-90} y={545} width={180} height={38} rx={10} fill={c.pink+"12"} stroke={c.pink} strokeWidth={1.5}/>
                <text x={cx} y={563} textAnchor="middle" fill={c.pink} fontSize={11} fontWeight={700} fontFamily="system-ui">🔍 VERIFY — Perplexity</text>
                <text x={cx} y={576} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">5–15 calls · TYPE 2 literature check</text>

                {["PROMOTED→T1", "HELD T2", "NOVEL", "CONTRADICTED"].map((st, i) => (
                  <g key={st}>
                    <rect x={cx+105} y={548+i*13} width={st.length*6+8} height={12} rx={3} fill={[c.green,c.amber,c.purple,c.red][i]+"25"} stroke={[c.green,c.amber,c.purple,c.red][i]} strokeWidth={0.5}/>
                    <text x={cx+105+(st.length*6+8)/2} y={557+i*13} textAnchor="middle" fill={[c.green,c.amber,c.purple,c.red][i]} fontSize={7} fontWeight={600} fontFamily="system-ui">{st}</text>
                  </g>
                ))}

                <line x1={cx} y1={583} x2={cx} y2={610} stroke={c.pink} strokeWidth={1.5} markerEnd={`url(#a${c.pink.replace('#','')})`}/>

                {/* LAB GATE */}
                <rect x={cx-75} y={615} width={150} height={34} rx={10} fill={c.red+"15"} stroke={c.red} strokeWidth={1.5}/>
                <text x={cx} y={632} textAnchor="middle" fill={c.red} fontSize={11} fontWeight={700} fontFamily="system-ui">🚪 Lab Gate</text>
                <text x={cx} y={644} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">Falsify · Feasible · Novel</text>
                
                <text x={cx+88} y={628} fill={c.red} fontSize={8} fontWeight={600} fontFamily="system-ui">FAIL → Human</text>
                <line x1={cx+75} y1={632} x2={cx+85} y2={632} stroke={c.red} strokeWidth={1} strokeDasharray="3 2"/>

                <text x={cx-8} y={663} fill={c.green} fontSize={8} fontWeight={600} fontFamily="system-ui">PASS</text>
                <line x1={cx} y1={649} x2={cx} y2={680} stroke={c.amber} strokeWidth={1.5} markerEnd={`url(#a${c.amber.replace('#','')})`}/>

                {/* OPERATIONAL */}
                <text x={20} y={682} fill={c.amber} fontSize={10} fontWeight={700} fontFamily="system-ui">OPERATIONAL — S4→S6</text>
                
                <rect x={cx-95} y={690} width={190} height={36} rx={10} fill={c.card} stroke={c.amber} strokeWidth={1}/>
                <text x={cx} y={708} textAnchor="middle" fill={c.amber} fontSize={11} fontWeight={700} fontFamily="system-ui">💡 S4 — Hypothesis + Params</text>
                <text x={cx} y={720} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">10–15 calls</text>

                <line x1={cx} y1={726} x2={cx} y2={750} stroke={c.amber} strokeWidth={1.5} markerEnd={`url(#a${c.amber.replace('#','')})`}/>

                <rect x={cx-80} y={755} width={160} height={36} rx={10} fill={c.card} stroke={c.amber} strokeWidth={1}/>
                <text x={cx} y={773} textAnchor="middle" fill={c.amber} fontSize={11} fontWeight={700} fontFamily="system-ui">🎲 S5 — Monte Carlo</text>
                <text x={cx} y={785} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">0 LLM calls · Python</text>

                <line x1={cx} y1={791} x2={cx} y2={815} stroke={c.amber} strokeWidth={1.5} markerEnd={`url(#a${c.amber.replace('#','')})`}/>

                <rect x={cx-85} y={820} width={170} height={36} rx={10} fill={c.card} stroke={c.amber} strokeWidth={1}/>
                <text x={cx} y={838} textAnchor="middle" fill={c.amber} fontSize={11} fontWeight={700} fontFamily="system-ui">📋 S6 — Protocol Package</text>
                <text x={cx} y={850} textAnchor="middle" fill={c.dim} fontSize={8} fontFamily="system-ui">5 calls · single round</text>

                <line x1={cx} y1={856} x2={cx} y2={880} stroke={c.amber} strokeWidth={1.5} markerEnd={`url(#a${c.amber.replace('#','')})`}/>

                {/* OUTPUT */}
                <text x={20} y={885} fill={c.blue} fontSize={10} fontWeight={700} fontFamily="system-ui">OUTPUT</text>
                <rect x={cx-100} y={892} width={200} height={40} rx={10} fill={c.card} stroke={c.blue} strokeWidth={1.5}/>
                <text x={cx} y={911} textAnchor="middle" fill={c.blue} fontSize={12} fontWeight={700} fontFamily="system-ui">📦 Research Package</text>
                <text x={cx} y={925} textAnchor="middle" fill={c.muted} fontSize={9} fontFamily="system-ui">Report · Hypotheses · CI · Protocol · Audit</text>

                {/* CALL SUMMARY */}
                <rect x={40} y={955} width={W-80} height={90} rx={10} fill={c.card} stroke={c.border} strokeWidth={1}/>
                <text x={60} y={977} fill={c.text} fontSize={10} fontWeight={700} fontFamily="system-ui">CALL BUDGET (Evo)</text>
                {[
                  ["C0:1", c.purple, 60], ["S1:5", c.green, 130], ["S2:50–75", c.green, 195], ["S3:15–25", c.green, 310],
                  ["VERIFY:5–15", c.pink, 420], ["GATE:1", c.red, 540], ["S4:10–15", c.amber, 610], ["S5:0", c.amber, 700], ["S6:5", c.amber, 755],
                ].map(([l, col, x]) => (
                  <g key={l}><rect x={x} y={985} width={l.length*6.5+8} height={18} rx={4} fill={col+"20"} stroke={col} strokeWidth={0.5}/><text x={x+(l.length*6.5+8)/2} y={997} textAnchor="middle" fill={col} fontSize={8} fontWeight={600} fontFamily="system-ui">{l}</text></g>
                ))}
                <text x={60} y={1025} fill={c.text} fontSize={11} fontWeight={700} fontFamily="system-ui">Total: 92–142 calls</text>
                <text x={300} y={1025} fill={c.green} fontSize={11} fontWeight={600} fontFamily="system-ui">~$1.50–4.00/run</text>
                <text x={500} y={1025} fill={c.dim} fontSize={10} fontFamily="system-ui">vs v2.0: 185–350 calls ($2–8)</text>

                <text x={cx} y={H-10} textAnchor="middle" fill={c.dim} fontSize={9} fontFamily="system-ui">IRIS Gate Evo · Vasquez, A.J. (2026) · Temple of Two</text>
              </svg>
            </div>
          )}

          {/* LEGEND */}
          {(view === "both" || view === "legend") && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 5, marginBottom: 4 }}>
                <button onClick={() => setExpanded(new Set(stages.map(s => s.id)))} style={{ background: c.card, border: `1px solid ${c.border}`, color: c.muted, borderRadius: 5, padding: "4px 10px", fontSize: 10, cursor: "pointer" }}>Expand All</button>
                <button onClick={() => setExpanded(new Set())} style={{ background: c.card, border: `1px solid ${c.border}`, color: c.muted, borderRadius: 5, padding: "4px 10px", fontSize: 10, cursor: "pointer" }}>Collapse</button>
              </div>

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px" }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.purple, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Pipeline Entry
              </div>
              {pipeStages.map(s => <Card key={s.id} stage={s} open={expanded.has(s.id)} onToggle={() => toggle(s.id)}/>)}

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: 8 }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.green, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Observation Layer — S1→S3
              </div>
              {obsStages.map(s => <Card key={s.id} stage={s} open={expanded.has(s.id)} onToggle={() => toggle(s.id)}/>)}

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: 8 }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.pink, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Literature Verification
              </div>
              <Card stage={stages.find(s => s.id === "verify")} open={expanded.has("verify")} onToggle={() => toggle("verify")}/>

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: 8 }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.red, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Quality Gate
              </div>
              <Card stage={stages.find(s => s.id === "labgate")} open={expanded.has("labgate")} onToggle={() => toggle("labgate")}/>

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: 8 }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.amber, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Operational Layer — S4→S6
              </div>
              {opStages.map(s => <Card key={s.id} stage={s} open={expanded.has(s.id)} onToggle={() => toggle(s.id)}/>)}

              <div style={{ color: c.dim, fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.5px", marginTop: 8 }}>
                <span style={{ display: "inline-block", width: 3, height: 10, background: c.teal, borderRadius: 1, marginRight: 6, verticalAlign: "middle" }}/>Infrastructure
              </div>
              <Card stage={stages.find(s => s.id === "engine")} open={expanded.has("engine")} onToggle={() => toggle("engine")}/>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
