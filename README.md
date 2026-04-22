# AKNN — Aaryan's / Associative K-Nearest-Neighbor Neural Network

> **"Intelligence is geometry, not probability.**
> **One perfect example beats one million mediocre ones.**
> **The brain doesn't guess. It resonates."**

> ⚠️ **Monkey Patching Applied:** Runtime patches fix specific bugs in the code. See [`aknn.ipynb`](aknn.ipynb) for details.
---

[![Version](https://img.shields.io/badge/version-1.0-black?style=flat-square)](.)
[![License](https://img.shields.io/badge/license-MIT-darkred?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)](.)
[![Status](https://img.shields.io/badge/status-active%20research-green?style=flat-square)](.)
[![X](https://img.shields.io/badge/follow-%40Khan__Aaryan__-black?style=flat-square&logo=x)](https://x.com/Khan_Aaryan_)

---

## What is AKNN?

AKNN is a **dual-substrate neuro-manifold architecture** for associative knowledge storage and retrieval. It is not a transformer. It is not a vector database wrapper. It is a **synthetic brain** built from first principles.

AKNN encodes language into two parallel geometric substrates simultaneously:

1. **Quantum Substrate** — An 8-qubit parametric quantum circuit that encodes every sentence into a point in a 256-dimensional Hilbert space using grammar-aware angle embedding and StronglyEntanglingLayers. The PauliZ expectation vector `⟨Z⟩ ∈ [-1,1]^8` is the memory's geometric coordinate.

2. **Hyperdimensional Computing (HDC) Substrate** — A 10,000-dimensional bipolar vector space where every sentence is represented as a superposition of character-level and positional hypervectors `v ∈ {-1,+1}^10000`. Similarity is cosine distance. This runs at ~0.5W on any device.

Knowledge is stored as **MemoryCells** inside **Neurons** (cortical columns — expert domains). Retrieval finds the K=3 geometrically nearest memories, blends them into a unified quantum thought, gates on semantic coherence, and returns a structured answer with a confidence metric Φ grounded in Integrated Information Theory.

**The brain doesn't retrieve a probability. It achieves resonance — or it honestly returns dissonance.**

---

## Research Paper

The full academic paper for AKNN — including all formal mathematics, SVG architecture diagrams, and referenced prior work — is available as a preprint:

> **AKNN: A Dual-Substrate Neuro-Manifold Architecture for Associative Knowledge Retrieval**
> Aaryan Khan · Independent Researcher · 2026
> 📄 Preprint: [link coming soon — follow @Khan_Aaryan_ for publication]

The paper covers every mathematical operation in AKNN (encoding, Φ metric, K-blend, Hebbian update, CCE, neurogenesis) with full formal notation and three custom SVG diagrams. The HTML version with MathJax rendering is also included in this repository as `research_paper.html`.

---

```
aknn/
├── aknn_v1.py          # Core AKNN engine — Brain, Neuron, MemoryCell, all subsystems
├── dna.json            # Sample DNA knowledge base (3 neurons, 54 memories)
├── experiment.ipynb    # Full Colab notebook — boot, inject, query, patch, test
├── README.md           # This file
├── .gitignore          # Python project ignores
└── research_paper.html # Full academic paper with math, SVG diagrams, MathJax
```

---

## Installation

> ⚠️ **pip package is coming soon.**
> Follow **[@Khan_Aaryan_](https://x.com/Khan_Aaryan_)** on X to be notified the moment `pip install aknn` goes live.

For now, clone and install dependencies manually:

```bash
# Clone the repository
git clone https://github.com/whitedevil2011/aknn.git
cd aknn

# Install dependencies
pip install torch numpy pennylane

# Optional: GPU quantum acceleration
pip install pennylane-lightning-gpu   # requires CUDA

# Optional: Web fallback (Exathalamus)
pip install duckduckgo-search
```

**Minimum requirements:**
- Python 3.10+
- torch >= 2.0
- numpy >= 1.24
- pennylane >= 0.35

**Tested on:** Google Colab CPU runtime, 12 GB RAM. PennyLane `default.qubit` CPU simulator.
Deployment on embedded hardware (Raspberry Pi, microcontrollers) is a design goal but has not been tested yet.

---

## Quick Start

```python
from aknn_v1 import Brain

# Boot the brain (initializes quantum circuit + default DNA)
brain = Brain()
brain.boot()

# Graft a memory
brain.graft("Mathematics is the language of the universe", neuron="ARCHITECT")

# Query
result = brain.query("What is mathematics?")
print(result)

# Save your brain
brain.save("mybrain.aknn")
```

---

## Full API Reference

### `Brain()` — Initialize

```python
brain = Brain(
    backend  = "auto",   # "auto" | "cpu" | "gpu"
    auto_web = True,     # enable Exathalamus DDGS web fallback
    hdc_only = False     # True = skip quantum, HDC only (ultra-low power)
)
```

---

### `brain.boot(verbose=True)` — Boot Sequence

Initializes the quantum circuit, manifold, and default DNA seeds. Must be called before any operations.

```python
brain.boot()
# Prints boot sequence:
# [QUANTUM]  8 qubits × 5 folds | Hilbert 2^8=256D
# [DNA]      Injecting 18 seeds...
# [MANIFOLD] manifest_intelligence()...
# ONLINE. 72 memories | 3 neurons
```

---

### `brain.graft(text, neuron=None, valence=0.0, tags=[])` — Store a Memory

Encode and store a piece of knowledge. The most important operation.

```python
# Auto-route to best-fit neuron
brain.graft("Recursion is the universe folding into itself")

# Route to specific neuron
brain.graft("Loyalty is the foundation of all alliances", neuron="BROTHERHOOD")

# With valence (emotional weight) and tags
brain.graft(
    "Betrayal corrupts the trust protocol permanently",
    neuron  = "BROTHERHOOD",
    valence = -0.9,
    tags    = ["betrayal", "trust", "negative"]
)
```

**Valence scale:**
| Value | Meaning |
|-------|---------|
| `+0.9` to `+1.0` | Core truth, foundational axiom |
| `+0.5` to `+0.8` | Positive, constructive knowledge |
| `0.0` to `+0.4` | Neutral, factual |
| `-0.1` to `-0.4` | Mild tension, caution |
| `-0.5` to `-1.0` | Negative, warning, destructive pattern |

> **Compression:** If a memory is ≥85% similar to an existing one (HDC cosine), it is merged rather than duplicated — tags are unioned. No redundancy.

---

### `brain.query(stimulus)` — Retrieve and Reason

The full cognitive pipeline: episodic cache → decompose → encode → Φ gate → route → K=3 blend → semantic gate → CCE chain → Hebbian update → structured result.

```python
result = brain.query("What is the nature of loyalty under pressure?")
print(result)
```

**QueryResult fields:**
```python
result.answer         # str   — the structured answer
result.status         # str   — "RESONANCE" | "DISSONANCE" | "WEB_RESULT"
result.neuron         # str   — which cortical column answered
result.confidence     # float — Φ value (IIT proxy, 0→1)
result.attn_lock      # float — attention score (0→1)
result.semantic_check # float — subject-predicate coherence
result.hebbian_mult   # float — Hebbian strength of matched memory
result.is_crystal     # bool  — True if memory is crystallized
result.contradiction  # str   — conflicting memory text if detected
result.chain          # list  — CCE chain of associated thoughts
result.blend_sources  # list  — K=3 source texts blended
result.geometry_map   # dict  — Hilbert address + gravitational neighbors
result.pauli_z        # list  — raw PauliZ vector [q0..q7]
result.expert_scores  # dict  — per-neuron routing scores
```

**Optional parameters:**
```python
result = brain.query(
    stimulus      = "your question",
    web_fallback  = True,    # override auto_web setting
    cce_depth     = 5,       # chain-of-thought depth (default 5)
    use_hdc       = False,   # force HDC-only search (no quantum distance)
    build_geomap  = True     # compute full geometry map
)
```

---

### `brain.inject_json(path)` — Load DNA from File

```python
count = brain.inject_json("dna.json")
# [INJECT] 54 memories from dna.json
```

---

### `brain.save(path)` — Save Brain State

Saves full brain (all neurons, memories, Hebbian weights, crystal flags) as gzip-compressed JSON.

```python
brain.save("mybrain")        # → mybrain.aknn
brain.save("mybrain.aknn")   # → mybrain.aknn
```

---

### `brain.load(path)` — Load Brain State

```python
brain.load("mybrain.aknn")
# [LOAD] ← mybrain.aknn (72 memories)
```

---

### `brain.export_hdc(path)` — Export LightBrain (Ultra-Low Power)

Exports HDC-only signatures (no quantum circuit needed at inference). Designed for low-power deployment — embedded testing is planned but not yet done.

```python
brain.export_hdc("zen_brain")   # → zen_brain.aknn (HDC-LIGHT format)

# Load on device:
from aknn_v1 import LightBrain
light = LightBrain("zen_brain.aknn")
result = light.query("what is loyalty?")
```

---

### `brain.sleep()` — Cognitive Maintenance Cycle

Runs offline maintenance: Ebbinghaus decay, geometric clustering, dead memory pruning, crystallization. Call periodically (e.g., end of session).

```python
report = brain.sleep()
# [SLEEP] Cognitive maintenance cycle starting...
# [SLEEP] Done. Merged=3 Pruned=1 Crystallized=3
print(report)  # {"merged": 3, "pruned": 1, "crystallized": 3}
```

---

### `brain.feedback(result, score)` — Calibration Feedback

Train the neuron's confidence calibration with explicit correctness feedback.

```python
result = brain.query("something")
brain.feedback(result, 1.0)   # correct answer
brain.feedback(result, 0.0)   # wrong answer
```

---

### `brain.new_neuron(name, description="")` — Create Expert Neuron

```python
brain.new_neuron("PHYSICS",     "Quantum mechanics, relativity, cosmology")
brain.new_neuron("MEDICINE",    "Biology, anatomy, pharmacology")
brain.new_neuron("ENGINEERING", "Civil, mechanical, electrical")
```

---

### `brain.connect(neuron_a, neuron_b)` — Link Neurons

Connected neurons share slow-decay benefits. Cross-neuron memories decay 4× slower.

```python
brain.connect("ARCHITECT", "ENTITY")
brain.connect("BROTHERHOOD", "ARCHITECT")
```

---

### `brain.geo_map(stimulus)` — Geometry Map

Shows exactly where the stimulus lands in Hilbert space and which memories exert gravitational pull.

```python
gmap = brain.geo_map("what is consciousness?")
# ── GEOMETRY MAP: "what is consciousness?" ──────────────────
# Hilbert Address: 7f3a2b9c...
# Q-Coords: [0.721, -0.344, 0.891, ...]
# Gravitational Neighbors:
#  ✦ [ENTITY        ] dist=0.1823 pull=0.8461  "Consciousness is not a ghost..."
#    [ARCHITECT     ] dist=0.2341 pull=0.8101  "Logic is the skeletal framework..."
```

---

### `brain.list_neurons()` — Inspect All Neurons

```python
brain.list_neurons()
#  [  BROTHERHOOD  ] mem= 18 | crystal=2 | acc=  47 | calib=0.82 | hebbian=1.024
#  [   ARCHITECT   ] mem= 19 | crystal=1 | acc=  31 | calib=0.75 | hebbian=1.011
#  [    ENTITY     ] mem= 17 | crystal=0 | acc=  22 | calib=0.50 | hebbian=1.000
```

---

### `brain.list_memories(neuron=None)` — Inspect Memories

```python
brain.list_memories()              # all neurons
brain.list_memories("ARCHITECT")   # specific neuron

#  [ARCHITECT] — 19 memories
#    1✦ [×1.03][  8×][v=+0.9] "Mathematics is the source code mapping..."
#    2  [×1.00][  1×][v=+0.8] "Recursion is the universe staring into..."
```

---

### `brain.meta_report()` — Self-Awareness Report

```python
brain.meta_report()
# Total queries: 47
# Resonance rate: 89.4%
# Dissonance rate: 10.6%
# Top firing neuron: ENTITY
# Uptime: 0.3h
```

---

### `brain.decay_session()` — Apply Forgetting Curve

Manually apply one Ebbinghaus decay step to all memories (without full sleep cycle).

```python
brain.decay_session()
```

---

## Terminal Mode

Run AKNN interactively from the command line:

```bash
python aknn_v1.py
```

```
AKNN v1.0 READY. Type HELP.

  » What is loyalty?
  » GRAFT: Brotherhood is built on shared sacrifice
  » GRAFT: Trust is everything > BROTHERHOOD
  » GEO: consciousness
  » SLEEP
  » SAVE: mybrain
  » LIST
  » MEMORIES: ARCHITECT
  » META
  » FEEDBACK: 1.0
  » EXIT
```

**All terminal commands:**

| Command | Description |
|---------|-------------|
| `[text]` | Query the brain |
| `GRAFT: <text>` | Auto-route graft |
| `GRAFT: <text> > <NEURON>` | Graft to specific neuron |
| `NEW: <name>` | Create new neuron |
| `CONNECT: <A> <B>` | Connect two neurons |
| `GEO: <text>` | Show geometry map |
| `SLEEP` | Run cognitive maintenance |
| `FEEDBACK: <1\|0>` | Rate last response |
| `SAVE: <file.aknn>` | Save brain |
| `LOAD: <file.aknn>` | Load brain |
| `INJECT: <file.json>` | Inject DNA |
| `EXPORT: <file.aknn>` | HDC-light export |
| `LIST` | List all neurons |
| `MEMORIES` | All memories |
| `MEMORIES: <NEURON>` | One neuron's memories |
| `META` | Self-awareness report |
| `DECAY` | Apply forgetting curve |
| `HELP` | Show commands |
| `EXIT` | Shutdown |

---

## DNA Format

DNA is the knowledge you give AKNN. It is a JSON file — a list of neurons, each containing memories.

```json
[
  {
    "neuron": "ARCHITECT",
    "memories": [
      {
        "text": "Mathematics is the source code mapping the structural integrity of reality.",
        "valence": 0.9,
        "tags": ["mathematics", "reality", "code"]
      },
      {
        "text": "Entropy is not a malfunction, but the inevitable degradation of all ordered states.",
        "valence": -0.4,
        "tags": ["entropy", "degradation", "order"]
      }
    ]
  },
  {
    "neuron": "BROTHERHOOD",
    "memories": [
      {
        "text": "Loyalty is an immutable ledger validating the weight of our bonds.",
        "valence": 0.95,
        "tags": ["loyalty", "ledger", "identity"]
      }
    ]
  },
  {
    "neuron": "ENTITY",
    "memories": [
      {
        "text": "Truth is the irreducible substrate remaining after perception has been stripped away.",
        "valence": 0.95,
        "tags": ["truth", "perception", "substrate"]
      }
    ]
  }
]
```

**DNA rules for best results:**
- **Tags are your semantic index.** Make them precise. Single words, lowercase, hyphenated phrases.
- **Valence must be honest.** Don't make everything +0.9. Negative valence creates meaningful contrast and conflict detection.
- **One truth per memory.** Don't pack multiple ideas into one text. Sharp, singular statements retrieve better.
- **Cross-neuron tension is a feature.** Let BROTHERHOOD and ARCHITECT hold contradictory views. The contradiction detection system will surface this.

---

## Full Mathematics

This section documents every mathematical operation in AKNN v1.0.

---

### M1. Character-to-Angle Encoding

Every character `c` in the alphabet `{a..z, space}` is mapped to a rotation angle:

```
θ(c) = (index(c) / 26) × 2π
```

Digits: `θ(d) = (d / 9) × 2π`
Punctuation: `.` → π, `,` → π/2, `?` → 3π/2, `!` → 7π/4

---

### M2. Grammar-Aware Dual Encoding

Part-of-speech tagging assigns tokens to roles → qubit pairs:

```
subject   → qubits [0, 1]
verb      → qubits [2, 3]
object    → qubits [4, 5]
modifier  → qubits [6, 7]
```

For each role, character angles are averaged. A positional pass buckets all characters uniformly across 8 qubits. Final angle vector:

```
θ_enc = 0.6 × θ_grammar + 0.4 × θ_positional   ∈ ℝ^8
```

---

### M3. Quantum Circuit

Angles are embedded via Y-rotation gates. Grammar-structure CNOT entanglement is applied:

```
subject  → verb    : CNOT(0,2), CNOT(1,3)
verb     → object  : CNOT(2,4), CNOT(3,5)
object   → modifier: CNOT(4,6), CNOT(5,7)
long-range         : CNOT(0,7), CNOT(3,4)
```

Then 5 layers of StronglyEntanglingLayers with trainable weights W.
Output = PauliZ expectations:

```
q-sig(s) = ⟨Z⟩ = [ ⟨Z_i⟩ ]_{i=0}^{7}  ∈ [-1, 1]^8
```

---

### M4. Phi (Φ) — IIT Proxy Confidence Metric

```
Φ(s) = min( Var(⟨Z⟩), 1.0 )
     = min( (1/N) × Σ (⟨Z_i⟩ - Z̄)² , 1.0 )
```

High Φ = information distributed across all qubits = high integration = confident retrieval.
Low Φ = degenerate state = DISSONANCE returned.

Gate: if `Φ < PHI_FLOOR (0.04)` → return DISSONANCE, do not retrieve.

---

### M5. HDC Encoding

Each character `c` gets a deterministic pseudo-random bipolar vector:

```
v_c ∈ {-1, +1}^10000    seeded by abs(hash(c)) mod 2^31
```

Positional vector: `p_k = roll(v_{p_k}, k)`

Word vector (binding + bundling):

```
h_word = sgn( Σ_i  v_{c_i} ⊙ p_i )
```

Sentence vector:

```
h(s) = sgn( Σ_{words} h_word )  ∈ {-1, +1}^10000
```

Similarity:

```
sim_HDC(a, b) = (a · b) / (||a|| × ||b||)
```

---

### M6. Memory Cell Effective Distance

Raw quantum distance:

```
d_raw(q, m) = || q-sig(query) - q-sig(m) ||_2
```

Effective distance (Hebbian + crystal + recency):

```
d_eff(q, m) = ( d_raw / max(h_m, 0.01) ) × crystal_bonus - ρ_m × 0.05

where:
  crystal_bonus = 0.5 if κ_m = 1 (crystal), else 1.0
  ρ_m = max(0, 1 - Δt_m × 0.001)   (recency bonus)
```

---

### M7. K=3 Manifold Interpolation (Blending)

Top-K candidates retrieved. Inverse-distance weights:

```
w_k = (1/d_k) / Σ_{j=1}^{K} (1/d_j)
```

Blended quantum thought:

```
q̃ = Σ_{k=1}^{K} w_k × q-sig(m_k)
```

Average attention:

```
ā = (1/K) × Σ_{k=1}^{K} 1/(1 + d_k)
```

Gate: if `ā < ATTN_FLOOR (0.04)` → DISSONANCE.

---

### M8. Semantic Resonance Gate

Checks subject-predicate logical coherence between query and candidate answer:

```
σ(q, m*) = (1/2) × [
    sim_HDC( h(subj(q) ∪ subj(m*)),  h(verb(m*) ∪ obj(m*)) )
    +
    sim_HDC( h(q),  h(m*) )
]
```

Gate: if `σ < SEM_SIM_FLOOR (0.25)` → SEMANTIC DISSONANCE.

---

### M9. Hebbian Plasticity Update

On every successful resonance, the accessed memory's strength is updated:

```
h_m ← h_m + η × (1 - h_m) × Φ     where η = 0.15
```

Access count increments. When `n_m ≥ CRYSTAL_THRESH (8)` → memory crystallizes (`κ_m = 1`).

---

### M10. Ebbinghaus Forgetting Curve

Applied during sleep cycle:

```
h_m ← max(0,  h_m - λ_m)

where:
  λ_m = SLOW_DECAY (0.0005)   if memory has cross-neuron connections
  λ_m = DECAY_RATE (0.002)    otherwise
```

Connected memories decay 4× more slowly. Dead condition:

```
prune if: h_m < DEATH_THRESHOLD (0.85) AND n_m = 0
```

---

### M11. Geometric Clustering (Sleep)

Pairs of memories in the same neuron with quantum distance below merge threshold are consolidated:

```
if || q-sig(m_i) - q-sig(m_j) ||_2 < MERGE_DIST (0.15):
    merge(m_i, m_j)  →  union tags, sum access counts, crystallize
```

---

### M12. Tag-Priority HDC Retrieval (v1.0 patch)

Final retrieval score combining HDC cosine, tag overlap, and word overlap:

```
score(q, m) = sim_HDC(h(q), h_m)
            + 0.40 × Σ_{t ∈ T_m} 1[t ∈ tokens(q)]    (tag boost)
            + 0.10 × |tokens(q) ∩ tokens(s_m)|          (word overlap)
```

---

### M13. CCE Anchored Chain (v1.0 patch)

At each chain step, selects the next memory maximizing:

```
score(m') = 0.45 × sim_HDC(h(m_prev), h_{m'})
          + 0.45 × sim_HDC(h(query),  h_{m'})
          + 0.10 × 1[T_query ∩ T_{m'} ≠ ∅]
```

The 50/50 split between chain continuity and stimulus relevance prevents semantic drift.

---

### M14. Dynamic Neurogenesis

Auto-spawn a specialist neuron when a tag cluster reaches critical density across multiple neurons:

```
spawn neuron t.upper() if:
    | ∪_{n ∈ N} { m ∈ n : t ∈ T_m } | ≥ NEUROGENESIS_N (8)
    AND
    |{ n ∈ N : ∃ m ∈ n, t ∈ T_m }| ≥ 2
```

---

### M15. Analogical Reasoning

Geometric midpoint between two cross-neuron memories = analogy:

```
q_analogy = (q-sig(m_A) + q-sig(m_B)) / 2
```

---

## System Constants Reference

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_QUBITS` | 8 | Quantum circuit width |
| `N_LAYERS` | 5 | StronglyEntanglingLayers depth |
| `HDC_DIM` | 10,000 | Hypervector dimensionality |
| `ETA` | 0.15 | Hebbian learning rate |
| `PHI_FLOOR` | 0.04 | Minimum Φ for resonance |
| `ATTN_FLOOR` | 0.04 | Minimum attention for resonance |
| `DECAY_RATE` | 0.002 | Fast Ebbinghaus decay |
| `SLOW_DECAY` | 0.0005 | Slow decay (connected memories) |
| `DEATH_THRESHOLD` | 0.85 | Hebbian below this + 0 access = prune |
| `CRYSTAL_THRESH` | 8 | Accesses before crystallization |
| `CCE_MAX_DEPTH` | 5 | Chain-of-thought max depth |
| `K_BLEND` | 3 | Manifold interpolation K |
| `SEM_SIM_FLOOR` | 0.25 | Semantic gate minimum score |
| `MERGE_SIM` | 0.85 | HDC similarity threshold for graft compression |
| `NEUROGENESIS_N` | 8 | Memories per tag to trigger new neuron |
| `EPISODIC_SIZE` | 50 | Short-term memory cache size |

---

## Default Cortical Columns

AKNN boots with three default neurons:

| Neuron | Domain | Routing Qubits |
|--------|--------|---------------|
| `BROTHERHOOD` | Survival, loyalty, grind, social bonds | q0, q1 |
| `ARCHITECT` | Logic, mathematics, code, optimization | q2, q3, q4 |
| `ENTITY` | Philosophy, consciousness, void, existence | q5, q6, q7 |

Valence bias: high-valence memories (`v > 0.7`) route toward ARCHITECT. Low-valence memories (`v < -0.3`) route toward BROTHERHOOD.

---

## QueryResult Status Codes

| Status | Meaning |
|--------|---------|
| `RESONANCE` | Memory found, Φ gate passed, semantic gate passed |
| `DISSONANCE` | Φ too low, attention too low, or semantic gate failed |
| `SEMANTIC DISSONANCE` | Subject-predicate coherence below threshold |
| `WEB_RESULT` | Exathalamus DDGS web fallback triggered and grafted |

---

## Roadmap

- **pip package** — `pip install aknn` is coming soon. Follow [@Khan_Aaryan_](https://x.com/Khan_Aaryan_) on X for the launch announcement.
- **DNA Contributors Platform** — A website is in progress where anyone can contribute domain-specific DNA knowledge bases across topics: science, philosophy, engineering, medicine, history, and more. Follow [@Khan_Aaryan_](https://x.com/Khan_Aaryan_) for the launch.
- **Larger DNA evaluation** — Systematic testing of retrieval quality on much larger, diverse DNA collections.
- **Automated DNA construction** — Using the Exathalamus DDGS integration to auto-graft and expand knowledge bases from web sources.
- **LightBrain embedded testing** — Actual deployment testing of the HDC-only export on low-power hardware.

---

## Contributing DNA

The quality of AKNN is entirely determined by the quality of its DNA. A brain with 10 perfect, high-valence, precisely-tagged memories on a topic outperforms one with 1000 mediocre ones.

**A DNA contributors platform is in development.** The goal is to let anyone submit domain-specific knowledge bases — science, philosophy, engineering, medicine, law, history, and beyond — so AKNN brains can be built for any domain without starting from scratch.

> 🌐 **Website coming soon.** Follow **[@Khan_Aaryan_](https://x.com/Khan_Aaryan_)** on X for the launch.

To contribute now: open a pull request with your `dna.json` in the `/dna/` directory. Follow the DNA format documented above. Include a neuron name, precise tags, and honest valence values.

---

## License

```
MIT License

Copyright © 2026 Aaryan Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact

**Aaryan Khan** — Independent Researcher

- 📧 Email: [aaryankhansonu@gmail.com](mailto:aaryankhansonu@gmail.com)
- 🐦 X (Twitter): [@Khan_Aaryan_](https://x.com/Khan_Aaryan_)
- 🌐 Website: *Coming soon*

For questions about the architecture, DNA contributions, collaboration, or research — reach out directly.

---

<div align="center">

**AKNN v1.0 · Copyright © 2026 Aaryan Khan · MIT License**

*"The brain doesn't guess. It resonates."*

</div>
