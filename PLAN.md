# Where the Equivalence Breaks
**Activation Geometry Governs Selective Memory in Test-Time Training**

Alexander Hackett · Santa Clara University · March 2026
Target: NeurIPS 2026 / ICLR 2027 Workshop

---

## Thesis

NVIDIA's LaCT linearization (Li et al., Feb 2026) freezes hidden-layer params Θ during the TTT inner loop, collapsing a dynamic nonlinear feature map φₜ(x) into a static one. This is nearly lossless on aggregate perplexity but **destroys activation-gated interference isolation** — the ability of an MLP inner model to suppress crosstalk between similar keys by placing them on different sides of ReLU boundaries.

We demonstrate this with a clustered-key associative recall benchmark and mechanistic metrics that perplexity cannot capture.

---

## Core Argument

### The Math

Interference between bindings i and j under an overwrite gradient g:

```
Interference(i, j) = J(kᵢ)ᵀ · gⱼ
```

- **Linear inner model / frozen-φ MLP:** J(kᵢ) ∝ kᵢ → interference ∝ kᵢᵀkⱼ. No mechanism to suppress similar-key crosstalk.
- **Dynamic nonlinear inner model:** J(kᵢ) depends on active gates. Similar keys can produce orthogonal Jacobians if they straddle an activation boundary.

### Why Perplexity Misses It

Perplexity averages over millions of tokens. Hard recall tokens (specific binding retrieval among similar keys) are rare and contribute negligibly. The Zoology/MQAR literature already established that matched-perplexity models diverge on recall. NVIDIA's own NVS ablation (PSNR 25.97 → 25.71 when collapsing MLP to linear) is consistent — NVS requires discriminating similar camera poses.

---

## Experiment Design

### Task: Clustered-Key Associative Recall

Modified MQAR:
- Sample C cluster centroids on the unit sphere
- Generate M keys per cluster with controlled intra-cluster cosine similarity ρ
- Each key maps to a unique random value
- Keys are hardcoded (not learned), matching real TTT inner-model usage

**Why this isn't adversarial:** Clustered keys model polysemy, entity disambiguation, coreference — real language phenomena, not synthetic edge cases.

### Architectures (matched parameter count)

| Architecture | Inner Model | Role |
|---|---|---|
| TTT-Linear (Wide) | f(x) = Wx | Primary baseline |
| TTT-ReLU-MLP | N×(Linear+ReLU) + Linear | Primary experimental |
| TTT-MLP (Last-Layer Only) | Same MLP, Θ frozen | NVIDIA's Variant 1 |

### Protocol

1. Meta-train all architectures on clustered-key MQAR to convergence (Zoology data infra)
2. On held-out key sets:
   - Write N bindings via sequential TTT gradient steps
   - Overwrite one binding with a new value
   - Query all bindings
3. Measure **update success** (did overwrite land?) and **collateral damage** (did other bindings shift?)

### Primary Sweep

Architecture × Intra-cluster similarity **ρ** (0.1 to 0.95)

Scale: d = 32–128, single GPU, days not weeks.

---

## Metrics

### 1. Jacobian-Gradient Interference (Core)

Before applying overwrite, compute J(kᵢ)ᵀ · g for all existing bindings i, where g = gradient of overwrite loss. First-order predicted interference, computed analytically.

For MLP: also log binary activation masks and compute pairwise Jaccard overlap with overwrite target.

### 2. MQAR Recall Accuracy

Standard recall accuracy on clustered-key MQAR across the ρ sweep. Expect TTT-MLP maintains accuracy while TTT-Linear degrades as ρ → 1.

### 3. ReLU Gating Mask Switch Rate

**Per-token metric:** fraction of ReLU units in the inner MLP whose binary on/off state changes between consecutive TTT gradient steps.

- **High switch rate** → the inner loop is actively reshaping activation boundaries, exploiting the nonlinearity. The dynamic φ is doing real work.
- **Low/zero switch rate** → gates are saturated or the inner LR is too small to move weights across boundaries. The MLP is effectively linear despite having ReLU — the nonlinearity is wasted compute.

This is a direct diagnostic: if switch rate is near zero even for the full MLP, there is no geometric mechanism to study and the thesis is dead. If switch rate is high for full MLP but zero for frozen-φ (by definition), we have a clean contrast.

Compute: record binary mask mₜ = 1[hₜ > 0] at each inner-loop step t, report mean Hamming distance d(mₜ, mₜ₋₁) / dim normalized across steps and bindings.

### 4. Activation Overlap vs Interference Correlation (Smoking Gun)

Scatter plot: Jaccard similarity of ReLU masks between binding pairs vs measured interference. Expect strong positive correlation (r > 0.7) for MLP, zero correlation for Linear.

---

## Expected Results (Hero Figures)

### Figure 1: The ρ-Sweep

X-axis: intra-cluster similarity ρ. Y-axis: mean collateral damage.

- Low ρ: all architectures show low interference (keys are ~orthogonal)
- High ρ: TTT-Linear interference climbs proportionally (exact by construction)
- TTT-MLP interference climbs slower
- Frozen-φ MLP tracks closer to Linear than to full MLP

### Figure 2: Activation Overlap → Interference

Scatter: Jaccard(ReLU masks) vs measured interference per binding pair.

- MLP: strong positive correlation
- Linear: uniform activation → no correlation (flat)

### Figure 3: Switch Rate Across ρ

X-axis: ρ. Y-axis: mean ReLU mask switch rate.

- Full MLP: switch rate increases with ρ (model works harder to separate similar keys)
- Frozen-φ: zero by definition (sanity check)
- If switch rate is flat/zero for full MLP → **kill the paper**

### Figure 4: MQAR Accuracy

X-axis: ρ. Y-axis: recall accuracy. TTT-MLP holds, TTT-Linear degrades.

---

## Timeline

| Week | Work | Kill Condition |
|---|---|---|
| 1 | Clustered-key data generation + TTT inner loop impl (Linear, ReLU-MLP, frozen-φ). Verify on orthogonal keys. | — |
| 2 | Meta-training + ρ-sweep. Generate hero figure. | **If interference curves don't separate at ρ > 0.7, stop. Paper is dead.** |
| 3 | eNTK heatmaps, activation viz, interference-overlap correlation, switch rate analysis. Write paper. | — |

---

## Implementation Notes

- Using SDPA instead of flash_attn (T4 GPU constraint). Materializing sliding window + causal mask manually.
- Training <150M param toy models
- Minimal LaCT implementation is isolated — not imported elsewhere in repo, safe to modify freely
- LaCT core (`block_causal_lact_swiglu`) is pure `torch.bmm`, no flash_attn dependency
- Flash attn only used for the sliding window attention component (swappable to SDPA)

---

## Scope (What We Are NOT Claiming)

- NOT claiming NVIDIA linearization is wrong
- NOT claiming MLP beats Linear on language modeling
- Claiming the equivalence is **lossy in a specific, measurable, linguistically motivated way** — and providing tools to measure it
