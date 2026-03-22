"""
Mechanistic metrics for TTT interference analysis.

Two core metrics:
  1. eNTK:  trace(J(q)^T J(k)) — structural coupling between bindings in weight space
  2. Jaccard / switch-rate — ReLU gate overlap and dynamics (MLP only)
"""

import torch
import torch.nn.functional as F
from ttt import TTTLinear, TTTMLP, TTTMLPFrozenPhi, TTTBackbone


# ──────────────────────────────────────────────────────────────────────────────
# eNTK: pairwise scalar eNTK = trace(J(q_i)^T J(k_j))
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_entk(
    backbone: TTTBackbone,
    params: dict,
    queries: torch.Tensor,   # [B, nh, M, dk]
    keys: torch.Tensor,      # [B, nh, N, dk]
) -> torch.Tensor:            # [B, nh, M, N]
    """Pairwise scalar eNTK: trace(J(q_i)^T · J(k_j)).

    For TTTLinear (f(x) = xW^T):
        J(x) = I ⊗ x  →  eNTK(q, k) = dk · (q · k)

    For TTTMLP (f(x) = ReLU(xW0^T)W1^T, both update):
        W1 part: dk · (a_q · a_k)     where a = ReLU(xW0^T)
        W0 part: (q · k) · Σ_b ||W1[:,b]||² · mask_q_b · mask_k_b
        Total = sum of both

    For TTTMLPFrozenPhi (only W1 updates):
        eNTK(q, k) = dk · (a_q · a_k)     (W0 frozen → no W0 Jacobian)
    """
    dk = keys.shape[-1]

    if isinstance(backbone, TTTLinear):
        return dk * (queries @ keys.transpose(-1, -2))

    elif isinstance(backbone, TTTMLPFrozenPhi):
        W0 = backbone.W0                                   # [nh, dh, dk] frozen
        a_q = F.relu(queries @ W0.transpose(-1, -2))       # [B, nh, M, dh]
        a_k = F.relu(keys @ W0.transpose(-1, -2))          # [B, nh, N, dh]
        return dk * (a_q @ a_k.transpose(-1, -2))

    elif isinstance(backbone, TTTMLP):
        W0 = params["W0"]                                  # [B, nh, dh, dk]
        W1 = params["W1"]                                  # [B, nh, dk, dh]

        h_q = queries @ W0.transpose(-1, -2)               # [B, nh, M, dh]
        h_k = keys @ W0.transpose(-1, -2)                  # [B, nh, N, dh]
        a_q = F.relu(h_q)
        a_k = F.relu(h_k)
        mask_q = (h_q > 0).float()
        mask_k = (h_k > 0).float()

        # W1 contribution: df/dW1 = I ⊗ a(x)
        entk_w1 = dk * (a_q @ a_k.transpose(-1, -2))      # [B, nh, M, N]

        # W0 contribution: df/dW0 = W1^T diag(mask) ⊗ x
        w1_col_norms_sq = (W1 ** 2).sum(-2)                # [B, nh, dh]
        weighted_mask_q = mask_q * w1_col_norms_sq.unsqueeze(-2)
        mask_overlap = weighted_mask_q @ mask_k.transpose(-1, -2)
        entk_w0 = (queries @ keys.transpose(-1, -2)) * mask_overlap

        return entk_w1 + entk_w0

    raise ValueError(f"Unknown backbone type: {type(backbone)}")


# ──────────────────────────────────────────────────────────────────────────────
# ReLU mask metrics (MLP backbones only)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_jaccard(masks: torch.Tensor) -> torch.Tensor:
    """Pairwise Jaccard similarity of binary ReLU masks.

    Args:
        masks: [B, nh, K, dh] binary tensor from TTTOutput.masks
    Returns:
        [B, nh, K, K] Jaccard similarity matrix
    """
    m = masks.float()
    intersection = m @ m.transpose(-1, -2)                  # [B, nh, K, K]
    counts = m.sum(-1)                                      # [B, nh, K]
    union = counts.unsqueeze(-1) + counts.unsqueeze(-2) - intersection
    return intersection / union.clamp(min=1e-8)


@torch.no_grad()
def compute_switch_rate(masks: torch.Tensor) -> torch.Tensor:
    """Fraction of ReLU units that flip between consecutive inner-loop steps.

    Args:
        masks: [B, nh, K, dh] binary tensor (K tokens processed sequentially)
    Returns:
        [B, nh, K-1] switch rate per step
    """
    return (masks[:, :, 1:] != masks[:, :, :-1]).float().mean(-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    # ── setup: 8 bindings, single head, small dims ──
    B, nh, N, dk, dh = 1, 1, 8, 16, 64
    eta = 1.0 / dk

    # clustered keys: 2 clusters of 4, high intra-cluster similarity
    c1 = F.normalize(torch.randn(1, dk), dim=-1)
    c2 = F.normalize(torch.randn(1, dk), dim=-1)
    noise = 0.15
    keys = torch.cat([
        F.normalize(c1 + noise * torch.randn(4, dk), dim=-1),  # cluster 1
        F.normalize(c2 + noise * torch.randn(4, dk), dim=-1),  # cluster 2
    ]).unsqueeze(0).unsqueeze(0)  # [1, 1, 8, dk]
    values = torch.randn(B, nh, N, dk)
    queries = keys.clone()  # query = key for analysis

    # ── build all three backbones ──
    backbones = {
        "Linear": TTTLinear(nh, dk),
        "MLP": TTTMLP(nh, dk, dh),
        "Frozen-φ": TTTMLPFrozenPhi(nh, dk, dh),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, (name, backbone) in enumerate(backbones.items()):
        # run inner loop token-by-token to get final params + masks
        params = backbone.init_fast_weights(B, keys.device)
        all_masks = []
        for i in range(N):
            result = backbone.compute_mini_batch(
                params,
                queries[:, :, i:i+1],
                keys[:, :, i:i+1],
                values[:, :, i:i+1],
                eta,
            )
            params = result.params
            if result.masks is not None:
                all_masks.append(result.masks)

        # eNTK at final weight state
        entk = compute_entk(backbone, params, queries, keys)
        entk_tril = torch.tril(entk[0, 0], diagonal=-1)

        ax = axes[0, col]
        im = ax.imshow(entk_tril.numpy(), cmap="RdBu_r", aspect="equal")
        ax.set_title(f"{name} — eNTK (tril)")
        ax.set_xlabel("key token m")
        ax.set_ylabel("query token n")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Jaccard (MLP backbones only)
        ax = axes[1, col]
        if len(all_masks) > 0:
            masks = torch.cat(all_masks, dim=2)  # [B, nh, N, dh]
            jaccard = compute_jaccard(masks)
            jaccard_tril = torch.tril(jaccard[0, 0], diagonal=-1)
            im = ax.imshow(jaccard_tril.numpy(), cmap="viridis", aspect="equal")
            ax.set_title(f"{name} — Jaccard (tril)")
            ax.set_xlabel("token m")
            ax.set_ylabel("token n")
            plt.colorbar(im, ax=ax, fraction=0.046)

            sr = compute_switch_rate(masks)
            print(f"{name} switch rate per step: {sr[0, 0].tolist()}")
        else:
            ax.text(0.5, 0.5, "N/A\n(no ReLU masks)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{name} — Jaccard")

    plt.tight_layout()
    plt.savefig("entk_jaccard_test.png", dpi=150)
    plt.show()
    print("Saved entk_jaccard_test.png")
