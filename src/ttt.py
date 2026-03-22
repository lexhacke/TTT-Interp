import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from einops import rearrange


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTTConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 6
    ttt_inner_lr: float = 1.0
    mini_batch_size: int = 64
    ffn_mult: int = 4
    ttt_hidden_mult: float = 4.0   # MLP inner model: dh = ttt_hidden_mult * head_dim
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    backbone_type: str = "linear"  # "linear" | "mlp" | "mlp_frozen_phi"

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads


# ──────────────────────────────────────────────────────────────────────────────
# Components
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# TTT Backbones
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TTTOutput:
    params: dict
    output: torch.Tensor
    masks: torch.Tensor | None = None  # ReLU gating masks, only from MLP backbone

class TTTBackbone(ABC, nn.Module):
    """Abstract base for TTT inner models (fast weights).

    The TTT inner loop works like this:
      1. We receive a chunk of K tokens, already projected into Q, K, V spaces
      2. K and V are the "training data" — we use them to compute a loss and update fast weights
      3. Q is the "test data" — we evaluate the updated fast weights on Q to produce output

    For each token i in the chunk:
      - TRAIN: feed k_i through the inner model, compute MSE loss against v_i
      - UPDATE: compute gradient of that loss, SGD-step the fast weights
      - EVAL: feed q_i through the *updated* inner model to get output o_i

    The fast weights (W, W0, W1, etc.) persist ACROSS chunks within a sequence
    but are reset at the start of each new sequence.

    Args to compute_mini_batch:
        params: dict of fast weight tensors, carried from the previous chunk.
                First chunk gets these from init_fast_weights().
        XQ:     queries  [B, nh, K, dk] — evaluated AFTER the weight update
        XK:     keys     [B, nh, K, dk] — input to the inner model (training input)
        XV:     values   [B, nh, K, dv] — target for MSE loss (training label)
        eta:    scalar learning rate for the inner SGD step (= config.ttt_inner_lr / head_dim)

    Returns:
        (updated_params, output) where output is [B, nh, K, dv]
    """

    @abstractmethod
    def init_fast_weights(self, B: int, device: torch.device) -> dict:
        """Return initial fast weight state, batched over B."""

    @abstractmethod
    def compute_mini_batch(
        self, params: dict,
        XQ: torch.Tensor,
        XK: torch.Tensor,
        XV: torch.Tensor,
        eta: float,
    ) -> TTTOutput:
        """Process one mini-batch. Returns (updated_params, output [B, nh, K, dv])."""


class TTTLinear(TTTBackbone):
    """Inner model: f(x) = x @ W^T    (single linear layer, no bias)

    Uses the DUAL FORM for efficiency (parallel within chunk):

    Instead of looping over tokens one by one, we compute all outputs at once:
      Z  = XK @ W^T                        # forward all keys through current W
      dZ = 2(Z - XV)                        # MSE gradient for every token at once
      Attn = tril(XQ @ XK^T)               # causal attention: token i sees j <= i
      O  = XQ @ W^T - eta * Attn @ dZ      # each query sees the effect of all
                                            #   prior gradient updates via Attn

    This is an APPROXIMATION: all gradients are computed at the initial W for this
    chunk, not at the progressively updated W. Exact in the limit of small eta.

    After the chunk, W is updated by the sum of all token gradients:
      W_new = W - eta * (dZ^T @ XK)        # carried to the next chunk
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, head_dim)))

    def init_fast_weights(self, B, device):
        return {"W": self.W.unsqueeze(0).expand(B, -1, -1, -1).clone()}

    def compute_mini_batch(self, params, XQ, XK, XV, eta):
        W = params["W"]                                  # [B, nh, hd, hd]

        # 1) Forward all keys: predict v from k
        Z = XK @ W.transpose(-1, -2)                     # [B, nh, K, hd]

        # 2) MSE gradient: dL/dz = 2(z - v) for each token
        dZ = 2.0 * (Z - XV)

        # 3) Dual form output: query i accumulates gradient corrections from tokens j <= i
        #    Attn[i,j] = dot(q_i, k_j) — how much token j's gradient affects token i's output
        Attn = torch.tril(XQ @ XK.transpose(-1, -2))     # [B, nh, K, K]
        O = XQ @ W.transpose(-1, -2) - eta * Attn @ dZ

        # 4) Update W for next chunk: apply all K tokens' gradients
        W_new = W - eta * (dZ.transpose(-1, -2) @ XK)
        return TTTOutput(params={"W": W_new}, output=O)


class TTTMLP(TTTBackbone):
    """Inner model: f(x) = x + RMSNorm(ReLU(x @ W0^T) @ W1^T)

    Prenorm residual MLP with RMSNorm for stability (per TTT paper §2.3).
    All params (W0 and W1) update at every token. RMSNorm weights are frozen
    during the inner loop (trained by outer loop only).

    For each token i in the chunk:
      FORWARD on key k_i:
        h_i  = k_i @ W0^T                      # [B, nh, 1, dh]
        a_i  = ReLU(h_i)                        # [B, nh, 1, dh]
        r_i  = a_i @ W1^T                       # [B, nh, 1, dk]
        z_i  = k_i + RMSNorm(r_i)               # residual + norm

      BACKWARD (manual grad of ||z_i - v_i||^2, chain through norm + residual):
        dz_i = 2(z_i - v_i)
        dr_i = dz_i ⊙ d(RMSNorm)/d(r_i)       # chain through RMSNorm
        dW1  = dr_i^T @ a_i
        da_i = dr_i @ W1
        dW0  = (da_i * mask_i)^T @ k_i

      SGD STEP: W0 -= eta * dW0, W1 -= eta * dW1

      EVALUATE on query q_i (updated weights):
        o_i = q_i + RMSNorm(ReLU(q_i @ W0^T) @ W1^T)
    """

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.W0 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, hidden_dim, head_dim)))
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, hidden_dim)))
        # RMSNorm weights: per-head, frozen during inner loop
        self.norm_weight = nn.Parameter(torch.ones(num_heads, 1, head_dim))
        self.norm_eps = 1e-6

    def _rms_norm(self, x):
        """x: [B, nh, 1, dk]. Uses self.norm_weight [nh, 1, dk] (broadcast over B)."""
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.norm_eps)
        return (x.float() * rms).to(x.dtype) * self.norm_weight

    def _rms_norm_backward(self, x, grad_out):
        """Manual backward for RMSNorm: d(loss)/d(x) given d(loss)/d(norm(x)).
        x, grad_out: [B, nh, 1, dk]
        """
        dk = x.shape[-1]
        x_f = x.float()
        rms_inv = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        x_hat = x_f * rms_inv
        # grad through norm_weight * x_hat
        grad_xhat = (grad_out.float() * self.norm_weight)
        # grad through x_hat = x * rms_inv
        grad_x = rms_inv * (grad_xhat - x_hat * (grad_xhat * x_hat).mean(-1, keepdim=True))
        return grad_x.to(x.dtype)

    def init_fast_weights(self, B, device):
        return {
            "W0": self.W0.unsqueeze(0).expand(B, -1, -1, -1).clone(),
            "W1": self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone(),
        }

    def compute_mini_batch(self, params, XQ, XK, XV, eta):
        W0 = params["W0"]                                # [B, nh, dh, dk]
        W1 = params["W1"]                                # [B, nh, dk, dh]
        K_len = XK.shape[-2]
        outputs = []
        masks = []

        for i in range(K_len):
            k_i = XK[:, :, i : i + 1, :]                 # [B, nh, 1, dk]
            v_i = XV[:, :, i : i + 1, :]                  # [B, nh, 1, dv]
            q_i = XQ[:, :, i : i + 1, :]                  # [B, nh, 1, dk]

            # ── forward on key: f(k) = k + RMSNorm(ReLU(k @ W0^T) @ W1^T) ──
            h_i = k_i @ W0.transpose(-1, -2)              # [B, nh, 1, dh]
            mask_i = (h_i > 0)
            a_i = h_i * mask_i                             # ReLU
            r_i = a_i @ W1.transpose(-1, -2)               # [B, nh, 1, dk]
            z_i = k_i + self._rms_norm(r_i)                # residual + norm

            # ── backward: manual grad of ||z_i - v_i||^2 ──
            dz_i = 2.0 * (z_i - v_i)                      # dL/dz  [B, nh, 1, dk]
            # chain through residual: dL/dr = dL/dz ⊙ d(RMSNorm)/dr
            dr_i = self._rms_norm_backward(r_i, dz_i)     # dL/dr  [B, nh, 1, dk]
            dW1 = dr_i.transpose(-1, -2) @ a_i            # dL/dW1 [B, nh, dk, dh]
            da_i = dr_i @ W1                               # dL/da  [B, nh, 1, dh]
            dW0 = (da_i * mask_i).transpose(-1, -2) @ k_i # dL/dW0 [B, nh, dh, dk]

            # ── SGD step ──
            W0 = W0 - eta * dW0
            W1 = W1 - eta * dW1

            # ── evaluate on query (using UPDATED W0, W1) ──
            h_q = q_i @ W0.transpose(-1, -2)
            o_i = q_i + self._rms_norm(F.relu(h_q) @ W1.transpose(-1, -2))
            outputs.append(o_i)
            masks.append(mask_i)

        return TTTOutput(params={"W0": W0, "W1": W1}, output=torch.cat(outputs, dim=-2), masks=torch.cat(masks, dim=-2))


class TTTMLPFrozenPhi(TTTBackbone):
    """Inner model: f(x) = x + RMSNorm(ReLU(x @ W0^T) @ W1^T)

    ONLY W1 updates. W0 is frozen during the inner loop (still trained by outer loop).
    This is "Variant 1" from the NVIDIA paper. Admits a dual form because the ReLU masks
    are fixed (W0 doesn't change).

    The residual + RMSNorm wraps the MLP output, matching the full MLP backbone.
    Since RMSNorm weights are frozen during the inner loop and the norm is applied
    per-token, it factors through the dual form cleanly:

      A_K = ReLU(XK @ W0^T)
      A_Q = ReLU(XQ @ W0^T)
      R   = A_K @ W1^T                      # raw MLP output
      Z   = XK + RMSNorm(R)                 # with residual + norm
      dZ  = 2(Z - XV)
      dR  = dZ ⊙ d(RMSNorm)/dR             # chain through norm
      ... standard dual form on dR ...
    """

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.W0 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, hidden_dim, head_dim)))
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, hidden_dim)))
        self.norm_weight = nn.Parameter(torch.ones(num_heads, 1, head_dim))
        self.norm_eps = 1e-6

    def _rms_norm(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.norm_eps)
        return (x.float() * rms).to(x.dtype) * self.norm_weight

    def _rms_norm_backward(self, x, grad_out):
        x_f = x.float()
        rms_inv = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        x_hat = x_f * rms_inv
        grad_xhat = grad_out.float() * self.norm_weight
        grad_x = rms_inv * (grad_xhat - x_hat * (grad_xhat * x_hat).mean(-1, keepdim=True))
        return grad_x.to(x.dtype)

    def init_fast_weights(self, B, device):
        return {"W1": self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()}

    def compute_mini_batch(self, params, XQ, XK, XV, eta):
        W0 = self.W0                                      # [nh, dh, dk] frozen
        W1 = params["W1"]                                  # [B, nh, dv, dh]

        A_K = F.relu(XK @ W0.transpose(-1, -2))           # [B, nh, K, dh]
        A_Q = F.relu(XQ @ W0.transpose(-1, -2))           # [B, nh, K, dh]

        # Forward: f(x) = x + RMSNorm(A_K @ W1^T)
        R_K = A_K @ W1.transpose(-1, -2)                  # [B, nh, K, dk]
        Z = XK + self._rms_norm(R_K)
        dZ = 2.0 * (Z - XV)

        # Chain through RMSNorm (per-token, so applies elementwise across K)
        dR = self._rms_norm_backward(R_K, dZ)             # [B, nh, K, dk]

        # Dual form on dR (same as before but with dR instead of dZ)
        Attn = torch.tril(A_Q @ A_K.transpose(-1, -2))   # [B, nh, K, K]
        R_Q = A_Q @ W1.transpose(-1, -2)                  # [B, nh, K, dk]
        O = XQ + self._rms_norm(R_Q - eta * Attn @ dR)
        W1_new = W1 - eta * (dR.transpose(-1, -2) @ A_K)
        return TTTOutput(params={"W1": W1_new}, output=O)


# ──────────────────────────────────────────────────────────────────────────────
# TTT Layer  (wraps any backbone into a sequence-modelling layer)
# ──────────────────────────────────────────────────────────────────────────────

class TTTLayer(nn.Module):
    def __init__(self, config: TTTConfig, backbone: TTTBackbone):
        super().__init__()
        self.config = config
        dk = config.head_dim
        D = config.hidden_size

        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.post_norm = RMSNorm(dk)
        self.backbone = backbone
        self.eta = config.ttt_inner_lr / dk

        cos, sin = precompute_rope(dk, config.max_seq_len, config.rope_theta)
        self.rope_cos: torch.Tensor
        self.rope_sin: torch.Tensor
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        K = self.config.mini_batch_size
        nh = self.config.num_heads
        assert L % K == 0, f"seq_len {L} must be divisible by mini_batch_size {K}"
        num_mb = L // K

        # ── project & reshape to [B, nh, num_mb, K, dk] ──
        Q  = rearrange(self.q_proj(x), 'b l (nh dk) -> b nh l dk', nh=nh)
        XK = rearrange(self.k_proj(x), 'b l (nh dk) -> b nh l dk', nh=nh)
        V  = rearrange(self.v_proj(x), 'b l (nh dk) -> b nh l dk', nh=nh)

        Q  = rearrange(Q,  'b nh (nm k) dk -> b nh nm k dk', k=K)
        XK = rearrange(XK, 'b nh (nm k) dk -> b nh nm k dk', k=K)
        V  = rearrange(V,  'b nh (nm k) dk -> b nh nm k dk', k=K)

        # ── RoPE (positions reset per mini-batch) ──
        cos = self.rope_cos[:K]                            # [K, dk//2]
        sin = self.rope_sin[:K]
        Q  = apply_rope(Q, cos, sin)
        XK = apply_rope(XK, cos, sin)

        # ── scan over mini-batches ──
        params = self.backbone.init_fast_weights(B, x.device)
        outputs = []
        for mb in range(num_mb):
            result = self.backbone.compute_mini_batch(
                params,
                Q[:, :, mb],                               # [B, nh, K, dk]
                XK[:, :, mb],
                V[:, :, mb],
                self.eta,
            )
            params = result.params
            outputs.append(result.output)

        O = torch.stack(outputs, dim=2)                    # [B, nh, num_mb, K, dk]
        O = rearrange(O, 'b nh nm k dk -> b nh (nm k) dk')
        O = self.post_norm(O)
        O = rearrange(O, 'b nh l dk -> b l (nh dk)')
        return self.o_proj(O)


# ──────────────────────────────────────────────────────────────────────────────
# Outer FFN  (standard SwiGLU — outer model is not restricted)
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ──────────────────────────────────────────────────────────────────────────────
# Block & Model
# ──────────────────────────────────────────────────────────────────────────────

def _make_backbone(config: TTTConfig) -> TTTBackbone:
    nh, dk = config.num_heads, config.head_dim
    dh = int(dk * config.ttt_hidden_mult)
    if config.backbone_type == "linear":
        return TTTLinear(nh, dk)
    if config.backbone_type == "mlp":
        return TTTMLP(nh, dk, dh)
    if config.backbone_type == "mlp_frozen_phi":
        return TTTMLPFrozenPhi(nh, dk, dh)
    raise ValueError(f"Unknown backbone: {config.backbone_type}")


class TTTBlock(nn.Module):
    def __init__(self, config: TTTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size)
        self.ttt = TTTLayer(config, _make_backbone(config))
        self.ln2 = RMSNorm(config.hidden_size)
        self.ffn = FeedForward(config.hidden_size, config.hidden_size * config.ffn_mult)

    def forward(self, x):
        x = x + self.ttt(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TTTModel(nn.Module):
    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TTTBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight            # tie weights

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    # Quick test to verify the model runs and produces gradients
    config = TTTConfig(
        vocab_size=100,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        ttt_inner_lr=0.1,
        mini_batch_size=8,
        ffn_mult=4,
        ttt_hidden_mult=4,
        max_seq_len=16,
        rope_theta=10000.0,
        backbone_type="mlp",
    )
    model = TTTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, targets)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    loss.backward()
    print("Gradients computed successfully.")