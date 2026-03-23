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
    num_inner_steps: int = 1        # number of passes over each mini-batch chunk

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
        eta: torch.Tensor | float,
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

        # 2) MSE gradient: dL/dz = 2(z - v)
        dZ = 2.0 * (Z - XV)

        # 3) Dual form output: query i accumulates gradient corrections from tokens j <= i
        #    eta is [B, nh, K, 1] — scales each token's contribution
        Attn = torch.tril(XQ @ XK.transpose(-1, -2))     # [B, nh, K, K]
        O = XQ @ W.transpose(-1, -2) - (eta * Attn) @ dZ

        # 4) Update W for next chunk: use last token's eta (accumulates all prior updates)
        eta_last = eta[:, :, -1:, :] if isinstance(eta, torch.Tensor) else eta
        W_new = W - eta_last * (dZ.transpose(-1, -2) @ XK)
        return TTTOutput(params={"W": W_new}, output=O)


def _rms_norm_fwd(x, weight, eps=1e-6):
    """RMSNorm forward: y = (x / rms) * weight."""
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * weight


def _rms_norm_fused_l2_bwd(x, l2_target, weight, eps=1e-6):
    """Fused RMSNorm + L2 loss backward: dL/dx for loss = ||RMSNorm(x) - target||^2.

    Adapted from ttt-reference.py ln_fused_l2_bwd, but without mean-centering (RMS not LN).
    """
    D = x.shape[-1]
    x_f = x.float()
    rms_inv = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    x_hat = x_f * rms_inv

    # forward: y = weight * x_hat
    y = weight * x_hat
    # L2 grad: dL/dy = 2(y - target)
    grad_output = 2.0 * (y - l2_target)
    # chain through weight: dL/d(x_hat) = grad_output * weight
    grad_x_hat = grad_output.float() * weight

    # chain through x_hat = x * rms_inv (no mean subtraction unlike LN)
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        * rms_inv
    )
    return z.to(x.dtype)


def _silu_bwd(x):
    """Derivative of SiLU: d/dx [x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
    sig = torch.sigmoid(x)
    return sig * (1.0 + x * (1.0 - sig))


class TTTMLP(TTTBackbone):
    """Inner model: f(x) = x + RMSNorm(SiLU(x @ W0^T) @ W1^T)

    SiLU (Swish) activation — smooth, differentiable everywhere. Enables stable
    multi-step inner optimization without the chaotic gradient landscape of ReLU.

    For each token i:
      FORWARD:  r_i = SiLU(k_i @ W0^T) @ W1^T           (raw MLP output)
                z_i = k_i + RMSNorm(r_i)                  (output with residual)

      LOSS:     L = ||RMSNorm(r_i) - (v_i - k_i)||^2

      BACKWARD: dr_i = rms_norm_fused_l2_bwd(r_i, v_i - k_i, weight)
                dW1 = dr_i^T @ a_i
                da_i = dr_i @ W1
                dW0 = (da_i * silu'(h_i))^T @ k_i         (smooth derivative)

      EVAL:     o_i = q_i + RMSNorm(SiLU(q_i @ W0^T) @ W1^T)
    """

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.W0 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, hidden_dim, head_dim)))
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, hidden_dim)))
        self.norm_weight = nn.Parameter(torch.ones(num_heads, 1, head_dim))
        self.norm_eps = 1e-6

    def init_fast_weights(self, B, device):
        return {
            "W0": self.W0.unsqueeze(0).expand(B, -1, -1, -1).clone(),
            "W1": self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone(),
        }

    def compute_mini_batch(self, params, XQ, XK, XV, eta):
        W0 = params["W0"]                                # [B, nh, dh, dk]
        W1 = params["W1"]                                # [B, nh, dk, dh]
        K_len = XK.shape[-2]

        # ── Phase 1: accumulate gradients over chunk (all at initial W) ──
        dW0_acc = torch.zeros_like(W0)
        dW1_acc = torch.zeros_like(W1)
        for i in range(K_len):
            k_i = XK[:, :, i : i + 1, :]                 # [B, nh, 1, dk]
            v_i = XV[:, :, i : i + 1, :]                  # [B, nh, 1, dv]
            eta_i = eta[:, :, i : i + 1, :] if isinstance(eta, torch.Tensor) else eta

            h_i = k_i @ W0.transpose(-1, -2)              # [B, nh, 1, dh]
            a_i = F.silu(h_i)                              # SiLU activation
            r_i = a_i @ W1.transpose(-1, -2)               # [B, nh, 1, dk]

            target_i = v_i - k_i
            dr_i = _rms_norm_fused_l2_bwd(r_i, target_i, self.norm_weight, self.norm_eps)
            dW1_acc = dW1_acc + eta_i * (dr_i.transpose(-1, -2) @ a_i)
            da_i = dr_i @ W1
            dW0_acc = dW0_acc + eta_i * ((da_i * _silu_bwd(h_i)).transpose(-1, -2) @ k_i)

        # ── Phase 2: apply accumulated gradient, then eval all queries ──
        W0 = W0 - dW0_acc
        W1 = W1 - dW1_acc

        H_Q = XQ @ W0.transpose(-1, -2)                   # [B, nh, K, dh]
        R_Q = F.silu(H_Q) @ W1.transpose(-1, -2)          # [B, nh, K, dk]
        O = XQ + _rms_norm_fwd(R_Q, self.norm_weight, self.norm_eps)

        return TTTOutput(params={"W0": W0, "W1": W1}, output=O, masks=None)


class TTTMLPFrozenPhi(TTTBackbone):
    """Inner model: f(x) = x + RMSNorm(ReLU(x @ W0^T) @ W1^T)

    ONLY W1 updates. W0 is frozen during the inner loop (still trained by outer loop).
    Uses fused RMSNorm+L2 backward matching the full MLP backbone.

    Loss: ||RMSNorm(A_K @ W1^T) - (XV - XK)||^2
    Dual form operates on dR from the fused backward.
    """

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.W0 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, hidden_dim, head_dim)))
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, hidden_dim)))
        self.norm_weight = nn.Parameter(torch.ones(num_heads, 1, head_dim))
        self.norm_eps = 1e-6

    def init_fast_weights(self, B, device):
        return {"W1": self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()}

    def compute_mini_batch(self, params, XQ, XK, XV, eta):
        W0 = self.W0                                      # [nh, dh, dk] frozen
        W1 = params["W1"]                                  # [B, nh, dv, dh]

        A_K = F.relu(XK @ W0.transpose(-1, -2))           # [B, nh, K, dh]
        A_Q = F.relu(XQ @ W0.transpose(-1, -2))           # [B, nh, K, dh]

        # Forward: raw MLP output (pre-norm)
        R_K = A_K @ W1.transpose(-1, -2)                  # [B, nh, K, dk]

        # Fused RMSNorm + L2 backward, target = V - K
        target = XV - XK
        dR = _rms_norm_fused_l2_bwd(R_K, target, self.norm_weight, self.norm_eps)

        # Dual form on dR
        Attn = torch.tril(A_Q @ A_K.transpose(-1, -2))   # [B, nh, K, K]
        R_Q = A_Q @ W1.transpose(-1, -2)                  # [B, nh, K, dk]
        O = XQ + _rms_norm_fwd(R_Q - (eta * Attn) @ dR, self.norm_weight, self.norm_eps)
        eta_last = eta[:, :, -1:, :] if isinstance(eta, torch.Tensor) else eta
        W1_new = W1 - eta_last * (dR.transpose(-1, -2) @ A_K)
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
        self.base_lr = config.ttt_inner_lr / dk

        # Learnable eta: per-head linear projection → sigmoid → scale by base_lr
        # Input token → eta, so each token gets its own learning rate
        self.eta_proj = nn.Linear(D, config.num_heads, bias=True)
        nn.init.zeros_(self.eta_proj.bias)
        nn.init.normal_(self.eta_proj.weight, std=0.02)

        # 1/i token decay: turns accumulated gradient sum into running average
        # Without this, token 32 gets 32x the effective update of token 1
        token_idx = 1.0 / torch.arange(1, config.mini_batch_size + 1)  # [1, 1/2, 1/3, ...]
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(config.mini_batch_size))

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

        # ── learnable eta: [B, L] → [B, nh, num_mb, K, 1] ──
        ttt_lr = torch.sigmoid(self.eta_proj(x)) * self.base_lr  # [B, L, nh]
        ttt_lr = rearrange(ttt_lr, 'b (nm k) nh -> b nh nm k 1', k=K)

        # ── 1/i token decay: running average instead of running sum ──
        token_decay = torch.clamp_min(self.token_idx + self.learnable_token_idx, 0.0)  # [K]
        token_decay = token_decay.view(1, 1, 1, K, 1)       # broadcast over [B, nh, nm, K, 1]
        eta = ttt_lr * token_decay

        # ── RoPE (positions reset per mini-batch) ──
        cos = self.rope_cos[:K]                            # [K, dk//2]
        sin = self.rope_sin[:K]
        Q  = apply_rope(Q, cos, sin)
        XK = apply_rope(XK, cos, sin)

        # ── scan over mini-batches, with optional multi-step inner optimization ──
        num_inner_steps = self.config.num_inner_steps
        params = self.backbone.init_fast_weights(B, x.device)
        outputs = []
        inner_losses = [[] for _ in range(num_inner_steps)]  # per-step inner losses
        for mb in range(num_mb):
            for step in range(num_inner_steps):
                result = self.backbone.compute_mini_batch(
                    params,
                    Q[:, :, mb],                           # [B, nh, K, dk]
                    XK[:, :, mb],
                    V[:, :, mb],
                    eta[:, :, mb],                         # [B, nh, K, 1]
                )
                # compute inner MSE: ||f(k) - v||^2 averaged over chunk
                with torch.no_grad():
                    inner_mse = (result.output - V[:, :, mb]).pow(2).mean()
                    inner_losses[step].append(inner_mse)
                params = result.params
            outputs.append(result.output)

        O = torch.stack(outputs, dim=2)                    # [B, nh, num_mb, K, dk]
        O = rearrange(O, 'b nh nm k dk -> b nh (nm k) dk')
        O = self.post_norm(O)
        O = rearrange(O, 'b nh l dk -> b l (nh dk)')

        # average inner losses across mini-batches: [num_inner_steps]
        self._inner_losses = [
            torch.stack(step_losses).mean() for step_losses in inner_losses
        ]

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