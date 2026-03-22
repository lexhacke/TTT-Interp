"""Train TTT on clustered-key MQAR. Self-contained — no zoology dependency."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ttt import TTTConfig, TTTModel, TTTMLP
from metrics import compute_entk, compute_jaccard, compute_switch_rate
import wandb


# ──────────────────────────────────────────────────────────────────────────────
# Clustered embedding codebook
# ──────────────────────────────────────────────────────────────────────────────

def make_clustered_codebook(
    num_clusters: int,
    keys_per_cluster: int,
    embed_dim: int,
    rho: float = 0.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate key embeddings clustered on the unit sphere.

    Samples C centroids uniformly on S^{d-1}, then generates M keys per cluster:
        key = normalize(centroid + sigma * noise)
    where sigma controls intra-cluster spread.

    Args:
        num_clusters:     C centroids
        keys_per_cluster: M keys per cluster
        embed_dim:        d, embedding dimension
        rho:              target pairwise cosine similarity within clusters
                          (0 → orthogonal, 1 → identical)
        seed:             random seed

    Returns:
        codebook:    [C * M, d] unit vectors
        cluster_ids: [C * M]    integer cluster assignment
    """
    torch.manual_seed(seed)

    centroids = F.normalize(torch.randn(num_clusters, embed_dim), dim=-1)

    def _sample(sigma):
        ks = []
        for c in range(num_clusters):
            noise = sigma * torch.randn(keys_per_cluster, embed_dim)
            ks.append(F.normalize(centroids[c] + noise, dim=-1))
        return torch.cat(ks, dim=0)

    def _measure_rho(codebook_):
        rhos = []
        for c in range(num_clusters):
            ck = codebook_[c * keys_per_cluster : (c + 1) * keys_per_cluster]
            sims = ck @ ck.T
            n = ck.shape[0]
            rhos.append((sims.sum() - n) / (n * (n - 1)))
        return torch.stack(rhos).mean().item()

    # binary search for sigma that gives target rho
    lo, hi = 1e-6, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _measure_rho(_sample(mid)) > rho:
            lo = mid
        else:
            hi = mid
    sigma = (lo + hi) / 2

    keys = []
    ids = []
    for c in range(num_clusters):
        noise = sigma * torch.randn(keys_per_cluster, embed_dim)
        cluster_keys = F.normalize(centroids[c] + noise, dim=-1)
        keys.append(cluster_keys)
        ids.append(torch.full((keys_per_cluster,), c, dtype=torch.long))

    codebook = torch.cat(keys, dim=0)
    cluster_ids = torch.cat(ids, dim=0)
    avg_rho = _measure_rho(codebook)
    print(f"Codebook: {num_clusters} clusters x {keys_per_cluster} keys, "
          f"d={embed_dim}, target rho={rho:.2f}, actual rho={avg_rho:.3f}")

    return codebook, cluster_ids


# ──────────────────────────────────────────────────────────────────────────────
# MQAR data generation (stolen from zoology, stripped of deps)
# ──────────────────────────────────────────────────────────────────────────────

def make_mqar(
    vocab_size: int = 8192,
    num_examples: int = 100_000,
    input_seq_len: int = 128,
    num_kv_pairs: int = 8,
    seed: int = 0,
    power_a: float = 0.01,
):
    """
    Generate MQAR sequences.

    Input:  [K1 V1 K2 V2 ... | random padding ... | K_query random ... K_query ...]
    Label:  [-100 ...         | -100 ...           | V_answer -100 ... V_answer ...]

    Keys from vocab[1 : vocab_size//2), values from vocab[vocab_size//2 : vocab_size).
    Queries are keys re-placed at power-law-distributed gaps in the second half.
    Model must recall the value associated with each queried key.
    """
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len  # need room for KV pairs + queries

    np.random.seed(seed)

    context_size = num_kv_pairs * 2
    key_vocab_size = vocab_size // 2

    # sample unique keys and values per example
    keys = np.stack([
        np.random.choice(np.arange(1, key_vocab_size), size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])
    values = np.stack([
        np.random.choice(np.arange(key_vocab_size, vocab_size), size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])

    # interleave keys and values: [K1 V1 K2 V2 ...]
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # power-law gaps for query placement
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    gaps = np.stack([
        np.random.choice(space, size=num_kv_pairs, replace=False, p=p)
        for _ in range(num_examples)
    ])

    # place queries in the second half
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    # labels: only at query answer positions
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])

    # fill padding zeros with random tokens
    inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return inputs, labels


def make_clustered_mqar(
    codebook: torch.Tensor,          # [num_keys, embed_dim] from make_clustered_codebook
    cluster_ids: torch.Tensor,       # [num_keys]
    num_examples: int = 100_000,
    input_seq_len: int = 128,
    num_kv_pairs: int = 8,
    num_value_tokens: int = 4096,    # how many distinct value tokens
    seed: int = 0,
    power_a: float = 0.01,
):
    """Generate MQAR with clustered keys.

    Same sequence format as make_mqar, but key token IDs index into a
    fixed clustered codebook.  The model's embedding layer should be
    initialized with this codebook (keys) + random embeddings (values/padding)
    and optionally frozen.

    Key tokens:   [0, num_keys)
    Value tokens: [num_keys, num_keys + num_value_tokens)
    Total vocab:  num_keys + num_value_tokens

    Returns:
        inputs:  [num_examples, input_seq_len]  int64 token IDs
        labels:  [num_examples, input_seq_len]  int64 (-100 at non-query positions)
    """
    num_keys = codebook.shape[0]
    vocab_size = num_keys + num_value_tokens

    assert input_seq_len % 2 == 0
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)
    context_size = num_kv_pairs * 2

    # sample keys (token IDs into codebook) — unique per example
    keys = np.stack([
        np.random.choice(num_keys, size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])
    # sample values from value token range
    values = np.stack([
        np.random.choice(np.arange(num_keys, vocab_size), size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])

    # interleave: [K1 V1 K2 V2 ...]
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # power-law query placement
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack([
        np.random.choice(space, size=num_kv_pairs, replace=False, p=p)
        for _ in range(num_examples)
    ])

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])

    # fill padding with random tokens from value range (avoid key tokens)
    pad_mask = inputs == 0
    inputs[pad_mask] = torch.randint(num_keys, vocab_size, size=inputs.shape)[pad_mask]

    return inputs, labels


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_on_queries(logits, labels):
    """Accuracy only on positions where labels != -100."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def inject_codebook(model: TTTModel, codebook: torch.Tensor):
    """Overwrite key token embeddings with clustered codebook, freeze them.

    Key tokens are [0, num_keys). The rest of the embedding table
    (value tokens, padding) stays learnable.
    """
    num_keys, embed_dim = codebook.shape
    with torch.no_grad():
        model.embed.weight[:num_keys] = codebook
    # freeze key embeddings only: hook to zero grad for key rows
    def _zero_key_grad(grad):
        grad[:num_keys] = 0
        return grad
    model.embed.weight.register_hook(_zero_key_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Post-training eval: eNTK / Jaccard / switch-rate on held-out bindings
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_mechanistic(model: TTTModel, codebook: torch.Tensor, cluster_ids: torch.Tensor,
                     num_bindings: int = 8, device: str = "cpu"):
    """Run bindings through the first TTT layer's backbone, compute metrics.

    Returns dict with entk, jaccard, switch_rate tensors (or None if N/A).
    """
    model.eval()
    backbone = model.blocks[0].ttt.backbone
    layer = model.blocks[0].ttt
    nh = model.config.num_heads
    dk = model.config.head_dim

    # sample one key per cluster (up to num_bindings)
    num_clusters = cluster_ids.max().item() + 1
    selected = []
    for c in range(min(num_clusters, num_bindings)):
        idxs = (cluster_ids == c).nonzero(as_tuple=True)[0]
        selected.append(idxs[torch.randint(len(idxs), (1,)).item()])
    selected = torch.stack(selected)

    # get embeddings, project to K/Q space
    emb = model.embed(selected.to(device))                # [N, D]
    keys = layer.k_proj(emb)                               # [N, D]
    queries = layer.q_proj(emb)                            # [N, D]
    values = layer.v_proj(emb)                             # [N, D]

    # reshape to [1, nh, N, dk]
    N = len(selected)
    keys = keys.view(1, N, nh, dk).permute(0, 2, 1, 3)
    queries = queries.view(1, N, nh, dk).permute(0, 2, 1, 3)
    values = values.view(1, N, nh, dk).permute(0, 2, 1, 3)

    # run inner loop token-by-token
    eta = layer.eta
    params = backbone.init_fast_weights(1, device)
    all_masks = []
    for i in range(N):
        result = backbone.compute_mini_batch(
            params, queries[:, :, i:i+1], keys[:, :, i:i+1], values[:, :, i:i+1], eta,
        )
        params = result.params
        if result.masks is not None:
            all_masks.append(result.masks)

    # eNTK at final state
    entk = compute_entk(backbone, params, queries, keys)   # [1, nh, N, N]
    entk_tril = torch.tril(entk[0], diagonal=-1)           # [nh, N, N]

    # Jaccard + switch rate (MLP only)
    jaccard_tril = None
    switch_rate = None
    if len(all_masks) > 0:
        masks = torch.cat(all_masks, dim=2)                 # [1, nh, N, dh]
        jaccard = compute_jaccard(masks)
        jaccard_tril = torch.tril(jaccard[0], diagonal=-1)  # [nh, N, N]
        switch_rate = compute_switch_rate(masks)[0]          # [nh, N-1]

    return {
        "entk": entk_tril,             # [nh, N, N]
        "jaccard": jaccard_tril,        # [nh, N, N] or None
        "switch_rate": switch_rate,     # [nh, N-1] or None
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    # model
    backbone_type: str = "linear",
    hidden_size: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    mini_batch_size: int = 16,
    ttt_inner_lr: float = 1.0,
    ttt_hidden_mult: int = 4,
    # data
    rho: float = 0.8,
    num_clusters: int = 4,
    keys_per_cluster: int = 8,
    seq_len: int = 128,
    num_kv_pairs: int = 8,
    num_value_tokens: int = 4096,
    num_train: int = 100_000,
    num_test: int = 3_000,
    # training
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 50,
    eval_every: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_wandb: bool = True,
):
    # ── codebook ──
    codebook, cluster_ids = make_clustered_codebook(
        num_clusters, keys_per_cluster, hidden_size, rho=rho,
    )
    vocab_size = num_clusters * keys_per_cluster + num_value_tokens

    # ── data ──
    print(f"Generating clustered MQAR: rho={rho}, seq_len={seq_len}, kv_pairs={num_kv_pairs}")
    train_x, train_y = make_clustered_mqar(
        codebook, cluster_ids, num_train, seq_len, num_kv_pairs, num_value_tokens, seed=0,
    )
    test_x, test_y = make_clustered_mqar(
        codebook, cluster_ids, num_test, seq_len, num_kv_pairs, num_value_tokens, seed=1,
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, drop_last=False,
    )

    # ── model ──
    config = TTTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        mini_batch_size=mini_batch_size,
        ttt_inner_lr=ttt_inner_lr,
        ttt_hidden_mult=ttt_hidden_mult,
        max_seq_len=seq_len,
        backbone_type=backbone_type,
    )
    model = TTTModel(config).to(device)
    inject_codebook(model, codebook.to(device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {backbone_type}, params: {n_params:,}, device: {device}")

    # ── wandb ──
    if use_wandb:
        wandb.init(
            project="ttt-interp",
            config={
                "backbone": backbone_type, "rho": rho,
                "hidden_size": hidden_size, "num_heads": num_heads,
                "num_layers": num_layers, "mini_batch_size": mini_batch_size,
                "ttt_inner_lr": ttt_inner_lr, "ttt_hidden_mult": ttt_hidden_mult,
                "num_clusters": num_clusters, "keys_per_cluster": keys_per_cluster,
                "seq_len": seq_len, "num_kv_pairs": num_kv_pairs,
                "batch_size": batch_size, "lr": lr, "epochs": epochs,
                "n_params": n_params,
            },
            name=f"{backbone_type}_rho{rho}",
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    # ── train loop ──
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── eval ──
        if epoch % eval_every == 0:
            model.eval()
            test_acc = 0.0
            test_n = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits, _ = model(batch_x)
                    test_acc += accuracy_on_queries(logits, batch_y) * batch_x.size(0)
                    test_n += batch_x.size(0)
            test_acc /= test_n
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | test acc {test_acc:.4f}")

            if use_wandb:
                wandb.log({"epoch": epoch, "train/loss": avg_loss, "eval/accuracy": test_acc})

            if test_acc > 0.99:
                print("Solved!")
                break

    # ── post-training mechanistic eval ──
    print("Running mechanistic eval...")
    mech = eval_mechanistic(model, codebook.to(device), cluster_ids.to(device),
                            num_bindings=num_clusters, device=device)

    if use_wandb:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # log eNTK heatmap (head 0)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(mech["entk"][0].cpu().numpy(), cmap="RdBu_r", aspect="equal")
        ax.set_title(f"eNTK (tril) — {backbone_type} rho={rho}")
        ax.set_xlabel("key token m")
        ax.set_ylabel("query token n")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        wandb.log({"eval/entk_heatmap": wandb.Image(fig)})
        plt.close(fig)

        # log Jaccard heatmap if available
        if mech["jaccard"] is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(mech["jaccard"][0].cpu().numpy(), cmap="viridis", aspect="equal")
            ax.set_title(f"Jaccard (tril) — {backbone_type} rho={rho}")
            ax.set_xlabel("token m")
            ax.set_ylabel("token n")
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            wandb.log({"eval/jaccard_heatmap": wandb.Image(fig)})
            plt.close(fig)

        # log scalar summaries
        entk_mean = mech["entk"].abs().sum() / (mech["entk"] != 0).sum().clamp(min=1)
        log_dict = {"eval/entk_mean_abs": entk_mean.item()}
        if mech["switch_rate"] is not None:
            log_dict["eval/switch_rate_mean"] = mech["switch_rate"].mean().item()
        if mech["jaccard"] is not None:
            jaccard_vals = mech["jaccard"][mech["jaccard"] != 0]
            if len(jaccard_vals) > 0:
                log_dict["eval/jaccard_mean"] = jaccard_vals.mean().item()
        wandb.log(log_dict)

        wandb.finish()

    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="linear", choices=["linear", "mlp", "mlp_frozen_phi"])
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--num-clusters", type=int, default=4)
    p.add_argument("--keys-per-cluster", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--kv-pairs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    train(
        backbone_type=args.backbone,
        rho=args.rho,
        num_clusters=args.num_clusters,
        keys_per_cluster=args.keys_per_cluster,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        mini_batch_size=args.mini_batch_size,
        ttt_inner_lr=args.inner_lr,
        seq_len=args.seq_len,
        num_kv_pairs=args.kv_pairs,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=not args.no_wandb,
    )
