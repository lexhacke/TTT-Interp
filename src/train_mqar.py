"""Train TTT on clustered-key MQAR. Lightning + wandb."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from ttt import TTTConfig, TTTModel
from metrics import compute_entk, compute_jaccard, compute_switch_rate


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

    Args:
        num_clusters:     C centroids
        keys_per_cluster: M keys per cluster
        embed_dim:        d
        rho:              target intra-cluster cosine similarity
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
        keys.append(F.normalize(centroids[c] + noise, dim=-1))
        ids.append(torch.full((keys_per_cluster,), c, dtype=torch.long))

    codebook = torch.cat(keys, dim=0)
    cluster_ids = torch.cat(ids, dim=0)
    actual_rho = _measure_rho(codebook)
    print(f"Codebook: {num_clusters}×{keys_per_cluster}, d={embed_dim}, "
          f"target ρ={rho:.2f}, actual ρ={actual_rho:.3f}")
    return codebook, cluster_ids


# ──────────────────────────────────────────────────────────────────────────────
# MQAR data generation
# ──────────────────────────────────────────────────────────────────────────────

def make_clustered_mqar(
    codebook: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_examples: int,
    input_seq_len: int = 128,
    num_kv_pairs: int = 8,
    num_value_tokens: int = 4096,
    seed: int = 0,
    power_a: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_keys = codebook.shape[0]
    vocab_size = num_keys + num_value_tokens

    assert input_seq_len % 2 == 0
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)
    context_size = num_kv_pairs * 2

    keys = np.stack([
        np.random.choice(num_keys, size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])
    values = np.stack([
        np.random.choice(np.arange(num_keys, vocab_size), size=num_kv_pairs, replace=False)
        for _ in range(num_examples)
    ])

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

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

    pad_mask = inputs == 0
    inputs[pad_mask] = torch.randint(num_keys, vocab_size, size=inputs.shape)[pad_mask]

    return inputs, labels


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────

class MQARDataModule(L.LightningDataModule):
    def __init__(
        self,
        codebook: torch.Tensor,
        cluster_ids: torch.Tensor,
        num_train: int = 100_000,
        num_test: int = 3_000,
        seq_len: int = 128,
        num_kv_pairs: int = 8,
        num_value_tokens: int = 4096,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["codebook", "cluster_ids"])
        self.codebook = codebook
        self.cluster_ids = cluster_ids

    def setup(self, stage=None):
        hp = self.hparams
        self.train_x, self.train_y = make_clustered_mqar(
            self.codebook, self.cluster_ids, hp.num_train,
            hp.seq_len, hp.num_kv_pairs, hp.num_value_tokens, seed=0,
        )
        self.test_x, self.test_y = make_clustered_mqar(
            self.codebook, self.cluster_ids, hp.num_test,
            hp.seq_len, hp.num_kv_pairs, hp.num_value_tokens, seed=1,
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_x, self.train_y),
            batch_size=self.hparams.batch_size, shuffle=True,
            drop_last=True, num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.test_x, self.test_y),
            batch_size=self.hparams.batch_size, shuffle=False,
            drop_last=False, num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# LightningModule
# ──────────────────────────────────────────────────────────────────────────────

class TTTLitModel(L.LightningModule):
    def __init__(
        self,
        config: TTTConfig,
        codebook: torch.Tensor,
        cluster_ids: torch.Tensor,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "codebook", "cluster_ids"])
        self.config = config
        self.model = TTTModel(config)
        self.codebook = codebook
        self.cluster_ids = cluster_ids
        self.lr = lr
        self.weight_decay = weight_decay

        # inject codebook into embedding layer
        self._inject_codebook()

    def _inject_codebook(self):
        num_keys = self.codebook.shape[0]
        with torch.no_grad():
            self.model.embed.weight[:num_keys] = self.codebook

        def _zero_key_grad(grad):
            grad[:num_keys] = 0
            return grad
        self.model.embed.weight.register_hook(_zero_key_grad)

    def forward(self, input_ids):
        return self.model(input_ids)

    def _shared_step(self, batch):
        x, y = batch
        logits, _ = self.model(x)
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size), y.view(-1), ignore_index=-100,
        )
        # accuracy on query positions only
        mask = y != -100
        acc = (logits.argmax(-1)[mask] == y[mask]).float().mean() if mask.any() else torch.tensor(0.0)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # mechanistic eval every validation epoch
        mech = self._eval_mechanistic()
        if mech is None:
            return

        entk_mean = mech["entk"].abs().sum() / (mech["entk"] != 0).sum().clamp(min=1)
        self.log("eval/entk_mean_abs", entk_mean)

        if mech["switch_rate"] is not None:
            self.log("eval/switch_rate_mean", mech["switch_rate"].mean())
        if mech["jaccard"] is not None:
            jvals = mech["jaccard"][mech["jaccard"] != 0]
            if len(jvals) > 0:
                self.log("eval/jaccard_mean", jvals.mean())

        # log heatmaps to wandb
        if self.logger and hasattr(self.logger, "experiment"):
            import wandb as wb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(mech["entk"][0].cpu().numpy(), cmap="RdBu_r", aspect="equal")
            ax.set_title(f"eNTK (tril) — step {self.global_step}")
            ax.set_xlabel("key token m"); ax.set_ylabel("query token n")
            plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout()
            self.logger.experiment.log({"eval/entk_heatmap": wb.Image(fig)})
            plt.close(fig)

            if mech["jaccard"] is not None:
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(mech["jaccard"][0].cpu().numpy(), cmap="viridis", aspect="equal")
                ax.set_title(f"Jaccard (tril) — step {self.global_step}")
                ax.set_xlabel("token m"); ax.set_ylabel("token n")
                plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout()
                self.logger.experiment.log({"eval/jaccard_heatmap": wb.Image(fig)})
                plt.close(fig)

    @torch.no_grad()
    def _eval_mechanistic(self, num_bindings: int = 8):
        backbone = self.model.blocks[0].ttt.backbone
        layer = self.model.blocks[0].ttt
        nh, dk = self.config.num_heads, self.config.head_dim
        device = self.device
        cluster_ids = self.cluster_ids.to(device)

        num_clusters = cluster_ids.max().item() + 1
        selected = []
        for c in range(min(num_clusters, num_bindings)):
            idxs = (cluster_ids == c).nonzero(as_tuple=True)[0]
            selected.append(idxs[torch.randint(len(idxs), (1,)).item()])
        if len(selected) < 2:
            return None
        selected = torch.stack(selected)

        emb = self.model.embed(selected.to(device))
        keys = layer.k_proj(emb)
        queries = layer.q_proj(emb)
        values = layer.v_proj(emb)

        N = len(selected)
        keys = keys.view(1, N, nh, dk).permute(0, 2, 1, 3)
        queries = queries.view(1, N, nh, dk).permute(0, 2, 1, 3)
        values = values.view(1, N, nh, dk).permute(0, 2, 1, 3)

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

        entk = compute_entk(backbone, params, queries, keys)
        entk_tril = torch.tril(entk[0], diagonal=-1)

        jaccard_tril = None
        switch_rate = None
        if all_masks:
            masks = torch.cat(all_masks, dim=2)
            jaccard_tril = torch.tril(compute_jaccard(masks)[0], diagonal=-1)
            switch_rate = compute_switch_rate(masks)[0]

        return {"entk": entk_tril, "jaccard": jaccard_tril, "switch_rate": switch_rate}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.estimated_stepping_batches,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--backbone", type=str, default="linear", choices=["linear", "mlp", "mlp_frozen_phi"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--mini-batch-size", type=int, default=16)
    p.add_argument("--inner-lr", type=float, default=1.0)
    p.add_argument("--ttt-hidden-mult", type=int, default=4)

    # data
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--num-clusters", type=int, default=4)
    p.add_argument("--keys-per-cluster", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--kv-pairs", type=int, default=8)
    p.add_argument("--num-train", type=int, default=100_000)
    p.add_argument("--num-test", type=int, default=3_000)
    p.add_argument("--num-value-tokens", type=int, default=4096)

    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-wandb", action="store_true")

    args = p.parse_args()

    # ── codebook ──
    codebook, cluster_ids = make_clustered_codebook(
        args.num_clusters, args.keys_per_cluster, args.hidden, rho=args.rho,
    )
    vocab_size = args.num_clusters * args.keys_per_cluster + args.num_value_tokens

    # ── data ──
    dm = MQARDataModule(
        codebook=codebook, cluster_ids=cluster_ids,
        num_train=args.num_train, num_test=args.num_test,
        seq_len=args.seq_len, num_kv_pairs=args.kv_pairs,
        num_value_tokens=args.num_value_tokens,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # ── model ──
    config = TTTConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        mini_batch_size=args.mini_batch_size,
        ttt_inner_lr=args.inner_lr,
        ttt_hidden_mult=args.ttt_hidden_mult,
        max_seq_len=args.seq_len,
        backbone_type=args.backbone,
    )
    lit = TTTLitModel(config, codebook, cluster_ids, lr=args.lr)

    # ── logger ──
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project="ttt-interp",
            name=f"{args.backbone}_rho{args.rho}",
            config=vars(args),
        )

    # ── callbacks ──
    callbacks = [
        ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1, filename="best-{epoch}-{val/acc:.4f}"),
        EarlyStopping(monitor="val/acc", mode="max", patience=10, min_delta=0.001),
    ]

    # ── train ──
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    trainer.fit(lit, dm)


if __name__ == "__main__":
    main()
