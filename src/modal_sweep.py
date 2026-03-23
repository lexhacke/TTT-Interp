"""Modal launcher for TTT-Interp sweep. Each (backbone, hidden, rho) gets its own GPU container."""

import json
import itertools
import modal

app = modal.App("ttt-interp-sweep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "lightning",
        "wandb",
        "einops",
        "numpy",
        "matplotlib",
    )
    .add_local_dir(".", remote_path="/root/ttt-interp", copy=True)
)


@app.function(
    image=image,
    gpu="T4",
    timeout=10800,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_single(
    backbone: str, hidden: int, rho: float, ttt_hidden_mult: float,
    heads: int = 4, num_kv_pairs: int = 8, num_overwrites: int = 0,
    seq_len: int = 128, num_inner_steps: int = 1,
):
    import os
    os.environ["WANDB_HTTP_TIMEOUT"] = "60"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    import sys
    sys.path.insert(0, "/root/ttt-interp")
    from train_mqar import (
        TTTLitModel, MQARDataModule, make_clustered_codebook,
    )
    from ttt import TTTConfig
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

    num_clusters, keys_per_cluster = 4, 8
    num_value_tokens = 4096
    batch_size, lr, epochs = 64, 1e-3, 50

    codebook, cluster_ids = make_clustered_codebook(
        num_clusters, keys_per_cluster, hidden, rho=rho,
    )
    vocab_size = num_clusters * keys_per_cluster + num_value_tokens

    dm = MQARDataModule(
        codebook=codebook, cluster_ids=cluster_ids,
        num_train=100_000, num_test=3_000,
        seq_len=seq_len, num_kv_pairs=num_kv_pairs,
        num_overwrites=num_overwrites,
        num_value_tokens=num_value_tokens,
        batch_size=batch_size, num_workers=2,
    )

    config = TTTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_heads=heads,
        num_layers=2,
        mini_batch_size=16,
        ttt_inner_lr=1.0,
        ttt_hidden_mult=ttt_hidden_mult,
        max_seq_len=seq_len,
        backbone_type=backbone,
        num_inner_steps=num_inner_steps,
    )
    lit = TTTLitModel(config, codebook, cluster_ids, lr=lr)

    ow_tag = f"_ow{num_overwrites}" if num_overwrites > 0 else ""
    steps_tag = f"_s{num_inner_steps}" if num_inner_steps > 1 else ""
    logger = WandbLogger(
        project="ttt-interp",
        name=f"{backbone}_h{hidden}_rho{rho}_kv{num_kv_pairs}{ow_tag}{steps_tag}",
        config={
            "backbone": backbone, "hidden": hidden, "rho": rho,
            "ttt_hidden_mult": ttt_hidden_mult, "heads": heads,
            "num_kv_pairs": num_kv_pairs, "num_overwrites": num_overwrites,
            "seq_len": seq_len, "num_inner_steps": num_inner_steps,
        },
    )

    callbacks = [
        ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1),
        EarlyStopping(monitor="val/acc", mode="max", patience=10, min_delta=0.001),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        precision="32-true",
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
    )
    trainer.fit(lit, dm)


@app.local_entrypoint()
def main():
    config = json.loads(open("config.json").read())

    backbones = ["linear", "mlp", "mlp_frozen_phi"]
    hidden_sizes = config["hidden_sizes"]
    rhos = config["rho"]
    heads = 4
    num_kv_pairs = config.get("num_kv_pairs", 8)
    num_overwrites = config.get("num_overwrites", 0)
    seq_len = config.get("seq_len", 128)
    inner_steps_list = config.get("num_inner_steps", [1])
    if isinstance(inner_steps_list, int):
        inner_steps_list = [inner_steps_list]

    runs = []
    for backbone, hidden, rho, steps in itertools.product(backbones, hidden_sizes, rhos, inner_steps_list):
        ttt_hidden_mult = 4.0 if backbone == "linear" else 0.5
        runs.append((backbone, hidden, rho, ttt_hidden_mult, heads, steps))

    print(f"Launching {len(runs)} runs on Modal (kv={num_kv_pairs}, ow={num_overwrites}, steps={inner_steps_list})...")

    handles = []
    for backbone, hidden, rho, mult, h, steps in runs:
        handles.append(run_single.spawn(
            backbone, hidden, rho, mult, h,
            num_kv_pairs=num_kv_pairs, num_overwrites=num_overwrites,
            seq_len=seq_len, num_inner_steps=steps,
        ))

    for handle in handles:
        handle.get()

    print("All runs complete.")
