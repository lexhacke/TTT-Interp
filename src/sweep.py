"""Local sweep launcher. Reads config.json and runs all combinations."""

import json
import subprocess
import sys
import itertools
from pathlib import Path


def main():
    config_path = Path(__file__).parent / "config.json"
    config = json.loads(config_path.read_text())

    backbones = config["backbones"]
    hidden_sizes = config["hidden_sizes"]
    rhos = config["rho"]

    heads = 4

    runs = []
    for backbone, hidden, rho in itertools.product(backbones, hidden_sizes, rhos):
        head_dim = hidden // heads
        if backbone == "linear":
            ttt_hidden_mult = 4.0  # doesn't matter, unused
        else:
            # param-match: linear has dk² params, MLP has 2*dk*dh
            # dk² = 2*dk*dh → dh = dk/2 → mult = 0.5
            ttt_hidden_mult = 0.5

        runs.append({
            "backbone": backbone,
            "hidden": hidden,
            "rho": rho,
            "ttt_hidden_mult": ttt_hidden_mult,
        })

        # also run frozen-phi for every MLP config
        if backbone == "mlp":
            runs.append({
                "backbone": "mlp_frozen_phi",
                "hidden": hidden,
                "rho": rho,
                "ttt_hidden_mult": ttt_hidden_mult,
            })

    print(f"Total runs: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{len(runs)}: {run}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(Path(__file__).parent / "train_mqar.py"),
            "--backbone", run["backbone"],
            "--hidden", str(run["hidden"]),
            "--heads", str(heads),
            "--rho", str(run["rho"]),
            "--ttt-hidden-mult", str(run["ttt_hidden_mult"]),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
