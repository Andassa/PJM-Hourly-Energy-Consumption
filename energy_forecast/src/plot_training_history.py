"""
Trace les courbes train / val (MSE) à partir des ``history_*.json`` produits par train.py.

Usage (racine du dépôt) :
    python energy_forecast/src/plot_training_history.py
    python energy_forecast/src/plot_training_history.py --config energy_forecast/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def plot_one(history_path: Path, out_dir: Path) -> Path:
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [int(r["epoch"]) for r in rows]
    train = [r["train_loss"] for r in rows]
    val = [r["val_loss"] for r in rows]

    tag = history_path.stem.replace("history_", "")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train, "o-", label="Train (MSE)", markersize=4)
    ax.plot(epochs, val, "s-", label="Validation (MSE)", markersize=4)
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE (charge normalisée)")
    ax.set_title(f"Perte d'entraînement — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"loss_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_combined(history_paths: list[Path], out_dir: Path) -> Path | None:
    if len(history_paths) < 2:
        return None
    n = len(history_paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, hp in zip(axes[0], history_paths):
        rows = json.loads(hp.read_text(encoding="utf-8"))
        epochs = [int(r["epoch"]) for r in rows]
        tag = hp.stem.replace("history_", "")
        ax.plot(epochs, [r["train_loss"] for r in rows], "o-", label="Train", markersize=3)
        ax.plot(epochs, [r["val_loss"] for r in rows], "s-", label="Val", markersize=3)
        ax.set_title(tag)
        ax.set_xlabel("Époque")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Comparaison des courbes de perte (MSE normalisée)", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "loss_all_models.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphiques train/val depuis history_*.json")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Dossier des history_*.json (défaut : depuis config.yaml)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Sortie PNG (défaut : energy_forecast/results/plots/)",
    )
    parser.add_argument("--no-combined", action="store_true", help="Ne pas générer loss_all_models.png")
    args = parser.parse_args()

    if args.config is not None or args.checkpoints_dir is None:
        cfg_path = (args.config or default_config_path()).resolve()
        root = cfg_path.parent
        ckpt_dir = root / load_yaml(cfg_path)["paths"]["checkpoints_dir"]
    else:
        ckpt_dir = args.checkpoints_dir.resolve()
        root = ckpt_dir.parent.parent

    plots_dir = args.plots_dir or (root / "results" / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    history_files = sorted(ckpt_dir.glob("history_*.json"))
    if not history_files:
        raise FileNotFoundError(f"Aucun history_*.json dans {ckpt_dir}")

    for hp in history_files:
        out = plot_one(hp, plots_dir)
        print("Figure :", out.resolve())

    if not args.no_combined:
        combined = plot_combined(history_files, plots_dir)
        if combined is not None:
            print("Figure :", combined.resolve())


if __name__ == "__main__":
    main()
