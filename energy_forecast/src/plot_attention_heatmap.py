"""
Heatmap des poids d'attention (LSTM + Bahdanau) sur le jeu de test.

Usage :
    python energy_forecast/src/plot_attention_heatmap.py --config energy_forecast/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval_utils import (  # noqa: E402
    default_config_path,
    load_model_from_checkpoint,
    load_test_bundle,
    load_yaml,
    predict_attention_weights,
)
from models import LSTMBahdanauForecaster  # noqa: E402


def save_heatmap(alpha_1d: np.ndarray, title: str, out_path: Path) -> None:
    """alpha_1d : (168,) poids sur les pas d'entrée."""
    fig, ax = plt.subplots(figsize=(12, 2.8))
    data = alpha_1d.reshape(1, -1)
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=data.max() or 1.0)
    ax.set_yticks([0])
    ax.set_yticklabels(["Poids α"])
    ax.set_xlabel("Position dans la fenêtre d'entrée (heure t−168 … t−1)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="α (softmax)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap attention Bahdanau.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--window-idx", type=int, default=1280, help="Index d'une fenêtre test.")
    parser.add_argument("--mean-windows", type=int, default=200, help="Moyenne α sur N fenêtres (0=désactivé).")
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    cfg_path = (args.config or default_config_path()).resolve()
    root = cfg_path.parent
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    device_str = str(cfg.get("device", "cpu")).lower()
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    bundle = load_test_bundle(root, paths, subset=args.subset)
    ckpt = (root / paths["checkpoints_dir"] / "best_lstm_attention.pth").resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint attention introuvable : {ckpt}")

    model, _ = load_model_from_checkpoint(ckpt, model_cfg, int(bundle["x"].shape[-1]), device)
    if not isinstance(model, LSTMBahdanauForecaster):
        raise TypeError("Ce script nécessite le modèle lstm_attention.")

    alphas = predict_attention_weights(
        model, bundle["x"], device, int(train_cfg["batch_size"])
    )

    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    idx = max(0, min(args.window_idx, alphas.shape[0] - 1))
    p1 = plots_dir / f"attn_heatmap_window_{idx}.png"
    save_heatmap(
        alphas[idx],
        f"Poids d'attention — fenêtre test #{idx} (LSTM+attention)",
        p1,
    )
    print("Figure :", p1.resolve())

    if args.mean_windows > 0:
        n = min(args.mean_windows, alphas.shape[0])
        mean_alpha = alphas[:n].mean(axis=0)
        p2 = plots_dir / f"attn_heatmap_mean_{n}windows.png"
        save_heatmap(
            mean_alpha,
            f"Poids d'attention moyens — {n} premières fenêtres test",
            p2,
        )
        print("Figure :", p2.resolve())


if __name__ == "__main__":
    main()
