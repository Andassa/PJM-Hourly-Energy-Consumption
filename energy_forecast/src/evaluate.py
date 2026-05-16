"""
Évaluation sur le jeu de test : charge le checkpoint, prédit, dénormalise la cible en MW,
écrit ``results/metrics.csv`` et des figures dans ``results/plots/``.

Usage (racine du dépôt) :
    python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml
    python energy_forecast/src/evaluate.py --config energy_forecast/config.yaml --checkpoint energy_forecast/results/checkpoints/best_lstm.pth
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import LSTMBahdanauForecaster, build_model  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def forward_predictions(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, LSTMBahdanauForecaster):
        y_hat, _ = model(x)
        return y_hat
    return model(x)


def inverse_target_mw(
    y_norm: np.ndarray,
    scaler,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """y_norm : (...) valeurs Min-Max [0,1] pour la seule colonne cible."""
    idx = feature_cols.index(target_col)
    dmin = float(scaler.data_min_[idx])
    dmax = float(scaler.data_max_[idx])
    return y_norm * (dmax - dmin) + dmin


@torch.no_grad()
def predict_all(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        pred = forward_predictions(model, xb)
        outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0)


def metrics_mw(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> dict[str, float]:
    y_true = y_true.astype(np.float64).ravel()
    y_pred = y_pred.astype(np.float64).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))
    return {"mae_mw": mae, "rmse_mw": rmse, "mape_pct": mape, "n_points": float(y_true.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation test (MW) + figures.")
    parser.add_argument("--config", type=Path, default=None, help="YAML energy_forecast/config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Fichier .pth (défaut : best_<model>.pth)")
    parser.add_argument("--subset", type=int, default=None, help="N premiers échantillons test (debug).")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Sortie CSV (défaut : energy_forecast/results/metrics.csv)",
    )
    args = parser.parse_args()

    cfg_path = (args.config if args.config is not None else default_config_path()).resolve()
    cfg = load_yaml(cfg_path)
    root = cfg_path.parent
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    device_str = str(cfg.get("device", "cpu")).lower()
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    test_pkl = root / paths["test_pkl"]
    scaler_path = root / paths["sequence_scaler"]
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        model_slug = str(model_cfg["name"]).strip().lower().replace(" ", "_")
        ckpt_path = root / paths["checkpoints_dir"] / f"best_{model_slug}.pth"
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    blob = joblib.load(scaler_path)
    scaler = blob["scaler"]
    feature_cols: list[str] = list(blob["feature_cols"])
    target_col: str = str(blob["target_col"])

    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
    x_te = torch.from_numpy(np.asarray(test_data["X"], dtype=np.float32))
    y_te = np.asarray(test_data["y"], dtype=np.float32)

    if args.subset is not None:
        n = max(1, int(args.subset))
        x_te = x_te[:n]
        y_te = y_te[:n]
        print(f"Mode subset : test={x_te.shape[0]}")

    input_dim = int(x_te.shape[-1])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_model = str(ckpt.get("model_name", model_cfg["name"]))
    model = build_model(
        name=ckpt_model,
        input_dim=int(ckpt.get("input_dim", input_dim)),
        hidden_size=int(ckpt.get("hidden_size", model_cfg["hidden_size"])),
        num_layers=int(ckpt.get("num_layers", model_cfg["num_layers"])),
        horizon=int(ckpt.get("horizon", model_cfg["horizon"])),
        dropout=float(ckpt.get("dropout", model_cfg["dropout"])),
        attn_dim=int(ckpt.get("attn_dim", model_cfg.get("attn_dim", 64))),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    y_hat = predict_all(model, x_te, device, int(train_cfg["batch_size"]))

    y_true_mw = inverse_target_mw(y_te, scaler, feature_cols, target_col)
    y_hat_mw = inverse_target_mw(y_hat, scaler, feature_cols, target_col)

    m = metrics_mw(y_true_mw, y_hat_mw)
    row = {
        "model": ckpt_model,
        "checkpoint": str(ckpt_path),
        "target_col": target_col,
        **m,
    }
    print(json.dumps(row, indent=2))

    metrics_out = args.metrics_csv
    if metrics_out is None:
        metrics_out = root / "results" / "metrics.csv"
    else:
        metrics_out = metrics_out.resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if metrics_out.is_file():
        df_old = pd.read_csv(metrics_out)
        df_row = pd.concat([df_old, df_row], ignore_index=True)
    df_row.to_csv(metrics_out, index=False)
    print("Métriques enregistrées :", metrics_out.resolve())

    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(ckpt_path.stem).name

    # Une fenêtre : 24 pas, vérité vs prédiction (MW)
    idx = int(np.random.default_rng(42).integers(0, y_true_mw.shape[0]))
    h = np.arange(1, y_true_mw.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h, y_true_mw[idx], "o-", label="Observé", markersize=5)
    ax.plot(h, y_hat_mw[idx], "s--", label="Prédit", markersize=5)
    ax.set_xlabel("Heure dans l'horizon (1…24)")
    ax.set_ylabel(f"{target_col} (MW)")
    ax.set_title(f"Test — fenêtre #{idx} ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = plots_dir / f"eval_{tag}_window_{idx}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("Figure :", p1.resolve())

    # Nuage de points (sous-échantillon si très grand)
    flat_y = y_true_mw.ravel()
    flat_p = y_hat_mw.ravel()
    max_scatter = 8000
    if flat_y.size > max_scatter:
        rng = np.random.default_rng(42)
        pick = rng.choice(flat_y.size, size=max_scatter, replace=False)
        flat_y = flat_y[pick]
        flat_p = flat_p[pick]
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = min(flat_y.min(), flat_p.min()), max(flat_y.max(), flat_p.max())
    ax.scatter(flat_y, flat_p, s=4, alpha=0.35, c="steelblue")
    ax.plot(lim, lim, "r--", lw=1.5, label="y = x")
    ax.set_xlabel(f"Observé ({target_col}, MW)")
    ax.set_ylabel(f"Prédit (MW)")
    ax.set_title(f"Test — nuage de points ({tag})")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    p2 = plots_dir / f"eval_{tag}_scatter.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("Figure :", p2.resolve())

    # Résiduels
    resid = (y_hat_mw - y_true_mw).ravel()
    if resid.size > 50_000:
        rng = np.random.default_rng(43)
        resid = resid[rng.choice(resid.size, size=50_000, replace=False)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(resid, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="darkred", lw=1.2)
    ax.set_xlabel("Erreur (MW) = prédit − observé")
    ax.set_title(f"Test — histogramme des résidus ({tag})")
    fig.tight_layout()
    p3 = plots_dir / f"eval_{tag}_residuals.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print("Figure :", p3.resolve())


if __name__ == "__main__":
    main()
