"""
Test de Diebold–Mariano entre paires de modèles (erreurs sur le jeu de test, MW).

Usage :
    python energy_forecast/src/diebold_mariano.py --config energy_forecast/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval_utils import (  # noqa: E402
    default_config_path,
    inverse_target_mw,
    load_model_from_checkpoint,
    load_test_bundle,
    load_yaml,
    predict_mw,
)


def loss_series(y_true: np.ndarray, y_pred: np.ndarray, loss: str) -> np.ndarray:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    if loss == "mse":
        return (y_true - y_pred) ** 2
    if loss == "mae":
        return np.abs(y_true - y_pred)
    raise ValueError(f"Loss inconnue : {loss!r} (mse | mae)")


def diebold_mariano(d: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Statistique DM et p-valeur bilatérale (approx. normale).
    d_t = L(model_a) - L(model_b) ; d_bar < 0 suggère que model_a est meilleur.
    """
    d = np.asarray(d, dtype=np.float64)
    t = len(d)
    if t < 10:
        return float("nan"), float("nan")
    d_bar = float(d.mean())
    gamma = np.zeros(h)
    gamma[0] = np.var(d, ddof=1)
    for k in range(1, h):
        gamma[k] = float(np.cov(d[k:], d[:-k], ddof=1)[0, 1])
    var_dbar = (gamma[0] + 2.0 * sum((1.0 - k / t) * gamma[k] for k in range(1, h))) / t
    if var_dbar <= 0:
        return float("nan"), float("nan")
    stat = d_bar / np.sqrt(var_dbar)
    p = float(2.0 * (1.0 - norm.cdf(abs(stat))))
    return stat, p


def interpret_dm(model_a: str, model_b: str, mean_d: float, p: float, alpha: float = 0.05) -> str:
    if np.isnan(p):
        return "Échantillon trop petit ou variance nulle."
    if p >= alpha:
        return f"Pas de difference significative (alpha={alpha}) entre {model_a} et {model_b}."
    if mean_d < 0:
        return f"{model_a} est significativement meilleur que {model_b} (alpha={alpha})."
    return f"{model_b} est significativement meilleur que {model_a} (alpha={alpha})."


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de Diebold–Mariano (paires de modèles).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--loss", choices=("mse", "mae"), default="mse")
    parser.add_argument("--h", type=int, default=24, help="Retard HAC (horizon=24 par défaut).")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=["lstm", "bilstm", "lstm_attention"],
        help="Noms de modèles (checkpoints best_<name>.pth).",
    )
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
    y_true_mw = inverse_target_mw(
        bundle["y"], bundle["scaler"], bundle["feature_cols"], bundle["target_col"]
    )

    ckpt_dir = root / paths["checkpoints_dir"]
    preds: dict[str, np.ndarray] = {}
    for name in args.models:
        slug = name.strip().lower().replace(" ", "_")
        ckpt = ckpt_dir / f"best_{slug}.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint manquant : {ckpt}")
        model, _ = load_model_from_checkpoint(ckpt, model_cfg, int(bundle["x"].shape[-1]), device)
        preds[name] = predict_mw(
            model,
            bundle["x"],
            bundle["y"],
            bundle["scaler"],
            bundle["feature_cols"],
            bundle["target_col"],
            device,
            int(train_cfg["batch_size"]),
        )
        print(f"Prédictions chargées : {name} ({ckpt.name})")

    rows = []
    for a, b in combinations(preds.keys(), 2):
        la = loss_series(y_true_mw, preds[a], args.loss)
        lb = loss_series(y_true_mw, preds[b], args.loss)
        d = la - lb
        stat, p = diebold_mariano(d, h=max(1, args.h))
        mean_d = float(np.mean(d))
        row = {
            "model_a": a,
            "model_b": b,
            "loss": args.loss,
            "dm_stat": stat,
            "p_value": p,
            "significant_5pct": bool(p < 0.05) if not np.isnan(p) else False,
            "mean_loss_diff": mean_d,
            "interpretation": interpret_dm(a, b, mean_d, p),
        }
        rows.append(row)
        print(f"\n{a} vs {b}: DM={stat:.4f}, p={p:.4f}")
        print(row["interpretation"])

    out_csv = root / "results" / "dm_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nRésultats :", out_csv.resolve())


if __name__ == "__main__":
    main()
