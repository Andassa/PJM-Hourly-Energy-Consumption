"""Utilitaires partagés pour evaluate, Diebold–Mariano et figures d'attention."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMBahdanauForecaster, build_model


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def inverse_target_mw(
    y_norm: np.ndarray,
    scaler,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    idx = feature_cols.index(target_col)
    dmin = float(scaler.data_min_[idx])
    dmax = float(scaler.data_max_[idx])
    return y_norm * (dmax - dmin) + dmin


def load_test_bundle(root: Path, paths: dict, subset: int | None = None) -> dict:
    blob = joblib.load(root / paths["sequence_scaler"])
    with open(root / paths["test_pkl"], "rb") as f:
        test_data = pickle.load(f)
    x = torch.from_numpy(np.asarray(test_data["X"], dtype=np.float32))
    y = np.asarray(test_data["y"], dtype=np.float32)
    if subset is not None:
        n = max(1, int(subset))
        x, y = x[:n], y[:n]
    return {
        "x": x,
        "y": y,
        "scaler": blob["scaler"],
        "feature_cols": list(blob["feature_cols"]),
        "target_col": str(blob["target_col"]),
    }


def load_model_from_checkpoint(
    ckpt_path: Path,
    model_cfg: dict,
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = str(ckpt.get("model_name", model_cfg["name"]))
    model = build_model(
        name=name,
        input_dim=int(ckpt.get("input_dim", input_dim)),
        hidden_size=int(ckpt.get("hidden_size", model_cfg["hidden_size"])),
        num_layers=int(ckpt.get("num_layers", model_cfg["num_layers"])),
        horizon=int(ckpt.get("horizon", model_cfg["horizon"])),
        dropout=float(ckpt.get("dropout", model_cfg["dropout"])),
        attn_dim=int(ckpt.get("attn_dim", model_cfg.get("attn_dim", 64))),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, name


@torch.no_grad()
def predict_mw(
    model: nn.Module,
    x: torch.Tensor,
    y_norm: np.ndarray,
    scaler,
    feature_cols: list[str],
    target_col: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        if isinstance(model, LSTMBahdanauForecaster):
            pred, _ = model(xb)
        else:
            pred = model(xb)
        outs.append(pred.cpu().numpy())
    y_hat = np.concatenate(outs, axis=0)
    return inverse_target_mw(y_hat, scaler, feature_cols, target_col)


@torch.no_grad()
def predict_attention_weights(
    model: LSTMBahdanauForecaster,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    alphas: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        _, alpha = model(xb)
        alphas.append(alpha.cpu().numpy())
    return np.concatenate(alphas, axis=0)
