"""
Entraînement : lit ``config.yaml`` (chemins relatifs au dossier ``energy_forecast/``),
charge ``train.pkl`` / ``val.pkl``, optimise MSE avec early stopping sur la val,
sauvegarde le meilleur ``state_dict`` dans ``results/checkpoints/``.

Usage (depuis la racine du dépôt Git) :
    python energy_forecast/src/train.py --config energy_forecast/config.yaml
    python energy_forecast/src/train.py --config energy_forecast/config.yaml --max-epochs 2 --subset 2048
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models import LSTMBahdanauForecaster, build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def forward_predictions(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, LSTMBahdanauForecaster):
        y_hat, _ = model(x)
        return y_hat
    return model(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pred = forward_predictions(model, xb)
            loss = criterion(pred, yb)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
            bs = xb.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement LSTM / Bi-LSTM / LSTM+attention (PJME).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Fichier YAML (défaut : energy_forecast/config.yaml à côté de src/)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Surcharge du nombre d'époques (ex. 2 pour un test rapide).",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="N premiers échantillons train/val (debug rapide ; défaut = tout le jeu).",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config is not None else default_config_path()
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config introuvable : {cfg_path}")

    root = cfg_path.parent
    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    device_str = str(cfg.get("device", "cpu")).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA demandé mais indisponible — utilisation du CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str if device_str in ("cuda", "cpu") else "cpu")

    set_seed(int(train_cfg["seed"]))

    train_pkl = root / paths["train_pkl"]
    val_pkl = root / paths["val_pkl"]
    ckpt_dir = root / paths["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(train_pkl, "rb") as f:
        train_blob = pickle.load(f)
    with open(val_pkl, "rb") as f:
        val_blob = pickle.load(f)

    x_tr = torch.from_numpy(np.asarray(train_blob["X"], dtype=np.float32))
    y_tr = torch.from_numpy(np.asarray(train_blob["y"], dtype=np.float32))
    x_va = torch.from_numpy(np.asarray(val_blob["X"], dtype=np.float32))
    y_va = torch.from_numpy(np.asarray(val_blob["y"], dtype=np.float32))

    if args.subset is not None:
        n = max(1, int(args.subset))
        x_tr, y_tr = x_tr[:n], y_tr[:n]
        n_va = min(n, x_va.shape[0])
        x_va, y_va = x_va[:n_va], y_va[:n_va]
        print(f"Mode subset : train={x_tr.shape[0]}, val={x_va.shape[0]}")

    f_dim = int(x_tr.shape[-1])
    cfg_dim = int(model_cfg["input_dim"])
    if f_dim != cfg_dim:
        print(f"Avertissement : input_dim config={cfg_dim} mais X a F={f_dim} — utilisation de F.")
    input_dim = f_dim

    g = torch.Generator()
    g.manual_seed(int(train_cfg["seed"]))
    train_ds = TensorDataset(x_tr, y_tr)
    val_ds = TensorDataset(x_va, y_va)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        name=str(model_cfg["name"]),
        input_dim=input_dim,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        horizon=int(model_cfg["horizon"]),
        dropout=float(model_cfg["dropout"]),
        attn_dim=int(model_cfg.get("attn_dim", 64)),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(train_cfg["max_epochs"])
    patience = int(train_cfg["patience"])
    best_val = float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, max_epochs + 1), desc="Epochs", unit="ep"):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss = run_epoch(model, val_loader, criterion, None, device, train=False)
        history.append({"epoch": float(epoch), "train_loss": tr_loss, "val_loss": va_loss})
        print(f"Epoch {epoch:03d}  train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

        if va_loss < best_val - 1e-9:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping (patience={patience}), meilleure val_mse={best_val:.6f}.")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val = va_loss

    name_safe = str(model_cfg["name"]).strip().lower().replace(" ", "_")
    ckpt_path = ckpt_dir / f"best_{name_safe}.pth"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": str(model_cfg["name"]),
            "input_dim": input_dim,
            "hidden_size": int(model_cfg["hidden_size"]),
            "num_layers": int(model_cfg["num_layers"]),
            "horizon": int(model_cfg["horizon"]),
            "dropout": float(model_cfg["dropout"]),
            "attn_dim": int(model_cfg.get("attn_dim", 64)),
            "best_val_mse": best_val,
            "config_path": str(cfg_path),
        },
        ckpt_path,
    )

    log_path = ckpt_dir / f"history_{name_safe}.json"
    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Checkpoint :", ckpt_path.resolve())
    print("Historique :", log_path.resolve())


if __name__ == "__main__":
    main()
