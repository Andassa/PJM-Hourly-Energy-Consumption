"""
Découpage chronologique 80/10/10, Min-Max ajusté sur le train seulement,
fenêtres entrée 168 h → cible 24 h (charge normalisée).

Lit le premier ``*_with_features.csv`` dans ``data/processed/`` et écrit :
  - ``train.pkl``, ``val.pkl``, ``test.pkl`` (dict avec X, y, méta)
  - ``sequence_scaler.joblib`` (MinMaxScaler + noms de colonnes)

Usage :
    python energy_forecast/src/build_splits_sequences.py
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_ROOT = Path(__file__).resolve().parent.parent
_PROCESSED = _ROOT / "data" / "processed"

INPUT_LEN = 168
HORIZON = 24
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1


def _find_with_features_csv(processed: Path) -> Path:
    candidates = sorted(processed.glob("*_with_features.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun *_with_features.csv dans {processed}. "
            "Lancer d'abord : python energy_forecast/src/preprocessing.py"
        )
    return candidates[0]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Coupe dans l'ordre du temps : train | val | test (sans mélange)."""
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def detect_target_column(df: pd.DataFrame) -> str:
    mw = [c for c in df.columns if c.endswith("_MW")]
    if mw:
        return mw[0]
    raise ValueError("Aucune colonne *_MW trouvée pour la cible.")


def feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Colonnes numériques du modèle (hors Datetime)."""
    exclude = {"Datetime"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if target_col not in cols:
        raise ValueError(f"Cible {target_col} absente ou non numérique.")
    # Ordre stable : cible d'abord puis le reste alphabétique
    others = sorted(c for c in cols if c != target_col)
    return [target_col] + others


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_len: int = INPUT_LEN,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X : (N, input_len, n_features)
    y : (N, horizon) — uniquement les 24 prochaines valeurs de la charge (déjà normalisées).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    data = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)
    t = len(df)
    n = t - input_len - horizon + 1
    if n <= 0:
        f = len(feature_cols)
        return (
            np.empty((0, input_len, f), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    # (t - input_len + 1, F, input_len) -> on garde n premières fenêtres, axes (n, input_len, F)
    sw_x = sliding_window_view(data, input_len, axis=0)[:n].transpose(0, 2, 1)
    sw_y = sliding_window_view(target, horizon, axis=0)
    y_out = sw_y[input_len : input_len + n].astype(np.float32)
    return sw_x.astype(np.float32), y_out


def run(
    input_csv: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Path]:
    processed = processed_dir or _PROCESSED
    in_path = input_csv or _find_with_features_csv(processed)
    df = pd.read_csv(in_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    target_col = detect_target_column(df)
    feat_cols = feature_columns(df, target_col)

    train_df, val_df, test_df = chronological_split(df)

    scaler = MinMaxScaler()
    scaler.fit(train_df[feat_cols].to_numpy())

    train_s = train_df.copy()
    val_s = val_df.copy()
    test_s = test_df.copy()
    train_s[feat_cols] = scaler.transform(train_df[feat_cols].to_numpy())
    val_s[feat_cols] = scaler.transform(val_df[feat_cols].to_numpy())
    test_s[feat_cols] = scaler.transform(test_df[feat_cols].to_numpy())

    X_train, y_train = make_windows(train_s, feat_cols, target_col)
    X_val, y_val = make_windows(val_s, feat_cols, target_col)
    X_test, y_test = make_windows(test_s, feat_cols, target_col)

    meta = {
        "input_len": INPUT_LEN,
        "horizon": HORIZON,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "target_col": target_col,
        "feature_cols": feat_cols,
        "source_csv": str(in_path.resolve()),
        "n_rows_total": len(df),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_test_rows": len(test_df),
        "shapes": {
            "train": {"X": list(X_train.shape), "y": list(y_train.shape)},
            "val": {"X": list(X_val.shape), "y": list(y_val.shape)},
            "test": {"X": list(X_test.shape), "y": list(y_test.shape)},
        },
        "train_datetime_range": [str(train_df["Datetime"].iloc[0]), str(train_df["Datetime"].iloc[-1])],
        "val_datetime_range": [str(val_df["Datetime"].iloc[0]), str(val_df["Datetime"].iloc[-1])],
        "test_datetime_range": [str(test_df["Datetime"].iloc[0]), str(test_df["Datetime"].iloc[-1])],
    }

    scaler_path = processed / "sequence_scaler.joblib"
    joblib.dump(
        {"scaler": scaler, "feature_cols": feat_cols, "target_col": target_col},
        scaler_path,
    )

    meta_path = processed / "sequence_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    paths = {}
    for name, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        p = processed / f"{name}.pkl"
        with open(p, "wb") as f:
            pickle.dump({"X": X, "y": y, "meta": meta}, f, protocol=4)
        paths[name] = p

    paths["scaler"] = scaler_path
    paths["meta_json"] = meta_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Split chronologique + MinMax train + fenêtres 168→24.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV avec features (défaut : premier *_with_features.csv)",
    )
    args = parser.parse_args()
    paths = run(input_csv=args.input)
    in_used = args.input if args.input is not None else _find_with_features_csv(_PROCESSED)
    print("Entrée features :", in_used.resolve())
    for k, p in paths.items():
        print(f"  {k}: {p.resolve()}")


if __name__ == "__main__":
    main()
