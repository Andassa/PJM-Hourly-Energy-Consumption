"""
Charge le CSV nettoyé (grille 1 h), applique le feature engineering calendaire,
écrit le tableau enrichi dans data/processed/.

Usage (depuis n'importe quel répertoire) :
    python path/to/energy_forecast/src/preprocessing.py
    python path/to/energy_forecast/src/preprocessing.py --all
Ou depuis energy_forecast :
    python -m src.preprocessing
    python -m src.preprocessing --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Imports locaux : exécution en script
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from feature_engineering import add_calendar_features  # noqa: E402

import build_splits_sequences  # noqa: E402


def default_paths(root: Path) -> tuple[Path, Path]:
    processed = root / "data" / "processed"
    candidates = sorted(processed.glob("*_clean.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier *_clean.csv dans {processed}. "
            "Exécuter d'abord le notebook EDA (nettoyage DST)."
        )
    in_path = candidates[0]
    return in_path, output_path_for(processed, in_path)


def output_path_for(processed: Path, in_path: Path) -> Path:
    stem = in_path.stem
    if stem.endswith("_clean"):
        stem = stem[: -len("_clean")]
    return processed / f"{stem}_with_features.csv"


def run(in_path: Path | None = None, out_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent.parent
    processed = root / "data" / "processed"
    if in_path is None:
        in_path, out_path = default_paths(root)
    elif out_path is None:
        out_path = output_path_for(processed, in_path)

    df = pd.read_csv(in_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df_enriched = add_calendar_features(df, datetime_col="Datetime", country="US")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Features calendaires sur série horaire nettoyée.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV nettoyé (défaut : premier *_clean.csv dans data/processed/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV de sortie (défaut : <stem>_with_features.csv)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Après le CSV avec features, lancer split + MinMax + fenêtres 168→24 (train/val/test.pkl).",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.input is None:
        in_path, _ = default_paths(root)
    else:
        in_path = args.input
    out = run(in_path=args.input, out_path=args.output)
    print("Entrée :", in_path.resolve())
    print("Sortie :", out.resolve())
    df = pd.read_csv(out, nrows=3)
    print("Aperçu colonnes :", list(df.columns))
    if args.all:
        seq_paths = build_splits_sequences.run(input_csv=out)
        print("Séquences :")
        for k, p in seq_paths.items():
            print(f"  {k}: {p.resolve()}")


if __name__ == "__main__":
    main()
