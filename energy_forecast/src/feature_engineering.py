"""Features calendaires pour séries horaires (projet PJM / README)."""

from __future__ import annotations

import pandas as pd
import holidays


def add_calendar_features(
    df: pd.DataFrame,
    datetime_col: str = "Datetime",
    country: str = "US",
) -> pd.DataFrame:
    """
    Ajoute des colonnes dérivées du calendrier (sans fuite : uniquement
    l'information disponible à l'instant t).

    Colonnes ajoutées :
    - hour : 0–23
    - dow : jour de la semaine (0 = lundi … 6 = dimanche, convention pandas)
    - month : 1–12
    - is_weekend : 1 si samedi ou dimanche, sinon 0
    - is_holiday : 1 si jour férié (calendrier fédéral US par défaut), sinon 0
    """
    out = df.copy()
    if datetime_col not in out.columns:
        raise KeyError(f"Colonne manquante : {datetime_col!r}")

    ts = out[datetime_col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        out[datetime_col] = pd.to_datetime(ts)
        ts = out[datetime_col]

    out["hour"] = ts.dt.hour.astype("int16")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    if country.upper() == "US":
        cal = holidays.US()
    else:
        cal = holidays.country_holidays(country)

    out["is_holiday"] = ts.map(lambda t: int(t in cal)).astype("int8")
    return out
