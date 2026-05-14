"""Shared file loading and validation helpers for groundwater time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesData:
    dates: np.ndarray
    water_level: np.ndarray
    rainfall_mm: np.ndarray


def _read_table(file_path: str | Path) -> pd.DataFrame:
    """Try a small set of common delimiters used by the project datasets."""
    last_error: Optional[Exception] = None
    for sep, engine in ((r"\s+", "python"), ("\t", "c"), (",", "c")):
        try:
            df = pd.read_csv(file_path, header=None, sep=sep, engine=engine)
        except Exception as exc:  # pragma: no cover - pandas parser detail
            last_error = exc
            continue
        if df.shape[1] >= 3:
            return df

    if last_error is not None:
        raise ValueError(f"Failed to read input file: {last_error}") from last_error
    raise ValueError("Input file must contain at least 3 columns: date, water level, rainfall.")


def load_timeseries_file(
    file_path: str | Path,
    *,
    interpolate_water_level: bool,
    rainfall_unit: str = "mm",
    require_dates: bool = False,
) -> TimeSeriesData:
    """
    Load a 3-column time series file and perform lightweight validation.

    Expected columns are:
    1) date-like values
    2) water level in meters
    3) rainfall in mm/day by default
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Input file does not exist: {path}")
    if path.is_dir():
        raise ValueError(f"Input path must be a file, not a directory: {path}")

    df = _read_table(path)

    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    wl = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    rain = pd.to_numeric(df.iloc[:, 2], errors="coerce")

    if require_dates and dates.isna().any():
        bad_rows = list(np.where(dates.isna())[0][:5])
        raise ValueError(f"Date parsing failed for rows {bad_rows}.")

    if wl.notna().sum() < 3:
        raise ValueError("Water level column has fewer than 3 numeric values.")

    if rain.notna().sum() == 0:
        raise ValueError("Rainfall column has no numeric values.")

    if dates.notna().sum() >= 2:
        valid_dates = dates.dropna()
        if valid_dates.duplicated().any():
            raise ValueError("Date column contains duplicate timestamps.")
        if not valid_dates.is_monotonic_increasing:
            raise ValueError("Date column must be sorted in increasing order.")

    wl_series = wl.astype(float)
    if interpolate_water_level:
        wl_values = (
            wl_series.interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=float)
        )
    else:
        wl_values = wl_series.to_numpy(dtype=float)

    if not np.isfinite(wl_values).any():
        raise ValueError("Water level column became entirely invalid after preprocessing.")

    rain_values = np.nan_to_num(rain.to_numpy(dtype=float), nan=0.0)
    if np.any(rain_values < 0):
        raise ValueError("Rainfall values must be non-negative.")

    MM_PER_M = 1000.0
    if rainfall_unit == "m":
        # Input data is in mm; convert to metres for the core simulation which
        # operates in metres.  The field name "rainfall_mm" is a legacy misnomer —
        # when rainfall_unit="m", the stored values are actually in metres.
        rain_values = rain_values / MM_PER_M
    elif rainfall_unit != "mm":
        raise ValueError(f"Unsupported rainfall_unit: {rainfall_unit}")

    return TimeSeriesData(
        dates=dates.to_numpy(),
        water_level=wl_values,
        rainfall_mm=rain_values,
    )
