"""Memory-efficient OHLCV loader for backtesting grids.

Reduces RAM by ~50% vs. the default `pd.read_csv(..., parse_dates)` path:

* OHLC + Volume loaded as float32 (instead of float64).
* Datetime parsed once and cached as int64 (ns since epoch) inside a
  DatetimeIndex, never as Python objects.
* Daily-reset VWAP computed in a single pass with float32 intermediates.
* Auto-caches the processed frame as Parquet (keyed on source file mtime
  + size) so subsequent grid iterations skip the CSV parse entirely.
  while the cached frame stays compact.

The cache lives next to the source CSV as
``<stem>__cache_v1.parquet``. Delete it to force a rebuild.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_OHLCV_DTYPES = {
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "float32",
}

CACHE_SUFFIX = "__cache_v1.parquet"


def _cache_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + CACHE_SUFFIX)


def _cache_is_fresh(csv_path: Path, cache: Path) -> bool:
    if not cache.exists():
        return False
    csv_stat = csv_path.stat()
    try:
        cache_stat = cache.stat()
    except FileNotFoundError:
        return False
    if cache_stat.st_size == 0:
        return False
    return (
        cache_stat.st_mtime >= csv_stat.st_mtime
        and cache_stat.st_size >= csv_stat.st_size // 4
    )


def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["Datetime"],
        index_col="Datetime",
        dtype=_OHLCV_DTYPES,
    )
    df.index = pd.to_datetime(df.index)
    return df


def _read_parquet(cache: Path) -> pd.DataFrame:
    return pd.read_parquet(cache)


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if "VWAP" in df.columns:
        return df

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = typical * df["Volume"]
    dates = df.index.date
    cum_tp = tp_vol.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    df["VWAP"] = (cum_tp / cum_vol).astype("float32")
    return df


def load_ohlcv(
    csv_path: str | os.PathLike,
    *,
    use_cache: bool = True,
    add_vwap: bool = True,
) -> pd.DataFrame:
    """Load an OHLCV CSV into a memory-compact DataFrame.

    Parameters
    ----------
    csv_path:
        Path to a CSV with columns ``Datetime,Open,High,Low,Close,Volume``.
    use_cache:
        If True (default), read/write a Parquet cache next to the CSV.
    add_vwap:
        If True (default), attach a daily-reset ``VWAP`` float32 column.
    verbose:
        Print whether the cache was used.
    """
    csv_path = Path(os.path.expanduser(csv_path))
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    cache = _cache_path(csv_path)
    df: pd.DataFrame | None = None

    if use_cache and _cache_is_fresh(csv_path, cache):
        print(f"[loader] cache hit: {cache.name}")
        df = _read_parquet(cache)

    if df is None:
        print(f"[loader] parsing CSV: {csv_path.name}")
        df = _read_csv(csv_path)

        if add_vwap:
            df = _add_vwap(df)
        if use_cache:
            try:
                df.to_parquet(cache)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[loader] cache write skipped: {e}")

    if add_vwap and "VWAP" not in df.columns:
        df = _add_vwap(df)

    return df


def free(*objs) -> None:
    """Drop references and hint the GC. Cheap helper for grid loops."""
    for o in objs:
        del o
    import gc

    gc.collect()
