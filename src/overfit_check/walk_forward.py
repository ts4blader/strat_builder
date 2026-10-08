from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

METRICS = {
    "sharpe_ratio": lambda pf: pf.sharpe_ratio(),
    "sortino_ratio": lambda pf: pf.sortino_ratio(),
    "calmar_ratio": lambda pf: pf.calmar_ratio(),
    "profit_factor": lambda pf: pf.trades.profit_factor(),
    "annual_return": lambda pf: pf.annualized_return(),
}


def _safe(x):
    """Convert to float, returning None for NaN/inf/non-numeric."""
    try:
        x = float(x)
    except TypeError, ValueError:
        return None
    return x if np.isfinite(x) else None


def _grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]


def _windows(data, is_years: int, oos_months: int):
    """Yield (is_start, is_end, oos_start, oos_end); step = OOS length."""
    start, last = data.index.min(), data.index.max()
    while True:
        is_end = start + pd.DateOffset(years=is_years) - pd.Timedelta(days=1)
        oos_start = is_end + pd.Timedelta(days=1)
        oos_end = oos_start + pd.DateOffset(months=oos_months) - pd.Timedelta(days=1)
        if oos_end > last:
            return
        yield start, is_end, oos_start, oos_end
        start = start + pd.DateOffset(months=oos_months)


def _evaluate(pf) -> dict:
    return {name: _safe(fn(pf)) for name, fn in METRICS.items()}


def walk_forward(
    data,
    run_backtest,
    param_grid: dict,
    is_years: int = 2,
    oos_months: int = 6,
    objective: str = "sharpe_ratio",
) -> list[dict]:
    if data.empty:
        raise ValueError("Loaded data is empty – cannot perform walk-forward test.")

    combos = _grid(param_grid)
    windows = list(
        _windows(data, is_years, oos_months)
    )  # materialize so tqdm has a total
    results = []

    for is_start, is_end, oos_start, oos_end in tqdm(
        windows, desc="Walk-forward windows"
    ):
        is_data = data.loc[is_start:is_end]
        oos_data = data.loc[oos_start:oos_end]
        if is_data.empty or oos_data.empty:
            continue

        # 1) Optimize on in-sample data.
        best_params, best_score = None, None
        for p in tqdm(combos, desc="  Optimizing", leave=False):
            pf, _ = run_backtest(is_data, p)
            score = _safe(METRICS[objective](pf))
            if score is not None and (best_score is None or score > best_score):
                best_params, best_score = p, score
        if best_params is None:
            continue

        # 2) Re-run best params on IS (for the metrics) and on unseen OOS data.
        is_pf, _ = run_backtest(is_data, best_params)
        oos_pf, _ = run_backtest(oos_data, best_params)

        is_m, oos_m = _evaluate(is_pf), _evaluate(oos_pf)
        results.append(
            {
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "params": best_params,
                **{f"is_{k}": v for k, v in is_m.items()},
                **{f"oos_{k}": v for k, v in oos_m.items()},
            }
        )
    return results


def _ratio(oos, is_):
    # A ratio is meaningless when the IS result is zero/negative.
    if oos is None or is_ is None or is_ <= 0:
        return None
    return oos / is_


def calculate_wfe(results: list[dict]) -> dict:
    if not results:
        return {}

    df = pd.DataFrame(results)

    # Per-window efficiency (annualized return, OOS / IS).
    df["wfe"] = [
        _ratio(o, i) for o, i in zip(df["oos_annual_return"], df["is_annual_return"])
    ]

    is_ret = df["is_annual_return"].dropna()
    oos_ret = df["oos_annual_return"].dropna()

    return {
        "n_windows": len(df),
        # Classic WFE: average OOS annualized return / average IS annualized return
        "wfe_aggregate": _ratio(oos_ret.mean(), is_ret.mean())
        if len(is_ret) and len(oos_ret)
        else None,
        "wfe_median_per_window": df["wfe"].median(),
        "pct_oos_profitable": (df["oos_annual_return"] > 0).mean(),
        "mean_is_sharpe": df["is_sharpe_ratio"].mean(),
        "mean_oos_sharpe": df["oos_sharpe_ratio"].mean(),
        "windows": df,
    }


def walk_forward_test(
    data,
    run_backtest,
    param_grid,
    is_years=2,
    oos_months=6,
    objective="sharpe_ratio",
):
    results = walk_forward(
        data, run_backtest, param_grid, is_years, oos_months, objective
    )
    wfe = calculate_wfe(results)
    windows = wfe.pop("windows", None)
    display(wfe)

    if windows is not None:
        display(windows)

    return wfe, windows
