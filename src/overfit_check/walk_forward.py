from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from dema_strat.utils import db_connect

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


def walk_forward_gen(
    data,
    run_backtest,
    param_grid: dict,
    table_name: str,
    force_write: bool = False,
    is_years: int = 2,
    oos_months: int = 6,
    objective: str = "profit_factor",
) -> list[dict]:

    # 1️⃣ Cache handling
    conn, cursor = db_connect()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    table_exists = cursor.fetchone() is not None
    if table_exists:
        if not force_write:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            conn.close()
            return [dict(zip(col_names, row)) for row in rows]
        else:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
    conn.close()

    # 2️⃣ Compute results (same as original implementation)
    combos = _grid(param_grid)
    windows = list(_windows(data, is_years, oos_months))
    results: list[dict] = []

    for is_start, is_end, oos_start, oos_end in tqdm(
        windows, desc="Walk-forward windows"
    ):
        is_data = data.loc[is_start:is_end]
        oos_data = data.loc[oos_start:oos_end]
        if is_data.empty or oos_data.empty:
            continue

        # Optimize on in‑sample data
        best_params, best_score = None, None
        for p in tqdm(combos, desc="  Optimizing", leave=False):
            pf, _ = run_backtest(is_data, p)
            score = _safe(METRICS[objective](pf))
            if score is not None and (best_score is None or score > best_score):
                best_params, best_score = p, score
        if best_params is None:
            continue

        # Re‑run with best parameters on IS and OOS
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

    # 3️⃣ Store results in DB
    if results:
        print(f"Store results in {table_name}")
        pf = pd.DataFrame(results)

        str_convert_cols = ["params"]
        for col in str_convert_cols:
            pf[col] = pf[col].astype(str)

        pf.to_sql(table_name, conn, if_exists="replace", index=False)
    return results


def calculate_wfe(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute WFE ratio for each metric
    for metric in METRICS:
        oos_col = f"oos_{metric}"
        is_col = f"is_{metric}"
        df[f"wfe_{metric}"] = df[oos_col] / df[is_col]

    return df


def walk_forward_test(
    data,
    run_backtest,
    param_grid,
    table_name,
    force_write: bool = False,
    is_years=2,
    oos_months=6,
    objective="sharpe_ratio",
):
    results = walk_forward_gen(
        data,
        run_backtest,
        param_grid,
        table_name,
        force_write,
        is_years,
        oos_months,
        objective,
    )

    wfe = calculate_wfe(results)

    # Plot WFE of each metric on the same Plotly chart
    # If wfe is empty, we skip plotting.
    if not wfe.empty:
        import plotly.graph_objects as go

        fig = go.Figure()
        # Use the index (window number) as x-axis; alternatively use is_end dates.
        x_vals = wfe.index
        for metric in METRICS:
            col = f"wfe_{metric}"
            if col in wfe.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=wfe[col],
                        mode="lines+markers",
                        name=metric,
                    )
                )
        # Add horizontal red line at y=0.5
        fig.add_shape(
            type="line",
            x0=min(x_vals),
            x1=max(x_vals),
            y0=0.5,
            y1=0.5,
            line={"color": "red", "dash": "dash"},
        )
        fig.update_layout(
            title=f"Walk‑Forward Evaluation (WFE) Ratios: {is_years}YRS window, {oos_months}MTHS step",
            xaxis_title="Window",
            yaxis_title="WFE Ratio",
            template="plotly_dark",
            font_family="JetBrainsMono Nerd Font",
        )
        # In a Jupyter environment, displaying the figure will render it.
        fig.show()

    return results, wfe
