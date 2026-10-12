import math
import sys
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from .loader import load_ohlcv

sys.path.insert(0, str(Path.cwd().parent))
import sqlite3
from pathlib import Path

from IPython.display import display

from constants import TABLE_STYLES


def get_project_root() -> Path:
    """Finds the root directory by looking for a marker file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent  # Fallback


# 1. Define your paths globally
ROOT_DIR = get_project_root()
DB_PATH = ROOT_DIR / "src" / "backtest_results.db"


def load_data(path):

    def load_file(file_path):
        return load_ohlcv(file_path, use_cache=True, add_vwap=True)

    file_path = Path(path).expanduser()
    df = load_file(file_path)

    return df


def load_data_and_split(path, split_ratio=0.8):

    def load_data(file_path):
        return load_ohlcv(file_path, use_cache=True, add_vwap=True)

    file_path = Path(path).expanduser()
    df = load_data(file_path)

    # Split 80/20 into in‑sample (IOS) and out‑of‑sample (OOS)
    split_idx = int(len(df) * split_ratio)
    df_ios = df.iloc[:split_idx]
    df_oos = df.iloc[split_idx:]

    return df_ios, df_oos


def stats_pretiier(stats):
    """Extract key performance metrics from a vectorbt Portfolio.stats() Series.

    Returns a dict with:
        sharp_ratio, sortino_ratio, profit_factor, calmar_ratio, ulcer_index,
        winrate, return_pct, num_trades, max_dd
    """
    mapping = {
        "sharp_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "profit_factor": "Profit Factor",
        "calmar_ratio": "Calmar Ratio",
        "winrate": "Win Rate [%]",
        "return_pct": "Total Return [%]",
        "num_trades": "Total Trades",
        "max_dd": "Max Drawdown [%]",
        "fee": "Total Fees Paid",
        "expectancy": "Expectancy",
    }
    return {k: stats.get(v) for k, v in mapping.items()}


SELECTED_METRICS = [
    "Total Return [%]",
    "Max Drawdown [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Profit Factor",
    "Total Trades",
    "Win Rate [%]",
    "Total Fees Paid",
]


def format_smart(x):
    if pd.isna(x) or x is None or math.isinf(x):
        return "N/A"
    if isinstance(x, (int, float)):
        # If the number has no decimal part, format as an integer
        if x == int(x):
            return f"{int(x):,}"
        # Otherwise, format with 3 decimal places
        return f"{x:,.3f}"
    return str(x)


def benchmark_porfolio(data):
    buy_hold_entries = pd.Series(False, index=data.index)
    buy_hold_entries.iloc[0] = True
    freq = data.index.to_series().diff().median()

    return vbt.Portfolio.from_signals(
        close=data["Close"], entries=buy_hold_entries, exits=None, freq=freq
    )


def compare_benchmark_summary(portfolio_ios, portfolio_oos, data_ios, data_oos):
    benchmark_ios = benchmark_porfolio(data_ios)
    benchmark_oos = benchmark_porfolio(data_oos)

    pf_all = pd.DataFrame(
        {
            "IOS": portfolio_ios.stats(),
            "OOS": portfolio_oos.stats(),
            "BENCHMARK IOS": benchmark_ios.stats(),
            "BENCHMARK OOS": benchmark_oos.stats(),
        }
    )

    pf_custom = pf_all.loc[SELECTED_METRICS]
    display(global_style(pf_custom))


def compare_benchmark(portfolio, data):
    # benchmark
    pf_benchmark = benchmark_porfolio(data)

    # Combine stats side-by-side
    pf_all = pd.DataFrame(
        {"STRATEGY": portfolio.stats(), "BENCHMARK": pf_benchmark.stats()}
    )

    pf_custom = pf_all.loc[SELECTED_METRICS]
    display(
        pf_custom.style.format(
            {"STRATEGY": format_smart, "BENCHMARK": format_smart},
            na_rep="N/A",  # Cleanly replaces NaN values
        ).set_table_styles(TABLE_STYLES)
    )


def db_connect():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    return conn, cursor


def global_style(df):
    """Applies a unified dark-header corporate theme to any DataFrame."""
    return df.style.set_table_styles(TABLE_STYLES).format(
        precision=2,
        thousands=",",
        na_rep="N/A",
    )  # Set standard number formatting


def global_style_loose(df):
    """Applies a unified dark-header corporate theme to any DataFrame."""
    return df.style.set_table_styles(TABLE_STYLES).format(
        precision=4,
        thousands=",",
        na_rep="N/A",
    )  # Set standard number formatting


def compare_table(dictionary):
    pf = pd.DataFrame(dictionary)
    display(global_style(pf))
