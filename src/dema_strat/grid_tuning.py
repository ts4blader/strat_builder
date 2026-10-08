import sqlite3
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def grid_tuning(params, table_name, run_backtest, data, replace_table=False):

    TABLE_NAME = table_name

    # maps vectorbt stat name -> db column name
    result_mapping = {
        "Profit Factor": "profit_factor",
        "Win Rate [%]": "winrate",
        "Expectancy": "expectancy",
        "Max Drawdown [%]": "max_dd",
        "Sharpe Ratio": "sharp_ratio",
        "Sortino Ratio": "sortino_ratio",
        "Calmar Ratio": "calmar_ratio",
        "Total Return [%]": "return_pct",
        "Total Trades": "n_trades",
        "Total Fees Paid": "fee",
        "Avg Winning Trade [%]": "avg_win_trade",
        "Avg Losing Trade [%]": "avg_lose_trade",
    }

    # build CREATE TABLE dynamically from params + result_mapping
    param_cols = ",\n    ".join(f"{k} {v['type']} NOT NULL" for k, v in params.items())
    result_cols = ",\n    ".join(f"{col} REAL" for col in result_mapping.values())

    create_sql = f"""
  CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE,
      {param_cols},
      {result_cols}
  )
  """

    db_path = Path("../backtest_results.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if replace_table:
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

    cursor.execute(create_sql)
    conn.commit()

    # build grid from params data
    grid_combinations = list(product(*(v["data"] for v in params.values())))

    param_names = list(params.keys())
    insert_cols = param_names + ["name"] + list(result_mapping.values())
    placeholders = ", ".join("?" for _ in insert_cols)
    insert_sql = f"INSERT OR IGNORE INTO {TABLE_NAME} ({', '.join(insert_cols)}) VALUES ({placeholders})"

    print(f"🔢 Total combinations: {len(grid_combinations):,}")
    print(f"💾 Storing results in table: '{TABLE_NAME}' @ {db_path.resolve()}")

    for combo in tqdm(grid_combinations, desc="Backtesting"):
        combo_params = dict(zip(param_names, combo))
        name = "_".join(str(v) for v in combo)

        try:
            portfolio, strat = run_backtest(data, params=combo_params)
            stats = portfolio.stats()

            result_values = []
            for vbt_key in result_mapping:
                val = stats.get(vbt_key, None)
                result_values.append(
                    float(val) if val is not None and pd.notna(val) else None
                )

            row = list(combo) + [name] + result_values
            cursor.execute(insert_sql, row)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Failed for {name}: {e}")

    print(f"✅ Done — {len(grid_combinations):,} combinations processed.")
    return len(grid_combinations)
