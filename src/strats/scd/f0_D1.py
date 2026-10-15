import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, str(Path.cwd().parent))


class STRAT:
    def __init__(self, CASH, COMMISSION, RISK_PER_TRADE):

        self.NAME = "SCD Random"
        self.CODE = "scd_rand"
        self.CASH = CASH
        self.COMMISSION = COMMISSION
        self.RISK_PER_TRADE = RISK_PER_TRADE

        self.PARAMS = {
            "std_step": {"type": "REAL", "data": [0.25, 0.5, 1], "default": 1},
            "rr_ratio": {
                "type": "REAL",
                "data": [0.5, 1.0, 1.5, 2.0],
                "default": 1,
            },
        }

        self.PARAMS_DEFAULT = {
            k: v.get("default", v["data"][0]) for k, v in self.PARAMS.items()
        }

    def _make_signals(self, data, params):
        rng = np.random.default_rng(42)

        day = pd.Series(data.index.normalize(), index=data.index)
        is_first = ~day.duplicated().to_numpy()  # first 4h bar of the day
        is_last = ~day.duplicated(keep="last").to_numpy()  # last 4h bar of the day

        # draw one direction per day, then apply it only on that day's first bar
        n_days = is_first.sum()
        day_long = rng.random(n_days) < 0.5
        go_long = np.zeros(len(data), dtype=bool)
        go_long[is_first] = day_long

        return pd.DataFrame(
            {
                "long": is_first & go_long,
                "short": is_first & ~go_long,
                "exit": is_last,
                "first": is_first,
            },
            index=data.index,
        )

    def run_backtest(self, data, params=None):
        if not params:
            params = self.PARAMS_DEFAULT

        signals = self._make_signals(data, params)
        freq = data.index.to_series().diff().median()

        # std_step is in percent: 1 -> 1% of entry price
        sl_pct = params["std_step"] / 100.0
        tp_pct = sl_pct * params["rr_ratio"]  # rr = take profit / stop loss

        # --- size from stop distance ---
        entry_price = data["Open"]  # entries fill at the day's first open
        sl_dist = entry_price * sl_pct  # stop distance in price terms
        size = self.RISK_PER_TRADE / sl_dist

        # no-leverage cap: notional can't exceed available cash
        max_size = (self.CASH * 0.99) / entry_price  # 0.99 leaves room for fees
        size = np.minimum(size, max_size)

        # fill price: Open on the entry bar, Close on the exit bar
        fill_price = np.where(signals["first"], data["Open"], data["Close"])

        portfolio = vbt.Portfolio.from_signals(
            close=data["Close"],
            open=data["Open"],
            high=data["High"],  # needed so stops trigger on intraday highs/lows
            low=data["Low"],
            price=fill_price,
            entries=signals["long"],
            exits=signals["exit"],  # end-of-day time exit
            short_entries=signals["short"],
            short_exits=signals["exit"],
            sl_stop=sl_pct,  # fraction of entry price
            tp_stop=tp_pct,
            stop_entry_price="fillprice",  # measure stops from our entry fill, not bar close
            init_cash=self.CASH,
            fees=self.COMMISSION,
            size=size,
            size_type="amount",
            freq=freq,
        )
        return portfolio, signals
