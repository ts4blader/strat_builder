import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from exits_builder import build_be_exits
from sizer import sl_based_sizer

sys.path.insert(0, str(Path.cwd().parent))


class STRAT:
    def __init__(self, CASH, COMMISSION, RISK_PER_TRADE):

        self.NAME = "DEMA be-exit"
        self.CASH = CASH
        self.COMMISSION = COMMISSION
        self.RISK_PER_TRADE = RISK_PER_TRADE

        self.PARAMS = {
            "ema_period": {"type": "INTEGER", "data": range(5, 15), "default": 9},
            "smoothing_period": {
                "type": "INTEGER",
                "data": range(6, 16),
                "default": 12,
            },
            "use_vwap": {"type": "BOOL", "data": [True, False], "default": True},
            "atr_period": {"type": "INTEGER", "data": range(14, 22, 2), "default": 18},
            "sl_mult": {"type": "REAL", "data": range(1, 3), "default": 1},
            "rr_ratio": {
                "type": "REAL",
                "data": [0.25, 0.5, 1.0, 1.5, 2.0],
                "default": 1,
            },
        }

        self.PARAMS_DEFAULT = {
            k: v.get("default", v["data"][0]) for k, v in self.PARAMS.items()
        }

    def _make_signals(self, data, params):
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        ema = vbt.indicators.MA.run(close, window=params["ema_period"], ewm=True)
        smoothed = vbt.indicators.MA.run(
            ema.ma, window=params["smoothing_period"], ewm=True
        )
        atr = vbt.indicators.ATR.run(high, low, close, window=params["atr_period"])

        long_cross = ema.ma_crossed_above(smoothed)
        short_cross = ema.ma_crossed_below(smoothed)

        valid_atr = np.isfinite(atr.atr) & (atr.atr > 0)
        long_mask = long_cross & valid_atr
        short_mask = short_cross & valid_atr

        if params["use_vwap"]:
            long_mask = long_mask & (close > data["VWAP"])
            short_mask = short_mask & (close < data["VWAP"])
        long_signal = long_mask
        short_signal = short_mask

        signals = pd.DataFrame(
            {
                "ema": ema.ma,
                "smoothed": smoothed.ma,
                "long": long_signal,
                "short": short_signal,
                "atr": atr.atr,
            }
        )

        return signals

    def run_backtest(self, data, params):
        if not params:
            params = self.PARAMS_DEFAULT

        signals = self._make_signals(data, params)
        freq = data.index.to_series().diff().median()

        rr_ratio = params.get("rr_ratio")
        sl_mult = params.get("sl_mult")

        long_signal = signals["long"].to_numpy()
        short_signal = signals["short"].to_numpy()

        long_exits, short_exits, sl_dist = build_be_exits(
            data, signals, rr_ratio=rr_ratio, sl_mult=sl_mult
        )
        sizer = sl_based_sizer(risk_per_trade=self.RISK_PER_TRADE, sl_dists=sl_dist)

        portfolio = vbt.Portfolio.from_signals(
            close=self.data["Close"],
            entries=long_signal,
            exits=long_exits,
            short_entries=short_signal,
            short_exits=short_exits,
            init_cash=self.CASH,
            fees=self.COMMISSION,
            size=sizer,
            size_type="amount",
            freq=freq,
        )

        return portfolio
