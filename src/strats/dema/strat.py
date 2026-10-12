import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, str(Path.cwd().parent))
from constants import CASH, COMMISSION


class DEMA_STRAT:
    def __init__(self, data, params):
        self.params = params or {
            "ema_period": 9,
            "smoothing_period": 12,
            "use_vwap": True,
            "atr_period": 18,
        }

        self.data = data
        self.freq = self.data.index.to_series().diff().median()
        self.signals = self._make_signals()

    def _make_signals(self):
        data = self.data
        params = self.params

        # keep float32 through indicator math; cast up only at the vectorbt boundary
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

    def _run_backtest(self, long_exits, short_exits, size=200, size_type="value"):
        if long_exits is None or short_exits is None:
            print("Please provide long exits and short exits")
            return

        signals = self.signals
        freq = self.data.index.to_series().diff().median()

        long_signal = signals["long"].to_numpy()
        short_signal = signals["short"].to_numpy()

        portfolio = vbt.Portfolio.from_signals(
            close=self.data["Close"],
            entries=long_signal,
            exits=long_exits,
            short_entries=short_signal,
            short_exits=short_exits,
            init_cash=CASH,
            fees=COMMISSION,
            size=size,
            size_type=size_type,
            freq=freq,
        )

        return portfolio
