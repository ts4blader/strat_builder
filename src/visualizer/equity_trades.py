import plotly.graph_objects as go
from plotly.subplots import make_subplots


def trades_visualizer(data, signals, portfolio, N=500):
    slc = data.iloc[:N]
    sig = signals.iloc[:N]
    trades = portfolio.trades.records_readable

    trades_cap = trades[trades["Entry Timestamp"] < slc.index[-1]]

    fig = make_subplots(
        rows=1,
        cols=1,
        # vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=slc.index,
            open=slc["Open"],
            high=slc["High"],
            low=slc["Low"],
            close=slc["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sig.index,
            y=sig["ema"],
            mode="lines",
            name="EMA",
            line={"width": 1, "color": "dodgerblue"},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sig.index,
            y=sig["smoothed"],
            mode="lines",
            name="Smoothed",
            line={"width": 1, "color": "orange"},
        ),
        row=1,
        col=1,
    )

    if "VWAP" in slc.columns:
        fig.add_trace(
            go.Scatter(
                x=slc.index,
                y=slc["VWAP"],
                mode="lines",
                name="VWAP",
                line={"width": 1, "dash": "dot", "color": "gray"},
            ),
            row=1,
            col=1,
        )

    # combine entry and exit timestamps per direction
    entry_ts = trades_cap.groupby("Direction")["Entry Timestamp"].apply(list).to_dict()
    exit_ts = trades_cap.groupby("Direction")["Exit Timestamp"].apply(list).to_dict()

    # helper to add marker traces with common args
    def _add_marker(x, y, name, symbol, color):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker={"symbol": symbol, "size": 9, "color": color},
            ),
            row=1,
            col=1,
        )

    # configuration for markers
    markers = [
        {
            "direction": "Long",
            "type": "entry",
            "ycol": "Low",
            "mult": 0.999,
            "symbol": "triangle-up",
            "color": "lime",
            "name": "Long Entry",
        },
        {
            "direction": "Long",
            "type": "exit",
            "ycol": "High",
            "mult": 1.001,
            "symbol": "x",
            "color": "lime",
            "name": "Long Exit",
        },
        {
            "direction": "Short",
            "type": "entry",
            "ycol": "High",
            "mult": 1.001,
            "symbol": "triangle-down",
            "color": "red",
            "name": "Short Entry",
        },
        {
            "direction": "Short",
            "type": "exit",
            "ycol": "Low",
            "mult": 0.999,
            "symbol": "x",
            "color": "red",
            "name": "Short Exit",
        },
    ]

    for cfg in markers:
        ts_list = (
            entry_ts[cfg["direction"]]
            if cfg["type"] == "entry"
            else exit_ts[cfg["direction"]]
        )
        mask = slc.index.isin(ts_list)
        if mask.any():
            _add_marker(
                x=slc.index[mask],
                y=slc[cfg["ycol"]].loc[mask].values * cfg["mult"],
                name=cfg["name"],
                symbol=cfg["symbol"],
                color=cfg["color"],
            )

    fig.update_layout(
        title=f"DEMA Strategy – First {N} Candles",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark",
        legend={
            "orientation": "v",
            "y": 0.5,
            "x": 1.02,
            "xanchor": "left",
            "yanchor": "middle",
        },
    )
    fig.update_yaxes(title_text="Price")
    fig.show()


def trades_visualizer_test(data, portfolio, N=500):
    slc = data.iloc[:N]
    trades = portfolio.trades.records_readable

    trades_cap = trades[trades["Entry Timestamp"] < slc.index[-1]]

    fig = make_subplots(
        rows=1,
        cols=1,
        # vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=slc.index,
            open=slc["Open"],
            high=slc["High"],
            low=slc["Low"],
            close=slc["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # combine entry and exit timestamps per direction
    entry_ts = trades_cap.groupby("Direction")["Entry Timestamp"].apply(list).to_dict()
    exit_ts = trades_cap.groupby("Direction")["Exit Timestamp"].apply(list).to_dict()

    # helper to add marker traces with common args
    def _add_marker(x, y, name, symbol, color):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker={"symbol": symbol, "size": 9, "color": color},
            ),
            row=1,
            col=1,
        )

    # configuration for markers
    markers = [
        {
            "direction": "Long",
            "type": "entry",
            "ycol": "Low",
            "mult": 0.999,
            "symbol": "triangle-up",
            "color": "lime",
            "name": "Long Entry",
        },
        {
            "direction": "Long",
            "type": "exit",
            "ycol": "High",
            "mult": 1.001,
            "symbol": "x",
            "color": "lime",
            "name": "Long Exit",
        },
        {
            "direction": "Short",
            "type": "entry",
            "ycol": "High",
            "mult": 1.001,
            "symbol": "triangle-down",
            "color": "red",
            "name": "Short Entry",
        },
        {
            "direction": "Short",
            "type": "exit",
            "ycol": "Low",
            "mult": 0.999,
            "symbol": "x",
            "color": "red",
            "name": "Short Exit",
        },
    ]

    for cfg in markers:
        ts_list = (
            entry_ts[cfg["direction"]]
            if cfg["type"] == "entry"
            else exit_ts[cfg["direction"]]
        )
        mask = slc.index.isin(ts_list)
        if mask.any():
            _add_marker(
                x=slc.index[mask],
                y=slc[cfg["ycol"]].loc[mask].values * cfg["mult"],
                name=cfg["name"],
                symbol=cfg["symbol"],
                color=cfg["color"],
            )

    fig.update_layout(
        title=f"DEMA Strategy – First {N} Candles",
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark",
        legend={
            "orientation": "v",
            "y": 0.5,
            "x": 1.02,
            "xanchor": "left",
            "yanchor": "middle",
        },
    )
    fig.update_yaxes(title_text="Price")
    fig.show()
