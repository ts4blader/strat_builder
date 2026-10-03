import numpy as np
import pandas as pd
from arch.bootstrap import OptimalBlockLength


def bootstrap_simulation(
    trades,
    risk_per_trade,
    # ev config
    profit_target,
    payout,
    cost,
    # strategy config
    risk_pct=0.01,
    start_equity=10_000,
    # simulation config
    n_sims=10_000,
    n_trades=1_000,
    drawdown_level=1,  # fraction of start_equity that may be lost (1 = 100%, 0.10 = 10%)
    seed: int | None = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed if seed is not None else 42)

    # ---- simulate ----
    trades_R = trades["PnL"] / risk_per_trade
    trades_R = trades_R.values

    idx = rng.integers(0, len(trades_R), size=(n_sims, n_trades))
    R = trades_R[idx]

    equity = start_equity * np.cumprod(1 + risk_pct * R, axis=1)
    equity = np.hstack([np.full((n_sims, 1), start_equity), equity])

    # ---- barriers ----
    target_equity = (
        profit_target if profit_target >= start_equity else start_equity + profit_target
    )
    ruin_equity = start_equity * (1 - drawdown_level)  # 0 when drawdown_level == 1

    hit_target = equity >= target_equity
    hit_ruin = equity <= ruin_equity

    n_steps = equity.shape[1]
    # first index at which each barrier is hit (n_steps = never hit)
    t_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), n_steps)
    t_ruin = np.where(hit_ruin.any(axis=1), hit_ruin.argmax(axis=1), n_steps)

    # 1) ruin is checked first: target only counts if reached strictly BEFORE ruin
    reached_drawdown = t_ruin < n_steps
    reached_target = (t_target < n_steps) & (t_target < t_ruin)
    ruined_before_target = reached_drawdown & ~reached_target

    # ---- metrics ----
    final = equity[:, -1]
    peak = np.maximum.accumulate(equity, axis=1)
    max_dd = ((peak - equity) / peak).max(axis=1)

    # ---- EV ----
    ev_per_sim = np.where(reached_target, payout, -cost)
    ev = ev_per_sim.mean()

    if verbose:
        print(f"Median final equity : {np.median(final):,.0f}")
        print(
            f"5th / 95th pct      : {np.percentile(final, 5):,.0f} / {np.percentile(final, 95):,.0f}"
        )
        print(f"Median max DD       : {np.median(max_dd):.1%}")
        print(f"95th pct max DD     : {np.percentile(max_dd, 95):.1%}")
        print(f"EV per simulation   : {ev:,.2f}")
        print(f"P(ruin @ {drawdown_level:.0%})      : {reached_drawdown.mean():.1%}")
        print(f"P(ruin before target): {ruined_before_target.mean():.1%}")
        print(f"P(payout)           : {reached_target.mean():.1%}")

    return {
        "Median final equity": f"{np.median(final):,.0f}",
        "5th / 95th pct": f"{np.percentile(final, 5):,.0f} / {np.percentile(final, 95):,.0f}",
        "Median max DD": f"{np.median(max_dd):.1%}",
        "95th pct max DD": f"{np.percentile(max_dd, 95):.1%}",
        "EV per simulation": f"{ev:,.2f}",
        f"P(ruin @ {drawdown_level:.0%})": f"{reached_drawdown.mean():.1%}",
        "P(ruin before target)": f"{ruined_before_target.mean():.1%}",
        "P(payout)": f"{reached_target.mean():.1%}",
    }


def block_bootstrap_simulation(
    trades: pd.DataFrame,
    risk_per_trade: float,
    # ev config
    profit_target: float,
    payout: float,
    cost: float,
    # bootstrap config
    # block_length: int = 10,
    method: str = "stationary",  # "stationary" (geometric) or "moving" (fixed-length overlapping)
    # strategy config
    risk_pct: float = 0.01,
    start_equity: float = 10_000,
    # simulation config
    n_sims: int = 10_000,
    n_trades: int = 1_000,
    drawdown_level: float = 1.0,  # fraction of start_equity lost (1.0 = 100%)
    seed: int | None = None,
    verbose: bool = False,
):
    """Monte Carlo simulation evaluating trading strategy performance using

    Block Bootstrapping (Stationary or Moving Block). Preserves autocorrelation
    and volatility clustering.
    """
    rng = np.random.default_rng(seed if seed is not None else 42)

    # Convert PnL to R-multiples
    trades_R = (trades["PnL"] / risk_per_trade).values
    n_historical = len(trades_R)

    # Fit optimal block length selector
    opt = OptimalBlockLength(trades_R)
    res = opt.compute()

    if n_historical == 0:
        raise ValueError("Trade log cannot be empty.")

    # ---- Block Resampling Matrix Generation ----
    # Pre-allocate indices matrix: shape (n_sims, n_trades)
    idx = np.empty((n_sims, n_trades), dtype=int)

    if method == "stationary":
        opt_sb = int(np.ceil(res["stationary"].values[0]))
        block_length = opt_sb

        # Stationary Bootstrap (Politis & Romano): Block length ~ Geometric(p = 1 / block_length)
        p = 1.0 / opt_sb

        for i in range(n_sims):
            pos = 0
            while pos < n_trades:
                # Random start point in historical trades
                start_idx = rng.integers(0, n_historical)
                # Random block size drawn from Geometric distribution
                b_len = rng.geometric(p)
                # Cap block size so it doesn't overrun n_trades
                actual_len = min(b_len, n_trades - pos)

                # Wrap around circular indexing to handle end-of-series boundary
                block_indices = (start_idx + np.arange(actual_len)) % n_historical
                idx[i, pos : pos + actual_len] = block_indices
                pos += actual_len

    elif method == "moving":
        opt_cbb = int(np.ceil(res["circular"].values[0]))
        block_length = opt_cbb

        # Moving Overlapping Block Bootstrap: Fixed block_length
        for i in range(n_sims):
            pos = 0
            while pos < n_trades:
                start_idx = rng.integers(0, n_historical)
                actual_len = min(opt_cbb, n_trades - pos)

                block_indices = (start_idx + np.arange(actual_len)) % n_historical
                idx[i, pos : pos + actual_len] = block_indices
                pos += actual_len
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'stationary' or 'moving'.")

    # Extract resampled trade R-multiples
    R = trades_R[idx]

    # ---- Simulate Equity Paths ----
    equity = start_equity * np.cumprod(1 + risk_pct * R, axis=1)
    equity = np.hstack([np.full((n_sims, 1), start_equity), equity])

    # ---- Barriers ----
    target_equity = (
        profit_target if profit_target >= start_equity else start_equity + profit_target
    )
    ruin_equity = start_equity * (1 - drawdown_level)

    hit_target = equity >= target_equity
    hit_ruin = equity <= ruin_equity

    n_steps = equity.shape[1]
    t_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), n_steps)
    t_ruin = np.where(hit_ruin.any(axis=1), hit_ruin.argmax(axis=1), n_steps)

    # Check first-touch constraints
    reached_drawdown = t_ruin < n_steps
    reached_target = (t_target < n_steps) & (t_target < t_ruin)
    ruined_before_target = reached_drawdown & ~reached_target

    # ---- Metrics ----
    final = equity[:, -1]
    peak = np.maximum.accumulate(equity, axis=1)
    max_dd = ((peak - equity) / peak).max(axis=1)

    # ---- EV Calculation ----
    ev_per_sim = np.where(reached_target, payout, -cost)
    ev = ev_per_sim.mean()

    if verbose:
        print(
            f"--- Block Bootstrap ({method.capitalize()}, Avg Block={block_length}) ---"
        )
        print(f"Median final equity : {np.median(final):,.0f}")
        print(
            f"5th / 95th pct      : {np.percentile(final, 5):,.0f} / {np.percentile(final, 95):,.0f}"
        )
        print(f"Median max DD       : {np.median(max_dd):.1%}")
        print(f"95th pct max DD     : {np.percentile(max_dd, 95):.1%}")
        print(f"EV per simulation   : {ev:,.2f}")
        print(f"P(ruin @ {drawdown_level:.0%})      : {reached_drawdown.mean():.1%}")
        print(f"P(ruin before target): {ruined_before_target.mean():.1%}")
        print(f"P(payout)           : {reached_target.mean():.1%}")

    return {
        "Median final equity": f"{np.median(final):,.0f}",
        "5th / 95th pct": f"{np.percentile(final, 5):,.0f} / {np.percentile(final, 95):,.0f}",
        "Median max DD": f"{np.median(max_dd):.1%}",
        "95th pct max DD": f"{np.percentile(max_dd, 95):.1%}",
        "EV per simulation": f"{ev:,.2f}",
        f"P(ruin @ {drawdown_level:.0%})": f"{reached_drawdown.mean():.1%}",
        "P(ruin before target)": f"{ruined_before_target.mean():.1%}",
        "P(payout)": f"{reached_target.mean():.1%}",
    }
