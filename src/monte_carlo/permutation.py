import numpy as np


def permutation_simulation(
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
    n_trades=None,  # None -> use every trade exactly once per path
    drawdown_level=1,  # fraction of start_equity that may be lost (1 = 100%, 0.10 = 10%)
    seed: int | None = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed if seed is not None else 42)

    # ---- simulate ----
    trades_R = (trades["PnL"] / risk_per_trade).values.astype(float)
    n_obs = len(trades_R)
    if n_trades is None:
        n_trades = n_obs

    # Each path = independent shuffles of the full trade set (without replacement),
    # concatenated if n_trades > n_obs, truncated if n_trades < n_obs.
    n_blocks = int(np.ceil(n_trades / n_obs))
    blocks = [
        trades_R[rng.random((n_sims, n_obs)).argsort(axis=1)] for _ in range(n_blocks)
    ]
    R = np.hstack(blocks)[:, :n_trades]

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

    # ruin is checked first: target only counts if reached strictly BEFORE ruin
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
