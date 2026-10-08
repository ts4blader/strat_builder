import numpy as np
import pandas as pd
from arch.bootstrap import optimal_block_length
from tqdm.auto import tqdm

# params = {
#     "tp" : 0.15,
#     "sl" : 0.10,
#     "daily_sl" : 0.05,
#     "trailing_sl" : False,
#     "bars_per_day" : 23,
#     "days_per_month" : 21,
#     "n_days" : 252,
#     "capital" : 100_000,
#     "challenge_cost" : 650,
#     "fixed_cost_month" : 15,
#     "payout_pct" : 0.80,
#     "bankroll" : 10000,
#     "block_size" : 1,
# }

# params["bars_per_month"] = params["days_per_month"] * params["bars_per_day"]
# params["n_bars"] = params["n_days"] * params["bars_per_day"]

# leverage_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def get_optimal_block_size(log_returns):
    """Compute optimal block length for circular block bootstrap via Politis & White (2004)."""
    returns_simple = (np.exp(log_returns.dropna()) - 1).values
    opt = optimal_block_length(returns_simple)
    block_size = round(opt.iloc[0]["circular"])
    return max(1, block_size)


def sample_bootstrap_path(returns_simple, p, rng):
    T = len(returns_simple)
    block_size = min(T, max(1, p["block_size"]))
    n_blocks = int(np.ceil(p["n_bars"] / block_size))
    max_start_idx = T - block_size

    start_idx = rng.integers(0, max_start_idx + 1, size=n_blocks)
    r = np.concatenate([returns_simple[s : s + block_size] for s in start_idx])[
        : p["n_bars"]
    ]
    return r


def propfirm_simulation(log_returns, leverage, p, n_sim, seed=None):
    returns = np.exp(log_returns.dropna()) - 1  # log -> simple
    T = len(returns)

    if "block_size" not in p:
        p["block_size"] = get_optimal_block_size(log_returns)

    block_size = min(T, max(1, p["block_size"]))
    n_blocks = int(np.ceil(p["n_bars"] / block_size))
    max_start_idx = T - block_size

    leveraged_returns = returns * leverage
    rng = np.random.default_rng(seed)

    result_list, bars_to_tp_list, bars_to_end_list = [], [], []
    final_return_list, loss_type_list = [], []
    max_drawdown_list, max_daily_drawdown_list = [], []

    for _ in range(n_sim):
        start_idx = rng.integers(0, max_start_idx + 1, size=n_blocks)
        r = np.concatenate(
            [leveraged_returns.values[s : s + block_size] for s in start_idx]
        )[: p["n_bars"]]

        cum = (1 + r).cumprod()
        tp_level = 1 + p["tp"]
        sl_level = 1 - p["sl"]

        if p["trailing_sl"]:
            drawdown = cum / np.maximum.accumulate(cum) - 1
            hit_sl_cum = drawdown <= -p["sl"]
        else:
            hit_sl_cum = cum <= sl_level

        n_days = p["n_bars"] // p["bars_per_day"]
        daily_r = np.zeros(p["n_bars"])
        for day in range(n_days):
            start = day * p["bars_per_day"]
            end = (day + 1) * p["bars_per_day"]
            daily_r[start:end] = np.cumprod(1 + r[start:end]) - 1
        hit_sl_daily = daily_r <= -p["daily_sl"]

        hit_sl = hit_sl_cum | hit_sl_daily
        hit_tp = cum >= tp_level

        tp_idx = np.argmax(hit_tp) if hit_tp.any() else None
        sl_idx = np.argmax(hit_sl) if hit_sl.any() else None

        sl_type = None
        if sl_idx is not None:
            if hit_sl_daily[sl_idx]:
                sl_type = "daily"
            elif hit_sl_cum[sl_idx]:
                sl_type = "cumulative"

        if tp_idx is not None and (sl_idx is None or tp_idx <= sl_idx):
            outcome = 1

            cum_after = cum[tp_idx:]
            daily_r_after = daily_r[tp_idx:]

            if p["trailing_sl"]:
                drawdown_after = cum_after / np.maximum.accumulate(cum_after) - 1
                hit_sl_cum_after = drawdown_after <= -p["sl"]
            else:
                hit_sl_cum_after = cum_after <= sl_level

            hit_sl_daily_after = daily_r_after <= -p["daily_sl"]
            hit_sl_after = hit_sl_cum_after | hit_sl_daily_after

            if hit_sl_after.any():
                sl_after_idx = np.argmax(hit_sl_after)
                end_idx = tp_idx + sl_after_idx
            else:
                end_idx = len(cum) - 1

            bars_to_tp = tp_idx
            bars_to_end = end_idx

            cum_funded = cum[tp_idx : end_idx + 1]
            max_idx = np.argmax(cum_funded) + tp_idx

            real_max_idx = (
                max_idx
                + rng.integers(0, p["bars_per_month"] + 1)
                - p["bars_per_month"] // 2
            )
            min_payout_idx = tp_idx + p["bars_per_month"]

            if min_payout_idx <= real_max_idx <= end_idx:
                final_return = cum[real_max_idx] - 1
            else:
                prev_real_max_idx = real_max_idx - p["bars_per_month"]
                if min_payout_idx <= prev_real_max_idx <= end_idx:
                    final_return = cum[prev_real_max_idx] - 1
                else:
                    final_return = 0

            daily_r_slice = daily_r[: end_idx + 1]
            cum_slice = cum[: end_idx + 1]

        elif sl_idx is not None:
            outcome = -1
            bars_to_tp = np.nan
            bars_to_end = sl_idx
            final_return = np.nan

            daily_r_slice = daily_r[: sl_idx + 1]
            cum_slice = cum[: sl_idx + 1]

        else:
            outcome = 0
            bars_to_tp = np.nan
            bars_to_end = len(cum) - 1
            final_return = np.nan

            daily_r_slice = daily_r
            cum_slice = cum

        drawdown_slice = cum_slice / np.maximum.accumulate(cum_slice) - 1
        max_drawdown_list.append(drawdown_slice.min())
        max_daily_drawdown_list.append(daily_r_slice.min())

        result_list.append(outcome)
        bars_to_tp_list.append(bars_to_tp)
        bars_to_end_list.append(bars_to_end)
        final_return_list.append(final_return)
        loss_type_list.append(sl_type)

    result_list = np.array(result_list)
    bars_to_tp_list = np.array(bars_to_tp_list, dtype=float)
    bars_to_end_list = np.array(bars_to_end_list, dtype=float)
    final_return_list = np.array(final_return_list, dtype=float)
    loss_type_list = np.array(loss_type_list)
    max_drawdown_list = np.array(max_drawdown_list, dtype=float)
    max_daily_drawdown_list = np.array(max_daily_drawdown_list, dtype=float)

    success_mask = result_list == 1
    loss_mask = result_list == -1

    loss_daily_rate = np.mean(loss_mask & (loss_type_list == "daily"))
    loss_cum_rate = np.mean(loss_mask & (loss_type_list == "cumulative"))

    success_rate = success_mask.mean()
    loss_rate = loss_mask.mean()
    no_event_rate = (result_list == 0).mean()

    avg_bars_to_tp = (
        np.nanmean(bars_to_tp_list[success_mask]) if success_mask.any() else np.nan
    )
    avg_bars_to_end = np.nanmean(bars_to_end_list)

    net_profit = np.maximum((final_return_list - p["tp"]) * p["capital"], 0)
    net_payout = net_profit * p["payout_pct"]

    monthly_cost = (bars_to_end_list / p["bars_per_month"]) * p["fixed_cost_month"]
    avg_monthly_cost = np.nanmean(monthly_cost)

    expected_payout = np.nanmean(net_payout[success_mask]) if success_mask.any() else 0

    # EV: the challenge fee is assumed refunded on passing -> weighted by (1-success_rate)
    # avg_monthly_cost is already an unconditional expectation, it should not be reweighted
    ev = (
        success_rate * expected_payout
        - (1 - success_rate) * p["challenge_cost"]
        - avg_monthly_cost
    )

    # cost_to_pass: expected gross outlay to get a funded account, refunded or not
    cost_to_pass = (
        (p["challenge_cost"] + avg_monthly_cost) / success_rate
        if success_rate > 0
        else 0
    )
    efficiency = ev / cost_to_pass if cost_to_pass > 0 else 0
    prob_ruin = (1 - success_rate) ** (
        p["bankroll"] / (p["challenge_cost"] + avg_monthly_cost)
    )

    return {
        "leverage": leverage,
        "success_rate": success_rate,
        "loss_rate": loss_rate,
        "no_event_rate": no_event_rate,
        "loss_daily_rate": loss_daily_rate,
        "loss_cumulative_rate": loss_cum_rate,
        "avg_max_drawdown": round(max_drawdown_list.mean(), 3),
        "avg_max_daily_drawdown": round(max_daily_drawdown_list.mean(), 3),
        "avg_days_to_tp": int(avg_bars_to_tp / p["bars_per_day"])
        if success_mask.any()
        else np.nan,
        "avg_days_to_end": int(avg_bars_to_end / p["bars_per_day"]),
        "avg_final_return": round(np.nanmean(final_return_list), 3)
        if success_mask.any()
        else 0,
        "avg_net_payout": int(np.nanmean(net_payout)) if success_mask.any() else 0,
        "expected_value": round(ev, 1),
        "cost_to_pass": round(cost_to_pass, 1) if success_mask.any() else np.nan,
        "efficiency": round(efficiency, 3) if success_mask.any() else np.nan,
        "prob_ruin": round(prob_ruin, 4),
    }


def propfirm_sim_leverage(log_returns, params, leverage_list):
    rows = [
        propfirm_simulation(log_returns, lev, params, n_sim=1000, seed=42)
        for lev in tqdm(leverage_list, desc="Propfirm simulations with leverage")
    ]
    results_df = pd.DataFrame(rows, index=leverage_list)
    return results_df.transpose()
