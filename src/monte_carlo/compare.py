from dema_strat import compare_table

from .bootstrap import block_bootstrap_simulation
from .parametric import parametric_simulation
from .permutation import permutation_simulation


def compare_monte_carlo(
    trades,
    risk_per_trade,
    equity,
    win_rate,
    profit_factor,
    rr_ratio,
    cost,
    payout,
    profit_target,
    risk_pct,
    drawdown_level,
):
    parametric = parametric_simulation(
        start_equity=equity,
        cost=cost,
        payout=payout,
        profit_target=profit_target,
        win_rate=win_rate,
        profit_factor=profit_factor,
        rr_ratio=rr_ratio,
        risk_pct=risk_pct,
        drawdown_level=drawdown_level,
    )

    bootstrap = block_bootstrap_simulation(
        trades=trades,
        risk_per_trade=risk_per_trade,
        start_equity=equity,
        cost=cost,
        payout=payout,
        profit_target=profit_target,
        risk_pct=risk_pct,
        drawdown_level=drawdown_level,
    )

    permutation = permutation_simulation(
        trades=trades,
        risk_per_trade=risk_per_trade,
        start_equity=equity,
        cost=cost,
        payout=payout,
        profit_target=profit_target,
        risk_pct=risk_pct,
        drawdown_level=drawdown_level,
    )

    compare_table(
        {
            "Parametric": parametric,
            "Bootstrap": bootstrap,
            "Permutation": permutation,
        }
    )
