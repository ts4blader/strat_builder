import numpy as np
import pandas as pd
from IPython.display import display

from core import global_style


def neighborhood_analysis(df, PARAMS, TARGET, radius=1):
    param_cols = list(PARAMS)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])

    # rank each param onto integer grid coordinates (handles uneven spacing)
    coords = (
        df[param_cols]
        .apply(lambda s: s.rank(method="dense").astype(int) - 1)
        .to_numpy()
    )
    vals = df[TARGET].to_numpy()

    # precompute Chebyshev distance matrix → neighbour mask
    # |coords[i] - coords[j]|_inf <= radius
    dist = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
    mask = dist <= radius

    df = df.copy()
    df["smoothed"] = np.nanmean(np.where(mask, vals[None, :], np.nan), axis=1)
    df["n_neighbours"] = mask.sum(axis=1)

    # edge combos have few neighbours — their smoothed value is unreliable
    core = df[df["n_neighbours"] >= 0.6 * df["n_neighbours"].max()]
    core["score"] = (
        core["smoothed"] * 0.7 + core[TARGET] * 0.3
    )  # adjust weights to taste

    print(f"Target: {TARGET}, SCORE = smoothed * 0.7 + TARGET * 0.3")
    display(
        global_style(
            core.nlargest(10, columns=["score"])[
                param_cols + [TARGET, "smoothed", "n_neighbours", "score"]
            ]
        )
    )

    best = core.nlargest(1, columns=["score"]).iloc[0]
    print("The best parameters combination: ")
    display(best)

    return best


def neighbours_for_radius(df, param_cols, radius):
    """
    Compute n_neighbours per row for a given radius.
    Returns df with an added 'n_neighbours' column.
    """
    grid = df[param_cols].apply(lambda s: s.rank(method="dense").astype(int) - 1)

    out_frames = []

    gg = grid.loc[grid.index].to_numpy()
    n_neigh = np.array([(np.abs(gg - r).max(axis=1) <= radius).sum() for r in gg])
    g = grid.copy()
    g["n_neighbours"] = n_neigh
    out_frames.append(g)

    return pd.concat(out_frames)


def radius_diagnostics(
    df,
    PARAMS,
    TARGET,
    radii=(1, 2, 3, 4),
    saturation_tol=0.02,
    min_axis_card=3,
):
    """
    Sweep candidate radii and report neighbour-count stats to help pick
    the most 'effective' radius: one where n_neighbours varies meaningfully
    (real edge effects being captured) rather than being uniform/maxed-out
    (saturation -> smoothing has swallowed whole parameter axes).

    saturation_tol: relative tolerance (on max-min spread vs the mean) below
    which a radius is flagged as fully 'saturated' (looks uniform).

    min_axis_card: axes with fewer than this many unique values (e.g. binary
    flags) are EXCLUDED from the "exceeds axis range" check. A 2-value axis
    saturates at radius=1 unavoidably -- that's not oversmoothing, there's
    simply no finer resolution possible on that axis. Only axes with real
    resolution to lose should count against a radius.
    """
    param_cols = [p for p in PARAMS if p not in ("asset", "timeframe")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])

    # per-parameter number of unique tested values -> tells us when a radius
    # will start to span (or exceed) a whole axis
    axis_sizes = {p: df[p].nunique() for p in param_cols}
    meaningful_axes = {p: s for p, s in axis_sizes.items() if s >= min_axis_card}

    rows = []
    for r in radii:
        dfr = neighbours_for_radius(df, param_cols, r)
        nn = dfr["n_neighbours"]

        spread = nn.max() - nn.min()
        rel_spread = spread / nn.mean() if nn.mean() else 0.0
        saturated = rel_spread < saturation_tol

        # does this radius already exceed a MEANINGFUL (non-binary-like) axis?
        exceeds_meaningful_axis = any(
            (2 * r + 1) >= size for size in meaningful_axes.values()
        )

        rows.append(
            {
                "radius": r,
                "n_neighbours_min": nn.min(),
                "n_neighbours_max": nn.max(),
                "n_neighbours_mean": round(nn.mean(), 1),
                "rel_spread": round(rel_spread, 4),
                "saturated": saturated,
                "exceeds_meaningful_axis": exceeds_meaningful_axis,
            }
        )

    report = pd.DataFrame(rows)

    # pick the largest radius that (a) is not fully saturated and
    # (b) has not yet exceeded any axis with real resolution to lose
    # (ignoring low-cardinality/binary axes, which saturate trivially).
    candidates = report[(~report["saturated"]) & (~report["exceeds_meaningful_axis"])]
    if len(candidates):
        recommended = candidates["radius"].max()
    else:
        recommended = report["radius"].min()

    print("Per-parameter unique tested values (axis sizes):")
    for p, size in axis_sizes.items():
        flag = (
            "  <- low-cardinality, excluded from saturation check"
            if size < min_axis_card
            else ""
        )
        sat_r = -(-(size - 1) // 2)  # ceil((size-1)/2)
        print(
            f"  {p}: {size} unique values -> saturates around radius >= {sat_r}{flag}"
        )
    print()
    print(report.to_string(index=False))
    print(f"\nRecommended radius: {recommended}")

    return report, recommended


# --- usage ---
# report, best_r = radius_diagnostics(df, PARAMS, TARGET, radii=(1, 2, 3, 4))
