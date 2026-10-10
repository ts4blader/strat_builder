import numpy as np
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core import global_style


def kmeans_params_finder(df, PARAMS):
    # clean: inf from zero-loss / zero-downside combos
    features = ["profit_factor", "winrate", "calmar_ratio", "sortino_ratio"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    param_cols = [p for p in PARAMS if p not in ("asset", "timeframe")]

    def find_param_plateaus(
        g, top_q=0.90, k_range=range(2, 10), target="profit_factor"
    ):
        top = g[g[target] >= g[target].quantile(top_q)]
        if len(top) < 20:
            return None
        X = StandardScaler().fit_transform(top[param_cols])

        best = max(
            (
                (silhouette_score(X, lab), k, lab)
                for k, lab in (
                    (
                        k,
                        KMeans(
                            k, init="k-means++", n_init=10, random_state=42
                        ).fit_predict(X),
                    )
                    for k in k_range
                    if k < len(top)
                )
            ),
            key=lambda t: t[0],
        )
        score, k, labels = best
        top = top.assign(cluster_id=labels)

        prof = top.groupby("cluster_id").agg(
            size=(target, "size"),
            median_target=(target, "median"),
            q25_target=(target, lambda s: s.quantile(0.25)),
            **{c: (c, "mean") for c in features},
            **{c: (c, "median") for c in param_cols},
        )
        # rank by worst-case, not best-case: a plateau you can actually trade
        return prof.sort_values(["q25_target", "size"], ascending=False), score, k

    # for (asset, tf), g in df.groupby(['asset', 'timeframe']):
    #     res = find_param_plateaus(g)
    #     if res is None:
    #         continue
    #     prof, score, k = res
    #     print(f"{asset} {tf} — k={k}, silhouette={score:.3f}, TARGET: {TARGET}")
    #     print(f"*All metrics are mean values {features}")
    #     display(global_style(prof))

    for target in features:
        res = find_param_plateaus(g=df, target=target)
        if res is None:
            continue
        prof, score, k = res
        print(f"k={k}, silhouette={score:.3f}, TARGET: {target}")
        print(f"*All metrics are mean values {features}")
        display(global_style(prof))
