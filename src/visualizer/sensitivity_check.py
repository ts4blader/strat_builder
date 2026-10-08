import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sensitivity_check_visualizer(df, PARAM_COLS, TARGET):
    n_params = len(PARAM_COLS)
    n_cols = 2
    n_rows = -(-n_params // n_cols)  # ceil division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    SERIES_COLOR = "#468fe9"
    PEAK_COLOR = "#e8183e"

    for i, col in enumerate(PARAM_COLS):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        grouped = (
            df.groupby(col)[TARGET]
            .agg(["mean", "std", "min", "max", "median"])
            .reset_index()
        )
        grouped = grouped.sort_values(col)

        # peak_row = grouped.loc[grouped["max"].idxmax()]
        # peak_param = peak_row[col]
        # peak_value = peak_row["max"]

        # Box plot for distribution
        fig.add_trace(
            go.Box(
                x=df[col].astype(str),
                y=df[TARGET].round(2),
                name=col.replace("_", " ").title(),
                marker_color=SERIES_COLOR,
                line_width=2,
                boxmean=True,
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )

        fig.update_xaxes(
            title_text=col.replace("_", " ").title(),
            row=row,
            col=col_idx,
            title_font_size=11,
            tickfont_size=10,
            gridwidth=1,
            showline=True,
            linewidth=1,
        )
        fig.update_yaxes(
            title_text=TARGET.replace("_", " ").title() if col_idx == 1 else None,
            row=row,
            col=col_idx,
            title_font_size=11,
            tickfont_size=10,
            gridwidth=1,
            showline=True,
            linewidth=1,
        )

    # Hide empty subplots if params don't fill the grid
    total_slots = n_rows * n_cols
    for j in range(n_params, total_slots):
        row = j // n_cols + 1
        col_idx = j % n_cols + 1
        fig.update_xaxes(visible=False, row=row, col=col_idx)
        fig.update_yaxes(visible=False, row=row, col=col_idx)

    fig.update_layout(
        title_text=f"Sensitivity Analysis — {TARGET.replace('_', ' ').title()} by Parameter",
        height=300 * n_rows,
        width=1100,
        font_family="JetBrainsMono Nerd Font",
        template="plotly_dark",
    )

    return fig
