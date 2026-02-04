import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_main_chart(
    df_slice,
    period_start,
    period_end,
    show_ichimoku: bool,
    show_bb: bool,
    show_kc: bool,
):
    """
    Build main chart:
    - Price (High–Low bars)
    - RSI
    - CMB Composite
    With optional overlays:
    - Ichimoku
    - Bollinger Bands
    - Keltner Channel
    """

    # -------------------------------------------------
    # Build categorical ticks (show date only on change)
    # -------------------------------------------------
    tickvals = []
    ticktext = []

    prev_date = None
    for x, d in zip(df_slice["x"], df_slice["date_only"]):
        if d != prev_date:
            tickvals.append(x)
            ticktext.append(d)
            prev_date = d

    # -------------------------------------------------
    # Create subplots
    # -------------------------------------------------
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.2, 0.25],
        vertical_spacing=0.04,
        subplot_titles=(
            "Price (High–Low)",
            "RSI",
            "CMB Composite",
        ),
    )

    # -------------------------------------------------
    # Price: TRUE high–low bars (Excel-style)
    # -------------------------------------------------
    x_vals = []
    y_vals = []

    for x, low, high in zip(df_slice["x"], df_slice["low"], df_slice["high"]):
        x_vals.extend([x, x, None])
        y_vals.extend([low, high, None])

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color="black", width=1),
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # -------------------------------------------------
    # Ichimoku Cloud
    # -------------------------------------------------
    if show_ichimoku:
        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["tenkan"],
                name="Tenkan",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["kijun"],
                name="Kijun",
                line=dict(color="red", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["senkou_a"],
                name="Senkou A",
                line=dict(color="rgba(0,200,0,0.6)", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["senkou_b"],
                name="Senkou B",
                line=dict(color="rgba(200,0,0,0.6)", width=1),
            ),
            row=1,
            col=1,
        )

    # -------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------
    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["bb_mid"],
                name="BB Mid",
                line=dict(color="gray", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["bb_upper"],
                name="BB Upper",
                line=dict(color="gray", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["bb_lower"],
                name="BB Lower",
                line=dict(color="gray", width=1),
            ),
            row=1,
            col=1,
        )

    # -------------------------------------------------
    # Keltner Channel
    # -------------------------------------------------
    if show_kc:
        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["kc_mid"],
                name="KC Mid",
                line=dict(color="orange", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["kc_upper"],
                name="KC Upper",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_slice["x"],
                y=df_slice["kc_lower"],
                name="KC Lower",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )

    # -------------------------------------------------
    # RSI
    # -------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["rsi"],
            name="RSI",
            line=dict(color="orange", width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)

    # -------------------------------------------------
    # CMB Composite
    # -------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci"],
            name="CI",
            line=dict(color="gray", width=2),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci_13"],
            name="CI 13",
            line=dict(color="dodgerblue", width=1.5),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci_33"],
            name="CI 33",
            line=dict(color="red", width=1.5),
        ),
        row=3,
        col=1,
    )

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # -------------------------------------------------
    # Selected period markers
    # -------------------------------------------------
    if period_start is not None and period_end is not None:
        x_start = period_start.strftime("%Y-%m-%d %H:%M")
        x_end = period_end.strftime("%Y-%m-%d %H:%M")

        fig.add_vline(
            x=x_start,
            line_dash="dash",
            line_color="black",
            line_width=2,
        )

        fig.add_vline(
            x=x_end,
            line_dash="dash",
            line_color="black",
            line_width=2,
        )

    return fig
