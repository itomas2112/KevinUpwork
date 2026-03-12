import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components

# Fixed EMA colors: shortest period = red, 2nd = cyan, 2nd longest = yellow, longest = green
EMA_FIXED_COLORS = ["red", "cyan", "yellow", "green"]

# Panel name → yaxis suffix mapping (y=price, y2..y9=oscillators)
PANEL_YAXIS = {
    "rsi": "y2",
    "cmb": "y3",
    "stoch": "y4",
    "adx": "y5",
    "atr": "y6",
    "macd": "y7",
    "obv": "y8",
    "accdist": "y9",
}


def build_main_chart(
    df_slice,
    period_start,
    period_end,
    show_ichimoku: bool,
    show_bb: bool,
    show_kc: bool,
    show_strategy: bool,
    rsi_upper_1: float = 70.0,
    rsi_upper_2: float = 67.0,
    rsi_lower_1: float = 33.0,
    rsi_lower_2: float = 30.0,
    cmb_lines: list = None,
    draw_mode: bool = False,
    chart_height: int = 920,
    ichi_show_tenkan: bool = True,
    ichi_show_kijun: bool = True,
    ichi_show_senkou_a: bool = True,
    ichi_show_senkou_b: bool = True,
    ichi_show_chikou: bool = True,
    ichi_show_senkou_a_current: bool = False,
    ichi_show_senkou_b_current: bool = False,
    ichi_show_chikou_decision: bool = False,
    bb_show_upper: bool = True,
    bb_show_middle: bool = True,
    bb_show_lower: bool = True,
    kc_show_upper: bool = True,
    kc_show_middle: bool = True,
    kc_show_lower: bool = True,
    chart_key: str = None,
    show_rsi: bool = True,
    show_cmb: bool = True,
    show_stoch: bool = True,
    show_adx: bool = False,
    show_atr: bool = False,
    show_macd: bool = False,
    show_obv: bool = False,
    show_accdist: bool = False,
    show_supertrend: bool = False,
    show_ema: bool = False,
    ema_periods: list = None,
    show_donchian: bool = False,
    dc_show_upper: bool = True,
    dc_show_middle: bool = True,
    dc_show_lower: bool = True,
    show_psar: bool = False,
):
    """
    Build main chart using a single x-axis with dynamic y-axis domains
    so the vertical crosshair spans all panels.
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
    # Y-axis domains — dynamic based on visible panels
    # Build list bottom-to-top: first entry = bottommost panel
    # -------------------------------------------------
    visible_panels = []
    if show_accdist:
        visible_panels.append("accdist")
    if show_obv:
        visible_panels.append("obv")
    if show_macd:
        visible_panels.append("macd")
    if show_atr:
        visible_panels.append("atr")
    if show_adx:
        visible_panels.append("adx")
    if show_stoch:
        visible_panels.append("stoch")
    if show_cmb:
        visible_panels.append("cmb")
    if show_rsi:
        visible_panels.append("rsi")

    if visible_panels:
        price_domain = [0.59, 1.00]
        osc_total = 0.55
        gap = 0.03
        n = len(visible_panels)
        usable = osc_total - gap * (n - 1) if n > 1 else osc_total
        panel_height = usable / n

        domains = {}
        bottom = 0.0
        for panel_name in visible_panels:
            domains[panel_name] = [round(bottom, 4), round(bottom + panel_height, 4)]
            bottom += panel_height + gap
    else:
        price_domain = [0.0, 1.00]
        domains = {}

    fig = go.Figure()

    # -------------------------------------------------
    # Price: TRUE high–low bars  (yaxis="y", default)
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
            line=dict(color="white", width=1),
            name="Price",
            showlegend=False,
        )
    )

    # -------------------------------------------------
    # Strategy markers  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_strategy:

        price_range = df_slice["high"].max() - df_slice["low"].min()
        marker_offset = price_range * 0.03

        entries = df_slice[df_slice["entry_signal"]]

        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["x"],
                    y=entries["low"] - marker_offset,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="yellow",
                        line=dict(color="white", width=1),
                    ),
                    name="Entry",
                )
            )

        if "exit_type" in df_slice.columns:
            target_exits = df_slice[
                (df_slice["exit_signal"]) & (df_slice["exit_type"] == "Target")
            ]
            if not target_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=target_exits["x"],
                        y=target_exits["high"] + marker_offset,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="green",
                            line=dict(color="white", width=1),
                        ),
                        name="Target Exit",
                    )
                )

            dynamic_stop_exits = df_slice[
                (df_slice["exit_signal"]) & (df_slice["exit_type"] == "Stop")
            ]
            if not dynamic_stop_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=dynamic_stop_exits["x"],
                        y=dynamic_stop_exits["high"] + marker_offset,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="orange",
                            line=dict(color="white", width=1),
                        ),
                        name="Dynamic Stop Exit",
                    )
                )

            initial_stop_exits = df_slice[
                (df_slice["exit_signal"]) & (df_slice["exit_type"] == "Initial Stop")
            ]
            if not initial_stop_exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=initial_stop_exits["x"],
                        y=initial_stop_exits["high"] + marker_offset,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="red",
                            line=dict(color="white", width=1),
                        ),
                        name="Static Stop Exit",
                    )
                )
        else:
            exits = df_slice[df_slice["exit_signal"]]

            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits["x"],
                        y=exits["high"] + marker_offset,
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color="green",
                            line=dict(color="white", width=1),
                        ),
                        name="Exit",
                    )
                )

    # -------------------------------------------------
    # Ichimoku Cloud  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_ichimoku:
        if ichi_show_tenkan:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["tenkan"], name="Tenkan",
                                     line=dict(color="red", width=1), showlegend=False))
        if ichi_show_kijun:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["kijun"], name="Kijun",
                                     line=dict(color="lightblue", width=1), showlegend=False))
        if ichi_show_senkou_a:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["senkou_a"], name="Senkou A",
                                     line=dict(color="yellow", width=1), showlegend=False))
        if ichi_show_senkou_b:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["senkou_b"], name="Senkou B",
                                     line=dict(color="green", width=1), showlegend=False))
        if ichi_show_chikou and "chikou" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["chikou"], name="Chikou",
                                     line=dict(color="#9932CC", width=1.5), showlegend=False))
        if ichi_show_senkou_a_current and "senkou_a_current" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["senkou_a_current"],
                                     name="Senkou A (current)",
                                     line=dict(color="yellow", width=1, dash="dot"), showlegend=False))
        if ichi_show_senkou_b_current and "senkou_b_current" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["senkou_b_current"],
                                     name="Senkou B (current)",
                                     line=dict(color="green", width=1, dash="dot"), showlegend=False))
        if ichi_show_chikou_decision:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["latest"],
                                     name="Chikou (decision)",
                                     line=dict(color="#9932CC", width=1, dash="dot"), showlegend=False))

    # -------------------------------------------------
    # Bollinger Bands  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_bb:
        if bb_show_middle:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["bb_mid"], name="BB Mid",
                                     line=dict(color="gray", width=1, dash="dot"), showlegend=False))
        if bb_show_upper:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["bb_upper"], name="BB Upper",
                                     line=dict(color="gray", width=1), showlegend=False))
        if bb_show_lower:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["bb_lower"], name="BB Lower",
                                     line=dict(color="gray", width=1), showlegend=False))

    # -------------------------------------------------
    # Keltner Channel  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_kc:
        if kc_show_middle:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["kc_mid"], name="KC Mid",
                                     line=dict(color="orange", width=1, dash="dot"), showlegend=False))
        if kc_show_upper:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["kc_upper"], name="KC Upper",
                                     line=dict(color="orange", width=1), showlegend=False))
        if kc_show_lower:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["kc_lower"], name="KC Lower",
                                     line=dict(color="orange", width=1), showlegend=False))

    # -------------------------------------------------
    # Supertrend Overlay  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_supertrend and "supertrend" in df_slice.columns and "supertrend_dir" in df_slice.columns:
        st_vals = df_slice["supertrend"]
        st_dir = df_slice["supertrend_dir"]
        x_arr = df_slice["x"]

        # Split into up/down segments for coloring
        up_x, up_y = [], []
        dn_x, dn_y = [], []
        for xi, yi, di in zip(x_arr, st_vals, st_dir):
            if di == 1:
                up_x.append(xi)
                up_y.append(yi)
                if dn_x:
                    # Connect the transition point
                    up_x.insert(len(up_x) - 1, xi)
                    up_y.insert(len(up_y) - 1, yi)
                    dn_x.append(xi)
                    dn_y.append(yi)
                    fig.add_trace(go.Scatter(x=dn_x, y=dn_y, name="Supertrend Down",
                                             line=dict(color="red", width=2), showlegend=False,
                                             hoverinfo="skip"))
                    dn_x, dn_y = [], []
            else:
                dn_x.append(xi)
                dn_y.append(yi)
                if up_x:
                    dn_x.insert(len(dn_x) - 1, xi)
                    dn_y.insert(len(dn_y) - 1, yi)
                    up_x.append(xi)
                    up_y.append(yi)
                    fig.add_trace(go.Scatter(x=up_x, y=up_y, name="Supertrend Up",
                                             line=dict(color="green", width=2), showlegend=False,
                                             hoverinfo="skip"))
                    up_x, up_y = [], []

        # Flush remaining segments
        if up_x:
            fig.add_trace(go.Scatter(x=up_x, y=up_y, name="Supertrend Up",
                                     line=dict(color="green", width=2), showlegend=False,
                                     hoverinfo="skip"))
        if dn_x:
            fig.add_trace(go.Scatter(x=dn_x, y=dn_y, name="Supertrend Down",
                                     line=dict(color="red", width=2), showlegend=False,
                                     hoverinfo="skip"))

    # -------------------------------------------------
    # EMA Overlay  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_ema and ema_periods:
        # Sort EMAs by period to assign colors: shortest=red, 2nd=lightblue, 2nd longest=yellow, longest=green
        indexed_periods = [(i, period) for i, period in enumerate(ema_periods)]
        sorted_by_period = sorted(indexed_periods, key=lambda x: x[1])
        n_emas = len(sorted_by_period)

        # Assign colors based on sorted position
        def _ema_color(sorted_pos, total):
            if total == 1:
                return "red"
            elif total == 2:
                return ["red", "cyan"][sorted_pos]
            elif total == 3:
                return ["red", "cyan", "yellow"][sorted_pos]
            elif total == 4:
                return EMA_FIXED_COLORS[sorted_pos]
            else:
                # For 5+: red for shortest, green for longest, distribute middle colors
                if sorted_pos == 0:
                    return "red"
                elif sorted_pos == total - 1:
                    return "green"
                elif sorted_pos == 1:
                    return "cyan"
                elif sorted_pos == total - 2:
                    return "yellow"
                else:
                    return "white"

        # Build color map: original index -> color
        color_map = {}
        for sorted_pos, (orig_idx, _) in enumerate(sorted_by_period):
            color_map[orig_idx] = _ema_color(sorted_pos, n_emas)

        for i, period in enumerate(ema_periods):
            col = f"ema_{i}"
            if col in df_slice.columns:
                color = color_map.get(i, "white")
                fig.add_trace(go.Scatter(
                    x=df_slice["x"], y=df_slice[col],
                    name=f"EMA({int(period)})",
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # -------------------------------------------------
    # Donchian Channel  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_donchian:
        if dc_show_middle and "dc_mid" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["dc_mid"], name="DC Mid",
                                     line=dict(color="cyan", width=1, dash="dot"), showlegend=False))
        if dc_show_upper and "dc_upper" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["dc_upper"], name="DC Upper",
                                     line=dict(color="cyan", width=1), showlegend=False))
        if dc_show_lower and "dc_lower" in df_slice.columns:
            fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["dc_lower"], name="DC Lower",
                                     line=dict(color="cyan", width=1), showlegend=False))

    # -------------------------------------------------
    # Parabolic SAR  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_psar and "psar" in df_slice.columns and "psar_dir" in df_slice.columns:
        psar_vals = df_slice["psar"]
        psar_dirs = df_slice["psar_dir"]
        x_arr = df_slice["x"]

        # Color dots by direction: green (uptrend/SAR below) and red (downtrend/SAR above)
        up_x, up_y = [], []
        dn_x, dn_y = [], []
        for xi, yi, di in zip(x_arr, psar_vals, psar_dirs):
            if di == 1:
                up_x.append(xi)
                up_y.append(yi)
            else:
                dn_x.append(xi)
                dn_y.append(yi)

        if up_x:
            fig.add_trace(go.Scatter(
                x=up_x, y=up_y, mode="markers", name="PSAR Up",
                marker=dict(color="green", size=3, symbol="circle"),
                showlegend=False, hoverinfo="skip"))
        if dn_x:
            fig.add_trace(go.Scatter(
                x=dn_x, y=dn_y, mode="markers", name="PSAR Down",
                marker=dict(color="red", size=3, symbol="circle"),
                showlegend=False, hoverinfo="skip"))

    # -------------------------------------------------
    # RSI  (yaxis="y2")
    # -------------------------------------------------
    if show_rsi:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["rsi"], name="RSI",
                                 line=dict(color="#9932CC", width=2), showlegend=False, yaxis="y2"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["rsi_13"], name="RSI 13 SMA",
                                 line=dict(color="lightblue", width=1), showlegend=False, yaxis="y2",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["rsi_33"], name="RSI 33 SMA",
                                 line=dict(color="yellow", width=1), showlegend=False, yaxis="y2",
                                 hoverinfo="skip"))
        fig.add_shape(type="rect", x0=0, x1=1, y0=rsi_upper_2, y1=rsi_upper_1,
                      xref="paper", yref="y2", fillcolor="red", opacity=0.3, line_width=0)
        fig.add_shape(type="rect", x0=0, x1=1, y0=rsi_lower_2, y1=rsi_lower_1,
                      xref="paper", yref="y2", fillcolor="blue", opacity=0.3, line_width=0)
        fig.add_shape(type="line", x0=0, x1=1, y0=50, y1=50, xref="paper", yref="y2",
                      line=dict(dash="solid", color="gray", width=1))
        for rsi_val in [rsi_upper_1, rsi_upper_2, rsi_lower_1, rsi_lower_2]:
            fig.add_shape(type="line", x0=0, x1=1, y0=rsi_val, y1=rsi_val, xref="paper", yref="y2",
                          line=dict(dash="dot", color="gray", width=1))

    # -------------------------------------------------
    # CMB Composite  (yaxis="y3")
    # -------------------------------------------------
    if show_cmb:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["ci"], name="CI",
                                 line=dict(color="#9932CC", width=2), showlegend=False, yaxis="y3"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["ci_13"], name="CI 13",
                                 line=dict(color="lightblue", width=1), showlegend=False, yaxis="y3",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["ci_33"], name="CI 33",
                                 line=dict(color="yellow", width=1), showlegend=False, yaxis="y3",
                                 hoverinfo="skip"))
        if cmb_lines:
            for val in cmb_lines:
                fig.add_shape(type="line", x0=0, x1=1, y0=val, y1=val, xref="paper", yref="y3",
                              line=dict(dash="dot", color="white", width=1))

    # -------------------------------------------------
    # Stochastic  (yaxis="y4")
    # -------------------------------------------------
    if show_stoch:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["stoch_k"], name="%K",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y4"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["stoch_d"], name="%D",
                                 line=dict(color="red", width=1.5), showlegend=False, yaxis="y4"))
        for lv, dash in [(80, "solid"), (50, "dot"), (20, "solid")]:
            fig.add_shape(type="line", x0=0, x1=1, y0=lv, y1=lv, xref="paper", yref="y4",
                          line=dict(dash=dash, color="gray", width=1))

    # -------------------------------------------------
    # ADX  (yaxis="y5")
    # -------------------------------------------------
    if show_adx:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["adx"], name="ADX",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y5"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["plus_di"], name="+DI",
                                 line=dict(color="green", width=1), showlegend=False, yaxis="y5"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["minus_di"], name="-DI",
                                 line=dict(color="red", width=1), showlegend=False, yaxis="y5"))

    # -------------------------------------------------
    # ATR  (yaxis="y6")
    # -------------------------------------------------
    if show_atr:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["atr"], name="ATR",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y6"))

    # -------------------------------------------------
    # MACD  (yaxis="y7")
    # -------------------------------------------------
    if show_macd:
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["macd_line"], name="MACD",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y7"))
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["macd_signal"], name="Signal",
                                 line=dict(color="red", width=1), showlegend=False, yaxis="y7"))
        # Histogram as bar chart
        fig.add_trace(go.Bar(x=df_slice["x"], y=df_slice["macd_hist"], name="Histogram",
                             marker_color="lightblue", opacity=0.8, showlegend=False, yaxis="y7"))
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0, xref="paper", yref="y7",
                      line=dict(dash="solid", color="gray", width=1))

    # -------------------------------------------------
    # OBV  (yaxis="y8")
    # -------------------------------------------------
    if show_obv and "obv" in df_slice.columns and df_slice["obv"].notna().any():
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["obv"], name="OBV",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y8"))

    # -------------------------------------------------
    # Accumulation/Distribution  (yaxis="y9")
    # -------------------------------------------------
    if show_accdist and "acc_dist" in df_slice.columns and df_slice["acc_dist"].notna().any():
        fig.add_trace(go.Scatter(x=df_slice["x"], y=df_slice["acc_dist"], name="Acc/Dist",
                                 line=dict(color="white", width=1.5), showlegend=False, yaxis="y9"))

    # -------------------------------------------------
    # Hover & Crosshair
    # -------------------------------------------------
    for trace in fig.data:
        if getattr(trace, 'hoverinfo', None) == 'skip':
            trace.hovertemplate = None
            continue
        yax = getattr(trace, 'yaxis', None) or 'y'
        if yax == 'y':
            if trace.name == 'Price':
                trace.hoverinfo = "x+y+name"
                trace.hovertemplate = "<b>%{fullData.name}</b>: %{y:.2f}<extra></extra>"
            else:
                trace.hoverinfo = "skip"
                trace.hovertemplate = None
        else:
            trace.hoverinfo = "x+y+name"
            trace.hovertemplate = "<b>%{fullData.name}</b>: %{y:.2f}<extra></extra>"

    # Y-axis spike options
    spike_y = dict(showspikes=True, spikemode="across+toaxis", spikethickness=1,
                   spikedash="dot", spikecolor="gray", spikesnap="cursor")
    spike_y_no_label = dict(showspikes=True, spikemode="across", spikethickness=1,
                            spikedash="dot", spikecolor="gray", spikesnap="cursor")
    grid_style = dict(gridcolor="rgba(255,255,255,0.08)", griddash="dash")

    # Determine bottommost panel for x-axis anchor
    bottom_panel = visible_panels[0] if visible_panels else None
    bottom_yaxis = PANEL_YAXIS.get(bottom_panel, "y") if bottom_panel else "y"

    # -------------------------------------------------
    # Layout
    # -------------------------------------------------
    layout_kwargs = dict(
        height=chart_height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        spikedistance=-1,
        uirevision=chart_key or "constant",
        xaxis=dict(
            anchor=bottom_yaxis,
            domain=[0, 1],
            type="category",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showspikes=True, spikemode="across+toaxis", spikethickness=1,
            spikedash="dot", spikecolor="gray", spikesnap="cursor",
            **grid_style,
        ),
        yaxis=dict(domain=price_domain,
                   anchor="x" if not visible_panels else "free", position=0,
                   **grid_style, **spike_y_no_label),
    )

    # Add y-axes for each panel (visible or not, to keep axis numbering stable)
    panel_configs = {
        "rsi":     ("y2", show_rsi, dict(tickmode="array",
                    tickvals=sorted([rsi_lower_2, rsi_lower_1, 50, rsi_upper_2, rsi_upper_1]))),
        "cmb":     ("y3", show_cmb, {}),
        "stoch":   ("y4", show_stoch, {}),
        "adx":     ("y5", show_adx, {}),
        "atr":     ("y6", show_atr, {}),
        "macd":    ("y7", show_macd, {}),
        "obv":     ("y8", show_obv, {}),
        "accdist": ("y9", show_accdist, {}),
    }

    for panel_name, (yax_key, is_visible, extra) in panel_configs.items():
        domain = domains.get(panel_name, [0, 0])
        is_bottom = (bottom_panel == panel_name)
        axis_dict = dict(
            domain=domain,
            anchor="x" if is_bottom else "free",
            position=0,
            **grid_style,
            **(spike_y if is_visible else {}),
            visible=is_visible,
            **extra,
        )
        # yaxis2 -> layout key "yaxis2"
        layout_kwargs[f"yaxis{yax_key[1:]}"] = axis_dict

    # Annotations for visible panels
    chart_annotations = [
        dict(text="Price (High\u2013Low)", x=0.02, y=price_domain[1] + 0.005,
             xref="paper", yref="paper", showarrow=False,
             font=dict(size=13, color="lightgray"), xanchor="left", yanchor="bottom"),
    ]
    panel_labels = {
        "rsi": "RSI", "cmb": "CMB Composite", "stoch": "Stochastic",
        "adx": "ADX", "atr": "ATR", "macd": "MACD",
        "obv": "OBV", "accdist": "Acc/Dist",
    }
    for panel_name in visible_panels:
        domain = domains[panel_name]
        chart_annotations.append(
            dict(text=panel_labels.get(panel_name, panel_name), x=0.02, y=domain[1] + 0.005,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=13, color="lightgray"), xanchor="left", yanchor="bottom"))
    layout_kwargs["annotations"] = chart_annotations

    if draw_mode:
        layout_kwargs["dragmode"] = "drawrect"
        layout_kwargs["newshape"] = dict(
            line=dict(width=0),
            fillcolor="rgba(255, 255, 255, 0.07)",
            layer="above",
        )

    fig.update_layout(**layout_kwargs)

    # -------------------------------------------------
    # Selected period markers
    # -------------------------------------------------
    if period_start is not None and period_end is not None:
        x_start = period_start.strftime("%d.%m.%Y_%H:%M")
        x_end = period_end.strftime("%d.%m.%Y_%H:%M")

        fig.add_shape(type="line", x0=x_start, x1=x_start, y0=0, y1=1,
                      xref="x", yref="paper", line=dict(dash="dash", color="white", width=2))
        fig.add_shape(type="line", x0=x_end, x1=x_end, y0=0, y1=1,
                      xref="x", yref="paper", line=dict(dash="dash", color="white", width=2))

    return fig


def _render_persistent_chart(fig, config, chart_key, height, draw_mode):
    """Render a Plotly chart with shape and zoom persistence across reruns."""

    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=config,
    )

    safe_key = (chart_key or "default").replace("'", "\\'")

    # Build axis persistence for all y-axes (y through y9)
    axis_keys = ["yR"] + [f"y{i}R" for i in range(2, 10)]
    axis_names = ["yaxis"] + [f"yaxis{i}" for i in range(2, 10)]

    save_lines = []
    restore_lines = []
    autorange_lines = []
    for ak, an in zip(axis_keys, axis_names):
        restore_lines.append(f"if (saved.{ak}) {{ upd['{an}.range'] = saved.{ak}; upd['{an}.autorange'] = false; }}")
        save_lines.append(
            f"if (ed['{an}.range[0]'] !== undefined) s.{ak} = [ed['{an}.range[0]'], ed['{an}.range[1]']];"
        )
        autorange_lines.append(f"if (ed['{an}.autorange']) delete s.{ak};")

    restore_js = "\n                ".join(restore_lines)
    save_js = "\n                ".join(save_lines)
    autorange_js = "\n                ".join(autorange_lines)

    persistence_js = f"""
    <script>
    (function() {{
        var KEY = '{safe_key}';
        var P  = window.parent;
        P.__plotlyPersist = P.__plotlyPersist || {{}};

        function setup() {{
            var pd = document.querySelector('.js-plotly-plot');
            if (!pd || !pd._fullLayout) {{ setTimeout(setup, 100); return; }}

            var baseShapes = JSON.parse(JSON.stringify(pd.layout.shapes || []));

            var saved = P.__plotlyPersist[KEY];
            if (saved) {{
                var upd = {{}};
                if (saved.shapes && saved.shapes.length)
                    upd.shapes = baseShapes.concat(saved.shapes);
                if (saved.xR) {{
                    upd['xaxis.range'] = saved.xR;
                    upd['xaxis.autorange'] = false;
                }}
                {restore_js}
                if (Object.keys(upd).length) Plotly.relayout(pd, upd);
            }}

            pd.on('plotly_relayout', function(ed) {{
                var s = P.__plotlyPersist[KEY] || {{}};

                var all = pd.layout.shapes || [];
                s.shapes = [];
                for (var i = 0; i < all.length; i++) {{
                    // Only persist user-drawn shading rectangles (not RSI/indicator zone bands)
                    if (all[i].type === 'rect' && all[i].fillcolor && all[i].fillcolor.indexOf('255, 255, 255') !== -1)
                        s.shapes.push(JSON.parse(JSON.stringify(all[i])));
                }}

                if (ed['xaxis.range[0]'] !== undefined) s.xR = [ed['xaxis.range[0]'], ed['xaxis.range[1]']];
                {save_js}

                if (ed['xaxis.autorange']) delete s.xR;
                {autorange_js}

                P.__plotlyPersist[KEY] = s;
            }});

            // Delete/Backspace removes the currently selected shape
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Delete' || e.key === 'Backspace') {{
                    var idx = pd._fullLayout._activeShapeIndex;
                    if (idx >= 0 && idx !== undefined) {{
                        var shapes = JSON.parse(JSON.stringify(pd.layout.shapes || []));
                        shapes.splice(idx, 1);
                        Plotly.relayout(pd, {{shapes: shapes}});
                        e.preventDefault();
                    }}
                }}
            }});
        }}
        setup();
    }})();
    </script>
    """

    cursor_css = ".shapelayer path { cursor: pointer !important; }" if draw_mode else ""

    full_html = (
        "<!DOCTYPE html><html><head>"
        "<style>body{margin:0;padding:0;background:#111111;overflow:hidden;}"
        f"{cursor_css}</style></head>"
        f"<body>{chart_html}{persistence_js}</body></html>"
    )

    _components.html(full_html, height=height, scrolling=False)


def render_charts(
    df_slice_1h,
    df_slice_15m,
    start_1h,
    end_1h,
    start_15m,
    end_15m,
    show_ichimoku,
    show_bb,
    show_kc,
    show_strategy,
    rsi_zones_1h=None,
    rsi_zones_15m=None,
    show_1h=True,
    chart_key=None,
    draw_mode=False,
    chart_height=920,
    show_rsi=True,
    show_cmb=True,
    show_stoch=True,
    show_adx=False,
    show_atr=False,
    show_macd=False,
    show_obv=False,
    show_accdist=False,
    show_supertrend=False,
    show_ema=False,
    ema_periods=None,
    show_donchian=False,
    show_psar=False,
):
    """Renders charts."""
    rsi_zones_1h = rsi_zones_1h or {}
    rsi_zones_15m = rsi_zones_15m or {}

    panel_kwargs = dict(
        show_rsi=show_rsi, show_cmb=show_cmb, show_stoch=show_stoch,
        show_adx=show_adx, show_atr=show_atr, show_macd=show_macd,
        show_obv=show_obv, show_accdist=show_accdist,
        show_supertrend=show_supertrend, show_ema=show_ema,
        show_donchian=show_donchian, show_psar=show_psar,
    )

    config = {"scrollZoom": True}
    if draw_mode:
        config["modeBarButtonsToAdd"] = ["drawrect", "eraseshape"]

    if show_1h:
        st.subheader("1H Chart")
        fig_1h = build_main_chart(
            df_slice=df_slice_1h, period_start=start_1h, period_end=end_1h,
            show_ichimoku=show_ichimoku, show_bb=show_bb, show_kc=show_kc,
            show_strategy=show_strategy, draw_mode=draw_mode,
            chart_height=chart_height, chart_key=chart_key,
            **panel_kwargs, **(rsi_zones_1h or {}),
        )
        _render_persistent_chart(fig_1h, config,
                                 f"1h_{chart_key}" if chart_key else "1h",
                                 chart_height, draw_mode)
    else:
        st.subheader("15m Chart")
        fig_15m = build_main_chart(
            df_slice=df_slice_15m, period_start=start_15m, period_end=end_15m,
            show_ichimoku=show_ichimoku, show_bb=show_bb, show_kc=show_kc,
            show_strategy=show_strategy, draw_mode=draw_mode,
            chart_height=chart_height, chart_key=chart_key,
            **panel_kwargs, **(rsi_zones_15m or {}),
        )
        _render_persistent_chart(fig_15m, config,
                                 f"15m_{chart_key}" if chart_key else "15m",
                                 chart_height, draw_mode)
