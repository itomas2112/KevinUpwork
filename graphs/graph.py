import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components


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
):
    """
    Build main chart using a single x-axis with 4 y-axis domains
    so the vertical crosshair spans all panels.

    Panels (top to bottom): Price, RSI, CMB Composite, Stochastic
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
    # Y-axis domains (bottom to top)
    # -------------------------------------------------
    stoch_domain = [0.00, 0.13]
    cmb_domain   = [0.17, 0.36]
    rsi_domain   = [0.40, 0.55]
    price_domain = [0.59, 1.00]

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

        # Offset to push markers away from price bars for visibility
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
            # Target exits — green
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

            # Dynamic stop exits (exit group stops) — orange
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

            # Static/Initial stop exits — red
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
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["tenkan"],
                    name="Tenkan",
                    line=dict(color="red", width=1),
                    showlegend=False,
                )
            )

        if ichi_show_kijun:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["kijun"],
                    name="Kijun",
                    line=dict(color="lightblue", width=1),
                    showlegend=False,
                )
            )

        if ichi_show_senkou_a:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["senkou_a"],
                    name="Senkou A",
                    line=dict(color="yellow", width=1),
                    showlegend=False,
                )
            )

        if ichi_show_senkou_b:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["senkou_b"],
                    name="Senkou B",
                    line=dict(color="green", width=1),
                    showlegend=False,
                )
            )

        if ichi_show_chikou and "chikou" in df_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["chikou"],
                    name="Chikou",
                    line=dict(color="#9932CC", width=1.5),
                    showlegend=False,
                )
            )

        # Decision lines (un-displaced) — independent of original line toggles
        if ichi_show_senkou_a_current and "senkou_a_current" in df_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["senkou_a_current"],
                    name="Senkou A (current)",
                    line=dict(color="yellow", width=1, dash="dot"),
                    showlegend=False,
                )
            )

        if ichi_show_senkou_b_current and "senkou_b_current" in df_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["senkou_b_current"],
                    name="Senkou B (current)",
                    line=dict(color="green", width=1, dash="dot"),
                    showlegend=False,
                )
            )

        if ichi_show_chikou_decision:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["latest"],
                    name="Chikou (decision)",
                    line=dict(color="#9932CC", width=1, dash="dot"),
                    showlegend=False,
                )
            )

    # -------------------------------------------------
    # Bollinger Bands  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_bb:
        if bb_show_middle:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["bb_mid"],
                    name="BB Mid",
                    line=dict(color="gray", width=1, dash="dot"),
                    showlegend=False,
                )
            )

        if bb_show_upper:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["bb_upper"],
                    name="BB Upper",
                    line=dict(color="gray", width=1),
                    showlegend=False,
                )
            )

        if bb_show_lower:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["bb_lower"],
                    name="BB Lower",
                    line=dict(color="gray", width=1),
                    showlegend=False,
                )
            )

    # -------------------------------------------------
    # Keltner Channel  (yaxis="y" — Price panel)
    # -------------------------------------------------
    if show_kc:
        if kc_show_middle:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["kc_mid"],
                    name="KC Mid",
                    line=dict(color="orange", width=1, dash="dot"),
                    showlegend=False,
                )
            )

        if kc_show_upper:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["kc_upper"],
                    name="KC Upper",
                    line=dict(color="orange", width=1),
                    showlegend=False,
                )
            )

        if kc_show_lower:
            fig.add_trace(
                go.Scatter(
                    x=df_slice["x"],
                    y=df_slice["kc_lower"],
                    name="KC Lower",
                    line=dict(color="orange", width=1),
                    showlegend=False,
                )
            )

    # -------------------------------------------------
    # RSI  (yaxis="y2")
    # -------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["rsi"],
            name="RSI",
            line=dict(color="#9932CC", width=2),
            showlegend=False,
            yaxis="y2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["rsi_13"],
            name="RSI 13 SMA",
            line=dict(color="lightblue", width=1),
            showlegend=False,
            yaxis="y2",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["rsi_33"],
            name="RSI 33 SMA",
            line=dict(color="yellow", width=1),
            showlegend=False,
            yaxis="y2",
            hoverinfo="skip",
        )
    )

    # RSI zones (shaded)
    fig.add_shape(
        type="rect",
        x0=0, x1=1, y0=rsi_upper_2, y1=rsi_upper_1,
        xref="paper", yref="y2",
        fillcolor="red", opacity=0.3, line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=0, x1=1, y0=rsi_lower_2, y1=rsi_lower_1,
        xref="paper", yref="y2",
        fillcolor="blue", opacity=0.3, line_width=0,
    )

    # RSI horizontal reference lines: 50 + 4 zone boundaries
    fig.add_shape(
        type="line", x0=0, x1=1, y0=50, y1=50,
        xref="paper", yref="y2",
        line=dict(dash="solid", color="gray", width=1),
    )
    for rsi_val in [rsi_upper_1, rsi_upper_2, rsi_lower_1, rsi_lower_2]:
        fig.add_shape(
            type="line", x0=0, x1=1, y0=rsi_val, y1=rsi_val,
            xref="paper", yref="y2",
            line=dict(dash="dot", color="gray", width=1),
        )

    # -------------------------------------------------
    # CMB Composite  (yaxis="y3")
    # -------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci"],
            name="CI",
            line=dict(color="#9932CC", width=2),
            showlegend=False,
            yaxis="y3",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci_13"],
            name="CI 13",
            line=dict(color="lightblue", width=1),
            showlegend=False,
            yaxis="y3",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["ci_33"],
            name="CI 33",
            line=dict(color="yellow", width=1),
            showlegend=False,
            yaxis="y3",
            hoverinfo="skip",
        )
    )

    # CMB horizontal reference lines
    if cmb_lines:
        for val in cmb_lines:
            fig.add_shape(
                type="line",
                x0=0, x1=1, y0=val, y1=val,
                xref="paper", yref="y3",
                line=dict(dash="dot", color="white", width=1),
            )

    # -------------------------------------------------
    # Stochastic  (yaxis="y4")
    # -------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["stoch_k"],
            name="%K",
            line=dict(color="white", width=1.5),
            showlegend=False,
            yaxis="y4",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_slice["x"],
            y=df_slice["stoch_d"],
            name="%D",
            line=dict(color="red", width=1.5),
            showlegend=False,
            yaxis="y4",
        )
    )

    # Stochastic horizontal reference lines
    fig.add_shape(type="line", x0=0, x1=1, y0=80, y1=80,
                  xref="paper", yref="y4",
                  line=dict(dash="solid", color="gray", width=1))
    fig.add_shape(type="line", x0=0, x1=1, y0=50, y1=50,
                  xref="paper", yref="y4",
                  line=dict(dash="dot", color="gray", width=1))
    fig.add_shape(type="line", x0=0, x1=1, y0=20, y1=20,
                  xref="paper", yref="y4",
                  line=dict(dash="solid", color="gray", width=1))

    # -------------------------------------------------
    # Hover & Crosshair
    # -------------------------------------------------
    # Price panel: no hover tooltip (just crosshair).
    # Indicator panels: show name + value.
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

    # Y-axis spike options (horizontal crosshair per panel)
    spike_y = dict(
        showspikes=True,
        spikemode="across+toaxis",
        spikethickness=1,
        spikedash="dot",
        spikecolor="gray",
        spikesnap="cursor",
    )
    # Price panel spike: no toaxis label (don't show price value at axis)
    spike_y_no_label = dict(
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="gray",
        spikesnap="cursor",
    )

    # -------------------------------------------------
    # Layout — single x-axis, 4 y-axis domains
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
            anchor="y4",
            domain=[0, 1],
            type="category",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showspikes=True,
            spikemode="across+toaxis",
            spikethickness=1,
            spikedash="dot",
            spikecolor="gray",
            spikesnap="cursor",
            gridcolor="rgba(255,255,255,0.08)",
            griddash="dash",
        ),
        yaxis=dict(domain=price_domain, anchor="free", position=0,
                   gridcolor="rgba(255,255,255,0.08)", griddash="dash",
                   **spike_y_no_label),
        yaxis2=dict(domain=rsi_domain, anchor="free", position=0,
                    tickmode="array",
                    tickvals=sorted([rsi_lower_2, rsi_lower_1, 50, rsi_upper_2, rsi_upper_1]),
                    gridcolor="rgba(255,255,255,0.08)", griddash="dash",
                    **spike_y),
        yaxis3=dict(domain=cmb_domain, anchor="free", position=0,
                    gridcolor="rgba(255,255,255,0.08)", griddash="dash",
                    **spike_y),
        yaxis4=dict(domain=stoch_domain, anchor="x",
                    gridcolor="rgba(255,255,255,0.08)", griddash="dash",
                    **spike_y),
        annotations=[
            dict(text="Price (High\u2013Low)", x=0.02, y=price_domain[1] + 0.005,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=13, color="lightgray"),
                 xanchor="left", yanchor="bottom"),
            dict(text="RSI", x=0.02, y=rsi_domain[1] + 0.005,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=13, color="lightgray"),
                 xanchor="left", yanchor="bottom"),
            dict(text="CMB Composite", x=0.02, y=cmb_domain[1] + 0.005,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=13, color="lightgray"),
                 xanchor="left", yanchor="bottom"),
            dict(text="Stochastic", x=0.02, y=stoch_domain[1] + 0.005,
                 xref="paper", yref="paper", showarrow=False,
                 font=dict(size=13, color="lightgray"),
                 xanchor="left", yanchor="bottom"),
        ],
    )

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

        fig.add_shape(
            type="line",
            x0=x_start, x1=x_start, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dash", color="white", width=2),
        )

        fig.add_shape(
            type="line",
            x0=x_end, x1=x_end, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dash", color="white", width=2),
        )

    return fig


def _render_persistent_chart(fig, config, chart_key, height, draw_mode):
    """Render a Plotly chart with shape and zoom persistence across reruns.

    Uses ``st.components.v1.html`` so the chart lives inside an iframe where
    we can attach Plotly JS event listeners.  Drawn shapes (rects) and axis
    ranges are stored on ``window.parent`` which survives Streamlit reruns.
    """

    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=config,
    )

    safe_key = (chart_key or "default").replace("'", "\\'")

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
                if (saved.xR)  upd['xaxis.range']   = saved.xR;
                if (saved.yR)  upd['yaxis.range']    = saved.yR;
                if (saved.y2R) upd['yaxis2.range']   = saved.y2R;
                if (saved.y3R) upd['yaxis3.range']   = saved.y3R;
                if (saved.y4R) upd['yaxis4.range']   = saved.y4R;
                if (Object.keys(upd).length) Plotly.relayout(pd, upd);
            }}

            pd.on('plotly_relayout', function(ed) {{
                var s = P.__plotlyPersist[KEY] || {{}};

                var all = pd.layout.shapes || [];
                s.shapes = [];
                for (var i = 0; i < all.length; i++) {{
                    if (all[i].type === 'rect')
                        s.shapes.push(JSON.parse(JSON.stringify(all[i])));
                }}

                if (ed['xaxis.range[0]']  !== undefined) s.xR  = [ed['xaxis.range[0]'],  ed['xaxis.range[1]']];
                if (ed['yaxis.range[0]']  !== undefined) s.yR  = [ed['yaxis.range[0]'],  ed['yaxis.range[1]']];
                if (ed['yaxis2.range[0]'] !== undefined) s.y2R = [ed['yaxis2.range[0]'], ed['yaxis2.range[1]']];
                if (ed['yaxis3.range[0]'] !== undefined) s.y3R = [ed['yaxis3.range[0]'], ed['yaxis3.range[1]']];
                if (ed['yaxis4.range[0]'] !== undefined) s.y4R = [ed['yaxis4.range[0]'], ed['yaxis4.range[1]']];

                if (ed['xaxis.autorange'])  delete s.xR;
                if (ed['yaxis.autorange'])  delete s.yR;
                if (ed['yaxis2.autorange']) delete s.y2R;
                if (ed['yaxis3.autorange']) delete s.y3R;
                if (ed['yaxis4.autorange']) delete s.y4R;

                P.__plotlyPersist[KEY] = s;
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
):
    """
    Renders charts. Side by side when 1H+15m, full width when 15m only.
    """
    rsi_zones_1h = rsi_zones_1h or {}
    rsi_zones_15m = rsi_zones_15m or {}

    # Plotly config: always enable scroll zoom for y-axis stretching
    config = {"scrollZoom": True}
    if draw_mode:
        config["modeBarButtonsToAdd"] = [
            "drawrect",
            "eraseshape",
        ]

    if show_1h:
        st.subheader("1H Chart")
        fig_1h = build_main_chart(
            df_slice=df_slice_1h,
            period_start=start_1h,
            period_end=end_1h,
            show_ichimoku=show_ichimoku,
            show_bb=show_bb,
            show_kc=show_kc,
            show_strategy=show_strategy,
            draw_mode=draw_mode,
            chart_height=chart_height,
            chart_key=chart_key,
            **(rsi_zones_1h or {}),
        )
        _render_persistent_chart(fig_1h, config,
                                 f"1h_{chart_key}" if chart_key else "1h",
                                 chart_height, draw_mode)
    else:
        st.subheader("15m Chart")
        fig_15m = build_main_chart(
            df_slice=df_slice_15m,
            period_start=start_15m,
            period_end=end_15m,
            show_ichimoku=show_ichimoku,
            show_bb=show_bb,
            show_kc=show_kc,
            show_strategy=show_strategy,
            draw_mode=draw_mode,
            chart_height=chart_height,
            chart_key=chart_key,
            **(rsi_zones_15m or {}),
        )
        _render_persistent_chart(fig_15m, config,
                                 f"15m_{chart_key}" if chart_key else "15m",
                                 chart_height, draw_mode)
