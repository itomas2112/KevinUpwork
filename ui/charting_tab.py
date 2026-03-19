"""
Charting tab (Tab 1) UI and logic
"""
import streamlit as st
import streamlit.components.v1 as _components
from data.loader import load_ohlc, load_drm, parse_drm_periods
from data.helpers import expand_selection
from indicators.calculate_indicators import calculate_indicators, slice_for_graph, migrate_indicator_settings, changed_groups, recalculate_groups
from graphs.graph import render_charts
from strategies.first_strategy import ichimoku_tenkan_kijun_strategy, execute_custom_strategy
import pandas as pd
import json


def _backtest_fingerprint(strategy, indicator_params, **overlay_flags):
    """Build a hashable fingerprint for backtest cache invalidation."""
    strat_str = json.dumps(strategy, sort_keys=True, default=str) if strategy else ""
    params_str = json.dumps(indicator_params, sort_keys=True, default=str)
    flags = tuple(sorted(overlay_flags.items()))
    return hash((strat_str, params_str, flags))


def _get_or_calculate(raw_df_key, features_key, params_key, current_params,
                      global_start_date=None, global_end_date=None):
    """
    Return cached df_features from session_state if params and raw data are unchanged.
    If only some indicator params changed, incrementally recalculate just those indicators.
    Otherwise do a full recalculation.

    global_start_date / global_end_date trim the raw data before any calculation.
    """
    raw_df = st.session_state[raw_df_key]

    # Apply global date range filter
    if global_start_date is not None:
        raw_df = raw_df[raw_df.index >= pd.Timestamp(global_start_date)]
    if global_end_date is not None:
        raw_df = raw_df[raw_df.index < pd.Timestamp(global_end_date) + pd.Timedelta(days=1)]

    if raw_df.empty:
        return raw_df

    data_fingerprint = (len(raw_df), raw_df.index[0], raw_df.index[-1])

    stored_params = st.session_state.get(params_key)
    stored_fingerprint = st.session_state.get(f"{params_key}_data_fp")
    stored_features = st.session_state.get(features_key)

    # Exact match — return cached
    if (stored_features is not None
            and stored_params == current_params
            and stored_fingerprint == data_fingerprint):
        return stored_features

    # If raw data is the same but only some indicator params changed,
    # do an incremental recalculation on the existing DataFrame.
    if (stored_features is not None
            and stored_params is not None
            and stored_fingerprint == data_fingerprint):
        groups = changed_groups(stored_params, current_params)
        if groups:
            df_features = stored_features.copy()
            recalculate_groups(df_features, groups, **current_params)
            st.session_state[features_key] = df_features
            st.session_state[params_key] = current_params.copy()
            return df_features

    # Full recalculation (new data or first run)
    df_features = calculate_indicators(df=raw_df, **current_params)
    st.session_state[features_key] = df_features
    st.session_state[params_key] = current_params.copy()
    st.session_state[f"{params_key}_data_fp"] = data_fingerprint
    return df_features


def _build_charting_fingerprint(sidebar_config, show_1h):
    """Build a fingerprint of all inputs that affect charting output.

    When this fingerprint differs from the last calculated one, the user
    sees a sticky 'Recalculate' bar at the bottom of the viewport.
    """
    display_only_keys = {'rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2',
                         'cmb_lines',
                         'ichi_show_tenkan', 'ichi_show_kijun', 'ichi_show_senkou_a',
                         'ichi_show_senkou_b', 'ichi_show_chikou',
                         'ichi_show_senkou_a_current', 'ichi_show_senkou_b_current',
                         'ichi_show_chikou_decision',
                         'bb_show_upper', 'bb_show_middle', 'bb_show_lower',
                         'kc_show_upper', 'kc_show_middle', 'kc_show_lower',
                         'dc_show_upper', 'dc_show_middle', 'dc_show_lower'}

    params_key = 'params_1h' if show_1h else 'params_15m'
    indicator_params = {k: v for k, v in sidebar_config[params_key].items() if k not in display_only_keys}

    # Include strategy settings
    strategy_settings = None
    if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
        actual_idx = st.session_state.get('selected_custom_strategy_actual_idx')
        if actual_idx is not None and actual_idx < len(st.session_state['saved_strategies']):
            strategy_settings = st.session_state['saved_strategies'][actual_idx].get('indicator_settings')
            if strategy_settings:
                strategy_settings = migrate_indicator_settings(strategy_settings)

    # Include combos, strategy, overlays, display params, date range
    charting_selections = sidebar_config['charting_selections']
    combos_str = json.dumps(charting_selections, sort_keys=True, default=str)
    params_str = json.dumps(indicator_params, sort_keys=True, default=str)
    strat_idx = st.session_state.get('selected_custom_strategy_idx', 0)
    strat_actual = st.session_state.get('selected_custom_strategy_actual_idx')
    strat_settings_str = json.dumps(strategy_settings, sort_keys=True, default=str) if strategy_settings else ""

    # Include display-only keys (RSI zones, visibility toggles) so toggling
    # a line also triggers the recalculate bar
    display_params = {k: sidebar_config[params_key].get(k) for k in display_only_keys
                      if k in sidebar_config[params_key]}
    display_str = json.dumps(display_params, sort_keys=True, default=str)

    overlay_keys = ('show_ichimoku', 'show_bb', 'show_kc', 'show_donchian', 'show_psar',
                    'show_rsi', 'show_cmb', 'show_stoch', 'show_adx', 'show_atr',
                    'show_macd', 'show_obv', 'show_accdist', 'show_supertrend', 'show_ema')
    overlays = tuple((k, sidebar_config.get(k)) for k in overlay_keys)
    g_start = sidebar_config.get('global_start_date')
    g_end = sidebar_config.get('global_end_date')
    chart_height = sidebar_config.get('chart_height')
    draw_mode = sidebar_config.get('draw_mode', False)

    return hash((combos_str, params_str, strat_idx, strat_actual, strat_settings_str,
                 display_str, overlays, str(g_start), str(g_end), chart_height, draw_mode))


def render_charting_tab(sidebar_config):
    """Render the charting tab content"""

    show_1h = sidebar_config['analysis_mode'] == "1H"

    # File uploaders
    render_file_uploaders(show_1h)

    # Check if data is loaded
    if not check_data_loaded(show_1h):
        return

    # Require date range before proceeding
    if not sidebar_config.get('date_range_applied', False):
        st.info("Please set a date range in the sidebar and click **Apply Training Set** to proceed.")
        return

    # Expand all charting selections into (pattern_type, primary, secondary) combos
    charting_selections = sidebar_config['charting_selections']
    all_combos = []
    seen_combos = set()
    for sel in charting_selections:
        for combo in expand_selection(sel):
            if combo not in seen_combos:
                seen_combos.add(combo)
                all_combos.append(combo)

    if not all_combos:
        st.info("Please configure pattern selections in the sidebar to display charts.")
        return

    # ── Recalculate gate ─────────────────────────────────────────────
    # Build a fingerprint of the current inputs.  When it differs from
    # the last-calculated fingerprint, show a sticky "Recalculate" bar
    # instead of re-running the expensive pipeline automatically.
    current_fp = _build_charting_fingerprint(sidebar_config, show_1h)
    last_calc_fp = st.session_state.get('_charting_calc_fp')

    calculate_clicked = st.button("Calculate", key="charting_calculate", type="primary")

    # First time — nothing cached yet, need an initial calculate
    if last_calc_fp is None and not calculate_clicked:
        st.info("Click **Calculate** to render charts.")
        return

    params_changed = (last_calc_fp is not None and current_fp != last_calc_fp)

    if params_changed and not calculate_clicked:
        # Show cached results (below) but also show sticky recalculate bar
        _inject_sticky_recalculate_bar()

    if not calculate_clicked and last_calc_fp is not None:
        # Show cached charting results without recalculating
        cached = st.session_state.get('_charting_cached_output')
        if cached:
            _display_cached_output(cached, sidebar_config, show_1h)
            return

    # ── Expensive pipeline (only runs on Calculate click) ────────────

    # Clear chart HTML cache for fresh rebuild
    st.session_state['_charting_html_cache'] = {}

    # Determine if custom strategy is selected (before indicator calc for overrides)
    show_custom_strategy = False
    selected_custom_strategy = None
    strategy_indicator_settings = None

    if st.session_state.get('selected_custom_strategy_idx', 0) > 0:
        actual_idx = st.session_state.get('selected_custom_strategy_actual_idx')
        if actual_idx is not None and actual_idx < len(st.session_state['saved_strategies']):
            show_custom_strategy = True
            selected_custom_strategy = st.session_state['saved_strategies'][actual_idx]
            strategy_indicator_settings = selected_custom_strategy.get('indicator_settings')
            if strategy_indicator_settings:
                strategy_indicator_settings = migrate_indicator_settings(strategy_indicator_settings)

    # Calculate indicators (exclude display-only params: RSI zones and CMB lines)
    # Results are stored in session_state and only recalculated when params or data change.
    display_only_keys = {'rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2',
                         'cmb_lines',
                         'ichi_show_tenkan', 'ichi_show_kijun', 'ichi_show_senkou_a',
                         'ichi_show_senkou_b', 'ichi_show_chikou',
                         'ichi_show_senkou_a_current', 'ichi_show_senkou_b_current',
                         'ichi_show_chikou_decision',
                         'bb_show_upper', 'bb_show_middle', 'bb_show_lower',
                         'kc_show_upper', 'kc_show_middle', 'kc_show_lower',
                         'dc_show_upper', 'dc_show_middle', 'dc_show_lower'}

    df_features_1h = None
    df_features_15m = None
    indicator_params_1h = None
    indicator_params_15m = None
    g_start = sidebar_config.get('global_start_date')
    g_end = sidebar_config.get('global_end_date')
    if show_1h:
        indicator_params_1h = {k: v for k, v in sidebar_config['params_1h'].items() if k not in display_only_keys}
        if strategy_indicator_settings:
            indicator_params_1h.update(strategy_indicator_settings)
        df_features_1h = _get_or_calculate(
            "df_1h", "df_features_1h", "_indicator_params_1h", indicator_params_1h,
            global_start_date=g_start, global_end_date=g_end,
        )
    else:
        indicator_params_15m = {k: v for k, v in sidebar_config['params_15m'].items() if k not in display_only_keys}
        if strategy_indicator_settings:
            indicator_params_15m.update(strategy_indicator_settings)
        df_features_15m = _get_or_calculate(
            "df_15m", "df_features_15m", "_indicator_params_15m", indicator_params_15m,
            global_start_date=g_start, global_end_date=g_end,
        )

    # Collect stats from all periods for global aggregation
    all_stats_1h = []
    all_stats_15m = []
    strategy_label = None
    total_periods = 0

    # Build backtest cache fingerprint — invalidate when strategy or params change
    active_indicator_params = indicator_params_1h if show_1h else indicator_params_15m
    bt_fingerprint = _backtest_fingerprint(
        selected_custom_strategy, active_indicator_params,
        show_ichimoku=sidebar_config['show_ichimoku'],
        show_bb=sidebar_config['show_bb'],
        show_kc=sidebar_config['show_kc'],
        show_donchian=sidebar_config.get('show_donchian', False),
        show_psar=sidebar_config.get('show_psar', False),
    )
    # Invalidate cache if fingerprint changed
    if st.session_state.get('_bt_fingerprint') != bt_fingerprint:
        st.session_state['_bt_cache'] = {}
        st.session_state['_bt_fingerprint'] = bt_fingerprint

    # Reserve a container at the top for the global performance summary
    global_perf_container = st.container()

    # Get DRM data
    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')

    # Collect rendered output for caching
    cached_output = {
        'combo_sections': [],
        'all_stats_1h': [],
        'all_stats_15m': [],
        'strategy_label': None,
        'total_periods': 0,
    }

    # Render periods for each combo
    for pattern_type, primary, secondary in all_combos:
        drm_df = drm_bullish if pattern_type == 'Bullish' else drm_bearish
        if drm_df is None:
            continue

        drm_periods = parse_drm_periods(drm_df, pattern_type, primary, secondary)
        if not drm_periods:
            continue

        # Sort periods newest first
        drm_periods.sort(key=lambda p: p[0], reverse=True)

        st.markdown(f"## {pattern_type} — {primary} → {secondary}")

        section = {
            'pattern_type': pattern_type,
            'primary': primary,
            'secondary': secondary,
            'periods': [],
        }

        for i, (start_dt, end_dt) in enumerate(drm_periods, start=1):
            chart_key = f"{pattern_type}_{primary}_{secondary}_{i}"
            stats_1h, stats_15m = render_period(
                i, start_dt, end_dt,
                df_features_1h, df_features_15m,
                sidebar_config,
                show_custom_strategy,
                selected_custom_strategy,
                show_1h,
                chart_key=chart_key,
            )

            section['periods'].append({
                'start_dt': start_dt,
                'end_dt': end_dt,
                'chart_key': chart_key,
            })

            if stats_1h is not None:
                all_stats_1h.append(stats_1h)
            if stats_15m is not None:
                all_stats_15m.append(stats_15m)

        cached_output['combo_sections'].append(section)
        total_periods += len(drm_periods)

    if show_custom_strategy and selected_custom_strategy is not None:
        strategy_label = selected_custom_strategy.get('strategy_name', 'Custom Strategy')

    # Render global performance summary at the top
    has_stats = bool(all_stats_1h) if show_1h else bool(all_stats_15m)
    if has_stats:
        with global_perf_container:
            render_global_performance(all_stats_1h, all_stats_15m, strategy_label, total_periods, show_1h)

    # Cache the output and fingerprint
    cached_output['all_stats_1h'] = all_stats_1h
    cached_output['all_stats_15m'] = all_stats_15m
    cached_output['strategy_label'] = strategy_label
    cached_output['total_periods'] = total_periods
    cached_output['show_custom_strategy'] = show_custom_strategy
    cached_output['selected_custom_strategy'] = selected_custom_strategy
    cached_output['chart_height'] = sidebar_config.get('chart_height', 920)
    st.session_state['_charting_cached_output'] = cached_output
    st.session_state['_charting_calc_fp'] = current_fp


def _display_cached_output(cached, sidebar_config, show_1h):
    """Display charts from cached HTML strings — no chart construction or data processing."""
    all_stats_1h = cached.get('all_stats_1h', [])
    all_stats_15m = cached.get('all_stats_15m', [])
    strategy_label = cached.get('strategy_label')
    total_periods = cached.get('total_periods', 0)
    show_custom_strategy = cached.get('show_custom_strategy', False)
    chart_height = cached.get('chart_height', 920)

    chart_html_cache = st.session_state.get('_charting_html_cache', {})

    # Reserve a container at the top for the global performance summary
    global_perf_container = st.container()

    for section in cached.get('combo_sections', []):
        pattern_type = section['pattern_type']
        primary = section['primary']
        secondary = section['secondary']

        st.markdown(f"## {pattern_type} — {primary} → {secondary}")

        for idx, period_info in enumerate(section['periods']):
            chart_key = period_info['chart_key']
            start_dt = period_info['start_dt']
            end_dt = period_info['end_dt']
            period_num = idx + 1

            st.markdown(f"### Period {period_num}: {start_dt} → {end_dt}")

            cached_entry = chart_html_cache.get(chart_key)
            if cached_entry:
                html_str = cached_entry['html']
                stats_data = cached_entry.get('stats')
                strat_label = cached_entry.get('strategy_label')

                if stats_data is not None:
                    col_charts, col_stats = st.columns([3, 1], gap="medium")
                    with col_charts:
                        _components.html(html_str, height=chart_height, scrolling=False)
                    with col_stats:
                        render_strategy_stats(
                            stats_data if show_1h else None,
                            stats_data if not show_1h else None,
                            strat_label, show_1h)
                else:
                    _components.html(html_str, height=chart_height, scrolling=False)
            else:
                st.info("Click **Calculate** to render this chart.")

            st.divider()

    # Render global performance summary
    has_stats = bool(all_stats_1h) if show_1h else bool(all_stats_15m)
    if has_stats:
        with global_perf_container:
            render_global_performance(all_stats_1h, all_stats_15m, strategy_label, total_periods, show_1h)


def _inject_sticky_recalculate_bar():
    """Inject CSS + HTML for a fixed bar at the bottom of the viewport."""
    st.markdown("""
    <style>
    .sticky-recalc-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-top: 2px solid #e67e22;
        padding: 12px 24px;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.4);
    }
    .sticky-recalc-bar span {
        color: #e67e22;
        font-weight: 600;
        font-size: 0.95rem;
    }
    </style>
    <div class="sticky-recalc-bar">
        <span>⚠ Parameters changed — click Calculate above to update charts</span>
    </div>
    """, unsafe_allow_html=True)


def render_file_uploaders(show_1h=False):
    """Render file upload section"""
    col_data, col_drm = st.columns([1, 1], gap="small")

    with col_data:
        if show_1h:
            uploaded_file_1h = st.file_uploader(
                "1H OHLC", type=["csv"], key="1h", label_visibility="collapsed"
            )
            st.caption("1H OHLC (.csv)")

            if uploaded_file_1h is not None:
                df_1h = load_ohlc(uploaded_file_1h)
                st.session_state["df_1h"] = df_1h
                st.success("1H data loaded")
        else:
            uploaded_file_15m = st.file_uploader(
                "15m OHLC", type=["csv"], key="15m", label_visibility="collapsed"
            )
            st.caption("15m OHLC (.csv)")

            if uploaded_file_15m is not None:
                df_15m = load_ohlc(uploaded_file_15m)
                st.session_state["df_15m"] = df_15m
                st.success("15m data loaded")

    with col_drm:
        uploaded_drm = st.file_uploader(
            "Date Range Manager", type=["xlsx"], key="drm_", label_visibility="collapsed"
        )
        st.caption("DRM (.xlsx)")

        if uploaded_drm is not None:
            # Load both sheets for multi-pattern support
            try:
                st.session_state['drm_bullish'] = load_drm(uploaded_drm, 'Bullish')
            except Exception:
                st.session_state['drm_bullish'] = None
            try:
                uploaded_drm.seek(0)
                st.session_state['drm_bearish'] = load_drm(uploaded_drm, 'Bearish')
            except Exception:
                st.session_state['drm_bearish'] = None

            # Keep legacy 'drm' key pointing to bullish by default
            if st.session_state.get('drm_bullish') is not None:
                st.session_state['drm'] = st.session_state['drm_bullish']
            elif st.session_state.get('drm_bearish') is not None:
                st.session_state['drm'] = st.session_state['drm_bearish']

            st.success("Date Range Manager loaded")

def check_data_loaded(show_1h=False):
    """Check if all required data is loaded"""
    if show_1h:
        if "df_1h" not in st.session_state:
            st.info("Please upload 1H data file and DRM file.")
            return False
    else:
        if "df_15m" not in st.session_state:
            st.info("Please upload 15m data file and DRM file.")
            return False

    drm_bullish = st.session_state.get('drm_bullish')
    drm_bearish = st.session_state.get('drm_bearish')
    if drm_bullish is None and drm_bearish is None:
        st.info("Please upload a DRM file.")
        return False

    return True


def render_period(period_num, start_dt, end_dt, df_features_1h, df_features_15m,
                  sidebar_config, show_custom_strategy, selected_custom_strategy, show_1h=True,
                  chart_key=None):
    """Render a single period with charts and stats. Returns (stats_1h, stats_15m) or (None, None)."""

    st.markdown(f"### Period {period_num}: {start_dt} → {end_dt}")

    # Slice data for the active timeframe only
    df_slice_1h, period_start_1h, period_end_1h = None, None, None
    df_slice_15m, period_start_15m, period_end_15m = None, None, None
    if show_1h:
        df_slice_1h, period_start_1h, period_end_1h = slice_for_graph(
            df=df_features_1h, start_date=start_dt, end_date=end_dt,
            show_ichimoku=sidebar_config['show_ichimoku'],
            show_bb=sidebar_config['show_bb'],
            show_kc=sidebar_config['show_kc'],
            show_donchian=sidebar_config.get('show_donchian', False),
            show_psar=sidebar_config.get('show_psar', False),
        )
    else:
        df_slice_15m, period_start_15m, period_end_15m = slice_for_graph(
            df=df_features_15m, start_date=start_dt, end_date=end_dt,
            show_ichimoku=sidebar_config['show_ichimoku'],
            show_bb=sidebar_config['show_bb'],
            show_kc=sidebar_config['show_kc'],
            show_donchian=sidebar_config.get('show_donchian', False),
            show_psar=sidebar_config.get('show_psar', False),
        )

    active_slice = df_slice_1h if show_1h else df_slice_15m
    if active_slice.empty:
        st.info("No data for this period.")
        return None, None

    # Execute strategies (with caching)
    stats_1h, stats_15m = None, None
    strategy_label = None

    if show_custom_strategy and selected_custom_strategy is not None:
        bt_cache = st.session_state.get('_bt_cache', {})
        cache_hit = chart_key in bt_cache

        if cache_hit:
            # Use cached backtest results
            cached_df, cached_stats = bt_cache[chart_key]
            if show_1h:
                df_slice_1h, stats_1h = cached_df, cached_stats
            else:
                df_slice_15m, stats_15m = cached_df, cached_stats
        else:
            # Run backtest and cache results
            if show_1h:
                df_slice_1h, stats_1h = execute_custom_strategy(df_slice_1h, selected_custom_strategy, period_start_1h, period_end_1h)
                bt_cache[chart_key] = (df_slice_1h, stats_1h)
            else:
                df_slice_15m, stats_15m = execute_custom_strategy(df_slice_15m, selected_custom_strategy, period_start_15m, period_end_15m)
                bt_cache[chart_key] = (df_slice_15m, stats_15m)
            st.session_state['_bt_cache'] = bt_cache

        strategy_label = selected_custom_strategy.get('strategy_name', 'Custom Strategy')

    # RSI zone and CMB line parameters per timeframe
    chart_param_keys = ['rsi_upper_1', 'rsi_upper_2', 'rsi_lower_1', 'rsi_lower_2',
                        'cmb_lines',
                        'ichi_show_tenkan', 'ichi_show_kijun', 'ichi_show_senkou_a',
                        'ichi_show_senkou_b', 'ichi_show_chikou',
                        'ichi_show_senkou_a_current', 'ichi_show_senkou_b_current',
                        'ichi_show_chikou_decision',
                        'bb_show_upper', 'bb_show_middle', 'bb_show_lower',
                        'kc_show_upper', 'kc_show_middle', 'kc_show_lower',
                        'dc_show_upper', 'dc_show_middle', 'dc_show_lower',
                        'ema_periods']
    rsi_zones_1h = {k: sidebar_config['params_1h'][k] for k in chart_param_keys} if show_1h else None
    rsi_zones_15m = {k: sidebar_config['params_15m'][k] for k in chart_param_keys} if not show_1h else None

    # Render charts
    draw_mode = sidebar_config.get('draw_mode', False)
    active_stats = stats_1h if show_1h else stats_15m
    show_strategy = show_custom_strategy and active_stats is not None

    panel_kwargs = {k: sidebar_config[k] for k in
                    ['show_rsi', 'show_cmb', 'show_stoch', 'show_adx', 'show_atr',
                     'show_macd', 'show_obv', 'show_accdist', 'show_supertrend', 'show_ema',
                     'show_donchian', 'show_psar']}

    chart_kwargs = dict(
        df_slice_1h=df_slice_1h, df_slice_15m=df_slice_15m,
        start_1h=period_start_1h, end_1h=period_end_1h,
        start_15m=period_start_15m, end_15m=period_end_15m,
        show_ichimoku=sidebar_config['show_ichimoku'],
        show_bb=sidebar_config['show_bb'],
        show_kc=sidebar_config['show_kc'],
        rsi_zones_1h=rsi_zones_1h,
        rsi_zones_15m=rsi_zones_15m,
        show_1h=show_1h,
        chart_key=chart_key,
        draw_mode=draw_mode,
        chart_height=sidebar_config['chart_height'],
        **panel_kwargs,
    )

    if show_strategy:
        col_charts, col_stats = st.columns([3, 1], gap="medium")

        with col_charts:
            chart_html = render_charts(show_strategy=True, **chart_kwargs)

        with col_stats:
            render_strategy_stats(stats_1h, stats_15m, strategy_label, show_1h)
    else:
        chart_html = render_charts(show_strategy=False, **chart_kwargs)

    # Cache the chart HTML for fast redisplay on subsequent reruns
    html_cache = st.session_state.get('_charting_html_cache', {})
    html_cache[chart_key] = {
        'html': chart_html,
        'stats': active_stats,
        'strategy_label': strategy_label,
    }
    st.session_state['_charting_html_cache'] = html_cache

    st.divider()

    return stats_1h, stats_15m


def render_strategy_stats(stats_1h, stats_15m, strategy_label, show_1h=True):
    """Render strategy statistics table"""
    st.subheader("Strategy Statistics")

    if strategy_label:
        st.caption(f"**{strategy_label}**")

    def _format_stats(stats):
        n = int(stats.loc['Number of trades', 'value'])
        win_rate = stats.loc['Win rate (%)', 'value']
        wins = round(n * win_rate / 100) if n > 0 else 0
        losses = n - wins
        total_win = stats.loc['Winning trades P&L (R)', 'value']
        total_lose = stats.loc['Losing trades P&L (R)', 'value']
        avg_profit = total_win / wins if wins > 0 else 0.0
        avg_loss = total_lose / losses if losses > 0 else 0.0
        return [
            f"{n}",
            f"{round(win_rate):.0f}%",
            f"{round(stats.loc['Loss rate (%)', 'value']):.0f}%",
            f"{avg_profit:.2f}R",
            f"{avg_loss:.2f}R",
            f"{stats.loc['Total P&L (R)', 'value']:.2f}R",
            f"{round(stats.loc['Target exit (%)', 'value']):.0f}%",
            f"{round(stats.loc['Static exit (%)', 'value']):.0f}%",
            f"{round(stats.loc['Dynamic exit (%)', 'value']):.0f}%",
            f"{stats.loc['Sharpe Ratio', 'value']:.2f}",
            f"{stats.loc['MAR Ratio', 'value']:.2f}",
            f"{stats.loc['SQN', 'value']:.2f}",
        ]

    if show_1h:
        data = {"1H": _format_stats(stats_1h)}
    else:
        data = {"15m": _format_stats(stats_15m)}

    stats_table = pd.DataFrame(
        data,
        index=[
            "Number of trades",
            "Win %",
            "Lose %",
            "Avg Profit",
            "Avg Loss",
            "Total P&L",
            "Target Exit %",
            "Static %",
            "Dynamic %",
            "Sharpe Ratio",
            "MAR Ratio",
            "SQN",
        ],
    )
    st.table(stats_table)


def _aggregate_stats(all_stats):
    """
    Aggregate stats across multiple periods.

    Each stats_df has these rows:
        Number of trades, Win rate (%), Loss rate (%),
        Total return (%), Total P&L (R), Avg P&L per trade (R),
        Winning trades P&L (R), Losing trades P&L (R)

    We sum the raw counts and dollars, then recompute rates.
    Sharpe, MaxDD, and MAR are computed from pooled individual trade R P&Ls.
    """
    import numpy as np

    all_trade_pnls = []  # pooled individual trade R P&Ls
    total_win_pnl = 0.0
    total_lose_pnl = 0.0
    total_static_alloc = 0.0
    total_dynamic_alloc = 0.0
    total_target_alloc = 0.0

    for stats_df in all_stats:
        total_win_pnl += stats_df.loc['Winning trades P&L (R)', 'value']
        total_lose_pnl += stats_df.loc['Losing trades P&L (R)', 'value']

        # Collect allocation-weighted exit type totals
        attrs = getattr(stats_df, 'attrs', {})
        total_static_alloc += attrs.get('total_static_alloc', 0.0)
        total_dynamic_alloc += attrs.get('total_dynamic_alloc', 0.0)
        total_target_alloc += attrs.get('total_target_alloc', 0.0)

        # Collect individual trade R P&Ls for pooled metric computation
        trade_pnls = attrs.get('trade_pnls_r', [])
        all_trade_pnls.extend(trade_pnls)

    # Derive counts directly from pooled trade P&Ls (avoids lossy percentage reconstruction)
    total_trades = len(all_trade_pnls)
    total_wins = sum(pnl > 0 for pnl in all_trade_pnls)
    total_losses = total_trades - total_wins
    total_pnl = total_win_pnl + total_lose_pnl

    if total_trades > 0:
        win_pct = (total_wins / total_trades) * 100
        lose_pct = (total_losses / total_trades) * 100
        target_exit_pct = total_target_alloc / total_trades
        static_exit_pct = total_static_alloc / total_trades
        dynamic_exit_pct = total_dynamic_alloc / total_trades

        avg_win_pnl = total_win_pnl / total_wins if total_wins > 0 else 0.0
        avg_lose_pnl = total_lose_pnl / total_losses if total_losses > 0 else 0.0

        # Expected Value = (Win% x Avg Profit) - (Lose% x Avg Loss)
        expected_value = (win_pct / 100 * avg_win_pnl) + (lose_pct / 100 * avg_lose_pnl)

        # Sharpe Ratio from pooled trades
        if len(all_trade_pnls) >= 2:
            pnl_std = np.std(all_trade_pnls, ddof=1)
            sharpe_ratio = (np.mean(all_trade_pnls) / pnl_std) if pnl_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max Drawdown from pooled cumulative R equity curve
        if all_trade_pnls:
            cumulative = np.cumsum(all_trade_pnls)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - peak
            max_drawdown = abs(drawdowns.min())
        else:
            max_drawdown = 0.0

        # MAR Ratio
        mar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else 0.0

        # SQN: sqrt(N) * mean(R P&Ls) / stdev(R P&Ls)
        if len(all_trade_pnls) >= 2:
            pnl_std = np.std(all_trade_pnls, ddof=1)
            sqn = (np.mean(all_trade_pnls) / pnl_std * np.sqrt(len(all_trade_pnls))) if pnl_std > 0 else 0.0
        else:
            sqn = 0.0
    else:
        win_pct = 0.0
        lose_pct = 0.0
        total_pnl = 0.0
        avg_win_pnl = 0.0
        avg_lose_pnl = 0.0
        expected_value = 0.0
        target_exit_pct = 0.0
        static_exit_pct = 0.0
        dynamic_exit_pct = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        mar_ratio = 0.0
        sqn = 0.0

    return {
        'num_trades': total_trades,
        'win_pct': win_pct,
        'lose_pct': lose_pct,
        'avg_win_pnl': avg_win_pnl,
        'avg_lose_pnl': avg_lose_pnl,
        'total_pnl': total_pnl,
        'expected_value': expected_value,
        'target_exit_pct': target_exit_pct,
        'static_exit_pct': static_exit_pct,
        'dynamic_exit_pct': dynamic_exit_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'mar_ratio': mar_ratio,
        'sqn': sqn,
    }


def render_global_performance(all_stats_1h, all_stats_15m, strategy_label, num_periods, show_1h=True):
    """Render the global performance summary across all date ranges"""

    st.markdown("---")
    st.header("Global Performance Metric of Group of Date Ranges")
    st.caption(f"Performance metrics across **{num_periods}** date range(s)")

    if strategy_label:
        st.caption(f"Strategy: **{strategy_label}**")

    def _format_agg(agg):
        return [
            f"{agg['num_trades']}",
            f"{agg['win_pct']:.0f}%",
            f"{agg['lose_pct']:.0f}%",
            f"{agg['avg_win_pnl']:.2f}R",
            f"{agg['avg_lose_pnl']:.2f}R",
            f"{agg['total_pnl']:.2f}R",
            f"{agg['expected_value']:.2f}R",
            f"{agg['target_exit_pct']:.0f}%",
            f"{agg['static_exit_pct']:.0f}%",
            f"{agg['dynamic_exit_pct']:.0f}%",
            f"{agg['sharpe_ratio']:.2f}",
            f"{agg['mar_ratio']:.2f}",
            f"{agg['sqn']:.2f}",
        ]

    if show_1h:
        agg = _aggregate_stats(all_stats_1h)
        data = {"1H": _format_agg(agg)}
    else:
        agg = _aggregate_stats(all_stats_15m)
        data = {"15m": _format_agg(agg)}

    global_table = pd.DataFrame(
        data,
        index=[
            "Number of Trades",
            "Win %",
            "Lose %",
            "Avg Profit",
            "Avg Loss",
            "Total P&L",
            "Expected Value",
            "Target Exit %",
            "Static %",
            "Dynamic %",
            "Sharpe Ratio",
            "MAR Ratio",
            "SQN",
        ],
    )

    st.table(global_table)
