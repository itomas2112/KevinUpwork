"""
Strategy Builder tab (Tab 2) UI and logic
"""
import streamlit as st
from config.constants import (
    PRICE_AND_INDICATORS,
    RSI_GROUP,
    CMB_GROUP,
    EVENT_TYPES,
    CONDITION_OPERATORS,
    CONDITION_COMPARE_TYPES
)
from strategies.strategy_manager import save_strategy_to_session, delete_strategy, delete_all_strategies


def render_strategy_builder_tab():
    """Render the strategy builder tab content"""

    col_left, col_center, col_right = st.columns([1, 1, 1])

    with col_center:
        st.header("Strategy Builder")

        if not st.session_state['strategy_started']:
            render_create_button()
        elif st.session_state['strategy_direction'] is None:
            render_direction_selection()
        else:
            render_strategy_form()

        st.divider()
        render_strategy_management()


def render_create_button():
    """Render the create new strategy button"""
    if st.button("➕ Create New Strategy", type="primary", use_container_width=True):
        st.session_state['strategy_started'] = True
        st.rerun()


def render_direction_selection():
    """Render Long/Short selection"""
    st.subheader("Step 1: Choose Strategy Direction")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Long", type="primary", use_container_width=True):
            st.session_state['strategy_direction'] = 'Long'
            st.rerun()

    with col2:
        if st.button("📉 Short", type="primary", use_container_width=True):
            st.session_state['strategy_direction'] = 'Short'
            st.rerun()


def render_strategy_form():
    """Render the main strategy creation form"""
    st.success(f"Strategy Direction: **{st.session_state['strategy_direction']}**")

    # Strategy name input
    strategy_name_input = st.text_input(
        "Strategy Name",
        value=st.session_state.get('strategy_name_input', ''),
        placeholder="Enter a name for your strategy...",
        key="strategy_name_field"
    )

    # Reset button
    if st.button("🔄 Reset Strategy", type="secondary"):
        reset_strategy_builder()

    st.divider()

    # Entry and Exit boxes
    render_entry_box()
    st.divider()
    render_exit_box()
    st.divider()

    # Save button
    render_save_button(strategy_name_input)


def render_entry_box():
    """Render entry strategy configuration box"""
    st.subheader("Entry Strategy")

    with st.container(border=True):
        # TRIGGER (Required) - Event between indicators/price
        st.markdown("#### Trigger (Required)")
        st.caption("Define an event between compatible elements")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Group selection for element 1
            entry_trigger_group1 = st.selectbox(
                "Select Group",
                ["Price & Indicators", "RSI Group", "CMB Group"],
                key="entry_trigger_group1"
            )

            if entry_trigger_group1 == "Price & Indicators":
                available_elements1 = PRICE_AND_INDICATORS
            elif entry_trigger_group1 == "RSI Group":
                available_elements1 = RSI_GROUP
            else:
                available_elements1 = CMB_GROUP

            entry_trigger_element1 = st.selectbox(
                "Element 1",
                available_elements1,
                key="entry_trigger_element1"
            )

        with col2:
            entry_trigger_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key="entry_trigger_event"
            )

        with col3:
            # Choose between indicator or fixed value
            entry_trigger_compare_type = st.radio(
                "Compare to",
                CONDITION_COMPARE_TYPES,
                key="entry_trigger_compare_type",
                horizontal=True
            )

            if entry_trigger_compare_type == "Indicator":
                # Element 2 must be from same group as Element 1
                compatible_elements = get_compatible_elements(entry_trigger_element1)

                entry_trigger_element2 = st.selectbox(
                    "Element 2",
                    [e for e in compatible_elements if e != entry_trigger_element1],
                    key="entry_trigger_element2"
                )
                st.caption(f"Example: {entry_trigger_element1} {entry_trigger_event} {entry_trigger_element2}")
            else:  # Fixed Value
                entry_trigger_value = st.number_input(
                    "Value/Level",
                    value=50.0,
                    key="entry_trigger_value",
                    help="e.g., RSI crosses above 50, Price crosses below 4000"
                )
                st.caption(f"Example: {entry_trigger_element1} {entry_trigger_event} {entry_trigger_value}")

        st.divider()

        # POSITION SIZE (Required) - Units of position
        st.markdown("#### Position Size (Required)")
        st.caption("Specify the units/size of the position to enter")

        entry_position_size = st.number_input(
            "Position Size (units)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="entry_position_size"
        )

        st.divider()

        # CONDITIONS (Optional, 0-10)
        st.markdown("#### Conditions")
        st.caption(
            "All conditions must be met for the trigger to activate. If any condition fails, entry will not occur.")

        # Add/Remove condition buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("➕ Add Condition", key="add_entry_condition"):
                if st.session_state['entry_conditions_count'] < 10:
                    st.session_state['entry_conditions_count'] += 1
                    st.rerun()
        with col2:
            if st.button("➖ Remove", key="remove_entry_condition"):
                if st.session_state['entry_conditions_count'] > 0:
                    st.session_state['entry_conditions_count'] -= 1
                    st.rerun()

        # Display conditions
        if st.session_state['entry_conditions_count'] > 0:
            st.markdown(f"**Active Conditions: {st.session_state['entry_conditions_count']}**")

            for i in range(st.session_state['entry_conditions_count']):
                with st.expander(f"Condition {i + 1}", expanded=True):
                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        # Group selection for condition element 1
                        cond_group1 = st.selectbox(
                            "Select Group",
                            ["Price & Indicators", "RSI Group", "CMB Group"],
                            key=f"entry_cond_{i}_group1"
                        )

                        if cond_group1 == "Price & Indicators":
                            cond_available_elements1 = PRICE_AND_INDICATORS
                        elif cond_group1 == "RSI Group":
                            cond_available_elements1 = RSI_GROUP
                        else:
                            cond_available_elements1 = CMB_GROUP

                        cond_element1 = st.selectbox(
                            "Element 1",
                            cond_available_elements1,
                            key=f"entry_cond_{i}_element1"
                        )

                    with col2:
                        cond_operator = st.selectbox(
                            "Operator",
                            CONDITION_OPERATORS,
                            key=f"entry_cond_{i}_operator"
                        )

                    with col3:
                        # Choose between indicator or fixed value
                        cond_compare_type = st.radio(
                            "Compare to",
                            CONDITION_COMPARE_TYPES,
                            key=f"entry_cond_{i}_compare_type",
                            horizontal=True
                        )

                        if cond_compare_type == "Indicator":
                            # Element 2 must be from same group as Element 1
                            cond_compatible_elements = get_compatible_elements(cond_element1)

                            cond_element2 = st.selectbox(
                                "Element 2",
                                [e for e in cond_compatible_elements if e != cond_element1],
                                key=f"entry_cond_{i}_element2"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_element2}")
                        else:  # Fixed Value
                            cond_value = st.number_input(
                                "Value",
                                value=50.0,
                                key=f"entry_cond_{i}_value"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_value}")
        else:
            st.info("No conditions added. Trigger will activate without additional requirements.")


def render_exit_box():
    """Render exit strategy configuration box"""
    st.subheader("Exit Strategy")

    with st.container(border=True):
        # TRIGGER (Required) - Event between indicators/price
        st.markdown("#### Trigger (Required)")
        st.caption("Define an event between compatible elements")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Group selection for element 1
            exit_trigger_group1 = st.selectbox(
                "Select Group",
                ["Price & Indicators", "RSI Group", "CMB Group"],
                key="exit_trigger_group1"
            )

            if exit_trigger_group1 == "Price & Indicators":
                available_elements1 = PRICE_AND_INDICATORS
            elif exit_trigger_group1 == "RSI Group":
                available_elements1 = RSI_GROUP
            else:
                available_elements1 = CMB_GROUP

            exit_trigger_element1 = st.selectbox(
                "Element 1",
                available_elements1,
                key="exit_trigger_element1"
            )

        with col2:
            exit_trigger_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key="exit_trigger_event"
            )

        with col3:
            # Choose between indicator or fixed value
            exit_trigger_compare_type = st.radio(
                "Compare to",
                CONDITION_COMPARE_TYPES,
                key="exit_trigger_compare_type",
                horizontal=True
            )

            if exit_trigger_compare_type == "Indicator":
                # Element 2 must be from same group as Element 1
                compatible_elements = get_compatible_elements(exit_trigger_element1)

                exit_trigger_element2 = st.selectbox(
                    "Element 2",
                    [e for e in compatible_elements if e != exit_trigger_element1],
                    key="exit_trigger_element2"
                )
                st.caption(f"Example: {exit_trigger_element1} {exit_trigger_event} {exit_trigger_element2}")
            else:  # Fixed Value
                exit_trigger_value = st.number_input(
                    "Value/Level",
                    value=50.0,
                    key="exit_trigger_value",
                    help="e.g., RSI crosses above 50, Price crosses below 4000"
                )
                st.caption(f"Example: {exit_trigger_element1} {exit_trigger_event} {exit_trigger_value}")

        st.divider()

        # POSITION SIZE (Required) - Units of position
        st.markdown("#### Position Size (Required)")
        st.caption("Specify the units/size of the position to exit")

        exit_position_size = st.number_input(
            "Position Size (units)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="exit_position_size"
        )

        st.divider()

        # CONDITIONS (Optional, 0-10)
        st.markdown("#### Conditions")
        st.caption(
            "All conditions must be met for the trigger to activate. If any condition fails, exit will not occur.")

        # Add/Remove condition buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("➕ Add Condition", key="add_exit_condition"):
                if st.session_state['exit_conditions_count'] < 10:
                    st.session_state['exit_conditions_count'] += 1
                    st.rerun()
        with col2:
            if st.button("➖ Remove", key="remove_exit_condition"):
                if st.session_state['exit_conditions_count'] > 0:
                    st.session_state['exit_conditions_count'] -= 1
                    st.rerun()

        # Display conditions
        if st.session_state['exit_conditions_count'] > 0:
            st.markdown(f"**Active Conditions: {st.session_state['exit_conditions_count']}**")

            for i in range(st.session_state['exit_conditions_count']):
                with st.expander(f"Condition {i + 1}", expanded=True):
                    col1, col2, col3 = st.columns([2, 1, 2])

                    with col1:
                        # Group selection for condition element 1
                        cond_group1 = st.selectbox(
                            "Select Group",
                            ["Price & Indicators", "RSI Group", "CMB Group"],
                            key=f"exit_cond_{i}_group1"
                        )

                        if cond_group1 == "Price & Indicators":
                            cond_available_elements1 = PRICE_AND_INDICATORS
                        elif cond_group1 == "RSI Group":
                            cond_available_elements1 = RSI_GROUP
                        else:
                            cond_available_elements1 = CMB_GROUP

                        cond_element1 = st.selectbox(
                            "Element 1",
                            cond_available_elements1,
                            key=f"exit_cond_{i}_element1"
                        )

                    with col2:
                        cond_operator = st.selectbox(
                            "Operator",
                            CONDITION_OPERATORS,
                            key=f"exit_cond_{i}_operator"
                        )

                    with col3:
                        # Choose between indicator or fixed value
                        cond_compare_type = st.radio(
                            "Compare to",
                            CONDITION_COMPARE_TYPES,
                            key=f"exit_cond_{i}_compare_type",
                            horizontal=True
                        )

                        if cond_compare_type == "Indicator":
                            # Element 2 must be from same group as Element 1
                            cond_compatible_elements = get_compatible_elements(cond_element1)

                            cond_element2 = st.selectbox(
                                "Element 2",
                                [e for e in cond_compatible_elements if e != cond_element1],
                                key=f"exit_cond_{i}_element2"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_element2}")
                        else:  # Fixed Value
                            cond_value = st.number_input(
                                "Value",
                                value=50.0,
                                key=f"exit_cond_{i}_value"
                            )
                            st.caption(f"{cond_element1} {cond_operator} {cond_value}")
        else:
            st.info("No conditions added. Trigger will activate without additional requirements.")


def render_save_button(strategy_name_input):
    """Render save strategy button"""
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("💾 Save Strategy", type="primary", use_container_width=True):
            count = save_strategy_to_session(strategy_name_input)

            # Reset the strategy builder to show "Create New Strategy" button again
            st.session_state['strategy_started'] = False
            st.session_state['strategy_direction'] = None
            st.session_state['entry_conditions_count'] = 0
            st.session_state['exit_conditions_count'] = 0
            st.session_state['strategy_name_input'] = ""

            st.success(f"✅ Strategy saved! Total: {count}")
            st.rerun()


def render_strategy_management():
    """Render strategy management section"""
    st.subheader("Strategy Management")

    if st.session_state['saved_strategies']:
        st.caption(f"Total strategies saved: {len(st.session_state['saved_strategies'])}")

        # Create a table view of strategies
        for idx, strategy in enumerate(st.session_state['saved_strategies']):
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.markdown(f"**{strategy.get('strategy_name', f'Strategy_{idx + 1}')}**")

                with col2:
                    direction_emoji = "📈" if strategy.get('direction') == 'Long' else "📉"
                    st.markdown(f"{direction_emoji} {strategy.get('direction', 'N/A')}")

                with col3:
                    st.caption(f"Created: {strategy.get('created_at', 'N/A')}")

                with col4:
                    if st.button("🗑️", key=f"delete_strategy_{idx}", help="Delete this strategy"):
                        delete_strategy(idx)
                        st.success(f"Strategy deleted!")
                        st.rerun()

                # Expandable details view
                with st.expander("View Strategy Details", expanded=False):

                    # Entry Strategy Section
                    st.markdown("### Entry Strategy")

                    with st.container(border=True):
                        entry = strategy.get('entry', {})
                        trigger = entry.get('trigger', {})

                        # Trigger
                        st.markdown("#### Trigger")
                        trigger_element1 = trigger.get('element1', 'N/A')
                        trigger_event = trigger.get('event', 'N/A')
                        trigger_compare_type = trigger.get('compare_type', 'Indicator')

                        if trigger_compare_type == "Fixed Value":
                            trigger_value = trigger.get('value', 'N/A')
                            st.info(f"**{trigger_element1}** {trigger_event} **{trigger_value}**")
                        else:
                            trigger_element2 = trigger.get('element2', 'N/A')
                            st.info(f"**{trigger_element1}** {trigger_event} **{trigger_element2}**")

                        # Position Size
                        st.markdown("#### Position Size")
                        position_size = entry.get('position_size', 'N/A')
                        st.info(f"**{position_size}** units")

                        # Conditions
                        st.markdown("#### Conditions")
                        conditions_count = entry.get('conditions_count', 0)

                        if conditions_count > 0:
                            st.markdown(f"**{conditions_count} condition(s) must be met:**")
                            for i, cond in enumerate(entry.get('conditions', []), 1):
                                cond_element1 = cond.get('element1', 'N/A')
                                cond_operator = cond.get('operator', 'N/A')
                                cond_compare_type = cond.get('compare_type', 'Indicator')

                                if cond_compare_type == "Fixed Value":
                                    cond_value = cond.get('value', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_value}")
                                else:
                                    cond_element2 = cond.get('element2', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_element2}")
                        else:
                            st.markdown("*No conditions - trigger activates immediately*")

                    st.divider()

                    # Exit Strategy Section
                    st.markdown("### Exit Strategy")

                    with st.container(border=True):
                        exit_cfg = strategy.get('exit', {})
                        exit_trigger = exit_cfg.get('trigger', {})

                        # Trigger
                        st.markdown("#### Trigger")
                        exit_trigger_element1 = exit_trigger.get('element1', 'N/A')
                        exit_trigger_event = exit_trigger.get('event', 'N/A')
                        exit_trigger_compare_type = exit_trigger.get('compare_type', 'Indicator')

                        if exit_trigger_compare_type == "Fixed Value":
                            exit_trigger_value = exit_trigger.get('value', 'N/A')
                            st.info(f"**{exit_trigger_element1}** {exit_trigger_event} **{exit_trigger_value}**")
                        else:
                            exit_trigger_element2 = exit_trigger.get('element2', 'N/A')
                            st.info(f"**{exit_trigger_element1}** {exit_trigger_event} **{exit_trigger_element2}**")

                        # Position Size
                        st.markdown("#### Position Size")
                        exit_position_size = exit_cfg.get('position_size', 'N/A')
                        st.info(f"**{exit_position_size}** units")

                        # Conditions
                        st.markdown("#### ⚙️ Conditions")
                        exit_conditions_count = exit_cfg.get('conditions_count', 0)

                        if exit_conditions_count > 0:
                            st.markdown(f"**{exit_conditions_count} condition(s) must be met:**")
                            for i, cond in enumerate(exit_cfg.get('conditions', []), 1):
                                cond_element1 = cond.get('element1', 'N/A')
                                cond_operator = cond.get('operator', 'N/A')
                                cond_compare_type = cond.get('compare_type', 'Indicator')

                                if cond_compare_type == "Fixed Value":
                                    cond_value = cond.get('value', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_value}")
                                else:
                                    cond_element2 = cond.get('element2', 'N/A')
                                    st.markdown(f"{i}. {cond_element1} **{cond_operator}** {cond_element2}")
                        else:
                            st.markdown("*No conditions - trigger activates immediately*")

                    # Advanced: Show JSON for debugging
                    with st.expander("🔧 Advanced: View Raw JSON", expanded=False):
                        st.json(strategy)

        # Bulk delete option
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Delete All Strategies", type="secondary", use_container_width=True):
                if st.session_state.get('confirm_delete_all', False):
                    delete_all_strategies()
                    st.session_state['confirm_delete_all'] = False
                    st.success("All strategies deleted!")
                    st.rerun()
                else:
                    st.session_state['confirm_delete_all'] = True
                    st.warning("⚠️ Click again to confirm deletion of ALL strategies")
                    st.rerun()
    else:
        st.info("No strategies saved yet. Create and save a strategy to see it here.")


def reset_strategy_builder():
    """Reset all strategy builder state"""
    st.session_state['strategy_started'] = False
    st.session_state['strategy_direction'] = None
    st.session_state['entry_conditions_count'] = 0
    st.session_state['exit_conditions_count'] = 0
    st.session_state['strategy_name_input'] = ""
    st.rerun()


def get_compatible_elements(selected_element):
    """Get compatible elements based on selection"""
    if selected_element in RSI_GROUP:
        return RSI_GROUP
    elif selected_element in CMB_GROUP:
        return CMB_GROUP
    else:
        return PRICE_AND_INDICATORS