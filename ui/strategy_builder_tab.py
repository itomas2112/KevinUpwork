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
    CONDITION_COMPARE_TYPES,
    EXIT_TYPES
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

    # Pattern selection for strategy
    st.markdown("#### Apply to Patterns")
    st.caption("Select which pattern combinations this strategy should apply to")

    # Get available patterns from PRIMARY_SECONDARY_MAP
    from data.helpers import PRIMARY_SECONDARY_MAP

    # Create pattern combination options
    pattern_options = []
    for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
        for secondary in secondaries:
            pattern_options.append(f"{primary} → {secondary}")

    # Multi-select for patterns
    selected_patterns = st.multiselect(
        "Pattern Combinations",
        options=pattern_options,
        default=st.session_state.get('strategy_patterns', []),
        key="strategy_patterns_select",
        help="This strategy will only appear when these pattern combinations are selected"
    )

    # Store in session state
    st.session_state['strategy_patterns'] = selected_patterns

    # Reset button
    if st.button("🔄 Reset Strategy", type="secondary"):
        reset_strategy_builder()

    st.divider()

    # Entry box
    render_entry_box()
    st.divider()

    # Initial Stop (Passive Stop) - NEW
    render_initial_stop_box()
    st.divider()

    # Exit Groups - NEW
    render_exit_groups()
    st.divider()

    # Validation and Save button
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
    """Render save strategy button with validation"""

    # Check if we're editing
    is_editing = st.session_state.get('editing_strategy', False)
    button_label = "Update Strategy" if is_editing else "Save Strategy"

    # Validation
    is_valid = validate_exit_groups()

    if not is_valid:
        st.error("⚠️ Total exit size must equal entry size!")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(button_label, type="primary", use_container_width=True, disabled=not is_valid):
            if is_editing:
                # Update existing strategy
                editing_idx = st.session_state.get('editing_strategy_idx')
                # Delete the old version
                st.session_state['saved_strategies'].pop(editing_idx)
                # Save the updated version
                count = save_strategy_to_session(strategy_name_input)

                # Clear editing flags
                st.session_state['editing_strategy'] = False
                st.session_state['editing_strategy_idx'] = None

                st.success(f"✅ Strategy updated!")
            else:
                # Create new strategy
                count = save_strategy_to_session(strategy_name_input)
                st.success(f"✅ Strategy saved! Total: {count}")

            # Reset the strategy builder
            st.session_state['strategy_started'] = False
            st.session_state['strategy_direction'] = None
            st.session_state['entry_conditions_count'] = 0
            st.session_state['exit_groups'] = []
            st.session_state['initial_stop'] = None
            st.session_state['strategy_name_input'] = ""

            st.rerun()

def render_strategy_management():
    """Render strategy management section"""
    st.subheader("Strategy Management")

    if st.session_state['saved_strategies']:
        st.caption(f"Total strategies saved: {len(st.session_state['saved_strategies'])}")

        # Create a table view of strategies
        for idx, strategy in enumerate(st.session_state['saved_strategies']):
            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])

                with col1:
                    st.markdown(f"**{strategy.get('strategy_name', f'Strategy_{idx + 1}')}**")

                with col2:
                    direction_emoji = "📈" if strategy.get('direction') == 'Long' else "📉"
                    st.markdown(f"{direction_emoji} {strategy.get('direction', 'N/A')}")

                with col3:
                    patterns = strategy.get('patterns', [])
                    if patterns:
                        st.caption(f"Patterns: {len(patterns)}")
                    else:
                        st.caption("Patterns: All")

                with col4:
                    if st.button("✏️", key=f"edit_strategy_{idx}", help="Edit this strategy"):
                        # Load strategy into builder for editing
                        load_strategy_for_editing(strategy, idx)
                        st.rerun()

                with col5:
                    if st.button("🗑️", key=f"delete_strategy_{idx}", help="Delete this strategy"):
                        delete_strategy(idx)
                        st.success(f"Strategy deleted!")
                        st.rerun()

                # Expandable details view
                with st.expander("View Strategy Details", expanded=False):

                    # Show patterns
                    st.markdown("### Applied to Patterns")
                    patterns = strategy.get('patterns', [])
                    if patterns:
                        for pattern in patterns:
                            st.markdown(f"- {pattern}")
                    else:
                        st.info("This strategy applies to all patterns")

                    st.divider()

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
                        st.markdown("#### Conditions")
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


def load_strategy_for_editing(strategy, strategy_idx):
    """Load a strategy into the builder for editing"""

    # Mark that we're editing an existing strategy
    st.session_state['editing_strategy'] = True
    st.session_state['editing_strategy_idx'] = strategy_idx

    # Load basic info
    st.session_state['strategy_started'] = True
    st.session_state['strategy_direction'] = strategy.get('direction', 'Long')
    st.session_state['strategy_name_input'] = strategy.get('strategy_name', '')
    st.session_state['strategy_patterns'] = strategy.get('patterns', [])

    # Load entry config
    entry = strategy.get('entry', {})
    entry_trigger = entry.get('trigger', {})

    st.session_state['entry_trigger_group1'] = entry_trigger.get('group', 'Price & Indicators')
    st.session_state['entry_trigger_element1'] = entry_trigger.get('element1')
    st.session_state['entry_trigger_event'] = entry_trigger.get('event')
    st.session_state['entry_trigger_compare_type'] = entry_trigger.get('compare_type', 'Indicator')

    if entry_trigger.get('compare_type') == 'Indicator':
        st.session_state['entry_trigger_element2'] = entry_trigger.get('element2')
    else:
        st.session_state['entry_trigger_value'] = entry_trigger.get('value', 50.0)

    st.session_state['entry_position_size'] = entry.get('position_size', 1.0)

    # Load entry conditions
    entry_conditions = entry.get('conditions', [])
    st.session_state['entry_conditions_count'] = len(entry_conditions)

    for i, cond in enumerate(entry_conditions):
        st.session_state[f'entry_cond_{i}_group1'] = cond.get('group', 'Price & Indicators')
        st.session_state[f'entry_cond_{i}_element1'] = cond.get('element1')
        st.session_state[f'entry_cond_{i}_operator'] = cond.get('operator')
        st.session_state[f'entry_cond_{i}_compare_type'] = cond.get('compare_type', 'Indicator')

        if cond.get('compare_type') == 'Indicator':
            st.session_state[f'entry_cond_{i}_element2'] = cond.get('element2')
        else:
            st.session_state[f'entry_cond_{i}_value'] = cond.get('value', 50.0)

    # Load exit config
    exit_cfg = strategy.get('exit', {})
    exit_trigger = exit_cfg.get('trigger', {})

    st.session_state['exit_trigger_group1'] = exit_trigger.get('group', 'Price & Indicators')
    st.session_state['exit_trigger_element1'] = exit_trigger.get('element1')
    st.session_state['exit_trigger_event'] = exit_trigger.get('event')
    st.session_state['exit_trigger_compare_type'] = exit_trigger.get('compare_type', 'Indicator')

    if exit_trigger.get('compare_type') == 'Indicator':
        st.session_state['exit_trigger_element2'] = exit_trigger.get('element2')
    else:
        st.session_state['exit_trigger_value'] = exit_trigger.get('value', 50.0)

    st.session_state['exit_position_size'] = exit_cfg.get('position_size', 1.0)

    # Load exit conditions
    exit_conditions = exit_cfg.get('conditions', [])
    st.session_state['exit_conditions_count'] = len(exit_conditions)

    for i, cond in enumerate(exit_conditions):
        st.session_state[f'exit_cond_{i}_group1'] = cond.get('group', 'Price & Indicators')
        st.session_state[f'exit_cond_{i}_element1'] = cond.get('element1')
        st.session_state[f'exit_cond_{i}_operator'] = cond.get('operator')
        st.session_state[f'exit_cond_{i}_compare_type'] = cond.get('compare_type', 'Indicator')

        if cond.get('compare_type') == 'Indicator':
            st.session_state[f'exit_cond_{i}_element2'] = cond.get('element2')
        else:
            st.session_state[f'exit_cond_{i}_value'] = cond.get('value', 50.0)


def validate_exit_groups():
    """Validate that total exit size equals entry size"""
    entry_size = st.session_state.get('entry_position_size', 0)

    total_exit_size = 0
    for exit_group in st.session_state.get('exit_groups', []):
        total_exit_size += exit_group.get('position_size', 0)

    return abs(total_exit_size - entry_size) < 0.001  # Small tolerance for float comparison


def add_exit_group():
    """Add a new exit group"""
    if 'exit_groups' not in st.session_state:
        st.session_state['exit_groups'] = []

    new_group = {
        'group_id': len(st.session_state['exit_groups']) + 1,
        'position_size': 1.0,
        'targets': [],
        'stops': []
    }

    st.session_state['exit_groups'].append(new_group)

def remove_exit_group(group_idx):
    """Remove an exit group"""
    if 'exit_groups' in st.session_state and 0 <= group_idx < len(st.session_state['exit_groups']):
        st.session_state['exit_groups'].pop(group_idx)


def add_exit_to_group(group_idx, exit_type):
    """Add a target or stop to a specific exit group"""
    if 'exit_groups' not in st.session_state:
        return

    if 0 <= group_idx < len(st.session_state['exit_groups']):
        exit_config = {
            'type': exit_type,
            'trigger': {},
            'conditions': []
        }

        if exit_type == 'Target':
            st.session_state['exit_groups'][group_idx]['targets'].append(exit_config)
        else:  # Stop
            st.session_state['exit_groups'][group_idx]['stops'].append(exit_config)


def remove_exit_from_group(group_idx, exit_type, exit_idx):
    """Remove a specific exit from a group"""
    if 'exit_groups' not in st.session_state:
        return

    if 0 <= group_idx < len(st.session_state['exit_groups']):
        if exit_type == 'Target':
            if 0 <= exit_idx < len(st.session_state['exit_groups'][group_idx]['targets']):
                st.session_state['exit_groups'][group_idx]['targets'].pop(exit_idx)
        else:  # Stop
            if 0 <= exit_idx < len(st.session_state['exit_groups'][group_idx]['stops']):
                st.session_state['exit_groups'][group_idx]['stops'].pop(exit_idx)


def render_initial_stop_box():
    """Render the initial stop (passive stop) configuration"""
    st.subheader("🛑 Initial Stop (Passive Stop)")
    st.caption("This stop is shared across ALL exit groups and used for risk calculation")

    with st.container(border=True):
        st.markdown("#### 🎯 Initial Stop Trigger")
        st.warning("⚠️ Must be a Price × Indicator event for proper 1R calculation")

        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            # Force Price group for initial stop
            st.info("Element 1: **Price** (fixed)")
            initial_stop_element1 = "Price"

        with col2:
            initial_stop_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key="initial_stop_event"
            )

        with col3:
            # Must be indicator for initial stop
            initial_stop_compare_type = st.radio(
                "Compare to",
                ["Indicator"],  # Only Indicator, no Fixed Value
                key="initial_stop_compare_type",
                horizontal=True
            )

            initial_stop_element2 = st.selectbox(
                "Indicator",
                PRICE_AND_INDICATORS[1:],  # Exclude "Price" from options
                key="initial_stop_element2"
            )
            st.caption(f"Example: Price {initial_stop_event} {initial_stop_element2}")

        # Store initial stop in session state
        st.session_state['initial_stop'] = {
            'element1': initial_stop_element1,
            'event': initial_stop_event,
            'compare_type': 'Indicator',
            'element2': initial_stop_element2
        }


def render_exit_groups():
    """Render all exit groups with targets and stops"""
    st.subheader("📤 Exit Strategy Groups")
    st.caption(
        "Each group handles a portion of your position. Targets and Stops within a group are OCO (One-Cancels-Other)")

    # Initialize exit groups if not exists
    if 'exit_groups' not in st.session_state:
        st.session_state['exit_groups'] = []

    # Calculate total sizes
    entry_size = st.session_state.get('entry_position_size', 0)
    total_exit_size = sum(group.get('position_size', 0) for group in st.session_state.get('exit_groups', []))

    # Validation message
    if len(st.session_state.get('exit_groups', [])) > 0:
        if abs(total_exit_size - entry_size) < 0.001:
            st.success(f"✓ Entry Size: {entry_size} units = Total Exit Size: {total_exit_size} units")
        else:
            st.error(f"⚠️ Entry Size ({entry_size}) ≠ Total Exit Size ({total_exit_size}) - Strategy invalid!")

    # Render each exit group
    for group_idx, exit_group in enumerate(st.session_state.get('exit_groups', [])):
        with st.container(border=True):
            # Group header
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"### Exit Group {group_idx + 1}")

            with col2:
                group_size = st.number_input(
                    "Group Size (units)",
                    min_value=0.0,
                    value=exit_group.get('position_size', 1.0),
                    step=0.1,
                    key=f"exit_group_{group_idx}_size"
                )
                st.session_state['exit_groups'][group_idx]['position_size'] = group_size

            with col3:
                if st.button("🗑️", key=f"remove_exit_group_{group_idx}", help="Remove this exit group"):
                    remove_exit_group(group_idx)
                    st.rerun()

            st.divider()

            # Targets section
            st.markdown("#### 🎯 Targets")
            targets = exit_group.get('targets', [])

            if len(targets) == 0:
                st.info("No targets added yet")
            else:
                for target_idx, target in enumerate(targets):
                    render_exit_config(group_idx, 'Target', target_idx, target)

            if st.button(f"➕ Add Target", key=f"add_target_group_{group_idx}"):
                add_exit_to_group(group_idx, 'Target')
                st.rerun()

            st.divider()

            # Stops section
            st.markdown("#### 🛑 Stops")
            st.caption("Initial Stop is automatically included in all groups")

            stops = exit_group.get('stops', [])

            # Show initial stop (always present)
            if st.session_state.get('initial_stop'):
                initial = st.session_state['initial_stop']
                st.info(
                    f"🔒 Initial Stop: {initial['element1']} {initial['event']} {initial['element2']} (auto-included)")

            if len(stops) == 0:
                st.info("No additional stops added")
            else:
                for stop_idx, stop in enumerate(stops):
                    render_exit_config(group_idx, 'Stop', stop_idx, stop)

            if st.button(f"➕ Add Stop", key=f"add_stop_group_{group_idx}"):
                add_exit_to_group(group_idx, 'Stop')
                st.rerun()

    # Add new exit group button
    st.divider()
    if st.button("➕ Add Exit Group", type="primary"):
        add_exit_group()
        st.rerun()


def render_exit_config(group_idx, exit_type, exit_idx, exit_config):
    """Render a single exit (target or stop) configuration"""
    icon = "🎯" if exit_type == "Target" else "🛑"

    with st.expander(f"{icon} {exit_type} {exit_idx + 1}", expanded=True):
        col_delete = st.columns([10, 1])

        with col_delete[1]:
            if st.button("×", key=f"remove_{exit_type}_{group_idx}_{exit_idx}",
                         help=f"Remove this {exit_type.lower()}"):
                remove_exit_from_group(group_idx, exit_type, exit_idx)
                st.rerun()

        # Trigger
        st.markdown("**Trigger**")
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            trigger_group = st.selectbox(
                "Select Group",
                ["Price & Indicators", "RSI Group", "CMB Group"],
                key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_group1"
            )

            if trigger_group == "Price & Indicators":
                available_elements = PRICE_AND_INDICATORS
            elif trigger_group == "RSI Group":
                available_elements = RSI_GROUP
            else:
                available_elements = CMB_GROUP

            trigger_element1 = st.selectbox(
                "Element 1",
                available_elements,
                key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_element1"
            )

        with col2:
            trigger_event = st.selectbox(
                "Event",
                EVENT_TYPES,
                key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_event"
            )

        with col3:
            trigger_compare_type = st.radio(
                "Compare to",
                CONDITION_COMPARE_TYPES,
                key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_compare_type",
                horizontal=True
            )

            if trigger_compare_type == "Indicator":
                compatible_elements = get_compatible_elements(trigger_element1)
                trigger_element2 = st.selectbox(
                    "Element 2",
                    [e for e in compatible_elements if e != trigger_element1],
                    key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_element2"
                )
            else:
                trigger_value = st.number_input(
                    "Value/Level",
                    value=50.0,
                    key=f"{exit_type}_{group_idx}_{exit_idx}_trigger_value"
                )

        # Conditions (optional)
        st.markdown("**Conditions (Optional)**")
        conditions_key = f"{exit_type}_{group_idx}_{exit_idx}_conditions_count"

        if conditions_key not in st.session_state:
            st.session_state[conditions_key] = 0

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Condition", key=f"add_cond_{exit_type}_{group_idx}_{exit_idx}"):
                if st.session_state[conditions_key] < 5:  # Limit to 5 conditions per exit
                    st.session_state[conditions_key] += 1
                    st.rerun()
        with col2:
            if st.button("➖ Remove", key=f"rem_cond_{exit_type}_{group_idx}_{exit_idx}"):
                if st.session_state[conditions_key] > 0:
                    st.session_state[conditions_key] -= 1
                    st.rerun()

        # Render conditions
        for cond_idx in range(st.session_state.get(conditions_key, 0)):
            render_exit_condition(group_idx, exit_type, exit_idx, cond_idx)


def render_exit_condition(group_idx, exit_type, exit_idx, cond_idx):
    """Render a condition for an exit"""
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        cond_group = st.selectbox(
            "Group",
            ["Price & Indicators", "RSI Group", "CMB Group"],
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_group"
        )

        if cond_group == "Price & Indicators":
            cond_available = PRICE_AND_INDICATORS
        elif cond_group == "RSI Group":
            cond_available = RSI_GROUP
        else:
            cond_available = CMB_GROUP

        cond_element1 = st.selectbox(
            "Element 1",
            cond_available,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_element1"
        )

    with col2:
        cond_operator = st.selectbox(
            "Operator",
            CONDITION_OPERATORS,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_operator"
        )

    with col3:
        cond_compare_type = st.radio(
            "Compare to",
            CONDITION_COMPARE_TYPES,
            key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_compare_type",
            horizontal=True
        )

        if cond_compare_type == "Indicator":
            cond_compatible = get_compatible_elements(cond_element1)
            cond_element2 = st.selectbox(
                "Element 2",
                [e for e in cond_compatible if e != cond_element1],
                key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_element2"
            )
        else:
            cond_value = st.number_input(
                "Value",
                value=50.0,
                key=f"{exit_type}_{group_idx}_{exit_idx}_cond_{cond_idx}_value"
            )


def reset_strategy_builder():
    """Reset all strategy builder state"""
    st.session_state['strategy_started'] = False
    st.session_state['strategy_direction'] = None
    st.session_state['entry_conditions_count'] = 0
    st.session_state['exit_groups'] = []
    st.session_state['initial_stop'] = None
    st.session_state['strategy_name_input'] = ""
    st.session_state['editing_strategy'] = False
    st.session_state['editing_strategy_idx'] = None
    st.rerun()