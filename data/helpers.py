PRIMARY_SECONDARY_MAP = {
    "W.(1)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(2)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(3)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(4)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(5)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(A)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(B)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(C)": [
        "W.1 Impulse",
        "W.3 Impulse",
        "W.5 Impulse",
    ],

    "W.(W)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(X)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],

    "W.(Y)": [
        "W.A Impulse",
        "W.C Impulse",
        "W.A/W Zigzag",
        "W.Y Zigzag",
    ],
}

ALL_UNIQUE_SECONDARIES = sorted(set(
    sec for secs in PRIMARY_SECONDARY_MAP.values() for sec in secs
))

PRIMARY_LIST = list(PRIMARY_SECONDARY_MAP.keys())


def expand_selection(selection):
    """
    Expand a selection dict into a list of (pattern_type, primary, secondary) tuples.
    Supports 6 modes: All Patterns, All Bullish, All Bearish,
    Specified Primary, Specified Secondary, Secondary Across Primaries.
    """
    mode = selection.get("mode", "All Patterns")

    if mode == "All Patterns":
        combos = []
        for ptype in ("Bullish", "Bearish"):
            for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
                for sec in secondaries:
                    combos.append((ptype, primary, sec))
        return combos

    if mode == "All Bullish":
        combos = []
        for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
            for sec in secondaries:
                combos.append(("Bullish", primary, sec))
        return combos

    if mode == "All Bearish":
        combos = []
        for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
            for sec in secondaries:
                combos.append(("Bearish", primary, sec))
        return combos

    ptype = selection.get("pattern_type", "Bullish")

    if mode == "Specified Primary":
        primary = selection.get("primary") or PRIMARY_LIST[0]
        secondaries = PRIMARY_SECONDARY_MAP.get(primary, [])
        return [(ptype, primary, sec) for sec in secondaries]

    if mode == "Specified Secondary":
        primary = selection.get("primary") or PRIMARY_LIST[0]
        secondary = selection.get("secondary")
        if secondary is None:
            secondaries = PRIMARY_SECONDARY_MAP.get(primary, [])
            secondary = secondaries[0] if secondaries else None
        if secondary is None:
            return []
        return [(ptype, primary, secondary)]

    if mode == "Secondary Across Primaries":
        secondary = selection.get("secondary")
        if secondary is None:
            secondary = ALL_UNIQUE_SECONDARIES[0] if ALL_UNIQUE_SECONDARIES else None
        if secondary is None:
            return []
        combos = []
        for primary, secondaries in PRIMARY_SECONDARY_MAP.items():
            if secondary in secondaries:
                combos.append((ptype, primary, secondary))
        return combos

    return []


def selection_label(selection):
    """Generate a readable label for a selection."""
    mode = selection.get("mode", "All Patterns")

    if mode == "All Patterns":
        return "All Patterns"
    if mode == "All Bullish":
        return "All Bullish"
    if mode == "All Bearish":
        return "All Bearish"

    ptype = selection.get("pattern_type", "Bullish")

    if mode == "Specified Primary":
        primary = selection.get("primary", "?")
        return f"{primary} {ptype}"

    if mode == "Specified Secondary":
        primary = selection.get("primary", "?")
        secondary = selection.get("secondary", "?")
        return f"{primary} → {secondary} {ptype}"

    if mode == "Secondary Across Primaries":
        secondary = selection.get("secondary", "?")
        return f"{secondary} (All Primaries) {ptype}"

    return mode

# ----------------------------------------------------------------------
# Fixed pattern columns (Performance / Grid Search / Test Set)
# ----------------------------------------------------------------------
# Every performance table shows four fixed columns — All Patterns, All
# Bullish, All Bearish, Global — followed by one column per user-added
# selection. "Global" is the aggregate of the user-added selections (the
# columns to its right); the three "All …" columns are always present, so
# they are no longer offered as add-a-selection modes in those tabs.

from collections import OrderedDict

FIXED_SELECTIONS = OrderedDict([
    ("All Patterns", {"mode": "All Patterns"}),
    ("All Bullish", {"mode": "All Bullish"}),
    ("All Bearish", {"mode": "All Bearish"}),
])

FIXED_COLUMNS = ["All Patterns", "All Bullish", "All Bearish", "Global"]

USER_SELECTION_MODES = [
    "Specified Primary",
    "Specified Secondary",
    "Secondary Across Primaries",
]


def all_combos():
    """Every (pattern_type, primary, secondary) tuple in the DRM universe
    (86: 43 Bullish + 43 Bearish), in PRIMARY_SECONDARY_MAP order."""
    return expand_selection(FIXED_SELECTIONS["All Patterns"])


def fixed_combo_map():
    """{"All Patterns": [...86], "All Bullish": [...43], "All Bearish": [...43]}."""
    return OrderedDict((label, expand_selection(sel))
                       for label, sel in FIXED_SELECTIONS.items())


def filter_combos_by_strategy(combos, strategy_patterns):
    """Keep only combos whose "primary → secondary" is in the strategy's
    pattern list. An empty / missing strategy_patterns passes every combo."""
    if not strategy_patterns:
        return list(combos)
    allowed = set(strategy_patterns)
    return [(ptype, primary, secondary)
            for ptype, primary, secondary in combos
            if f"{primary} → {secondary}" in allowed]


def unique_label(label, taken):
    """De-duplicate `label` against `taken` as "X", "X (2)", "X (3)", ..."""
    if label not in taken:
        return label
    n = 2
    while f"{label} ({n})" in taken:
        n += 1
    return f"{label} ({n})"


def build_pattern_columns(user_selections, strategy_patterns, stats_by_combo,
                          aggregate_fn, empty_fn):
    """Return OrderedDict label -> agg in the fixed order:
    All Patterns, All Bullish, All Bearish, Global, then one entry per user
    selection (labels de-duplicated "X", "X (2)", ...).

    Global = union of the user selections' combos, each unique combo counted
    once; empty_fn() when there are no user selections (or none of their
    combos carry stats).

    strategy_patterns filtering (the strategy's own pattern list) applies to
    every column — including the three fixed ones. stats_by_combo maps
    (pattern_type, primary, secondary) -> [stats, ...]; combos absent from
    it contribute nothing. A column with no stats gets empty_fn()."""
    def _agg(combos):
        stats = []
        for combo in combos:
            stats.extend(stats_by_combo.get(combo, []))
        return aggregate_fn(stats) if stats else empty_fn()

    columns = OrderedDict()
    for label, combos in fixed_combo_map().items():
        columns[label] = _agg(filter_combos_by_strategy(combos, strategy_patterns))

    # User selections: per-selection aggs plus the de-duplicated union for Global
    user_columns = OrderedDict()
    global_combos = []
    global_seen = set()
    for sel in user_selections or []:
        label = unique_label(selection_label(sel), set(columns) | set(user_columns) | {"Global"})
        combos = filter_combos_by_strategy(expand_selection(sel), strategy_patterns)
        for combo in combos:
            if combo not in global_seen:
                global_seen.add(combo)
                global_combos.append(combo)
        user_columns[label] = _agg(combos)

    columns["Global"] = _agg(global_combos) if global_combos else empty_fn()
    columns.update(user_columns)
    return columns
