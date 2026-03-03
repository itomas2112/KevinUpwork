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