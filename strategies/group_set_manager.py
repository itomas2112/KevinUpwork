"""
Group Set management — save, load, delete, import/export for Grid Search group sets.

Two modes per set:
  - MODE_RUNTIME ("events_at_runtime"): legacy / default. Candidates store
    only (group, element1, element2/value); event(s) are chosen at search time
    and cross-producted across every candidate.
  - MODE_PER_CANDIDATE ("events_per_candidate"): each candidate also stores
    its own `event`; no cross-product at search time.
"""
import streamlit as st
import json
import os

from config.constants import (
    GROUP_MAP, R_PROFIT_LOSS_ELEMENTS, ATR_TARGET_ELEMENTS,
    ELEMENT_TO_WFO_GROUP,
)

GROUP_SETS_FILE = "saved_group_sets.json"

MODE_RUNTIME = "events_at_runtime"
MODE_PER_CANDIDATE = "events_per_candidate"
VALID_MODES = (MODE_RUNTIME, MODE_PER_CANDIDATE)

# Group that owns ATR-related specials (ATR Stop, ATR Target).
_ATR_VOLUME_GROUP_NAME = "ATR / Volume Group"


def get_mode(group_set):
    """Return the mode of a group set, defaulting to MODE_RUNTIME for legacy data."""
    return group_set.get("mode", MODE_RUNTIME)


# ---------------------------------------------------------------------------
# Set-level "view" filtering: toggle indicators on/off + per-group value ranges
# ---------------------------------------------------------------------------

# Reverse lookup: element name -> indicator group name.
# Built lazily and cached because GROUP_MAP doesn't include EMA entries
# (those are appended dynamically); EMA elements all map back to
# "Price & Indicators" by construction.
_ELEMENT_TO_GROUP = None


def _element_to_group():
    global _ELEMENT_TO_GROUP
    if _ELEMENT_TO_GROUP is None:
        m = {}
        for grp, elements in GROUP_MAP.items():
            for el in elements:
                m[el] = grp
        _ELEMENT_TO_GROUP = m
    return _ELEMENT_TO_GROUP


def candidate_groups(candidate):
    """Return the set of indicator GROUP_NAMES this candidate touches.

    - ATR Stop and ATR Target both touch the ATR / Volume group.
    - Standard candidates touch the group of element1 (from candidate['group'])
      AND, when comparing to another indicator, the group of element2.
    - R Profit / R Loss elements aren't tied to an indicator group — they
      contribute nothing on their own. EMA "EMA N" elements always belong
      to Price & Indicators.
    """
    if candidate is None:
        return set()

    if candidate.get("stop_type") == "ATR":
        return {_ATR_VOLUME_GROUP_NAME}

    e1 = candidate.get("element1")
    if e1 == "ATR Target":
        return {_ATR_VOLUME_GROUP_NAME}

    groups = set()

    # element1 group: trust candidate['group'] when present, fall back to lookup
    if candidate.get("group") in GROUP_MAP:
        if e1 not in R_PROFIT_LOSS_ELEMENTS:
            groups.add(candidate["group"])
    elif e1 and e1 not in R_PROFIT_LOSS_ELEMENTS:
        g1 = _element_to_group().get(e1)
        if g1:
            groups.add(g1)
        elif isinstance(e1, str) and e1.startswith("EMA "):
            groups.add("Price & Indicators")

    # element2 group: only when comparing against another indicator
    if candidate.get("compare_type", "Indicator") == "Indicator":
        e2 = candidate.get("element2")
        if e2:
            g2 = _element_to_group().get(e2)
            if g2:
                groups.add(g2)
            elif isinstance(e2, str) and e2.startswith("EMA "):
                groups.add("Price & Indicators")

    return groups


def extract_groups_from_set(candidates):
    """Union of indicator groups touched by any candidate. Sorted list."""
    found = set()
    for c in candidates:
        found.update(candidate_groups(c))
    return sorted(found)


def extract_value_eligible_groups(candidates):
    """Groups that have at least one Fixed Value candidate. These are the
    groups for which a numeric (low, high, step) range filter is meaningful.
    Sorted list."""
    eligible = set()
    for c in candidates:
        if c.get("compare_type") != "Fixed Value":
            continue
        if c.get("value") is None:
            continue
        # The value is on element1's side of the comparison
        eligible.update(candidate_groups(c))
    return sorted(eligible)


def value_in_filter(value, vf):
    """Return True iff `value` is within [low, high] AND, if step > 0, lies on
    the (low + k*step) sequence within float tolerance.
    `vf` is {"low": float, "high": float, "step": float}."""
    if value is None or vf is None:
        return True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True

    low = float(vf.get("low", float("-inf")))
    high = float(vf.get("high", float("inf")))
    step = float(vf.get("step", 0) or 0)
    eps = 1e-6

    if v < low - eps or v > high + eps:
        return False

    if step > 0:
        # nearest grid value
        k = round((v - low) / step)
        if k < 0:
            return False
        grid_v = low + k * step
        if abs(grid_v - v) > eps + step * 1e-9:
            return False
        if grid_v > high + eps:
            return False

    return True


def apply_view(candidates, active_groups=None, value_filters=None):
    """Filter candidates by the set's view — toggled-off indicator groups
    and per-group value range constraints.

    `active_groups`: iterable of GROUP_NAMES that are kept. None or empty
      iterable both mean "no group filter" (everything passes).
      Passing the empty set explicitly via set() *does* mean "drop all".
    `value_filters`: {group_name: {low, high, step}}; absent groups skip the check.
    """
    out = []
    if active_groups is None:
        active_set = None
    else:
        active_set = set(active_groups)
    value_filters = value_filters or {}

    for c in candidates:
        groups = candidate_groups(c)
        # Group toggle filter: drop candidate if any of the groups it touches
        # is *not* in the active set.
        if active_set is not None:
            if groups and not groups.issubset(active_set):
                continue

        # Value range filter: applies to Fixed Value candidates only,
        # checked against any group filter that overlaps with this candidate.
        if c.get("compare_type") == "Fixed Value" and c.get("value") is not None:
            keep = True
            for g in groups:
                vf = value_filters.get(g)
                if vf and not value_in_filter(c["value"], vf):
                    keep = False
                    break
            if not keep:
                continue

        out.append(c)

    return out


# ---------------------------------------------------------------------------
# Indicator ranges per group set — Grid Search variant generation
# ---------------------------------------------------------------------------

# Each rangeable indicator group has ONE primary parameter that the user
# varies. Multi-param ranges within a group can be added later; for v1 we
# keep the offset numbering one-dimensional and easy to read in run labels.
INDICATOR_PRIMARY_PARAM = {
    "rsi": "rsi_window",
    "stoch": "stoch_k_period",
    "adx": "adx_period",
    "atr": "atr_period",
    "macd": "macd_fast",
    "supertrend": "supertrend_period",
    "bb": "bb_upper_period",
    "kc": "kc_upper_ema",
    "donchian": "dc_upper_period",
    "psar": "psar_af_max",
    "willr": "willr_period",
    "roc": "roc_period",
    "cci": "cci_period",
    "lr": "lr_period",
    # ema intentionally omitted — per-EMA ranges have a different shape
    # (one range per EMA index) and aren't supported in v1.
}

# Param dtype hints — int parameters get rounded; float parameters don't.
_INT_PARAMS = {
    "rsi_window", "stoch_k_period", "stoch_k_smooth", "stoch_d_smooth",
    "adx_period", "atr_period", "macd_fast", "macd_slow", "macd_signal",
    "supertrend_period",
    "bb_upper_period", "bb_mid_period", "bb_lower_period",
    "kc_upper_ema", "kc_mid_ema", "kc_lower_ema", "kc_atr_period",
    "dc_upper_period", "dc_mid_period", "dc_lower_period", "dc_offset",
    "willr_period", "roc_period", "roc_signal_period", "cci_period",
    "lr_period",
    "pc_upper_period", "pc_lower_period",
}


def candidate_wfo_groups(candidate):
    """Return the set of WFO group keys (e.g. 'rsi', 'bb') that this candidate
    references on either side of its comparison. Used to figure out which
    indicator ranges are relevant when expanding the candidate into variants.

    Empty set means the candidate doesn't touch any rangeable indicator group
    (e.g. Price vs Tenkan only references untunable indicators)."""
    if candidate is None:
        return set()

    if candidate.get("stop_type") == "ATR":
        return {"atr"}

    e1 = candidate.get("element1")
    if e1 == "ATR Target":
        return {"atr"}
    if e1 in R_PROFIT_LOSS_ELEMENTS:
        e1 = None

    groups = set()
    for el in (e1, candidate.get("element2") if candidate.get("compare_type", "Indicator") == "Indicator" else None):
        if not el:
            continue
        if el.startswith("EMA "):
            # EMA isn't supported as a ranged group in v1 — skip.
            continue
        g = ELEMENT_TO_WFO_GROUP.get(el)
        if g:
            groups.add(g)
    return groups


def enumerate_variant_values(min_val, max_val, step):
    """Return the list of values from min to max in `step` increments,
    inclusive of `max` when it lies on the grid (within float tolerance).

    Always returns at least one value: if range is degenerate (min == max
    or step <= 0), returns [min_val]."""
    try:
        lo = float(min_val)
        hi = float(max_val)
        st_ = float(step)
    except (TypeError, ValueError):
        return [min_val]

    if st_ <= 0 or hi <= lo:
        return [lo]

    out = []
    eps = 1e-9
    n = int(round((hi - lo) / st_))
    for k in range(n + 1):
        v = lo + k * st_
        if v > hi + eps:
            break
        out.append(v)
    if not out:
        out.append(lo)
    return out


def _coerce_param_value(param_name, v):
    """Round/cast a numeric variant value to the dtype expected by the indicator function."""
    if param_name in _INT_PARAMS:
        return int(round(float(v)))
    return float(v)


def compute_offsets(values):
    """Return a list of (offset_int, value) for the input value sequence.

    offset 0 is assigned to the value closest to the midpoint of [min, max].
    On ties the LOWER value gets offset 0. Other values get +1, +2 stepping up
    and -1, -2 stepping down from there."""
    if not values:
        return []
    if len(values) == 1:
        return [(0, values[0])]

    sorted_vals = sorted(values)
    lo = sorted_vals[0]
    hi = sorted_vals[-1]
    midpoint = (lo + hi) / 2.0

    # Pick the index of the "(0)" value: closest to midpoint, lower-bias on tie.
    best_idx = 0
    best_dist = float("inf")
    for i, v in enumerate(sorted_vals):
        d = abs(v - midpoint)
        # strict < so a later (higher) value with the same distance does NOT win
        if d < best_dist - 1e-12:
            best_dist = d
            best_idx = i

    return [(i - best_idx, v) for i, v in enumerate(sorted_vals)]


def offset_label(offset):
    """Return the user-facing offset prefix: '(0)', '(-1)', '(+2)'."""
    if offset == 0:
        return "(0)"
    sign = "+" if offset > 0 else "-"
    return f"({sign}{abs(offset)})"


def group_variant_combos(group_set, base_settings=None):
    """For each ranged indicator group in this set, return its variant list.

    Returns: {group_key: [(offset_int, params_dict), ...]}
    where params_dict is the indicator-param overrides for that variant
    (e.g. {"rsi_window": 15} for an RSI variant).

    Groups with no indicator_ranges entry are absent from the result.
    Groups where the primary param's range is degenerate (one value) appear
    with a single (0, params) entry."""
    base_settings = base_settings or {}
    ranges = group_set.get("indicator_ranges") or {}
    out = {}

    for group_key, params in ranges.items():
        primary_param = INDICATOR_PRIMARY_PARAM.get(group_key)
        if not primary_param:
            continue
        spec = params.get(primary_param)
        if not spec:
            continue
        try:
            lo, hi, step = float(spec[0]), float(spec[1]), float(spec[2])
        except (TypeError, ValueError, IndexError):
            continue

        raw_values = enumerate_variant_values(lo, hi, step)
        coerced_values = [_coerce_param_value(primary_param, v) for v in raw_values]
        # Dedupe consecutive duplicates that can arise from int-rounding
        deduped = []
        for v in coerced_values:
            if not deduped or deduped[-1] != v:
                deduped.append(v)

        offsets = compute_offsets(deduped)
        out[group_key] = [
            (offset, {primary_param: value})
            for offset, value in offsets
        ]
    return out


# ---------------------------------------------------------------------------
# Variant assignment helpers — mapping a candidate to a list of (variant_id,
# offsets_by_group) it should run as.
# ---------------------------------------------------------------------------


def variant_id_from_assignment(assignment):
    """Build a canonical, hashable, picklable id from a {group_key: offset} dict.
    Empty assignment -> None (the "default" variant)."""
    if not assignment:
        return None
    return tuple(sorted(assignment.items()))


def candidate_variant_assignments(candidate, variant_groups):
    """For a single candidate and a {group_key: [(offset, params)]} map of
    available indicator variants, return a list of variant assignments (one
    per Cartesian combination of variants for groups the candidate touches).

    Each item is (variant_id, offsets_by_group, params_overrides):
      - variant_id: canonical hashable key (or None for the default variant)
      - offsets_by_group: {group_key: offset_int}
      - params_overrides: merged param dict (e.g. {"rsi_window": 15})

    A candidate that touches no ranged group yields one item with None id."""
    if not variant_groups:
        return [(None, {}, {})]

    touched = candidate_wfo_groups(candidate) & set(variant_groups.keys())
    if not touched:
        return [(None, {}, {})]

    # Cartesian product of variants over the touched groups
    import itertools
    touched_sorted = sorted(touched)
    variant_lists = [variant_groups[g] for g in touched_sorted]

    out = []
    for combo in itertools.product(*variant_lists):
        offsets = {}
        params = {}
        for g, (offset, p) in zip(touched_sorted, combo):
            offsets[g] = offset
            params.update(p)
        vid = variant_id_from_assignment(offsets)
        out.append((vid, offsets, params))
    return out


def collect_unique_variants(candidates, variant_groups):
    """Walk every candidate and return the set of unique variant ids needed,
    plus a mapping variant_id -> params_overrides for that variant.

    Used by Grid Search to compute exactly the right number of indicator
    DataFrames (no duplication of work across candidates that share variants).
    """
    seen = {}  # variant_id -> params
    if not variant_groups:
        seen[None] = {}
        return seen

    for c in candidates:
        for vid, _offsets, params in candidate_variant_assignments(c, variant_groups):
            if vid not in seen:
                seen[vid] = params
    if not seen:
        seen[None] = {}
    return seen


# ---------------------------------------------------------------------------
# Generator — build a set's candidates from a selection of elements
# ---------------------------------------------------------------------------
#
# A group set may carry a "generator" block:
#   {"elements": [...], "fixed_values": {el: {min, max, step}},
#    "exclusions": [[a, b], ...], "exclude_same_indicator": bool,
#    "allow_cross_group": bool,
#    "r_profit" / "r_loss": {min, max, step} | None,
#    "atr_target" / "atr_stop": {period, min, max, step} | None}
# When present, `candidates` is regenerated from it on save / update / import;
# the stored candidates list stays the cache the search reads.

# Bands / lines of one indicator — "exclude_same_indicator" drops pairs inside
# one family. Ichimoku lines are deliberately not a family (Tenkan vs Kijun is
# a legitimate pair).
INDICATOR_FAMILIES = {
    "bb": {"BB Upper Band", "BB Middle Band", "BB Lower Band"},
    "kc": {"KC Upper Band", "KC Middle Band", "KC Lower Band"},
    "dc": {"DC Upper Band", "DC Middle Band", "DC Lower Band"},
    "supertrend": {"Supertrend", "Supertrend Upper", "Supertrend Lower"},
    "psar": {"PSAR", "PSAR Upper", "PSAR Lower"},
    "lr": {"LR Upper", "LR Middle", "LR Lower"},
    "pc": {"Price Upper", "Price Lower"},
}

# Price as Element 1 uses the bar high/low rule, so pairs with Price are
# generated in both directions.
_BOTH_WAYS_ELEMENTS = {"Price"}

_R_GROUP = "Price & Indicators"


def generator_element_group(element, ema_count=None):
    """GROUP_MAP group of a generator element, or None if unknown.
    `EMA N` belongs to Price & Indicators; with `ema_count` given, N must be
    within 1..ema_count."""
    g = _element_to_group().get(element)
    if g:
        return g
    if isinstance(element, str) and element.startswith("EMA "):
        try:
            n = int(element.split(" ", 1)[1])
        except ValueError:
            return None
        if n < 1 or (ema_count is not None and n > ema_count):
            return None
        return "Price & Indicators"
    return None


def _same_family(a, b):
    return any(a in fam and b in fam for fam in INDICATOR_FAMILIES.values())


def _unique(seq):
    return list(dict.fromkeys(seq))


def _range_values(spec):
    """Values of a {min, max, step} spec via enumerate_variant_values,
    rounded to kill float drift (1.0 + 3*0.2 -> 1.6)."""
    return [round(float(v), 10)
            for v in enumerate_variant_values(spec.get("min"), spec.get("max"),
                                              spec.get("step"))]


def _generate_parts(generator):
    """Return (pairs, fixed, r, atr) candidate lists for a generator spec."""
    generator = generator or {}
    elements = _unique(generator.get("elements") or [])
    exclusions = {frozenset(p) for p in (generator.get("exclusions") or [])
                  if isinstance(p, (list, tuple)) and len(p) == 2}
    same_ind = generator.get("exclude_same_indicator", True)
    cross = generator.get("allow_cross_group", False)

    def pair(a, b):
        return {"group": generator_element_group(a), "element1": a,
                "compare_type": "Indicator", "element2": b, "value": None}

    pairs = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            if a == b:
                continue
            if not cross and generator_element_group(a) != generator_element_group(b):
                continue
            if frozenset((a, b)) in exclusions:
                continue
            if same_ind and _same_family(a, b):
                continue
            pairs.append(pair(a, b))
            if b in _BOTH_WAYS_ELEMENTS or a in _BOTH_WAYS_ELEMENTS:
                pairs.append(pair(b, a))

    fixed = []
    fixed_values = generator.get("fixed_values") or {}
    for el in elements:
        spec = fixed_values.get(el)
        if not spec:
            continue
        for v in _range_values(spec):
            fixed.append({"group": generator_element_group(el), "element1": el,
                          "compare_type": "Fixed Value", "element2": None,
                          "value": v})

    r = []
    for key, el in (("r_profit", "R Profit"), ("r_loss", "R Loss")):
        spec = generator.get(key)
        if spec:
            for v in _range_values(spec):
                r.append({"group": _R_GROUP, "element1": el,
                          "compare_type": "Fixed Value", "element2": None,
                          "value": v})

    atr = []
    spec = generator.get("atr_target")
    if spec:
        for v in _range_values(spec):
            atr.append({"element1": "ATR Target",
                        "atr_period": int(spec.get("period", 14)),
                        "atr_multiplier": v})
    spec = generator.get("atr_stop")
    if spec:
        for v in _range_values(spec):
            atr.append({"stop_type": "ATR",
                        "atr_period": int(spec.get("period", 14)),
                        "atr_multiplier": v})

    return pairs, fixed, r, atr


def _dedupe_list(candidates):
    gs = {"candidates": candidates}
    _deduplicate_candidates(gs)
    return gs["candidates"]


def generate_candidates(generator, ema_count=0):
    """Deterministic, ordered, de-duplicated candidate list from a generator spec.
    Order: pairs in element-list order (a before b), Price both ways, then fixed values
    per element in element order, then R Profit, R Loss, ATR Target, ATR Stop.

    `ema_count` is accepted for symmetry with validate_generator; generation
    itself doesn't depend on it (run validate_generator first)."""
    pairs, fixed, r, atr = _generate_parts(generator)
    return _dedupe_list(pairs + fixed + r + atr)


def count_candidates(generator, ema_count=0):
    """{"pairs": n, "fixed": n, "r": n, "atr": n, "total": n} for a generator spec."""
    pairs, fixed, r, atr = _generate_parts(generator)
    return {"pairs": len(_dedupe_list(pairs)), "fixed": len(_dedupe_list(fixed)),
            "r": len(_dedupe_list(r)), "atr": len(_dedupe_list(atr)),
            "total": len(generate_candidates(generator, ema_count))}


def _range_errors(label, spec, need_period=False):
    if not isinstance(spec, dict):
        return [f"{label}: expected an object with min / max / step."]
    errs = []
    try:
        lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    except (KeyError, TypeError, ValueError):
        return [f"{label}: min / max / step must be numbers."]
    if lo > hi:
        errs.append(f"{label}: min ({lo:g}) is greater than max ({hi:g}).")
    if step <= 0:
        errs.append(f"{label}: step must be greater than 0.")
    if need_period:
        try:
            period = float(spec.get("period", 14))
            if period < 1 or period != int(period):
                errs.append(f"{label}: period must be a whole number ≥ 1.")
        except (TypeError, ValueError):
            errs.append(f"{label}: period must be a whole number ≥ 1.")
    return errs


def validate_generator(generator, ema_count=0):
    """Return a list of human-readable problems with a generator spec
    ([] = valid). `ema_count=None` skips the EMA-count limit."""
    if not isinstance(generator, dict):
        return ["Generator must be an object."]
    errs = []
    elements = generator.get("elements") or []
    if not isinstance(elements, list):
        return ["Generator 'elements' must be a list."]

    def check_element(el, where):
        if generator_element_group(el, ema_count) is not None:
            return True
        if (isinstance(el, str) and el.startswith("EMA ")
                and generator_element_group(el) is not None):
            errs.append(f"{where}: {el} is not available — the selected strategy "
                        f"has {ema_count} EMA(s).")
        else:
            errs.append(f"{where}: unknown element {el!r}.")
        return False

    for el in elements:
        check_element(el, "Elements")

    fixed_values = generator.get("fixed_values") or {}
    if not isinstance(fixed_values, dict):
        errs.append("Fixed values must be an object keyed by element.")
        fixed_values = {}
    for el, spec in fixed_values.items():
        if el not in elements:
            errs.append(f"Fixed values: {el!r} is not one of the selected elements.")
            continue
        errs.extend(_range_errors(f"Fixed values for {el}", spec))

    exclusions = generator.get("exclusions") or []
    if not isinstance(exclusions, list):
        errs.append("Exclusions must be a list of pairs.")
        exclusions = []
    for i, p in enumerate(exclusions, 1):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            errs.append(f"Exclusion #{i}: expected a pair of two elements.")
            continue
        ok = all([check_element(el, f"Exclusion #{i}") for el in p])
        if ok and p[0] == p[1]:
            errs.append(f"Exclusion #{i}: {p[0]} vs itself is never generated.")

    for key, label in (("r_profit", "R Profit"), ("r_loss", "R Loss")):
        if generator.get(key):
            errs.extend(_range_errors(label, generator[key]))
    for key, label in (("atr_target", "ATR Target"), ("atr_stop", "ATR Stop")):
        if generator.get(key):
            errs.extend(_range_errors(label, generator[key], need_period=True))

    if not errs and not generate_candidates(generator, ema_count):
        errs.append("The selection generates no candidates.")
    return errs


def _apply_generator(group_set, ema_count=None):
    """If `group_set` has a generator block, validate it and regenerate its
    candidates in place. Returns the number of candidates generated, or None
    for sets without a generator. Raises ValueError on an invalid spec."""
    generator = group_set.get("generator")
    if generator is None:
        return None
    errors = validate_generator(generator, ema_count)
    if errors:
        raise ValueError("Invalid generator: " + " ".join(errors))
    group_set["mode"] = MODE_RUNTIME
    group_set["candidates"] = generate_candidates(generator, ema_count)
    return len(group_set["candidates"])


# ---------------------------------------------------------------------------


def _deduplicate_candidates(group_set):
    """Remove duplicate candidates from a group set in-place. Returns count of removed duplicates."""
    candidates = group_set.get("candidates", [])
    seen = set()
    unique = []
    for c in candidates:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    removed = len(candidates) - len(unique)
    group_set["candidates"] = unique
    return removed


def _strip_legacy_fields(data):
    """Strip old type-specific fields from imported/legacy group sets.

    `event` on a candidate is legacy junk in MODE_RUNTIME but is meaningful
    (and required) in MODE_PER_CANDIDATE — only strip it for runtime mode.
    `operator` is always legacy.
    """
    data.pop("type", None)
    mode = get_mode(data)
    for cand in data.get("candidates", []):
        if mode == MODE_RUNTIME:
            cand.pop("event", None)
        cand.pop("operator", None)


def load_group_sets():
    """Load group sets from disk. Returns list of dicts."""
    if os.path.exists(GROUP_SETS_FILE):
        try:
            with open(GROUP_SETS_FILE, "r") as f:
                sets = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
        # Backfill missing mode field on legacy data
        for gs in sets:
            gs.setdefault("mode", MODE_RUNTIME)
        return sets
    return []


def save_group_sets_to_file():
    """Write session state group sets to disk."""
    with open(GROUP_SETS_FILE, "w") as f:
        json.dump(st.session_state.get("saved_group_sets", []), f, indent=4)


def save_group_set(group_set, ema_count=None):
    """Append a group set (after deduplication) and persist. Returns number of duplicates removed.

    Sets with a generator block get their candidates regenerated first
    (`ema_count` from the selected strategy; None skips the EMA limit check).
    Raises ValueError if the generator is invalid."""
    _apply_generator(group_set, ema_count)
    group_set.setdefault("mode", MODE_RUNTIME)
    removed = _deduplicate_candidates(group_set)
    st.session_state.setdefault("saved_group_sets", []).append(group_set)
    save_group_sets_to_file()
    return removed


def update_group_set(idx, group_set, ema_count=None):
    """Replace a group set at index (after deduplication) and persist. Returns number of duplicates removed.

    Sets with a generator block get their candidates regenerated first (see
    save_group_set). Raises ValueError if the generator is invalid."""
    _apply_generator(group_set, ema_count)
    group_set.setdefault("mode", MODE_RUNTIME)
    removed = _deduplicate_candidates(group_set)
    sets = st.session_state.get("saved_group_sets", [])
    if 0 <= idx < len(sets):
        sets[idx] = group_set
        save_group_sets_to_file()
    return removed


def delete_group_set(idx):
    """Delete a group set by index and persist."""
    sets = st.session_state.get("saved_group_sets", [])
    if 0 <= idx < len(sets):
        sets.pop(idx)
        save_group_sets_to_file()


def export_group_set(group_set):
    """Return JSON string for download."""
    return json.dumps(group_set, indent=2)


def import_group_set(json_str, ema_count=None):
    """Parse and validate a group set from JSON string. Returns dict or raises ValueError.

    Accepts both legacy format (no mode field) and the two current modes.
    Legacy fields are stripped automatically.

    Files with a generator block are validated (`ema_count` from the selected
    strategy; None skips the EMA limit) and their candidates regenerated —
    any `candidates` in the file are ignored. The count is reported back as
    `_regenerated` on the returned dict.
    """
    data = json.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    if "name" not in data:
        raise ValueError("Missing required field: 'name'")
    if data.get("generator") is not None:
        data.pop("_duplicates_removed", None)
        data["_regenerated"] = _apply_generator(data, ema_count)
        return data
    if "candidates" not in data:
        raise ValueError("Missing required field: 'candidates'")
    if not isinstance(data["candidates"], list):
        raise ValueError("'candidates' must be a list.")

    mode = data.get("mode", MODE_RUNTIME)
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}. Expected one of {VALID_MODES}.")
    data["mode"] = mode

    # Strip legacy type-specific fields (mode-aware: keeps `event` for MODE_PER_CANDIDATE)
    _strip_legacy_fields(data)

    if mode == MODE_PER_CANDIDATE:
        for i, cand in enumerate(data["candidates"]):
            if not cand.get("event"):
                raise ValueError(
                    f"Candidate #{i + 1}: missing 'event' (required for "
                    f"mode={MODE_PER_CANDIDATE!r}).")

    removed = _deduplicate_candidates(data)
    if removed:
        data["_duplicates_removed"] = removed
    return data
