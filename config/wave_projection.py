"""
Projecting wave markings between the canonical timeframe and a displayed one.

A wave count belongs to the price action, not to the aggregation it happened to
be drawn on. So markings are stored once, at the *base* timeframe of the
uploaded CSV (the canonical resolution), and every other aggregation on screen
is a projection of that one list. An edit made at 1D is refined back down to a
base bar before it is stored.

The snapping rule is the client's, verbatim: a marking snaps to the highest
high (or lowest low) of the period it is associated with, and when two periods
of the smaller timeframe carry that same extreme, the later one wins.

Two directions fall out of it:

* **Coarsen** -- a canonical point lands in whichever display period contains
  its time, and takes *that period's* high (for ``kind: "high"``) or low (for
  ``kind: "low"``). The pivot genuinely moves to the period's extreme; that is
  the rule, not an approximation of it.
* **Refine** -- among the base bars inside a display period, the ones whose
  high equals the period high (or low equals the period low) are candidates,
  and the last of them wins.

``to_canonical(to_display(p))`` is therefore **not** the identity, and that is
correct rather than a bug: a 15m pivot that is not its day's extreme coarsens
to the day's extreme, which refines back to a different 15m bar. The canonical
list is never rewritten from a projection -- only an edit the client actually
made at that aggregation travels back down. ``to_display(to_canonical(q))``
*is* the identity for every display point ``q``, and that is the direction the
tests pin.

Pure and Streamlit-free, like everything else under ``config/``.
"""
import math
from collections import namedtuple
from itertools import repeat

import numpy as np
import pandas as pd

from config.wave_analysis import POINT_KINDS

# Mirrors the ``tf_map`` inside ``data.loader.resample_ohlc``. Duplicated rather
# than imported so ``config/`` stays a leaf package; the grid-agreement test
# compares ``display_times`` against ``resample_ohlc`` itself on every
# timeframe, so the two cannot drift apart unnoticed.
RESAMPLE_RULES = {"1H": "1h", "4H": "4h", "1D": "1D"}

DISPLAY = "display"
CANONICAL = "canonical"

PeriodMap = namedtuple("PeriodMap", [
    "timeframe",
    "base_timeframe",
    "display_times",
    "bucket_of",
    "extreme_time",
    "extreme_price",
])
PeriodMap.__doc__ = """Everything a projection between two timeframes needs.

Plain tuples, dicts and Python scalars throughout, so the whole thing pickles
(the grid-search workers' constraint, and a cheap property to keep).

* ``timeframe`` / ``base_timeframe`` -- the two ends of the projection.
* ``display_times`` -- the display bars' times as ints, ascending. Equal to
  ``[int(ts.timestamp()) for ts in resample_ohlc(df, tf, base).index]``,
  including the ``dropna(subset=['high'])`` that removes empty bins: a point
  projected onto a bar the chart does not have would simply vanish.
* ``bucket_of`` -- base time -> display time, for every base bar whose display
  period survived that dropna.
* ``extreme_time`` -- ``(display time, kind)`` -> the base time of the bar
  carrying that period's extreme, the last one on a tie.
* ``extreme_price`` -- ``(display time, kind)`` -> that period's high or low.
  This doubles as the refined point's price: the bar named by ``extreme_time``
  is by construction the one whose high *is* the period high (or whose low is
  the period low), so a separate base-bar price table would hold the same
  number under a different key. Not storing one matters -- on the client's
  ten-year 15m file it was 520k entries and the bulk of the map's memory.
"""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _price_of(point):
    """A point's price as a finite float, or None when it has no usable one.

    A NaN would poison every comparison it takes part in -- ``<`` and ``==``
    are both False against it -- so the pivot carrying one is refused outright
    rather than left to win a contest by never losing one.
    """
    value = point.get("price")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _epoch_seconds(index):
    """A DatetimeIndex as the ints ``int(ts.timestamp())`` would produce.

    The payload the frontend holds is built with ``int(ts.timestamp())``, so the
    whole map has to speak in exactly those numbers. Converting through
    ``datetime64[s]`` rather than dividing the raw integer view matters: an
    index's storage unit is not always nanoseconds (a CSV loaded under pandas 3
    lands on microseconds), so the divisor is not a constant.
    """
    values = index
    if getattr(values, "tz", None) is not None:
        values = values.tz_convert("UTC").tz_localize(None)
    return values.to_numpy(dtype="datetime64[s]").astype("int64")


def _extreme_rows(codes, values, kind):
    """Row positions of each bin's extreme bar, last one on a tie.

    ``codes`` is the bin number per row and ``values`` the column being
    extremised. Sorting by ``(bin, value, position)`` puts each bin's rows in a
    contiguous block ordered by value, so the block's **last** row is the bin's
    maximum and -- because position is the final key -- the latest of any rows
    sharing that maximum. That last part is the client's tie-break, and without
    ``np.arange(n)`` in the key it would resolve on whatever order the sort
    happened to produce rather than on time.

    Lows flip the sign so "last of the block" keeps meaning "the extreme".
    Missing prices are pushed to the losing end so that a bar with no high can
    never be picked as its period's high -- which is also what ``resample``'s
    NaN-skipping max does, and the two must agree.

    Vectorised on purpose: this runs over a quarter of a million rows on the
    client's ten-year Gold file, where a ``groupby().apply()`` would be seconds.
    """
    if len(codes) == 0:
        return np.empty(0, dtype="int64")

    if kind == "high":
        keys = np.where(np.isnan(values), -np.inf, values)
    else:
        keys = np.where(np.isnan(values), -np.inf, -values)

    count = len(codes)
    order = np.lexsort((np.arange(count), keys, codes))
    ordered_codes = codes[order]
    # Last row of each block: the next row starts a different bin, or there is
    # no next row.
    block_end = np.empty(count, dtype=bool)
    block_end[-1] = True
    np.not_equal(ordered_codes[1:], ordered_codes[:-1], out=block_end[:-1])
    return order[block_end]


def _identity_map(df_base, base_timeframe):
    """The map for ``timeframe == base_timeframe``: every bar is its own period.

    Built directly instead of running a resample, which is both wasted work and
    a needless chance to disagree with ``resample_ohlc``'s own shortcut (it
    returns the frame unchanged, empty bins and all, so no dropna applies here).
    """
    times = _epoch_seconds(df_base.index).tolist()
    highs = df_base["high"].to_numpy(dtype="float64").tolist()
    lows = df_base["low"].to_numpy(dtype="float64").tolist()

    prices = dict(zip(zip(times, repeat("high")), highs))
    prices.update(zip(zip(times, repeat("low")), lows))
    self_times = dict(zip(zip(times, repeat("high")), times))
    self_times.update(zip(zip(times, repeat("low")), times))

    return PeriodMap(
        timeframe=base_timeframe,
        base_timeframe=base_timeframe,
        display_times=times,
        bucket_of=dict(zip(times, times)),
        extreme_time=self_times,
        extreme_price=prices,
    )


def period_map(df_base, timeframe, base_timeframe="15m"):
    """Everything a projection between two timeframes needs, built once.

    ``df_base`` is the base-resolution OHLC frame, already deduped exactly as
    ``ui.wave_analysis_tab.dedupe_bars`` leaves it -- the caller is responsible
    for that, because the payload the frontend holds is built from the same
    frame and the two must agree on which bars exist.
    """
    if timeframe == base_timeframe:
        return _identity_map(df_base, base_timeframe)

    rule = RESAMPLE_RULES.get(timeframe)
    if rule is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # ``df.resample(rule)`` *is* ``df.groupby(pd.Grouper(freq=rule))``, so going
    # through the grouper gives bin edges identical to ``resample_ohlc``'s by
    # construction, and hands back the per-row bin number in the same breath.
    # Computing bucket boundaries by floor/modulo instead would be a coin toss
    # on pandas' origin handling, and a one-bar offset is invisible in a unit
    # test and catastrophic on the client's chart.
    grouped = df_base[["high", "low"]].groupby(pd.Grouper(freq=rule))
    all_periods = grouped.agg({"high": "max", "low": "min"})
    # ``ngroup`` numbers the bins by their position in that aggregate, empty
    # bins included, which is exactly the index the codes below are used as.
    codes = grouped.ngroup().to_numpy()

    # The same dropna ``resample_ohlc`` applies, expressed as a mask so it can
    # also be asked of a base row: a bar whose whole period was dropped has no
    # bar on screen to project onto.
    kept = all_periods["high"].notna().to_numpy()
    periods = all_periods[kept]

    period_times = _epoch_seconds(all_periods.index)
    display_times = period_times[kept]
    base_times = _epoch_seconds(df_base.index)

    # ``ngroup`` returns -1 for a row it could not place at all, which would
    # otherwise index the *last* bin and misbin it silently.
    in_view = (codes >= 0) & kept[codes]
    view_codes = codes[in_view]
    view_times = base_times[in_view]
    view_time_list = view_times.tolist()
    view_high = df_base["high"].to_numpy(dtype="float64")[in_view]
    view_low = df_base["low"].to_numpy(dtype="float64")[in_view]

    bucket_of = dict(zip(view_time_list, period_times[view_codes].tolist()))

    display_list = display_times.tolist()
    extreme_price = dict(zip(zip(display_list, repeat("high")),
                             periods["high"].to_numpy(dtype="float64").tolist()))
    extreme_price.update(zip(zip(display_list, repeat("low")),
                             periods["low"].to_numpy(dtype="float64").tolist()))

    extreme_time = {}
    for kind, values in (("high", view_high), ("low", view_low)):
        rows = _extreme_rows(view_codes, values, kind)
        extreme_time.update(zip(
            zip(period_times[view_codes[rows]].tolist(), repeat(kind)),
            view_times[rows].tolist(),
        ))

    return PeriodMap(
        timeframe=timeframe,
        base_timeframe=base_timeframe,
        display_times=display_list,
        bucket_of=bucket_of,
        extreme_time=extreme_time,
        extreme_price=extreme_price,
    )


# ----------------------------------------------------------------- projecting


def to_display(point, pmap):
    """A canonical point coarsened onto a display bar, or None if unmappable.

    Unmappable means the point's time is not a bar of the frame the map was
    built from -- the data changed under the marking. The caller drops the
    whole pattern rather than half of it.
    """
    if not isinstance(point, dict):
        return None
    time = point.get("time")
    kind = point.get("kind")
    if not _is_int(time) or kind not in POINT_KINDS:
        return None

    bucket = pmap.bucket_of.get(time)
    if bucket is None:
        return None
    price = pmap.extreme_price.get((bucket, kind))
    if price is None:
        return None
    return {"time": bucket, "price": price, "kind": kind}


def to_canonical(point, pmap):
    """A display point refined down to the base bar carrying its extreme.

    None when the point does not sit on a display bar of this map.
    """
    if not isinstance(point, dict):
        return None
    time = point.get("time")
    kind = point.get("kind")
    if not _is_int(time) or kind not in POINT_KINDS:
        return None

    base_time = pmap.extreme_time.get((time, kind))
    if base_time is None:
        return None
    # The extreme bar's own high (or low) *is* the period's, so the period's
    # price is the refined point's price -- see PeriodMap's note on why no
    # per-base-bar price table is kept.
    price = pmap.extreme_price.get((time, kind))
    if price is None:
        return None
    return {"time": base_time, "price": price, "kind": kind}


def _project_points(points, pmap, direction):
    """Every point projected, or None if the pattern cannot survive it."""
    if direction == DISPLAY:
        project = to_display
    elif direction == CANONICAL:
        project = to_canonical
    else:
        raise ValueError(f"Unknown projection direction: {direction!r}")

    if not isinstance(points, list) or not points:
        return None

    projected = []
    for point in points:
        moved = project(point, pmap)
        if moved is None:
            return None
        projected.append(moved)

    # Collapse: two consecutive points landing on the same bar. A chart cannot
    # draw a zero-length leg and ``is_valid_pattern`` demands strictly
    # increasing times, so the pattern is dropped whole. Nudging a time to make
    # it fit would silently corrupt the client's marking instead.
    for earlier, later in zip(projected, projected[1:]):
        if later["time"] <= earlier["time"]:
            return None
    return projected


def project_pattern(pattern, pmap, direction=DISPLAY):
    """A pattern with its points projected, or None if it cannot be drawn.

    Identity, type, variation, degree and colour ride along untouched -- those
    live in canonical space and a projection has no opinion about them. Pure:
    a new dict comes out and the input is never touched.
    """
    if not isinstance(pattern, dict):
        return None
    projected = _project_points(pattern.get("points"), pmap, direction)
    if projected is None:
        return None
    return dict(pattern, points=projected)


def project_patterns(patterns, pmap, direction=DISPLAY):
    """A pattern list projected, dropping whatever cannot survive the trip."""
    if not isinstance(patterns, list):
        return []
    projected = []
    for pattern in patterns:
        moved = project_pattern(pattern, pmap, direction)
        if moved is not None:
            projected.append(moved)
    return projected


def projection_collapses(pattern, pmap, direction=DISPLAY):
    """True when ``project_patterns`` would drop this pattern.

    Both reasons count -- adjacent points sharing a display bar, and a point on
    a bar the frame no longer has -- because what the caller needs is the count
    of markings that will not appear, so it can say so rather than let them go
    missing in silence.
    """
    return project_pattern(pattern, pmap, direction) is None


# ------------------------------------------------------------------- magnetism


def pivot_magnets(patterns, pmap):
    """``{(display time, kind): [candidate, ...]}`` for every existing pivot.

    A candidate is ``(canonical time, price, pattern id, point index)``: what an
    incoming click in that display period, on that side of the bar, could attach
    to rather than snapping to the period's own extreme.

    This is not a special case bolted onto refinement -- it is how wave marking
    works. Elliott counts share pivots constantly: chained siblings share an
    endpoint, and every child shares both endpoints with a parent leg. Without
    it a parent drawn at 15m and a child drawn at 1D can never meet, because the
    parent's pivot is usually *not* its day's extreme, so the child's endpoints
    refine to different base bars and the parent/child relation -- which matches
    on exact (time, kind) -- is never established.

    Every pattern takes part, red ones included: a red pattern's pivot is still
    a real pivot the client may be counting from, and excluding it would make
    magnetism come and go as an unrelated overlap is created and resolved.

    The price rides along with the time rather than being looked up later:
    ``PeriodMap`` holds no per-base-bar price table (Phase 11a dropped it
    deliberately, it was the bulk of the map's memory) and a magnetised point
    must carry its own bar's high or low, not the period's extreme. A canonical
    pivot's stored price already *is* that number -- it either came out of
    ``extreme_price`` for the very bar it names, or straight off the base bar the
    client clicked at the base timeframe -- so nothing has to be recomputed.

    **Every** pivot in the display bar is offered, not just one, and the list is
    ordered best-first: nearest that period's extreme, ties to the later time.
    The display bar is drawn *at* its extreme, so the pivot closest to it is the
    one the client is looking at, and a caller with no way to choose between them
    can take the head of the list and behave exactly as the price-distance rule
    always did. What the rest of the list buys is intent: at a coarse aggregation
    every same-kind pivot inside a bar projects onto the *same* display point and
    their glyphs stack, so the click itself cannot separate them, and only what
    the finished pattern would *mean* can (see ``refine_event``).

    One pivot can be offered several times over when patterns share it, which is
    the normal case in a nest: the same bar is a parent's leg end and its child's
    terminus. Every (pattern, point index) pairing is kept, so the list records
    which counts touch the bar rather than only that something does. Refinement
    itself no longer reads those two fields -- it recognises a shared pivot by
    ``(time, kind)``, the same way ``child_leg_index`` does, and collapses the
    copies -- but they are what makes the order below total, and they say at a
    glance which count a candidate came from when one has to be debugged.

    Beyond distance and time the order falls back to the pattern's position in
    the list and then the point's index. Those two decide nothing a client could
    perceive -- they only separate entries that name the very same bar at the very
    same price -- but without them a tie would resolve on dict iteration order,
    and the choice a caller makes off this list has to be reproducible.

    Empty for a ``pmap`` of None: at the base timeframe display *is* canonical,
    so there is no period to be misplaced inside and nothing to pull anywhere.
    """
    if pmap is None or not isinstance(patterns, list):
        return {}

    ranked = {}
    for order, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            continue
        points = pattern.get("points")
        if not isinstance(points, list):
            continue
        pattern_id = pattern.get("id")
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                continue
            time = point.get("time")
            kind = point.get("kind")
            if not _is_int(time) or kind not in POINT_KINDS:
                continue
            price = _price_of(point)
            if price is None:
                continue
            bucket = pmap.bucket_of.get(time)
            if bucket is None:
                continue                # the data changed under this marking
            extreme = pmap.extreme_price.get((bucket, kind))
            if extreme is None:
                continue
            # Distance leads; the negated time only decides when they are equal,
            # which for a fixed extreme means the two pivots sit at the same
            # price, and then the later one wins.
            ranked.setdefault((bucket, kind), []).append(
                ((abs(price - extreme), -time, order, index), (time, price, pattern_id, index)))

    return {key: [candidate for _rank, candidate in sorted(entries, key=lambda e: e[0])]
            for key, entries in ranked.items()}


def _candidates_for(point, magnets):
    """The magnets offered to a display point, best-first. Empty when none are."""
    if not magnets or not isinstance(point, dict):
        return ()
    time = point.get("time")
    kind = point.get("kind")
    if not _is_int(time) or kind not in POINT_KINDS:
        return ()
    return magnets.get((time, kind)) or ()


MAX_CANDIDATES_PER_POINT = 8


def _point_key(point):
    """A pivot's identity for the relation: the bar, and which side of it.

    The two fields ``child_leg_index`` matches on, so two keys that compare equal
    here name a pivot the relation will read as shared once the pattern is
    stored. Spelled out rather than imported from ``config.wave_analysis``: this
    module needs to recognise a shared pivot, not to reason about parents, and a
    copy of two comparisons is cheaper than a dependency that would tempt the
    rest of the relation in behind it.
    """
    return (point["time"], point["kind"])


def _point_candidates(point, pmap, magnets):
    """Every canonical bar one display click could refine to, best-first.

    The pivots already marked in that display period lead, in the order
    ``pivot_magnets`` ranked them -- nearest the period's extreme first -- and the
    period's own extreme comes last, as the answer that is always available. A
    point whose display bar carries no pivot at all therefore has a one-element
    list holding exactly the plain Phase 12 refinement.

    Keeping the extreme on the end is what stops stronger magnetism from costing
    the client a marking. Two consecutive clicks can magnet onto the same base
    bar, and a pattern whose times do not strictly increase is not a pattern; were
    the candidates only pivots, such a pair would have no ordered assignment at
    all and the whole completion would be refused -- four clicks lost to a rule
    meant to make them stick. The extreme is where Phase 13's ordering fallback
    already sent an offender, now offered to the search instead of applied to its
    result.

    Duplicates collapse by time: several patterns sharing one bar all offer that
    bar, and the search reads a candidate only through its ``(time, kind)`` key,
    so the copies would be work that cannot change an answer.
    """
    chosen = []
    seen = set()
    for candidate in _candidates_for(point, magnets):
        # Bounded so that a pathological pile-up of pivots inside a single
        # display bar cannot blow the search up: eight candidates a point over at
        # most six points is a few thousand comparisons, and the pivots dropped
        # are the farthest from the period's extreme -- the ones the click, which
        # was drawn at that extreme, resembles least.
        if len(chosen) >= MAX_CANDIDATES_PER_POINT - 1:
            break
        if candidate[0] in seen:
            continue
        seen.add(candidate[0])
        # Every candidate reached here was keyed on this point's own (time,
        # kind), so the kind is the one it was filed under.
        chosen.append({"time": candidate[0], "price": candidate[1],
                       "kind": point.get("kind")})

    plain = to_canonical(point, pmap)
    if plain is not None and plain["time"] not in seen:
        chosen.append(plain)
    return chosen


def _relation_pairs(patterns, exclude_id=None):
    """(consecutive-pair keys, endpoint-pair keys) over the existing patterns.

    Two sets of ``(key, key)``, built once per call because the search reads them
    many times over:

    * **consecutive pairs** -- every ``(points[k], points[k + 1])`` of every
      pattern. A new pattern whose own two ends match one of these spans that
      pattern's leg exactly, which is the whole of what makes it a *child* of it.
    * **endpoint pairs** -- every pattern's ``(points[0], points[-1])``. A new
      pattern one of whose legs matches one of these has that pattern hanging off
      the leg, so it *adopts* it as a child.

    Keys rather than (pattern, index) pairs, because that is what the relation
    itself compares: a candidate offered by pattern A can name the very bar
    pattern B's leg ends on, and ``child_leg_index`` would see the leg either way.

    Every pattern takes part regardless of colour, matching how ``pivot_magnets``
    already gathers pivots. Colour is an output of ``settle`` and may change on
    the very next pass, so letting it decide which relations are worth forming
    would make the choice flicker as an unrelated overlap is created and resolved.

    ``exclude_id`` drops one pattern, which a ``move_point`` needs: a pattern is
    never its own child, so its own pivots must not be able to score a relation
    against the rest of itself.
    """
    consecutive = set()
    endpoints = set()
    if not isinstance(patterns, list):
        return consecutive, endpoints

    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        if exclude_id is not None and pattern.get("id") == exclude_id:
            continue
        points = pattern.get("points")
        if not isinstance(points, list) or len(points) < 2:
            continue
        keys = []
        for point in points:
            if (not isinstance(point, dict) or not _is_int(point.get("time"))
                    or point.get("kind") not in POINT_KINDS):
                keys = None
                break
            keys.append(_point_key(point))
        if not keys:
            continue                    # half a pattern names no relation at all
        consecutive.update(zip(keys, keys[1:]))
        endpoints.add((keys[0], keys[-1]))

    return consecutive, endpoints


def _best_assignment(candidates, consecutive, endpoints):
    """One candidate per point, expressing as much structure as it can. Or None.

    A complete assignment ``c_0 ... c_n`` scores one point for the new pattern
    becoming a child -- ``(c_0, c_n)`` matching a consecutive pair of an existing
    pattern -- plus one for each of its legs matching an existing pattern's two
    ends, which adopts that one. The total is maximised, so a pattern that both
    slots into a parent and adopts two children beats one that merely slots in.

    That is what the client's own workflow needs. He marks bottom-up -- small
    degrees first, larger ones drawn over them afterwards -- so a new pattern's
    *interior* points are exactly what decide which children it adopts, and
    choosing them by price distance leaves the children unadopted, the degrees
    underived, and the patterns colliding at one degree and going red.

    Ordering is a constraint *inside* the search rather than a check after it: a
    transition is only ever offered when time strictly increases, so every
    assignment considered is already one ``is_valid_pattern`` will accept.
    Preferring structure first and discovering afterwards that the result cannot
    be stored is how a marking gets lost, or a point gets moved somewhere the
    client never clicked to rescue the sequence.

    Ties -- and a score of zero, which is every pattern expressing nothing -- go
    to the assignment whose candidates come earliest in their lists, position 0
    first. Earliest is the ``pivot_magnets`` order, nearest each period's extreme,
    so a pattern with no relation to express lands exactly where the plain
    price-distance rule of Phase 13 would have put it.

    None only when no ordering of the candidates exists at all.
    """
    count = len(candidates)
    if count == 0 or any(not offered for offered in candidates):
        return None

    keys = [[_point_key(point) for point in offered] for offered in candidates]

    best = None                     # ((-score, indices), indices)
    for start in range(len(candidates[0])):
        # One forward pass per choice of first point: the adoption term is local
        # to consecutive pairs, but the child-of term couples the two ends, so the
        # first point has to be pinned before the rest can be scored on their own.
        # State is the candidate chosen at a position; what is carried is the best
        # score reaching it and, among the paths scoring that, the earliest.
        reached = {start: (0, (start,))}
        for position in range(count - 1):
            following = {}
            for index, (score, path) in reached.items():
                time = candidates[position][index]["time"]
                for step, candidate in enumerate(candidates[position + 1]):
                    if candidate["time"] <= time:
                        continue
                    total = score + (
                        1 if (keys[position][index], keys[position + 1][step])
                        in endpoints else 0)
                    walked = path + (step,)
                    standing = following.get(step)
                    if standing is None or (-total, walked) < (-standing[0], standing[1]):
                        following[step] = (total, walked)
            reached = following
            if not reached:
                break

        for index, (score, path) in reached.items():
            total = score
            if count > 1 and (keys[0][start], keys[count - 1][index]) in consecutive:
                total += 1
            rank = (-total, path)
            if best is None or rank < best[0]:
                best = (rank, path)

    if best is None:
        return None
    return [candidates[position][index] for position, index in enumerate(best[1])]


def _pattern_by_id(patterns, pattern_id):
    """The canonical pattern carrying that id, or None."""
    if not isinstance(patterns, list) or not isinstance(pattern_id, str) or not pattern_id:
        return None
    for pattern in patterns:
        if isinstance(pattern, dict) and pattern.get("id") == pattern_id:
            return pattern
    return None


# --------------------------------------------------------------------- events


def _moved_point(event, point, pmap, magnets, patterns):
    """Where a dragged point lands: the candidate expressing the most structure.

    One point moves while the pattern's others stand still, so the assignment is
    fixed but for this one slot and the two terms of ``_best_assignment`` can be
    scored candidate by candidate against the rest of the pattern held as it is.

    An interior point matters as much as an endpoint here. An endpoint decides
    whether *this* pattern is a child of another; an interior point decides which
    children hang off it, and shaking one loose by dragging the pivot it hangs
    from is the same loss of structure arrived at from the other side.

    A candidate on or past a neighbour is **skipped**, not chosen and then refused
    by ``_apply_move_point``, which would swallow the drag: the client would watch
    his point snap back with nothing to show for it. When no candidate is in
    bounds there is nothing to choose between and the head of the list stands,
    exactly as it did before -- the reducer's verdict on that is unchanged too.
    """
    offered = _point_candidates(point, pmap, magnets)
    if not offered:
        return None

    pattern = _pattern_by_id(patterns, event.get("id"))
    points = pattern.get("points") if isinstance(pattern, dict) else None
    index = event.get("point_index")
    if (not isinstance(points, list) or len(points) < 2 or not _is_int(index)
            or not 0 <= index < len(points)):
        return offered[0]

    keys = []
    for other in points:
        if (not isinstance(other, dict) or not _is_int(other.get("time"))
                or other.get("kind") not in POINT_KINDS):
            return offered[0]       # a half-formed pattern names no relation
        keys.append(_point_key(other))

    before = keys[index - 1][0] if index > 0 else None
    after = keys[index + 1][0] if index + 1 < len(points) else None
    consecutive, endpoints = _relation_pairs(patterns, exclude_id=pattern.get("id"))

    best = None                     # ((-score, position), candidate)
    for position, candidate in enumerate(offered):
        if before is not None and candidate["time"] <= before:
            continue
        if after is not None and candidate["time"] >= after:
            continue
        trial = list(keys)
        trial[index] = _point_key(candidate)
        score = sum(1 for pair in zip(trial, trial[1:]) if pair in endpoints)
        if (trial[0], trial[-1]) in consecutive:
            score += 1
        rank = (-score, position)
        if best is None or rank < best[0]:
            best = (rank, candidate)

    return offered[0] if best is None else best[1]


def refine_event(event, pmap, magnets=None, patterns=None):
    """An event's coordinates moved from display space into canonical space.

    Returns the refined event, or None when it cannot be honoured -- a point
    that does not sit on a display bar of this map. ``pmap`` of None means the
    display *is* canonical and the event passes through unchanged.

    ``pattern_completed`` and ``move_point`` carry coordinates; every other
    event type is addressed by id alone and passes through untouched.

    The refined price always comes from the map, never from the event -- but
    not because the event's price is wrong. It is the display bar's extreme,
    and the base bar ``extreme_time`` names is by construction the one whose own
    high *is* that period's high, so the two numbers agree (swept over the
    client's ten-year file: 0 mismatches in 171k display bar/kind pairs). The
    map is used because it is the single source of truth for what a canonical
    point may say: a price that arrives stale from an un-acked overlay, rounded
    by a JSON round trip, or hand-crafted, cannot then be written to disk
    against a bar that never carried it.

    ``magnets`` -- see ``pivot_magnets`` -- redirects a click that landed on a
    display period already carrying a pivot of the same kind onto *that* pivot
    instead of the period's extreme, price included. None or empty keeps the
    plain behaviour exactly. Magnetism is only ever offered for a projection:
    at the base timeframe there is no ``pmap`` at all, and the client clicked
    the exact bar he meant, so pulling that click onto a neighbouring pivot
    would turn a deliberate mark into a silent correction.

    ``patterns`` -- the canonical list -- makes the choice *between* competing
    magnets intent-aware, which price distance alone cannot be. At a coarse
    aggregation every same-kind pivot inside a bar draws at the same display
    point, so the click cannot say which was meant; what can is what the finished
    pattern would mean. Every point of the incoming pattern is chosen together,
    for the assignment that expresses the most structure: the new pattern
    slotting into an existing one as a child, and each of its legs adopting an
    existing pattern as one. See ``_best_assignment`` for the scoring and for why
    the client's bottom-up workflow makes the *interior* points matter as much as
    the ends. With no relation available anywhere every point takes the head of
    its list, which is the plain price-distance behaviour of Phase 13, so nothing
    that already worked can regress.

    Both terms are read off the canonical list, so without it there is nothing to
    recognise and the choice degrades to that same head of each list. The tab
    always passes it; a caller that does not is asking for Phase 13.

    Pure: a new event dict comes out and the input is never touched. A
    structurally broken event is handed on unchanged rather than refused, so
    the reducers stay the single place that decides what is malformed.
    """
    if pmap is None or not isinstance(event, dict):
        return event

    event_type = event.get("type")

    if event_type == "pattern_completed":
        pattern = event.get("pattern")
        if not isinstance(pattern, dict) or not isinstance(pattern.get("points"), list):
            return event
        points = pattern["points"]
        if not points:
            return event                # malformed; the reducers say so, not us

        # A pattern is never its own child, so a completion carrying an id the
        # list already holds must not score a relation against itself.
        consecutive, endpoints = _relation_pairs(patterns, exclude_id=pattern.get("id"))
        chosen = _best_assignment(
            [_point_candidates(point, pmap, magnets) for point in points],
            consecutive, endpoints)
        if chosen is None:
            return None
        return dict(event, pattern=dict(pattern, points=chosen))

    if event_type == "move_point":
        point = {"time": event.get("time"), "price": event.get("price"),
                 "kind": event.get("kind")}
        moved = _moved_point(event, point, pmap, magnets, patterns)
        if moved is None:
            return None
        return dict(event, time=moved["time"], price=moved["price"], kind=moved["kind"])

    return event
