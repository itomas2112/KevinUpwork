// Wave Analysis chart frontend.
//
// Charting library: TradingView Lightweight Charts v5.2.0 (standalone production
// build, Apache-2.0), vendored verbatim next to this file. v5 is required for the
// multi-pane API (chart.addSeries(..., paneIndex) / chart.panes()) and for the
// series-primitive plugin API the wave labels are drawn with.
//
// The Streamlit component protocol is implemented by hand below so the repo needs
// no Node/npm and no build step.

(function () {
    "use strict";

    var LWC = window.LightweightCharts;
    var DOTTED = (LWC.LineStyle && LWC.LineStyle.Dotted !== undefined) ? LWC.LineStyle.Dotted : 1;

    var COLOR_BG = "#121212";
    var COLOR_TEXT = "#9e9e9e";
    var COLOR_BORDER = "#2a2a2a";
    var COLOR_GRID = "#2b2b2b";
    var COLOR_BAR = "#c8c8c8";
    var COLOR_MAGENTA = "#e040fb";
    var COLOR_GREEN = "#4caf50";
    var COLOR_GREEN_LIGHT = "#8bc34a";
    var COLOR_CYAN = "#00bcd4";
    var COLOR_RED = "#ef5350";

    // Wave label palette. "red" marks a same-degree overlap violation; Python
    // owns the choice and sends it down on the pattern.
    var LABEL_COLORS = { yellow: "#FFD700", red: COLOR_RED };
    var COLOR_PROVISIONAL = "#9e9e9e";
    var COLOR_SNAP_RING = "#ef5350";
    var COLOR_SNAP_DOT = "#2196f3";

    var COLOR_SELECT_LINE = "#e0e0e0";
    var COLOR_HANDLE = "#2962ff";

    var LABEL_GAP_BAR = 6;      // px between the bar extreme and the first label
    var LABEL_GAP_STACK = 3;    // px between stacked labels
    var GHOST_ALPHA = 0.6;

    // Per-point annotation blocks on the selected pattern (price / leg range /
    // leg bar count), mirroring MotiveWave.
    var ANNOT_FONT = "10px Arial, sans-serif";
    var ANNOT_COLOR = "#b0b0b0";
    var ANNOT_LINE_HEIGHT = 12; // px per annotation line
    var ANNOT_GAP_STACK = 4;    // px between the label stack's outer edge and the block
    var ANNOT_OFFSET_X = 8;     // px right of bar centre, clear of the handle
    var ANNOT_OBSTACLE_PAD = 2; // px grown around a wave label when resolving collisions
    var ANNOT_PUSH_MARGIN = 3;  // px of clearance a collision push leaves behind
    var ANNOT_EDGE_MARGIN = 4;  // px of the pane's top/bottom a block may not enter
    var ANNOT_LEADER_MIN = 24;  // px of push past which a block earns a leader line
    var ANNOT_LEADER_COLOR = "#555555";

    var CLICK_SLOP = 4;         // px of movement that still counts as a click
    var LABEL_HIT_MIN = 6;      // px grown around a label's box when hit-testing
    var HANDLE_SIZE = 6;        // px side of a selection handle square
    var HANDLE_HIT = 6;         // px radius around a handle that starts a drag
    var SEGMENT_HIT = 5;        // px from a selected pattern's line that keeps it
    var PICK_SEGMENT_HIT = 6;   // px from any pattern's line that selects it

    var container = document.getElementById("wave-chart");
    var toolbarRoot = document.getElementById("wave-toolbar");
    // The element that goes fullscreen: chart + toolbar together, so the
    // toolbar stays reachable while fullscreen.
    var rootEl = document.getElementById("wave-root");

    var chart = null;
    var series = {};
    var lastFingerprint = null;
    var lastConfig = {};
    var lastLogicalRange = null;
    var readySent = false;
    var needsInitialScroll = false;
    var wheelTarget = null;

    // ---- wave state -------------------------------------------------
    var waveDefs = null;          // pattern/degree definitions from Python
    var degreeMap = {};           // degree name -> {index, letter, numeral, decoration, font}
    var degreeNames = [];         // degree names, most senior first
    var patterns = [];            // authoritative list (Python is the source of truth)
    var rendered = [];            // authoritative list + outbox replayed on top
    var outbox = [];              // events sent but not yet acked by Python
    var bars = [];                // current bar series data (rows with full OHLC)
    var allTimes = [];            // every payload time -- the time scale's index space
    var slotToBar = [];           // time-scale slot -> index into bars
    var timeToSlot = {};          // payload time -> time-scale slot
    var marking = null;           // in-progress pattern, or null
    var snap = null;              // current snap candidate {time, price, kind}
    var selectedId = null;        // selected pattern id (JS-local, never sent up)
    var drag = null;              // {id, index, orig} while a handle is being dragged
    var lastLayout = [];          // label geometry from the most recent draw
    var lastHandles = [];         // handle geometry from the most recent draw
    var eventSeq = 1;             // increments per event sent to Python
    var requestUpdate = null;     // primitive redraw hook
    var toolbarBuilt = false;
    var degreeSelect = null;
    var statusChip = null;
    var statusText = null;
    var deleteButton = null;
    var armedButton = null;
    var flashTimer = null;
    var pressStart = null;

    // ---- fullscreen state -------------------------------------------
    var fullscreen = false;       // authoritative flag, whichever mechanism ran
    var fsMode = null;            // "native" | "fallback" | null
    var fsPending = false;        // a native request is in flight
    var fsNativeWarned = false;   // the "native refused" warning is logged once
    var fsButton = null;
    var savedFrameStyle = null;   // the iframe's inline style attribute string
    var savedBodyOverflow = "";   // parent <body> overflow before we hid it
    var savedBodyOverflowSet = false;

    // -----------------------------------------------------------------
    // Streamlit component protocol
    // -----------------------------------------------------------------
    function post(msg) {
        msg.isStreamlitMessage = true;
        window.parent.postMessage(msg, "*");
    }

    function setFrameHeight(height) {
        post({ type: "streamlit:setFrameHeight", height: height });
    }

    function setComponentValue(value) {
        post({ type: "streamlit:setComponentValue", value: value, dataType: "json" });
    }

    // Protocol v2. Streamlit holds exactly one component value and re-delivers
    // it on every rerun, so a second event fired before Python has read the
    // first would simply overwrite it. Instead the whole outbox is posted every
    // time: Python applies every event above its stored seq and hands back an
    // ack, which is what finally drops events from the outbox here.
    function postOutbox() {
        setComponentValue({
            seq: outbox.length ? outbox[outbox.length - 1].eseq : eventSeq - 1,
            events: outbox.slice(),
        });
    }

    function sendEvent(event) {
        event.eseq = eventSeq;
        eventSeq += 1;
        outbox.push(event);
        postOutbox();
        refreshRendered();
    }

    function applyAck(ack) {
        if (typeof ack !== "number" || !isFinite(ack)) return;
        // The iframe can be remounted with Python's seq already far ahead
        // (Streamlit rebuilds the component on some reruns); restarting the
        // counter at 1 would make every new event look stale to Python.
        if (ack >= eventSeq) eventSeq = ack + 1;
        if (!outbox.length) return;
        var kept = outbox.filter(function (e) { return e.eseq > ack; });
        if (kept.length !== outbox.length) outbox = kept;
    }

    // -----------------------------------------------------------------
    // Payload -> series data
    // -----------------------------------------------------------------
    function barData(payload) {
        var out = [];
        for (var i = 0; i < payload.time.length; i++) {
            if (payload.open[i] === null || payload.high[i] === null ||
                payload.low[i] === null || payload.close[i] === null) {
                continue;
            }
            out.push({
                time: payload.time[i],
                open: payload.open[i],
                high: payload.high[i],
                low: payload.low[i],
                close: payload.close[i],
            });
        }
        return out;
    }

    function lineData(times, values) {
        var out = [];
        for (var i = 0; i < times.length; i++) {
            if (values[i] === null || values[i] === undefined) continue;
            out.push({ time: times[i], value: values[i] });
        }
        return out;
    }

    function intraday(timeframe) {
        return timeframe !== "1D";
    }

    function lineOptions(color) {
        return {
            color: color,
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
        };
    }

    // -----------------------------------------------------------------
    // Degree typography -- mirrors config/wave_analysis.render_glyph
    // -----------------------------------------------------------------
    var ROMAN = { "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V" };

    function glyphFor(label, letterStyle, numeralStyle) {
        if (label === "0") return "0";
        if (ROMAN[label]) {
            if (numeralStyle === "arabic") return label;
            return numeralStyle === "roman_lower" ? ROMAN[label].toLowerCase() : ROMAN[label];
        }
        return letterStyle === "lower_serif" ? label.toLowerCase() : label.toUpperCase();
    }

    function fontSpec(deg) {
        // Roman numerals ride along with the serif family even though they are
        // selected by numeral_style rather than letter_style.
        var serif = deg.letter === "lower_serif" || deg.numeral !== "arabic";
        return deg.font + "px " + (serif ? "Georgia, 'Times New Roman', serif" : "Arial, sans-serif");
    }

    function indexDegrees(defs) {
        degreeMap = {};
        degreeNames = [];
        var list = (defs && defs.degrees) || [];
        for (var i = 0; i < list.length; i++) {
            degreeNames.push(list[i][0]);
            degreeMap[list[i][0]] = {
                index: i,
                name: list[i][0],
                letter: list[i][1],
                numeral: list[i][2],
                decoration: list[i][3],
                font: list[i][4],
            };
        }
    }

    function labelsFor(patternType, variation) {
        var variations = waveDefs && waveDefs.pattern_defs && waveDefs.pattern_defs[patternType];
        if (!variations) return null;
        for (var i = 0; i < variations.length; i++) {
            if (variations[i][0] === variation) return ["0"].concat(variations[i][1]);
        }
        return null;
    }

    // -----------------------------------------------------------------
    // Optimistic overlay
    //
    // What is drawn is always "authoritative patterns + outbox replayed on
    // top" (+ the live drag preview). A mutation therefore shows instantly and
    // survives every rerun until Python acks it; once acked the event leaves
    // the outbox and the authoritative list alone decides. An event Python
    // *rejected* simply stops being replayed at that moment, so the labels
    // snap back on their own -- no extra plumbing needed.
    // -----------------------------------------------------------------
    function withPoint(pattern, index, point) {
        var points = pattern.points.slice();
        points[index] = point;
        var copy = {}, key;
        for (key in pattern) if (Object.prototype.hasOwnProperty.call(pattern, key)) copy[key] = pattern[key];
        copy.points = points;
        return copy;
    }

    function replayEvent(list, event) {
        var i;
        // Undo is deliberately not replayed: the snapshot it restores lives in
        // Python's session state, so there is nothing to apply locally. The
        // labels move when the authoritative rerun lands a moment later.
        if (event.type === "undo") return list;
        if (event.type === "pattern_completed") {
            for (i = 0; i < list.length; i++) {
                if (list[i].id === event.pattern.id) return list;
            }
            return list.concat([event.pattern]);
        }
        if (event.type === "delete_pattern") {
            return list.filter(function (p) { return p.id !== event.id; });
        }
        if (event.type === "move_point") {
            return list.map(function (p) {
                if (p.id !== event.id || !p.points) return p;
                if (event.point_index < 0 || event.point_index >= p.points.length) return p;
                return withPoint(p, event.point_index,
                                 { time: event.time, price: event.price, kind: event.kind });
            });
        }
        if (event.type === "shift_degree") {
            // Only the named pattern shifts locally: the cascade through its
            // nest is Python's job, and duplicating that here would mean
            // duplicating the containment logic too. The rest of the component
            // catches up on the authoritative rerun a moment later.
            return list.map(function (p) {
                if (p.id !== event.id) return p;
                var deg = degreeMap[p.degree];
                if (!deg) return p;
                var name = degreeNames[deg.index - event.delta];
                if (!name) return p;
                var copy = {}, key;
                for (key in p) if (Object.prototype.hasOwnProperty.call(p, key)) copy[key] = p[key];
                copy.degree = name;
                return copy;
            });
        }
        return list;
    }

    function refreshRendered() {
        var list = patterns.slice();
        for (var i = 0; i < outbox.length; i++) list = replayEvent(list, outbox[i]);
        // The point being dragged rides the snap candidate so its label, the
        // connecting lines and its handle all follow the cursor together.
        if (drag && snap) {
            list = list.map(function (p) {
                if (p.id !== drag.id || !p.points || drag.index >= p.points.length) return p;
                return withPoint(p, drag.index,
                                 { time: snap.time, price: snap.price, kind: snap.kind });
            });
        }
        rendered = list;
        if (selectedId && !findRendered(selectedId)) selectedId = null;
        updateStatus();
        redraw();
    }

    function findRendered(id) {
        for (var i = 0; i < rendered.length; i++) {
            if (rendered[i].id === id) return rendered[i];
        }
        return null;
    }

    // -----------------------------------------------------------------
    // Label geometry / drawing
    // -----------------------------------------------------------------
    function measureLabel(ctx, deg, glyph) {
        var text = deg.decoration === "parens" ? "(" + glyph + ")" : glyph;
        ctx.font = fontSpec(deg);
        var textWidth = ctx.measureText(text).width;
        var capHeight = deg.font;
        var metrics = { text: text, width: textWidth, height: capHeight, circle: null };
        if (deg.decoration === "circle") {
            // The circle must clearly surround the glyph rather than hug it --
            // in the reference the diameter runs ~2x the font size.
            var pad = deg.font * 0.45;
            var ry = capHeight / 2 + pad;
            var rx = Math.max(textWidth / 2 + pad, ry);
            metrics.circle = { rx: rx, ry: ry };
            metrics.width = 2 * rx;
            metrics.height = 2 * ry;
        }
        return metrics;
    }

    function drawLabel(ctx, x, centerY, deg, metrics, color, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.font = fontSpec(deg);
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(metrics.text, x, centerY);
        if (metrics.circle) {
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(x, centerY, metrics.circle.rx, metrics.circle.ry, 0, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();
    }

    // Every label that belongs on the price pane right now: rendered patterns
    // (authoritative + optimistic), the in-progress pattern and its ghost.
    // ``patternId``/``pointIndex`` are null for anything not yet a pattern.
    function collectLabels() {
        var items = [];

        for (var p = 0; p < rendered.length; p++) {
            var pattern = rendered[p];
            var deg = degreeMap[pattern.degree];
            var labels = labelsFor(pattern.pattern_type, pattern.variation);
            if (!deg || !labels || !pattern.points) continue;
            for (var i = 0; i < pattern.points.length && i < labels.length; i++) {
                items.push({
                    patternId: pattern.id,
                    pointIndex: i,
                    time: pattern.points[i].time,
                    price: pattern.points[i].price,
                    kind: pattern.points[i].kind,
                    deg: deg,
                    glyph: glyphFor(labels[i], deg.letter, deg.numeral),
                    color: LABEL_COLORS[pattern.color] || LABEL_COLORS.yellow,
                    alpha: 1,
                });
            }
        }

        if (marking) {
            var mdeg = degreeMap[marking.degree];
            if (mdeg) {
                for (var j = 0; j < marking.points.length; j++) {
                    items.push({
                        patternId: null,
                        pointIndex: null,
                        time: marking.points[j].time,
                        price: marking.points[j].price,
                        kind: marking.points[j].kind,
                        deg: mdeg,
                        glyph: glyphFor(marking.labels[j], mdeg.letter, mdeg.numeral),
                        color: LABEL_COLORS.yellow,
                        alpha: 1,
                    });
                }
                // Ghost of the next label, riding the snap candidate.
                if (snap && marking.points.length < marking.labels.length) {
                    items.push({
                        patternId: null,
                        pointIndex: null,
                        time: snap.time,
                        price: snap.price,
                        kind: snap.kind,
                        deg: mdeg,
                        glyph: glyphFor(marking.labels[marking.points.length], mdeg.letter, mdeg.numeral),
                        color: LABEL_COLORS.yellow,
                        alpha: GHOST_ALPHA,
                    });
                }
            }
        }
        return items;
    }

    // The single geometry pass: stack every label into its per-(time, kind)
    // group and return the placed boxes. The renderer draws from this and the
    // click hit-test reads the very same output, so what is clickable is by
    // construction exactly what is on screen.
    function layoutGroups(ctx, width) {
        var items = collectLabels();
        var placed = [];
        if (!items.length) return placed;

        var timeScale = chart.timeScale();
        var groups = {};
        var order = [];
        for (var i = 0; i < items.length; i++) {
            var key = items[i].time + "|" + items[i].kind;
            if (!groups[key]) { groups[key] = []; order.push(key); }
            groups[key].push(items[i]);
        }

        for (var g = 0; g < order.length; g++) {
            var group = groups[order[g]];
            var x = timeScale.timeToCoordinate(group[0].time);
            if (x === null || x === undefined) continue;
            if (x < -200 || x > width + 200) continue;   // cull off-screen bars
            var y = series.bars.priceToCoordinate(group[0].price);
            if (y === null || y === undefined) continue;

            // Least senior sits closest to the bar, each more senior degree
            // further out -- parent above child at a high, below it at a low.
            group.sort(function (a, b) { return b.deg.index - a.deg.index; });

            var up = group[0].kind === "high";
            var offset = LABEL_GAP_BAR;
            for (var k = 0; k < group.length; k++) {
                var item = group[k];
                var metrics = measureLabel(ctx, item.deg, item.glyph);
                var centerY = up ? y - offset - metrics.height / 2
                                 : y + offset + metrics.height / 2;
                placed.push({
                    patternId: item.patternId,
                    pointIndex: item.pointIndex,
                    time: item.time,
                    kind: item.kind,
                    x: x,
                    y: centerY,
                    // Circle decorations are already folded into the metrics,
                    // so the box always covers what is actually painted.
                    width: metrics.width,
                    height: metrics.height,
                    deg: item.deg,
                    metrics: metrics,
                    color: item.color,
                    alpha: item.alpha,
                });
                offset += metrics.height + LABEL_GAP_STACK;
            }
        }
        return placed;
    }

    function drawLabels(ctx, width) {
        lastLayout = layoutGroups(ctx, width);
        for (var i = 0; i < lastLayout.length; i++) {
            var item = lastLayout[i];
            drawLabel(ctx, item.x, item.y, item.deg, item.metrics, item.color, item.alpha);
        }
    }

    // -----------------------------------------------------------------
    // Per-point annotations (selected pattern only)
    //
    // Point 0 carries just its price; every later point also carries the leg
    // into it -- its price range and its length in bars. Everything is derived
    // from ``rendered``, which already has the outbox and the live drag preview
    // folded in, so the numbers follow a handle while it is being dragged and
    // stay honest after an undo or a degree shift with no extra plumbing.
    // -----------------------------------------------------------------
    function annotationLines(points, i) {
        var lines = [points[i].price.toFixed(1)];
        if (i === 0) return lines;
        lines.push("(" + Math.abs(points[i].price - points[i - 1].price).toFixed(1) + ")");
        var from = barIndexAtTime(points[i - 1].time);
        var to = barIndexAtTime(points[i].time);
        if (from !== null && to !== null) lines.push(Math.abs(to - from) + " bars");
        return lines;
    }

    // Outer edge of the whole label stack sitting on this (time, kind) -- read
    // from the boxes layoutGroups() has just placed, so a block at a shared
    // pivot clears every stacked degree and not merely its own label.
    function stackEdge(time, kind, fallbackY) {
        var up = kind === "high";
        var edge = null;
        for (var i = 0; i < lastLayout.length; i++) {
            var item = lastLayout[i];
            if (item.time !== time || item.kind !== kind) continue;
            var value = up ? item.y - item.height / 2 : item.y + item.height / 2;
            if (edge === null) edge = value;
            else edge = up ? Math.min(edge, value) : Math.max(edge, value);
        }
        if (edge !== null) return edge;
        return up ? fallbackY - LABEL_GAP_BAR : fallbackY + LABEL_GAP_BAR;
    }

    function rectsOverlap(a, b) {
        return a.left < b.right && b.left < a.right
            && a.top < b.bottom && b.top < a.bottom;
    }

    // Every wave label on screen, grown by ANNOT_OBSTACLE_PAD. All patterns,
    // not just the selected one: a neighbouring pattern's label at a nearby
    // pivot is just as much in the way as one of our own.
    function annotationObstacles() {
        var boxes = [];
        for (var i = 0; i < lastLayout.length; i++) {
            var item = lastLayout[i];
            boxes.push({
                left: item.x - item.width / 2 - ANNOT_OBSTACLE_PAD,
                right: item.x + item.width / 2 + ANNOT_OBSTACLE_PAD,
                top: item.y - item.height / 2 - ANNOT_OBSTACLE_PAD,
                bottom: item.y + item.height / 2 + ANNOT_OBSTACLE_PAD,
            });
        }
        return boxes;
    }

    function moveBlock(block, top) {
        block.top = top;
        block.bottom = top + block.height;
    }

    function firstOverlap(block, obstacles, settled) {
        var i;
        for (i = 0; i < obstacles.length; i++) {
            if (rectsOverlap(block, obstacles[i])) return obstacles[i];
        }
        for (i = 0; i < settled.length; i++) {
            if (rectsOverlap(block, settled[i])) return settled[i];
        }
        return null;
    }

    // Greedy outward sweep. Each block is pushed away from its bar -- up at a
    // high, down at a low -- until it clears every label and every block that
    // settled before it, and then joins the obstacles for the ones after it.
    // Each push moves strictly further out than the obstacle it cleared, so
    // the loop terminates; the pane edge is the backstop either way.
    function resolveAnnotations(blocks, obstacles, paneHeight) {
        var order = blocks.slice().sort(function (a, b) {
            if (a.up !== b.up) return a.up ? -1 : 1;
            return a.barX - b.barX;
        });
        var settled = [];
        var minTop = ANNOT_EDGE_MARGIN;
        for (var i = 0; i < order.length; i++) {
            var block = order[i];
            var maxTop = paneHeight - block.height - ANNOT_EDGE_MARGIN;
            if (maxTop < minTop) maxTop = minTop;
            for (;;) {
                var hit = firstOverlap(block, obstacles, settled);
                if (!hit) break;
                var top = block.up ? hit.top - ANNOT_PUSH_MARGIN - block.height
                                   : hit.bottom + ANNOT_PUSH_MARGIN;
                // At the pane edge, keep the overlap rather than bounce back
                // and forth: a dozen points crammed against one edge simply
                // cannot all be separated.
                if (block.up ? top < minTop : top > maxTop) {
                    moveBlock(block, block.up ? Math.min(block.top, minTop)
                                              : Math.max(block.top, maxTop));
                    break;
                }
                moveBlock(block, top);
            }
            settled.push(block);
        }
        return order;
    }

    // Runs after drawLabels(): it reads that pass's ``lastLayout``.
    //
    // Measure every block at the position it wants, resolve the collisions
    // between the blocks and the labels, then draw. Blocks only ever move
    // vertically, so each one stays alongside its own bar, and a block with
    // nothing in its way lands exactly where it did before this pass existed.
    function drawAnnotations(ctx, width, height) {
        if (marking || !selectedId) return;
        var pattern = findRendered(selectedId);
        if (!pattern || !pattern.points || !pattern.points.length) return;

        var timeScale = chart.timeScale();
        ctx.save();
        ctx.font = ANNOT_FONT;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";

        // Pass 1 -- measure.
        var blocks = [];
        var i, k;
        for (i = 0; i < pattern.points.length; i++) {
            var point = pattern.points[i];
            var x = timeScale.timeToCoordinate(point.time);
            if (x === null || x === undefined) continue;
            if (x < -200 || x > width + 200) continue;   // cull off-screen bars
            var y = series.bars.priceToCoordinate(point.price);
            if (y === null || y === undefined) continue;

            var lines = annotationLines(pattern.points, i);
            var textWidth = 0;
            for (k = 0; k < lines.length; k++) {
                var lineWidth = ctx.measureText(lines[k]).width;
                if (lineWidth > textWidth) textWidth = lineWidth;
            }
            var up = point.kind === "high";
            var blockHeight = lines.length * ANNOT_LINE_HEIGHT;
            var edge = stackEdge(point.time, point.kind, y);
            // Beyond the stack: above it at a high, below it at a low. Lines
            // read top-to-bottom in both cases.
            var top = up ? edge - ANNOT_GAP_STACK - blockHeight
                         : edge + ANNOT_GAP_STACK;
            blocks.push({
                lines: lines,
                up: up,
                barX: x,
                edge: edge,
                height: blockHeight,
                naturalTop: top,
                left: x + ANNOT_OFFSET_X,
                right: x + ANNOT_OFFSET_X + textWidth,
                top: top,
                bottom: top + blockHeight,
            });
        }

        // Pass 2 -- resolve.
        var settled = resolveAnnotations(blocks, annotationObstacles(), height);

        // Pass 3 -- draw. Leaders go down first so no text line is crossed by
        // one; a block that travelled far enough for its bar to be in doubt
        // gets a thin line back to the pivot it belongs to.
        var block;
        ctx.strokeStyle = ANNOT_LEADER_COLOR;
        ctx.lineWidth = 1;
        for (i = 0; i < settled.length; i++) {
            block = settled[i];
            var push = block.up ? block.naturalTop - block.top
                                : block.top - block.naturalTop;
            if (push <= ANNOT_LEADER_MIN) continue;
            var leaderX = Math.round(block.left) + 0.5;
            ctx.beginPath();
            ctx.moveTo(leaderX, block.up ? block.bottom : block.top);
            ctx.lineTo(leaderX, block.up ? block.edge - ANNOT_OBSTACLE_PAD
                                         : block.edge + ANNOT_OBSTACLE_PAD);
            ctx.stroke();
        }
        ctx.fillStyle = ANNOT_COLOR;
        for (i = 0; i < settled.length; i++) {
            block = settled[i];
            for (k = 0; k < block.lines.length; k++) {
                ctx.fillText(block.lines[k], block.left,
                             block.top + k * ANNOT_LINE_HEIGHT);
            }
        }
        ctx.restore();
    }

    // Connecting polyline + square handles for the selected pattern. Drawn
    // under the labels so a label is never obscured by its own handle.
    function drawSelection(ctx) {
        lastHandles = [];
        var pattern = selectedId ? findRendered(selectedId) : null;
        if (!pattern || !pattern.points || !pattern.points.length) return;

        var timeScale = chart.timeScale();
        var coords = [];
        for (var i = 0; i < pattern.points.length; i++) {
            var x = timeScale.timeToCoordinate(pattern.points[i].time);
            var y = series.bars.priceToCoordinate(pattern.points[i].price);
            if (x === null || x === undefined || y === null || y === undefined) {
                coords.push(null);
                continue;
            }
            coords.push({ x: x, y: y });
            lastHandles.push({ index: i, x: x, y: y });
        }

        ctx.save();
        ctx.strokeStyle = COLOR_SELECT_LINE;
        ctx.lineWidth = 1;
        ctx.beginPath();
        var started = false;
        for (i = 0; i < coords.length; i++) {
            if (!coords[i]) { started = false; continue; }
            if (!started) { ctx.moveTo(coords[i].x, coords[i].y); started = true; }
            else ctx.lineTo(coords[i].x, coords[i].y);
        }
        ctx.stroke();

        ctx.fillStyle = COLOR_HANDLE;
        var half = HANDLE_SIZE / 2;
        for (i = 0; i < lastHandles.length; i++) {
            ctx.fillRect(lastHandles[i].x - half, lastHandles[i].y - half,
                         HANDLE_SIZE, HANDLE_SIZE);
        }
        ctx.restore();
    }

    function drawProvisional(ctx) {
        if (!marking || marking.points.length < 2) return;
        var timeScale = chart.timeScale();
        ctx.save();
        ctx.strokeStyle = COLOR_PROVISIONAL;
        ctx.lineWidth = 1;
        ctx.beginPath();
        var started = false;
        for (var i = 0; i < marking.points.length; i++) {
            var x = timeScale.timeToCoordinate(marking.points[i].time);
            var y = series.bars.priceToCoordinate(marking.points[i].price);
            if (x === null || x === undefined || y === null || y === undefined) continue;
            if (!started) { ctx.moveTo(x, y); started = true; }
            else ctx.lineTo(x, y);
        }
        if (started) ctx.stroke();
        ctx.restore();
    }

    function drawSnapIndicator(ctx) {
        if ((!marking && !drag) || !snap) return;
        var x = chart.timeScale().timeToCoordinate(snap.time);
        var y = series.bars.priceToCoordinate(snap.price);
        if (x === null || x === undefined || y === null || y === undefined) return;
        ctx.save();
        ctx.strokeStyle = COLOR_SNAP_RING;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = COLOR_SNAP_DOT;
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    // Series primitive: the chart repaints it as part of every pane render, so
    // labels stay glued to their bars while panning with no rAF loop of our own.
    var wavePaneView = {
        zOrder: function () { return "top"; },
        renderer: function () {
            return {
                draw: function (target) {
                    target.useMediaCoordinateSpace(function (scope) {
                        if (!chart || !series.bars) return;
                        drawSelection(scope.context);
                        drawLabels(scope.context, scope.mediaSize.width);
                        drawAnnotations(scope.context, scope.mediaSize.width,
                                        scope.mediaSize.height);
                        drawProvisional(scope.context);
                        drawSnapIndicator(scope.context);
                    });
                },
            };
        },
    };

    var wavePrimitive = {
        attached: function (param) { requestUpdate = param.requestUpdate; },
        detached: function () { requestUpdate = null; },
        updateAllViews: function () {},
        paneViews: function () { return [wavePaneView]; },
    };

    function redraw() {
        if (requestUpdate) requestUpdate();
    }

    // -----------------------------------------------------------------
    // Chart construction
    // -----------------------------------------------------------------
    function buildChart(payload, config) {
        chart = LWC.createChart(container, {
            width: document.body.clientWidth,
            height: config.height,
            autoSize: false,
            layout: {
                background: { color: COLOR_BG },
                textColor: COLOR_TEXT,
                panes: {
                    separatorColor: COLOR_BORDER,
                    separatorHoverColor: "#3b3b3b",
                    enableResize: true,
                },
            },
            grid: {
                vertLines: { visible: true, color: COLOR_GRID, style: DOTTED },
                horzLines: { visible: false },
            },
            crosshair: {
                vertLine: { color: COLOR_TEXT, width: 1, style: DOTTED, labelBackgroundColor: COLOR_BORDER },
                horzLine: { color: COLOR_TEXT, width: 1, style: DOTTED, labelBackgroundColor: COLOR_BORDER },
            },
            rightPriceScale: {
                visible: true,
                borderColor: COLOR_BORDER,
                autoScale: true,
                scaleMargins: { top: 0.15, bottom: 0.15 },
            },
            timeScale: {
                borderColor: COLOR_BORDER,
                barSpacing: config.bar_spacing,
                rightOffset: 5,
                timeVisible: intraday(payload.timeframe),
                secondsVisible: false,
                fixLeftEdge: false,
                fixRightEdge: false,
                shiftVisibleRangeOnNewBar: false,
            },
            // Zoom lockout: no wheel zoom, no pinch, no axis-drag scaling,
            // no double-click reset. Wheel and drag pan through time instead.
            handleScale: false,
            handleScroll: {
                // Off: the library drops vertical wheel deltas whenever
                // handleScale.mouseWheel is false, so onWheel() below replaces
                // its wheel handling entirely (also avoids double-panning on
                // trackpad deltaX).
                mouseWheel: false,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: false,
            },
        });

        // Pane 0 -- price
        series.bars = chart.addSeries(LWC.BarSeries, {
            thinBars: true,
            upColor: COLOR_BAR,
            downColor: COLOR_BAR,
            openVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
        }, 0);
        series.bars.attachPrimitive(wavePrimitive);

        // Pane 1 -- RSI
        series.rsi = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_MAGENTA), 1);
        series.rsi_13 = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_GREEN), 1);
        series.rsi_33 = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_GREEN_LIGHT), 1);
        series.rsi.createPriceLine({
            price: 70, color: COLOR_RED, lineWidth: 1, lineStyle: DOTTED, axisLabelVisible: true, title: "",
        });
        series.rsi.createPriceLine({
            price: 30, color: COLOR_GREEN, lineWidth: 1, lineStyle: DOTTED, axisLabelVisible: true, title: "",
        });

        // Pane 2 -- CMB
        series.ci = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_MAGENTA), 2);
        series.ci_13 = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_CYAN), 2);
        series.ci_33 = chart.addSeries(LWC.LineSeries, lineOptions(COLOR_GREEN), 2);

        chart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
            if (range) lastLogicalRange = range;
        });
        chart.subscribeCrosshairMove(onCrosshairMove);

        // passive: false -- the handler always preventDefault()s so the wheel can
        // never leak through to the Streamlit page behind the iframe.
        container.addEventListener("wheel", onWheel, { passive: false });
        // Clicks are detected here rather than via subscribeClick so that the
        // mouse-up ending a drag-pan never places a point.
        container.addEventListener("mousedown", onMouseDown);
        // Move/up ride on the document so a drag that wanders off the chart
        // still tracks and still releases (leaving panning disabled forever
        // would be far worse than a clamped candidate).
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);

        window.addEventListener("resize", onResize);
        // The iframe also changes width without a window resize -- when the tab
        // becomes visible, or when the Streamlit sidebar is collapsed.
        if (window.ResizeObserver) {
            new ResizeObserver(onResize).observe(document.body);
        }
    }

    // The time scale's logical index counts every point on the chart, not every
    // *bar*: a row whose OHLC is incomplete is skipped by barData() while the
    // RSI/CMB series still put that timestamp on the scale. Mapping slot ->
    // nearest bar keeps snapping honest if that ever happens.
    function buildBarIndex(payload) {
        allTimes = payload.time.slice();
        var timeToBarIndex = {};
        for (var i = 0; i < bars.length; i++) timeToBarIndex[bars[i].time] = i;

        slotToBar = new Array(allTimes.length);
        timeToSlot = {};
        var last = -1;
        for (i = 0; i < allTimes.length; i++) {
            timeToSlot[allTimes[i]] = i;
            var index = timeToBarIndex[allTimes[i]];
            if (index === undefined) {
                slotToBar[i] = last;
            } else {
                slotToBar[i] = index;
                last = index;
            }
        }
        // Leading slots that precede the first real bar fall back to it.
        for (i = 0; i < slotToBar.length && slotToBar[i] === -1; i++) slotToBar[i] = 0;
    }

    // A point's bar index, resolved through the very same slot -> bar map that
    // snapping uses, so a counted leg spans exactly the bars the user sees.
    function barIndexAtTime(time) {
        var slot = timeToSlot[time];
        if (slot === undefined) return null;
        var index = slotToBar[slot];
        if (index === undefined || index < 0) return null;
        return index;
    }

    function barAtLogical(logical) {
        if (!bars.length || !slotToBar.length) return null;
        var slot = Math.round(logical);
        if (slot < 0) slot = 0;
        if (slot > slotToBar.length - 1) slot = slotToBar.length - 1;
        var index = slotToBar[slot];
        if (index === undefined || index < 0) return null;
        return bars[index] || null;
    }

    function setAllData(payload) {
        bars = barData(payload);
        buildBarIndex(payload);
        series.bars.setData(bars);
        var t = payload.time;
        series.rsi.setData(lineData(t, payload.rsi));
        series.rsi_13.setData(lineData(t, payload.rsi_13));
        series.rsi_33.setData(lineData(t, payload.rsi_33));
        series.ci.setData(lineData(t, payload.ci));
        series.ci_13.setData(lineData(t, payload.ci_13));
        series.ci_33.setData(lineData(t, payload.ci_33));
    }

    // pane.setHeight() behaves as a stretch factor and each sequential call steals
    // height from the panes set before it, so the requested pixels never land.
    // Stretch factors are plain relative weights -- feeding the requested pixel
    // values in as weights in a single pass reproduces the requested proportions
    // regardless of time-axis / separator overhead.
    function applyPaneHeights(heights, totalHeight) {
        if (!heights) return;
        var panes = chart.panes();
        var rsi = heights[1] || 0;
        var cmb = heights[2] || 0;
        var price = (totalHeight || 0) - rsi - cmb;
        if (!(price > 0)) price = 100;
        var weights = [price, rsi, cmb];
        for (var i = 0; i < weights.length && i < panes.length; i++) {
            if (weights[i] > 0) panes[i].setStretchFactor(weights[i]);
        }
    }

    // -----------------------------------------------------------------
    // Fullscreen
    //
    // Two mechanisms, tried in order. (A) real fullscreen on #wave-root --
    // may be refused outright because Streamlit's component iframe carries no
    // ``allow="fullscreen"``, which shows up as either a false
    // ``document.fullscreenEnabled`` or a rejected promise. (B) the fallback:
    // the component styles its *own* iframe from the inside so it covers the
    // parent viewport. Streamlit serves component iframes same-origin, so
    // ``window.frameElement`` and the parent document are reachable -- but
    // every such access is wrapped, because a future sandbox flag would turn
    // them into throws rather than a graceful degradation.
    //
    // ``fullscreen`` is the single flag the rest of the file keys off:
    // onResize() sizes from the viewport while it is set, and render() defers
    // height changes until it clears.
    // -----------------------------------------------------------------
    function updateFullscreenButton() {
        if (!fsButton) return;
        fsButton.textContent = fullscreen ? "✕ Exit" : "⛶";
        if (fullscreen) fsButton.classList.add("wa-on");
        else fsButton.classList.remove("wa-on");
    }

    function setFullscreenState(active) {
        fullscreen = active;
        if (active) document.body.classList.add("wa-fs");
        else document.body.classList.remove("wa-fs");
        updateFullscreenButton();
        // Both mechanisms change the iframe's box, so the ResizeObserver on
        // <body> normally fires too -- but it is asynchronous and does not fire
        // at all when only the height changed under native fullscreen, so the
        // resize is driven explicitly either way. A redundant resize is free.
        onResize();
        // Nothing posted a new frame height while fullscreen; re-post the
        // configured one on the way out so a height change that arrived
        // meanwhile lands now.
        if (!active) setFrameHeight(lastConfig.height);
    }

    function restoreFallback() {
        try {
            var frame = window.frameElement;
            if (frame) {
                // An empty saved string is treated as "no attribute": Streamlit
                // re-renders the iframe between toggles and sometimes leaves a
                // bare style="" behind, and restoring that verbatim would make
                // repeated toggles accumulate a difference from the original.
                if (!savedFrameStyle) frame.removeAttribute("style");
                else frame.setAttribute("style", savedFrameStyle);
            }
        } catch (err) {
            console.warn("[wave-chart] could not restore the iframe style: " + err);
        }
        try {
            if (savedBodyOverflowSet) {
                window.parent.document.body.style.overflow = savedBodyOverflow;
            }
        } catch (err) {
            console.warn("[wave-chart] could not restore the parent page scroll: " + err);
        }
        savedFrameStyle = null;
        savedBodyOverflow = "";
        savedBodyOverflowSet = false;
    }

    function enterFallback() {
        var frame = null;
        try {
            frame = window.frameElement;
        } catch (err) {
            frame = null;
        }
        if (!frame) {
            console.warn("[wave-chart] fullscreen unavailable: the component iframe " +
                         "is not reachable from inside and native fullscreen was refused.");
            return;
        }
        try {
            savedFrameStyle = frame.getAttribute("style");
            var parentBody = window.parent.document.body;
            if (parentBody) {
                savedBodyOverflow = parentBody.style.overflow;
                savedBodyOverflowSet = true;
                parentBody.style.overflow = "hidden";
            }
            frame.style.position = "fixed";
            frame.style.top = "0";
            frame.style.left = "0";
            frame.style.width = "100vw";
            frame.style.height = "100vh";
            // Above Streamlit's own chrome: its header stacks at 999990 and the
            // sidebar at 999991, so anything lower leaves the toolbar -- and
            // with it the exit button -- buried under them.
            frame.style.zIndex = "2147483647";
            frame.style.background = COLOR_BG;
        } catch (err) {
            console.warn("[wave-chart] fullscreen fallback failed: " + err);
            restoreFallback();
            return;
        }
        fsMode = "fallback";
        setFullscreenState(true);
    }

    function enterFullscreen() {
        if (fullscreen || fsPending) return;
        var promise = null;
        if (document.fullscreenEnabled && rootEl && rootEl.requestFullscreen) {
            try {
                promise = rootEl.requestFullscreen();
            } catch (err) {
                promise = null;
            }
        }
        if (!promise || typeof promise.then !== "function") {
            // Either the API is missing/disabled, or a legacy implementation
            // that switches synchronously and has already fired its event.
            if (document.fullscreenElement) return;
            enterFallback();
            return;
        }
        fsPending = true;
        promise.then(function () {
            fsPending = false;
            // The fullscreenchange listener owns the state transition.
        }, function (err) {
            fsPending = false;
            // Once only: the refusal is a property of how Streamlit embeds the
            // component, so it repeats on every single toggle.
            if (!fsNativeWarned) {
                fsNativeWarned = true;
                console.warn("[wave-chart] native fullscreen was refused (" + err +
                             "); using the viewport-fill fallback.");
            }
            enterFallback();
        });
    }

    function exitFullscreen() {
        if (fsMode === "native") {
            if (document.fullscreenElement) {
                try {
                    var promise = document.exitFullscreen();
                    if (promise && typeof promise.then === "function") {
                        promise.then(null, function (err) {
                            console.warn("[wave-chart] exiting fullscreen failed: " + err);
                        });
                    }
                } catch (err) {
                    console.warn("[wave-chart] exiting fullscreen failed: " + err);
                }
            } else {
                onFullscreenChange();   // already out; just reconcile
            }
            return;
        }
        if (fsMode === "fallback") {
            restoreFallback();
            fsMode = null;
            setFullscreenState(false);
        }
    }

    function toggleFullscreen() {
        if (fullscreen) exitFullscreen();
        else enterFullscreen();
    }

    // The browser exits native fullscreen on its own Escape without ever
    // delivering a keydown, so this -- not the key handler -- is what keeps the
    // flag, the button label and the chart size honest.
    function onFullscreenChange() {
        var native = !!document.fullscreenElement;
        if (native) {
            fsMode = "native";
            if (!fullscreen) setFullscreenState(true);
            else { updateFullscreenButton(); onResize(); }
        } else if (fsMode === "native") {
            fsMode = null;
            setFullscreenState(false);
        }
    }

    // -----------------------------------------------------------------
    // Toolbar
    // -----------------------------------------------------------------
    function closeMenus(except) {
        var menus = toolbarRoot.querySelectorAll(".wa-menu");
        for (var i = 0; i < menus.length; i++) {
            if (menus[i] !== except) menus[i].classList.remove("wa-open");
        }
    }

    function buildToolbar() {
        if (toolbarBuilt || !waveDefs) return;
        toolbarRoot.innerHTML = "";
        var row = document.createElement("div");
        row.className = "wa-row";
        toolbarRoot.appendChild(row);

        Object.keys(waveDefs.pattern_defs).forEach(function (patternType) {
            var slot = document.createElement("div");
            slot.className = "wa-slot";

            var button = document.createElement("button");
            button.type = "button";
            button.textContent = patternType;

            var menu = document.createElement("div");
            menu.className = "wa-menu";

            waveDefs.pattern_defs[patternType].forEach(function (entry) {
                var variation = entry[0];
                var item = document.createElement("button");
                item.type = "button";
                item.textContent = variation;
                item.addEventListener("click", function (event) {
                    event.stopPropagation();
                    menu.classList.remove("wa-open");
                    arm(patternType, variation, button);
                });
                menu.appendChild(item);
            });

            button.addEventListener("click", function (event) {
                event.stopPropagation();
                var open = menu.classList.contains("wa-open");
                closeMenus();
                if (!open) menu.classList.add("wa-open");
            });

            slot.appendChild(button);
            slot.appendChild(menu);
            row.appendChild(slot);
        });

        degreeSelect = document.createElement("select");
        (waveDefs.degrees || []).forEach(function (degree) {
            var option = document.createElement("option");
            option.value = degree[0];
            option.textContent = degree[0];
            degreeSelect.appendChild(option);
        });
        degreeSelect.value = waveDefs.default_degree;
        degreeSelect.addEventListener("click", function (event) { event.stopPropagation(); });
        row.appendChild(degreeSelect);

        // Last in the row, so it sits at its right end.
        fsButton = document.createElement("button");
        fsButton.type = "button";
        fsButton.className = "wa-fs-btn";
        fsButton.title = "Fullscreen (Esc to exit)";
        fsButton.addEventListener("click", function (event) {
            event.stopPropagation();
            closeMenus();
            toggleFullscreen();
        });
        row.appendChild(fsButton);
        updateFullscreenButton();

        // One chip, two jobs: the marking progress readout and the selection
        // readout. They are mutually exclusive states, so they share the slot.
        statusChip = document.createElement("span");
        statusChip.className = "wa-chip";
        statusText = document.createElement("span");
        statusChip.appendChild(statusText);

        deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "wa-del";
        deleteButton.textContent = "✕ Delete";
        deleteButton.addEventListener("click", function (event) {
            event.stopPropagation();
            deleteSelected();
        });
        statusChip.appendChild(deleteButton);
        toolbarRoot.appendChild(statusChip);

        // The controls sit outside the chart container, so the wheel would
        // otherwise escape to the Streamlit page behind the iframe.
        toolbarRoot.addEventListener("wheel", function (event) { event.preventDefault(); },
                                     { passive: false });
        document.addEventListener("click", function () { closeMenus(); });

        toolbarBuilt = true;
    }

    function updateStatus() {
        if (!statusChip) return;
        if (marking) {
            statusChip.classList.add("wa-on");
            deleteButton.style.display = "none";
            statusText.textContent = "Marking: " + marking.variation + " (" + marking.degree +
                ") — point " + marking.points.length + " of " + (marking.labels.length - 1) +
                " — Esc to cancel";
            return;
        }
        var selected = selectedId ? findRendered(selectedId) : null;
        if (selected) {
            statusChip.classList.add("wa-on");
            deleteButton.style.display = "";
            statusText.textContent = "Selected: " + selected.variation + " (" + selected.degree +
                ") — Del to delete · Ctrl+/− degree · Ctrl+Z undo · Esc to deselect";
            return;
        }
        statusChip.classList.remove("wa-on");
        deleteButton.style.display = "none";
        statusText.textContent = "";
    }

    function flashStatus() {
        if (!statusChip) return;
        statusChip.classList.add("wa-flash");
        if (flashTimer) clearTimeout(flashTimer);
        flashTimer = setTimeout(function () {
            statusChip.classList.remove("wa-flash");
            flashTimer = null;
        }, 300);
    }

    // -----------------------------------------------------------------
    // Marking
    // -----------------------------------------------------------------
    function arm(patternType, variation, button) {
        var labels = labelsFor(patternType, variation);
        if (!labels) return;
        if (armedButton) armedButton.classList.remove("wa-armed");
        armedButton = button;
        button.classList.add("wa-armed");
        // Marking and selection are mutually exclusive modes.
        if (drag) endDrag();
        selectedId = null;
        // The degree is captured now: changing the dropdown mid-marking must
        // not affect the pattern being placed.
        marking = {
            patternType: patternType,
            variation: variation,
            degree: degreeSelect.value,
            labels: labels,
            points: [],
        };
        snap = null;
        updateStatus();
        redraw();
    }

    function disarm() {
        marking = null;
        snap = null;
        if (armedButton) armedButton.classList.remove("wa-armed");
        armedButton = null;
        updateStatus();
        redraw();
    }

    function uuidHex() {
        if (window.crypto && window.crypto.randomUUID) {
            return window.crypto.randomUUID().replace(/-/g, "");
        }
        var hex = "";
        if (window.crypto && window.crypto.getRandomValues) {
            var buf = new Uint8Array(16);
            window.crypto.getRandomValues(buf);
            for (var i = 0; i < buf.length; i++) hex += ("0" + buf[i].toString(16)).slice(-2);
            return hex;
        }
        while (hex.length < 32) hex += Math.floor(Math.random() * 16).toString(16);
        return hex.slice(0, 32);
    }

    function onCrosshairMove(param) {
        // While a handle is being dragged the candidate is driven by the raw
        // mousemove handler instead, so the crosshair must not clear it.
        if (drag) return;
        if (!marking) {
            if (snap) { snap = null; redraw(); }
            return;
        }
        var next = (param && param.point && param.paneIndex === 0)
            ? candidateAtPoint(param.point.x, param.point.y) : null;
        setSnap(next);
    }

    function setSnap(next) {
        var changed = (next === null) !== (snap === null) ||
            (next && snap && (next.time !== snap.time || next.kind !== snap.kind));
        snap = next;
        if (!changed) return;
        if (drag) refreshRendered();     // the dragged point rides the candidate
        else redraw();
    }

    // Cursor -> the nearest bar's high or low. Never a free price.
    function candidateAtPoint(x, y) {
        if (!bars.length) return null;
        var logical = chart.timeScale().coordinateToLogical(x);
        if (logical === null || logical === undefined || isNaN(logical)) return null;
        var bar = barAtLogical(logical);
        if (!bar) return null;

        var yHigh = series.bars.priceToCoordinate(bar.high);
        var yLow = series.bars.priceToCoordinate(bar.low);
        if (yHigh === null || yLow === null) return null;
        var kind = Math.abs(y - yHigh) <= Math.abs(y - yLow) ? "high" : "low";
        return {
            time: bar.time,
            price: kind === "high" ? bar.high : bar.low,
            kind: kind,
        };
    }

    function placePoint() {
        var points = marking.points;
        // Points must run strictly forward in time.
        if (points.length && snap.time <= points[points.length - 1].time) {
            flashStatus();
            return;
        }
        points.push({ time: snap.time, price: snap.price, kind: snap.kind });
        if (points.length === marking.labels.length) completePattern();
        else updateStatus();
        redraw();
    }

    function completePattern() {
        var pattern = {
            id: uuidHex(),
            pattern_type: marking.patternType,
            variation: marking.variation,
            degree: marking.degree,
            color: "yellow",
            points: marking.points,
        };
        // Drawn optimistically by the outbox replay until Python acks it.
        sendEvent({ type: "pattern_completed", pattern: pattern });
        disarm();
    }

    // -----------------------------------------------------------------
    // Selection
    // -----------------------------------------------------------------
    function select(id) {
        if (selectedId === id) { updateStatus(); return; }
        selectedId = id;
        updateStatus();
        redraw();
    }

    // Chart-container coordinates. Pane 0 sits at the container's top-left, so
    // these are the same coordinates the primitive draws in.
    function localPoint(event) {
        var rect = container.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    // The pad scales with the degree so a 9 px Pico glyph is no harder to hit
    // than a 22 px Millennium one; the floor keeps every degree comfortably
    // clickable at the first attempt.
    function labelHitPad(deg) {
        return Math.max(LABEL_HIT_MIN, deg.font * 0.5);
    }

    // Last drawn wins, so the topmost label of an overlapping stack is picked.
    function labelAt(x, y) {
        for (var i = lastLayout.length - 1; i >= 0; i--) {
            var item = lastLayout[i];
            if (item.patternId === null || item.patternId === undefined) continue;
            var pad = labelHitPad(item.deg);
            if (Math.abs(x - item.x) <= item.width / 2 + pad &&
                Math.abs(y - item.y) <= item.height / 2 + pad) {
                return item;
            }
        }
        return null;
    }

    function handleAt(x, y) {
        for (var i = 0; i < lastHandles.length; i++) {
            var h = lastHandles[i];
            if (Math.abs(x - h.x) <= HANDLE_HIT && Math.abs(y - h.y) <= HANDLE_HIT) return h;
        }
        return null;
    }

    function distanceToSegment(x, y, a, b) {
        var dx = b.x - a.x;
        var dy = b.y - a.y;
        var lengthSq = dx * dx + dy * dy;
        var t = lengthSq ? ((x - a.x) * dx + (y - a.y) * dy) / lengthSq : 0;
        if (t < 0) t = 0;
        else if (t > 1) t = 1;
        var px = a.x + t * dx - x;
        var py = a.y + t * dy - y;
        return Math.sqrt(px * px + py * py);
    }

    // Clicking the connecting line of the selected pattern keeps it selected --
    // the lines are part of what the user sees as "the selection".
    function onSelectedSegments(x, y) {
        for (var i = 1; i < lastHandles.length; i++) {
            if (lastHandles[i].index !== lastHandles[i - 1].index + 1) continue;
            if (distanceToSegment(x, y, lastHandles[i - 1], lastHandles[i]) <= SEGMENT_HIT) {
                return true;
            }
        }
        return false;
    }

    // Clicking anywhere along a pattern's connecting lines selects it. Those
    // lines are only *drawn* for the selected pattern, but their geometry is
    // fully determined by the points, so the whole zigzag is a target -- far
    // easier to hit than one small label. The nearest segment wins when two
    // patterns' lines cross near the cursor.
    function patternAtSegment(x, y) {
        var timeScale = chart.timeScale();
        var bestId = null;
        var bestDistance = PICK_SEGMENT_HIT;
        for (var p = 0; p < rendered.length; p++) {
            var pattern = rendered[p];
            if (!pattern.points) continue;
            var previous = null;
            for (var i = 0; i < pattern.points.length; i++) {
                var px = timeScale.timeToCoordinate(pattern.points[i].time);
                var py = series.bars.priceToCoordinate(pattern.points[i].price);
                var current = (px === null || px === undefined ||
                               py === null || py === undefined)
                    ? null : { x: px, y: py };
                if (previous && current) {
                    var distance = distanceToSegment(x, y, previous, current);
                    if (distance <= bestDistance) {
                        bestDistance = distance;
                        bestId = pattern.id;
                    }
                }
                previous = current;
            }
        }
        return bestId;
    }

    function deleteSelected() {
        if (!selectedId) return;
        sendEvent({ type: "delete_pattern", id: selectedId });
        select(null);
    }

    function shiftSelectedDegree(delta) {
        if (!selectedId) return;
        sendEvent({ type: "shift_degree", id: selectedId, delta: delta });
    }

    // -----------------------------------------------------------------
    // Dragging a point onto another bar
    // -----------------------------------------------------------------
    function setPanning(enabled) {
        if (!chart) return;
        chart.applyOptions({
            handleScroll: {
                mouseWheel: false,
                pressedMouseMove: enabled,
                horzTouchDrag: enabled,
                vertTouchDrag: false,
            },
        });
    }

    function startDrag(handle) {
        var pattern = findRendered(selectedId);
        if (!pattern || !pattern.points || !pattern.points[handle.index]) return;
        drag = {
            id: selectedId,
            index: handle.index,
            orig: pattern.points[handle.index],
        };
        snap = null;
        // Otherwise the chart pans out from under the point being dragged.
        setPanning(false);
        redraw();
    }

    function endDrag() {
        drag = null;
        snap = null;
        setPanning(true);
        refreshRendered();
    }

    function commitDrag() {
        var d = drag;
        var candidate = snap;
        if (!d) return;
        if (!candidate) { endDrag(); return; }

        var pattern = findRendered(d.id);
        if (!pattern || !pattern.points) { endDrag(); return; }
        var points = pattern.points;
        var before = d.index > 0 ? points[d.index - 1].time : null;
        var after = d.index < points.length - 1 ? points[d.index + 1].time : null;
        if ((before !== null && candidate.time <= before) ||
            (after !== null && candidate.time >= after)) {
            flashStatus();
            endDrag();
            return;
        }
        if (candidate.time === d.orig.time && candidate.kind === d.orig.kind) {
            endDrag();
            return;
        }

        drag = null;
        snap = null;
        setPanning(true);
        sendEvent({
            type: "move_point",
            id: d.id,
            point_index: d.index,
            time: candidate.time,
            price: candidate.price,
            kind: candidate.kind,
        });
    }

    // -----------------------------------------------------------------
    // Pointer / keyboard
    // -----------------------------------------------------------------
    function onMouseDown(event) {
        pressStart = { x: event.clientX, y: event.clientY };
        // Keys only reach this document once something inside the iframe has
        // focus, so Delete / Ctrl+- work right after the click that selects.
        if (container.focus) container.focus({ preventScroll: true });
        if (marking || !selectedId) return;
        var local = localPoint(event);
        var handle = handleAt(local.x, local.y);
        if (handle) startDrag(handle);
    }

    function onMouseMove(event) {
        if (!drag) return;
        var local = localPoint(event);
        setSnap(candidateAtPoint(local.x, local.y));
    }

    function onMouseUp(event) {
        var start = pressStart;
        pressStart = null;
        if (drag) { commitDrag(); return; }
        if (!start || !container.contains(event.target)) return;
        // A mouse-up that ended a drag-pan is not a click.
        if (Math.abs(event.clientX - start.x) > CLICK_SLOP ||
            Math.abs(event.clientY - start.y) > CLICK_SLOP) return;

        if (marking) {
            if (snap) placePoint();
            return;
        }

        // Labels win over lines: a label sits on a point that several patterns
        // may share, so it is the more specific target of the two.
        var local = localPoint(event);
        var hit = labelAt(local.x, local.y);
        if (hit) { select(hit.patternId); return; }
        if (selectedId && onSelectedSegments(local.x, local.y)) return;
        var segmentId = patternAtSegment(local.x, local.y);
        if (segmentId) { select(segmentId); return; }
        select(null);
    }

    function onKeyDown(event) {
        if (event.key === "Escape") {
            if (drag) { event.preventDefault(); endDrag(); return; }
            if (marking) { event.preventDefault(); disarm(); return; }
            if (selectedId) { event.preventDefault(); select(null); return; }
            // Last in line: leaving fullscreen must never cost the user an
            // in-progress marking or a selection. (Under native fullscreen the
            // browser may act on Escape first and exit regardless; the
            // fullscreenchange listener reconciles that.)
            if (fullscreen) { event.preventDefault(); exitFullscreen(); }
            return;
        }
        // Undo needs no selection, so it is handled ahead of the guard below.
        // Not while marking or dragging: those have their own escape hatch and
        // undoing out from under a live gesture would only confuse.
        if ((event.ctrlKey || event.metaKey) && (event.key === "z" || event.key === "Z")) {
            if (marking || drag) return;
            event.preventDefault();
            sendEvent({ type: "undo" });
            return;
        }
        if (marking || !selectedId) return;

        if (event.key === "Delete" || event.key === "Backspace") {
            event.preventDefault();
            deleteSelected();
            return;
        }
        // Ctrl (or Cmd on a Mac) +/- are the browser's page-zoom shortcuts, so
        // these must be swallowed rather than merely handled.
        if (!event.ctrlKey && !event.metaKey) return;
        if (event.key === "+" || event.key === "=" || event.code === "NumpadAdd") {
            event.preventDefault();
            shiftSelectedDegree(1);
        } else if (event.key === "-" || event.key === "_" || event.code === "NumpadSubtract") {
            event.preventDefault();
            shiftSelectedDegree(-1);
        }
    }

    // -----------------------------------------------------------------
    // Wheel panning
    // -----------------------------------------------------------------
    function onWheel(event) {
        // Unconditional: the page must never scroll while the cursor is over any
        // pane of the chart.
        event.preventDefault();
        if (!chart) return;

        var horizontal = Math.abs(event.deltaX) > Math.abs(event.deltaY);
        // Verified in Chrome: wheel forward/up (deltaY < 0) and trackpad
        // swipe-left (deltaX > 0) both move the chart forward in time, which is
        // an increasing time-scale scroll position.
        var d = horizontal ? event.deltaX : -event.deltaY;
        if (event.deltaMode === 1) d *= 33;          // lines
        else if (event.deltaMode === 2) d *= 100;    // pages
        if (!d) return;

        var speed = lastConfig.wheel_speed || 8;
        var bars_ = (d / 100) * speed;

        var ts = chart.timeScale();
        // scrollPosition() only catches up with a scrollToPosition() on the next
        // repaint, so a fast spin delivering several wheel events inside one frame
        // would keep reading the same stale position and collapse into a single
        // notch. Accumulate within the frame, then resync with the real (possibly
        // clamped) position so drag-pans and edge clamping are never fought.
        if (wheelTarget === null) {
            var pos = ts.scrollPosition();
            if (pos === null || pos === undefined || isNaN(pos)) return;
            wheelTarget = pos;
            requestAnimationFrame(function () { wheelTarget = null; });
        }
        wheelTarget += bars_;
        // animate: false -- fast wheel spins must not queue up animations.
        ts.scrollToPosition(wheelTarget, false);
    }

    // The one place that knows how tall the chart should be. While fullscreen
    // the viewport is the answer under both mechanisms: native fullscreen
    // expands the iframe itself to the screen, and the fallback pins it to
    // 100vw/100vh -- either way the iframe's own window *is* the target box.
    function onResize() {
        if (!chart) return;
        var width = fullscreen ? window.innerWidth : document.body.clientWidth;
        var height = fullscreen ? window.innerHeight : lastConfig.height;
        if (!width || !height) return;
        chart.resize(width, height);
        // The chart may have been built while the tab was hidden (width 0), in
        // which case the initial scroll-to-latest never landed.
        if (needsInitialScroll) {
            chart.timeScale().scrollToPosition(5, false);
            needsInitialScroll = false;
        }
    }

    // -----------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------
    function setPatterns(list) {
        patterns = Array.isArray(list) ? list : [];
        refreshRendered();
    }

    function render(payload, config, patternList, defs, ack) {
        if (!payload || !payload.time || !payload.time.length) return;

        if (defs && !waveDefs) {
            waveDefs = defs;
            indexDegrees(defs);
        }

        // Ack first: pruning the outbox before the authoritative list lands
        // means one recompute, not two, and no flicker in between.
        applyAck(ack);

        if (!chart) {
            buildChart(payload, config);
            lastConfig = config;
            setAllData(payload);
            applyPaneHeights(config.pane_heights, config.height);
            chart.timeScale().scrollToPosition(5, false);
            needsInitialScroll = !document.body.clientWidth;
            lastFingerprint = payload.fingerprint;
            buildToolbar();
            setPatterns(patternList);
            setFrameHeight(config.height);
            if (!readySent) {
                setComponentValue({ seq: 0, events: [] });
                readySent = true;
            }
            return;
        }

        buildToolbar();

        // Config-only updates -- never touch the time scale, so the user's
        // view position survives Streamlit reruns.
        var prevConfig = lastConfig;
        lastConfig = config;
        // A rerun landing mid-fullscreen must not resize the chart back down or
        // re-post a frame height -- either would knock the component out of the
        // viewport-filling layout. The new height is applied on exit instead,
        // by setFullscreenState().
        if (config.height !== prevConfig.height && !fullscreen) {
            onResize();
            setFrameHeight(config.height);
        }
        // Only on an actual config change, so manual separator drags survive
        // unrelated reruns. The price weight is derived from the total height, so
        // a height change has to reapply too.
        if (JSON.stringify(config.pane_heights) !== JSON.stringify(prevConfig.pane_heights) ||
            config.height !== prevConfig.height) {
            applyPaneHeights(config.pane_heights, config.height);
        }
        if (config.bar_spacing !== prevConfig.bar_spacing) {
            chart.timeScale().applyOptions({ barSpacing: config.bar_spacing });
        }

        // Same data -> only the pattern list can have changed, and that must
        // never touch the time scale either.
        if (payload.fingerprint === lastFingerprint) {
            setPatterns(patternList);
            return;
        }

        // New data: any in-progress, selected or unacked pattern refers to bars
        // that are no longer on the chart.
        if (drag) endDrag();
        if (marking) disarm();
        selectedId = null;
        outbox = [];
        chart.timeScale().applyOptions({ timeVisible: intraday(payload.timeframe) });
        setAllData(payload);
        chart.timeScale().scrollToPosition(5, false);
        lastFingerprint = payload.fingerprint;
        setPatterns(patternList);
    }

    function onMessage(event) {
        var data = event.data;
        if (!data || data.type !== "streamlit:render") return;
        var args = data.args || {};
        render(args.payload, args.config || {}, args.patterns, args.wave_defs, args.ack);
    }

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    window.addEventListener("message", onMessage);
    post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
