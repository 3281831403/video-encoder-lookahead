"""Look-ahead bit allocation planner for a hybrid video encoder.

Terminology
-----------
Display order
    Frames are indexed ``0 .. n-1`` in presentation order.

Frame types
    ``"I"``  intra anchor: decoded independently and a random-access point.
    ``"P"``  inter anchor: references the nearest preceding I/P anchor.
    ``"B"``  bidirectional frame: references the nearest preceding anchor and
             the nearest following anchor inside its own GOP.

GOP
    The run of frames starting at an I and ending just before the next I.

Budget
    Bits are integers.  Conservation tolerance is one bit, inclusive:
    ``abs(sum(bits) - budget) <= 1``.

Coded order
    :func:`reorder_bframes` returns a permutation of display indices; every
    reference of a frame must appear earlier in coded order.

All eight public functions below are stubs.  Implement them without changing
their signatures, return shapes, or the tests in ``tests/test_plan.py``.
"""

import math

TYPE_WEIGHT = {"I": 6.0, "P": 3.0, "B": 1.0}
BIT_TOLERANCE = 1


def complexity(frames):
    """Return one normalized complexity value in ``[0.0, 1.0]`` per frame.

    ``frames`` is a non-empty sequence of frames; each frame is a non-empty
    sequence of numeric samples, and all frames have the same length.  For
    frame ``i`` the raw signal is::

        mean(abs(frame[i][k] - frame[i-1][k])) + population_variance(frame[i])

    with the difference term equal to ``0.0`` for ``i == 0``.  The result is
    the raw signal divided by the largest raw signal, so the maximum is
    ``1.0`` whenever any raw signal is non-zero, and every value is ``0.0``
    for an all-constant clip.  Normalization makes values comparable across
    clips with different contrast or sample magnitude.

    Returns ``list[float]`` of length ``len(frames)``.  Values are monotone
    in the raw signal: a frame with a larger raw signal never scores lower.
    """
    n = len(frames)
    if n == 0:
        return []

    raw = []
    previous = None
    for i, frame in enumerate(frames):
        samples = [float(value) for value in frame]
        count = len(samples)

        mean = sum(samples) / count
        variance = sum((value - mean) ** 2 for value in samples) / count

        if i == 0:
            difference = 0.0
        else:
            difference = sum(
                abs(value - previous[k]) for k, value in enumerate(samples)
            ) / count

        raw.append(difference + variance)
        previous = samples

    peak = max(raw)
    if peak <= 0.0:
        return [0.0] * n
    return [value / peak for value in raw]


def scene_scores(complexity, window):
    """Sliding-window normalized score used to locate scene cuts.

    ``complexity`` is the normalized list produced by :func:`complexity`.
    ``window >= 1`` is the number of *preceding* frames used as reference.
    The window is strictly causal -- it never contains the frame being
    scored -- so a cut at display index ``i`` is visible at index ``i`` and is
    not delayed by one frame::

        score[0] = 0.0
        lo = max(0, i - window)
        score[i] = complexity[i] / mean(complexity[lo:i])

    If that reference mean is ``0.0`` the score is ``0.0``.  Scores are
    ``>= 0.0``; a value well above ``1.0`` marks a candidate cut at exactly
    that index (the caller applies its own threshold, see
    :func:`plan_batch`).  Returns ``list[float]`` of the same length.
    """
    n = len(complexity)
    if n == 0:
        return []

    values = list(complexity)
    scores = [0.0] * n
    for i in range(1, n):
        lo = max(0, i - window)
        reference = values[lo:i]
        if not reference:
            scores[i] = 0.0
            continue
        mean = sum(reference) / len(reference)
        scores[i] = values[i] / mean if mean != 0.0 else 0.0
    return scores


def plan_gop(n_frames, keyint, scene_cuts):
    """Build display-order frame types for one clip.

    Parameters
    ----------
    n_frames:
        Number of frames, ``>= 1``.
    keyint:
        Maximum allowed distance between consecutive I frames, ``>= 1``.
        The I-to-I interval is compared inclusively: an interval of exactly
        ``keyint`` frames is legal, an interval of ``keyint + 1`` is not.
    scene_cuts:
        Iterable of display indices that start a new scene.  Frame ``0`` is
        always I and every index in ``scene_cuts`` is always I.  Duplicates
        are harmless.

    Deterministic layout
    --------------------
    * I frames go to ``0``, to every scene cut, and periodically every
      ``keyint`` frames measured from the previous I (a scene cut restarts
      the periodic grid).
    * Inside one GOP (I up to, but excluding, the next I) the last frame
      before the next I is P: it is the future reference for that GOP's B
      frames.  Every other non-I frame is B.
    * A GOP therefore ends in P, or is a single I when the clip ends on an I.
      B frames always sit between two anchors of their own GOP.

    Returns ``list[str]`` of length ``n_frames`` over ``{"I", "P", "B"}``.
    """
    if n_frames < 1:
        return []

    cuts = {index for index in scene_cuts if 0 <= index < n_frames}

    is_i = [False] * n_frames
    is_i[0] = True
    last_i = 0
    for index in range(1, n_frames):
        # An interval of exactly keyint is legal; only a longer one forces a
        # new I.  Checking at each frame (rather than with <=) keeps every
        # I-to-I interval at most keyint.
        if index in cuts or index - last_i >= keyint:
            is_i[index] = True
            last_i = index

    types = ["B"] * n_frames
    i_positions = [index for index in range(n_frames) if is_i[index]]
    for index in i_positions:
        types[index] = "I"

    for position, start in enumerate(i_positions):
        if position + 1 < len(i_positions):
            end = i_positions[position + 1]
        else:
            end = n_frames
        # Last frame of the GOP takes the future-reference role; a GOP that
        # is a single I keeps its I.
        if end - 1 > start:
            types[end - 1] = "P"

    return types


def assign_bits(types, budget, complexity):
    """Distribute an integer bit budget over frames.

    ``types`` is the display-order type list from :func:`plan_gop` and
    ``complexity`` is the matching normalized list (values normally in
    ``[0, 1]``).  ``budget`` is a non-negative integer number of bits.

    The weight of frame ``i`` is::

        TYPE_WEIGHT[types[i]] * (1.0 + complexity[i])

    with ``TYPE_WEIGHT = {"I": 6.0, "P": 3.0, "B": 1.0}``.  Thus at equal
    complexity an I frame receives twice the bits of a P frame.  Complexity
    is only a relative modifier: its magnitude must not change the conserved
    total.

    The result is ``list[int]`` with ``len(result) == len(types)``,
    non-negative values, and a total conserved exactly: ``sum(result) ==
    budget`` for every integer budget (rounding residue is handed out by
    largest fractional part, ties broken by lower display index).
    """
    n = len(types)
    if n == 0:
        return []
    if budget <= 0:
        return [0] * n

    weights = [
        TYPE_WEIGHT[frame_type] * (1.0 + complexity[i])
        for i, frame_type in enumerate(types)
    ]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        # Degenerate weights cannot happen with the published constants, but
        # fall back to an equal split instead of dividing by zero.
        weights = [1.0] * n
        total_weight = float(n)

    exact = [budget * weight / total_weight for weight in weights]
    bits = [int(math.floor(value)) for value in exact]

    residue = budget - sum(bits)
    order = sorted(range(n), key=lambda i: (bits[i] - exact[i], i))
    for k in range(residue):
        bits[order[k % n]] += 1

    return bits


def reorder_bframes(types):
    """Convert display order to coded order.

    ``types`` is a display-order list produced by :func:`plan_gop`.  The
    result is a permutation of ``range(len(types))``: entry ``k`` is the
    display index coded at position ``k``.

    A GOP is the run starting at an I and ending just before the next I.
    Within each GOP the coded order is the I, then the GOP's final P (the
    future reference), then the B frames in increasing display order::

        display  I B1 B2 P   ->   coded  I P B1 B2

    GOPs are concatenated in display order.  No frame may cross an I
    boundary: every frame of GOP ``k`` is coded before any frame of GOP
    ``k + 1``, and a run of B frames keeps its display order (never reversed).
    """
    n = len(types)
    if n == 0:
        return []

    gops = []
    current = []
    for index, frame_type in enumerate(types):
        if frame_type == "I" and current:
            gops.append(current)
            current = []
        current.append(index)
    if current:
        gops.append(current)

    coded = []
    for gop in gops:
        # I leads, then anchors (future references) in display order, then
        # the B run in display order.  A single-I GOP contributes just the I.
        coded.extend(index for index in gop if types[index] == "I")
        coded.extend(index for index in gop if types[index] == "P")
        coded.extend(index for index in gop if types[index] == "B")
    return coded


def budget_check(bits, budget):
    """Summarize budget conservation for one plan.

    Returns ``{"total": int, "diff": int, "ok": bool}`` where
    ``total = sum(bits)``, ``diff = total - budget`` and
    ``ok = abs(diff) <= 1``.  The tolerance is one bit, inclusive: exactly
    one bit over or under is acceptable, two bits is not.  A zero budget is
    not special-cased; an empty or all-zero plan is ok, while one bit of
    drift is still within tolerance.
    """
    total = sum(bits)
    diff = total - budget
    return {"total": total, "diff": diff, "ok": abs(diff) <= BIT_TOLERANCE}


def validate_plan(types, bits, budget, keyint):
    """Check one plan and return a list of error strings (empty == valid).

    ``types`` is the display-order type list, ``bits`` the matching per-frame
    integers, ``budget`` the target total, and ``keyint`` the maximum I-to-I
    distance.  Checks, in this order:

    1. ``len(types) == len(bits)`` and the plan is non-empty.
    2. Every type is one of ``"I"``, ``"P"``, ``"B"``; frame 0 is ``"I"``.
    3. Consecutive I positions differ by at most ``keyint`` (inclusive).
    4. ``abs(sum(bits) - budget) <= 1`` (one-bit tolerance).
    5. Every B frame has at least one I/P anchor after it in display order;
       that following anchor is its future reference.  A trailing B, or a B
       whose only anchors are in the past, cannot be decoded.

    Return one human-readable string per violated rule.  An empty list means
    the plan is valid; it never raises for a well-formed argument list.
    """
    errors = []

    if not types or not bits or len(types) != len(bits):
        errors.append("plan is empty or types/bits lengths differ")
        return errors

    valid_types = {"I", "P", "B"}
    if any(frame_type not in valid_types for frame_type in types):
        errors.append("unknown frame type present")
    if types[0] != "I":
        errors.append("frame 0 must be I")

    i_positions = [index for index, frame_type in enumerate(types) if frame_type == "I"]
    for left, right in zip(i_positions, i_positions[1:]):
        if right - left > keyint:
            errors.append(
                "I-to-I interval %d..%d exceeds keyint %d" % (left, right, keyint)
            )

    if abs(sum(bits) - budget) > BIT_TOLERANCE:
        errors.append("bit total deviates from budget by more than one bit")

    for index, frame_type in enumerate(types):
        if frame_type == "B":
            if not any(t in ("I", "P") for t in types[index + 1 :]):
                errors.append(
                    "B frame at %d has no future I/P reference" % index
                )

    return errors


def plan_batch(clips, config):
    """Plan several clips independently and report the batch result.

    ``clips`` is a non-empty list of ``{"name": str, "frames": frames}``
    mappings.  ``config`` supplies the same settings for every clip::

        {"keyint": int, "budget": int, "window": int,
         "scene_threshold": float}

    For each clip, in isolation:

    1. ``c = complexity(clip["frames"])``
    2. ``s = scene_scores(c, window)``
    3. ``cuts = [0] + [i for i in range(1, n) if s[i] >= scene_threshold]``
    4. ``types = plan_gop(n, keyint, cuts)``
    5. ``bits = assign_bits(types, budget, c)``

    Returns::

        {
          "clips": [
            {"name": str, "types": list[str], "bits": list[int],
             "complexity": list[float],
             "check": {"total": int, "diff": int, "ok": bool}},
            ...
          ],
          "total_bits": int,   # sum over every clip's bits
          "ok": bool,          # all checks ok and all validate_plan empty
        }

    Clips are independent: entry ``k`` of a batch must equal the only entry
    produced when clip ``k`` is planned alone.  No complexity array, plan, or
    bit allocation may be reused across clips.
    """
    keyint = config["keyint"]
    budget = config["budget"]
    window = config["window"]
    scene_threshold = config["scene_threshold"]

    planned = []
    total_bits = 0
    all_ok = True

    for clip in clips:
        frames = clip["frames"]
        # Every clip gets freshly computed arrays; nothing is shared.
        frame_complexity = complexity(frames)
        scores = scene_scores(frame_complexity, window)
        cuts = [0] + [
            index
            for index in range(1, len(frames))
            if scores[index] >= scene_threshold
        ]
        types = plan_gop(len(frames), keyint, cuts)
        bits = assign_bits(types, budget, frame_complexity)
        check = budget_check(bits, budget)

        if not check["ok"] or validate_plan(types, bits, budget, keyint):
            all_ok = False
        total_bits += sum(bits)

        planned.append(
            {
                "name": clip["name"],
                "types": types,
                "bits": bits,
                "complexity": frame_complexity,
                "check": check,
            }
        )

    return {"clips": planned, "total_bits": total_bits, "ok": all_ok}
