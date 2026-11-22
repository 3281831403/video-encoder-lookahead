"""Specification tests for the look-ahead bit allocation planner.

Run from the repository root with ``python -m pytest tests -q``.
Every test exercises the public API in ``lookahead.plan``.  The shipped
module contains only ``NotImplementedError`` stubs, so the initial state is
exactly 15 red tests.
"""

import pytest

from lookahead.plan import (
    assign_bits,
    budget_check,
    complexity,
    plan_batch,
    plan_gop,
    reorder_bframes,
    scene_scores,
    validate_plan,
)


def _mean(values):
    values = list(values)
    assert values
    return sum(values) / len(values)


def _i_positions(types):
    return [i for i, frame_type in enumerate(types) if frame_type == "I"]


def _coded_positions(encoded):
    return {frame: slot for slot, frame in enumerate(encoded)}


def _references(types, index):
    """Return display indices that frame ``index`` depends on.

    I frames have no references.  P frames depend on the nearest preceding
    I/P anchor.  B frames depend on the nearest preceding anchor and on the
    nearest following anchor (their future reference).
    """
    preceding = [i for i, t in enumerate(types[:index]) if t in ("I", "P")]
    refs = [preceding[-1]] if preceding else []
    if types[index] == "B":
        following = [
            i
            for i, t in enumerate(types[index + 1 :], start=index + 1)
            if t in ("I", "P")
        ]
        refs.append(following[0])
    return refs


def test_plan_gop_last_gop_does_not_exceed_keyint():
    # Defect 1: deciding "still inside the GOP" with <= schedules the next I
    # one frame late, so the last interval can become keyint + 1.
    n_frames, keyint = 100, 12
    types = plan_gop(n_frames, keyint, [0])
    i_positions = _i_positions(types)

    assert i_positions[0] == 0
    assert all(right - left <= keyint for left, right in zip(i_positions, i_positions[1:]))
    assert i_positions == list(range(0, n_frames, keyint))
    assert types[12] == "I"


def test_plan_gop_scene_cut_in_b_slot_becomes_i():
    # Defect 2: cuts that land on a B slot must force an I frame there; a
    # planner that only promotes P slots silently drops the cut.
    n_frames, keyint, cuts = 20, 6, [3, 9, 14]
    types = plan_gop(n_frames, keyint, cuts)

    assert len(types) == n_frames
    for cut in cuts:
        assert types[cut] == "I"
    # These indices are not on the periodic grid, so only the cut rule can
    # turn them into I frames.
    assert types[3] == "I"
    assert types[14] == "I"


def test_assign_bits_keeps_i_premium_after_normalization():
    # Defect 3: allocating I frames and then re-normalizing the whole list
    # can scale the I premium away; an I frame must stay clearly above P.
    types = ["I"] + ["B"] * 5 + ["P"]
    frame_complexity = [0.5] * len(types)
    bits = assign_bits(types, 2100, frame_complexity)

    assert bits[0] == max(bits)
    assert bits[0] > 1.5 * bits[-1]


def test_assign_bits_total_independent_of_complexity_scale():
    # Defect 4: weighting with raw (un-normalized) complexity makes the total
    # depend on the magnitude of the values instead of only on the budget.
    types = ["I", "B", "P", "B", "B", "P"]
    low = [0.02] * len(types)
    high = [0.98] * len(types)

    low_bits = assign_bits(types, 1000, low)
    high_bits = assign_bits(types, 1000, high)

    assert abs(sum(low_bits) - 1000) <= 1
    assert abs(sum(high_bits) - 1000) <= 1
    assert sum(low_bits) == sum(high_bits)


def test_reorder_bframes_preserves_b_display_order():
    # Defect 5: a B run must be coded in display order (B1 B2 B3), never
    # reversed (B3 B2 B1).
    types = ["I", "B", "B", "B", "P"]
    encoded = reorder_bframes(types)

    assert sorted(encoded) == list(range(5))
    assert encoded == [0, 4, 1, 2, 3]
    assert encoded[2:] == [1, 2, 3]


def test_reorder_bframes_never_crosses_gop_boundary():
    # Defect 6: swapping the last B of a GOP with the next GOP's I mixes two
    # GOPs and breaks the closed reference chain.
    types = plan_gop(12, 5, [0])
    encoded = reorder_bframes(types)
    positions = _coded_positions(encoded)

    assert sorted(encoded) == list(range(12))
    assert max(positions[i] for i in range(0, 5)) < min(positions[i] for i in range(5, 10))
    assert max(positions[i] for i in range(5, 10)) < min(positions[i] for i in range(10, 12))
    assert encoded == [0, 4, 1, 2, 3, 5, 9, 6, 7, 8, 10, 11]


def test_budget_check_one_bit_tolerance_is_inclusive():
    # Defect 7: ``abs(diff) < 1`` rejects the allowed one-bit drift; the
    # tolerance is inclusive and an exact zero-budget plan is ok.
    assert budget_check([], 0) == {"total": 0, "diff": 0, "ok": True}
    assert budget_check([300], 300) == {"total": 300, "diff": 0, "ok": True}
    assert budget_check([301], 300) == {"total": 301, "diff": 1, "ok": True}
    assert budget_check([299], 300) == {"total": 299, "diff": -1, "ok": True}
    assert budget_check([1], 0) == {"total": 1, "diff": 1, "ok": True}
    assert budget_check([302], 300)["ok"] is False
    assert budget_check([298], 300)["ok"] is False
    assert set(budget_check([301], 300)) == {"total", "diff", "ok"}


def test_validate_plan_rejects_b_without_future_reference():
    # Defect 8: a trailing B frame has no future I/P reference and cannot be
    # decoded, but a checker that only looks at I spacing and the total misses
    # it.
    errors = validate_plan(["I", "P", "B"], [100, 100, 100], 300, 10)
    assert errors, "trailing B frame must be reported as an error"

    assert validate_plan(["I", "P"], [150, 150], 300, 10) == []


def test_scene_scores_window_excludes_current_frame():
    # Defect 9: if the window contains the current frame, a cut at index k is
    # measured against a window that already includes the spike, which shifts
    # the decision by one frame.
    frame_complexity = [0.05, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05]
    scores = scene_scores(frame_complexity, 3)

    assert len(scores) == len(frame_complexity)
    assert scores[0] == 0.0
    expected = 0.9 / ((0.05 + 0.05 + 0.05) / 3.0)
    assert scores[4] == pytest.approx(expected)
    assert max(range(len(scores)), key=lambda i: scores[i]) == 4
    assert scores[4] > scores[5]


def test_plan_batch_second_clip_uses_own_complexity():
    # Defect 10: reusing the first clip's complexity array makes the second
    # clip plan from the wrong content.
    flat = [[20] * 8 for _ in range(24)]
    busy = []
    for i in range(24):
        if i % 2 == 0:
            busy.append([0, 200, 0, 200, 0, 200, 0, 200])
        else:
            busy.append([200, 0, 200, 0, 200, 0, 200, 0])
    clips = [{"name": "flat", "frames": flat}, {"name": "busy", "frames": busy}]
    config = {"keyint": 8, "budget": 2400, "window": 4, "scene_threshold": 2.0}

    batch = plan_batch(clips, config)
    solo_busy = plan_batch([clips[1]], config)

    assert batch["clips"][1] == solo_busy["clips"][0]
    assert batch["clips"][1]["complexity"] == complexity(busy)
    assert batch["clips"][0]["complexity"] != batch["clips"][1]["complexity"]


def test_plan_gop_i_interval_never_exceeds_keyint():
    # Cross-check for many keyints and clip lengths: with frame 0 forced to I
    # and no scene cuts, the I-to-I distance must always be <= keyint.
    for keyint in (1, 2, 3, 5, 7, 8, 13):
        for n_frames in (1, 2, keyint, keyint + 1, 3 * keyint + 2):
            types = plan_gop(n_frames, keyint, [0])
            assert len(types) == n_frames
            assert types[0] == "I"
            positions = _i_positions(types)
            assert all(
                right - left <= keyint
                for left, right in zip(positions, positions[1:])
            )


def test_assign_bits_total_equals_budget_within_one_bit():
    # Budget conservation at edges: zero, tiny, and large budgets all stay
    # within the inclusive one-bit tolerance.
    types = plan_gop(37, 8, [0, 20])
    frame_complexity = [((i * 7) % 11) / 10.0 for i in range(37)]

    for budget in (0, 1, 2, 37, 1000, 12345):
        bits = assign_bits(types, budget, frame_complexity)
        assert len(bits) == len(types)
        assert all(isinstance(value, int) and value >= 0 for value in bits)
        assert abs(sum(bits) - budget) <= 1


def test_i_average_bits_strictly_exceed_p_average():
    # The type premium must survive the full plan + allocation pipeline.
    types = plan_gop(30, 6, [0])
    frame_complexity = [0.5] * 30
    bits = assign_bits(types, 6000, frame_complexity)

    i_average = _mean(value for value, t in zip(bits, types) if t == "I")
    p_average = _mean(value for value, t in zip(bits, types) if t == "P")
    assert i_average > p_average


def test_reorder_bframes_decode_order_respects_references():
    # Coded order must place every reference before the frame that needs it.
    types = plan_gop(21, 4, [0, 13])
    encoded = reorder_bframes(types)
    positions = _coded_positions(encoded)

    assert sorted(encoded) == list(range(21))
    for index, frame_type in enumerate(types):
        if frame_type == "I":
            continue
        refs = _references(types, index)
        assert refs, "P and B frames need at least one reference"
        for reference in refs:
            assert positions[reference] < positions[index]


def test_plan_batch_matches_separate_plans():
    # Batching must not leak state between clips: clip k of a batch is
    # identical to planning clip k alone.
    first = {
        "name": "a",
        "frames": [[10, 20, 10, 20, 10, 20, 10, 20] for _ in range(20)],
    }
    second = {"name": "b", "frames": [[40] * 8 for _ in range(20)]}
    config = {"keyint": 5, "budget": 1500, "window": 3, "scene_threshold": 1.8}

    both = plan_batch([first, second], config)
    solo_first = plan_batch([first], config)
    solo_second = plan_batch([second], config)

    assert both["clips"][0] == solo_first["clips"][0]
    assert both["clips"][1] == solo_second["clips"][0]
    assert both["total_bits"] == solo_first["total_bits"] + solo_second["total_bits"]
    assert both["ok"] is True
