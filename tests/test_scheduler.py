"""PriorityGate: formula, ordering, tail fade, slot fairness."""
import threading
import time

import pytest

from arc3.scheduler import GameProgress, PriorityGate, priority, tail_value


def started(gid, **kw):
    return GameProgress(game_id=gid, started=True, **kw)


def test_formula_matches_published_terms():
    p = started("g", level=2, actions_on_level=25, tokens_on_level=62000, levels_total=8)
    n, l, a, t = 8, 3, 25, 62000
    a_term = l * (25 / (25 + a)) ** 2 * 55 / (n * (n + 1) / 2)
    b_term = 8.0                                    # 5 levels after level 3 of 8
    c_term = 0.25 * 2 ** (-(a / 115) ** 2) + 0.75 * 2 ** (-(t / 62000) ** 2)
    assert priority(p, 0.0) == pytest.approx((a_term + b_term) * c_term)


def test_levels_clipped_and_unknown_defaults_to_ten():
    assert priority(started("a", levels_total=3)) == priority(started("b", levels_total=6))
    assert priority(started("a", levels_total=0)) == priority(started("b", levels_total=10))
    assert priority(started("a", levels_total=14)) == priority(started("b", levels_total=10))


def test_tail_lookup_by_levels_remaining():
    assert [tail_value(r, 0.0) for r in (0, 1, 2, 3, 7)] == [0, 5, 7, 8, 8]
    on_last = started("g", level=5, levels_total=6)
    one_before = started("g", level=4, levels_total=6)
    assert priority(one_before) > priority(on_last)


def test_tail_fades_linearly_over_last_20_percent():
    assert tail_value(3, 0.5) == 8.0
    assert tail_value(3, 0.8) == pytest.approx(8.0)
    assert tail_value(3, 0.9) == pytest.approx(4.0)
    assert tail_value(3, 1.0) == pytest.approx(0.0)


def test_more_actions_and_tokens_lower_priority():
    fresh = started("a", level=1, levels_total=7)
    stuck = started("b", level=1, levels_total=7, actions_on_level=300, tokens_on_level=150000)
    assert priority(fresh) > priority(stuck)


def test_unstarted_outranks_everything():
    assert priority(GameProgress("new")) > priority(started("x", level=9, levels_total=10))


def _wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def _acquire_in_thread(gate, progress, order):
    def go():
        gate.acquire(progress)
        order.append(progress.game_id)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_capacity_and_reacquire_is_noop():
    gate = PriorityGate(max_active=2)
    a, b, c = GameProgress("a"), GameProgress("b"), GameProgress("c")
    assert gate.acquire(a) and gate.acquire(b)
    assert gate.acquire(a, timeout=0.01)            # already holds: immediate
    assert not gate.acquire(c, timeout=0.05)        # full
    assert a.started and b.started and not c.started
    gate.release(a)
    assert gate.acquire(c, timeout=0.5)


def test_free_slot_goes_to_highest_priority_waiter():
    gate = PriorityGate(max_active=1)
    holder = GameProgress("holder")
    gate.acquire(holder)
    low = started("low", level=0, levels_total=7, actions_on_level=400, tokens_on_level=200000)
    high = started("high", level=4, levels_total=7)
    order = []
    threads = [_acquire_in_thread(gate, low, order)]
    assert _wait_for(lambda: "low" in [w for w, _ in gate.snapshot()["waiting"]])
    threads.append(_acquire_in_thread(gate, high, order))
    assert _wait_for(lambda: len(gate.snapshot()["waiting"]) == 2)
    gate.release(holder)
    assert _wait_for(lambda: order == ["high"])
    gate.release(high)
    assert _wait_for(lambda: order == ["high", "low"])
    for t in threads:
        t.join(1)


def test_unstarted_games_first_in_arrival_order():
    gate = PriorityGate(max_active=1)
    holder = started("holder", level=3, levels_total=7)
    gate.acquire(holder)
    order, threads = [], []
    veteran = started("veteran", level=6, levels_total=7)
    for p in (GameProgress("n1"), veteran, GameProgress("n2")):
        threads.append(_acquire_in_thread(gate, p, order))
        assert _wait_for(lambda pid=p.game_id: pid in [w for w, _ in gate.snapshot()["waiting"]])
    for expected in (["n1"], ["n1", "n2"], ["n1", "n2", "veteran"]):
        gate.release(GameProgress(order[-1]) if order else holder)
        assert _wait_for(lambda e=expected: order == e), order
    for t in threads:
        t.join(1)


def test_release_then_reacquire_queues_behind_equal_waiter():
    gate = PriorityGate(max_active=1)
    a, b = started("a", levels_total=7), started("b", levels_total=7)
    gate.acquire(a)
    order = []
    t = _acquire_in_thread(gate, b, order)
    assert _wait_for(lambda: len(gate.snapshot()["waiting"]) == 1)
    gate.release(a)
    assert _wait_for(lambda: order == ["b"])
    assert not gate.acquire(a, timeout=0.05)        # b holds the only slot now
    gate.release(b)
    assert gate.acquire(a, timeout=0.5)
    t.join(1)


def test_finish_frees_slot_and_cancels_pending_acquire():
    gate = PriorityGate(max_active=1, poll_seconds=0.02)
    a, b = GameProgress("a"), GameProgress("b")
    gate.acquire(a)
    result = []
    t = threading.Thread(target=lambda: result.append(gate.acquire(b)), daemon=True)
    t.start()
    assert _wait_for(lambda: len(gate.snapshot()["waiting"]) == 1)
    gate.finish(b)
    t.join(1)
    assert result == [False]
    gate.finish(a)
    assert gate.snapshot()["active"] == []


def test_run_fraction_follows_clock():
    now = [100.0]
    gate = PriorityGate(clock=lambda: now[0])
    gate.set_run_window(100.0, 200.0)
    now[0] = 190.0
    assert gate.run_fraction() == pytest.approx(0.9)


def test_level_completed_resets_counters():
    p = started("g", actions_on_level=50, tokens_on_level=9000)
    p.level_completed(0)
    assert p.actions_on_level == 50
    p.level_completed(1)
    assert (p.level, p.actions_on_level, p.tokens_on_level) == (1, 0, 0)


def test_many_threads_never_exceed_capacity():
    gate = PriorityGate(max_active=3)
    active, peak, lock = [0], [0], threading.Lock()

    def game(i):
        p = GameProgress(f"g{i}")
        for _ in range(5):
            gate.acquire(p)
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.001)
            with lock:
                active[0] -= 1
            gate.release(p)
        gate.finish(p)
    threads = [threading.Thread(target=game, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert peak[0] <= 3 and not any(t.is_alive() for t in threads)
