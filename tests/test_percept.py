"""Perception tests on synthetic grids and real public-game frames."""
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc3 import percept
from arc3.env import Env
from arc3.types import COLOR_LETTERS as L, Action

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "data" / "environments"

L_SHAPE = np.array([[1, 0], [1, 0], [1, 1]], dtype=np.uint8)


def place(grid, shape, r, c, color):
    rr, cc = np.nonzero(shape)
    grid[rr + r, cc + c] = color


def test_segment_basic_and_children():
    g = np.zeros((10, 10), np.uint8)
    g[1:6, 1:6] = 3          # ring of colour 3 ...
    g[2:5, 2:5] = 0          # ... with a hole ...
    g[3, 3] = 8              # ... holding a dot
    g[8, 8] = 9
    seg = percept.segment(g)
    # raster order: background (0,0) first, then the ring, then hole background, dot, corner dot
    colors = [n.color for n in seg.nodes]
    assert colors == [0, 3, 0, 8, 9]
    ring, hole, dot = seg.nodes[1], seg.nodes[2], seg.nodes[3]
    assert ring.bbox == [1, 1, 5, 5] and ring.pixels == 16 and ring.letter == L[3]
    assert ring.children == [hole.id] and hole.children == [dot.id]
    assert dot.parent == hole.id and hole.parent == ring.id
    assert ring.rotational_symmetry == 4 and ring.rotation is None
    assert (0, 1) in seg.adjacency and (1, 2) in seg.adjacency and (2, 3) in seg.adjacency
    assert (0, 3) not in seg.adjacency
    assert seg.node_at(3, 3).id == dot.id and seg.neighbors(dot.id) == [hole.id]
    assert sorted(seg.nodes[0].children) == [ring.id, 4]


def test_shape_hashes_rotation():
    nodes = []
    for k in range(4):
        g = np.zeros((8, 8), np.uint8)
        place(g, np.rot90(L_SHAPE, -k), 2, 2, 4)
        nodes.append([n for n in percept.segment(g).nodes if n.color == 4][0])
    assert len({n.shape_hash for n in nodes}) == 1
    assert len({n.pose_hash for n in nodes}) == 4
    assert [n.rotational_symmetry for n in nodes] == [1] * 4
    rots = [n.rotation for n in nodes]
    assert sorted(rots) == [0, 90, 180, 270]
    assert all((rots[k] - rots[0]) % 360 == 90 * k for k in range(4))
    # colour changes colour_shape_hash only
    g = np.zeros((8, 8), np.uint8)
    place(g, L_SHAPE, 2, 2, 5)
    red = [n for n in percept.segment(g).nodes if n.color == 5][0]
    assert red.shape_hash == nodes[0].shape_hash and red.color_shape_hash != nodes[0].color_shape_hash
    # 2-fold symmetric bar: rotation in {0, 90}
    g = np.zeros((8, 8), np.uint8)
    g[2, 1:5] = 7
    bar = [n for n in percept.segment(g).nodes if n.color == 7][0]
    assert bar.rotational_symmetry == 2 and bar.rotation in (0, 90)


def test_hash_stability():
    g = np.arange(64 * 64, dtype=np.uint32).reshape(64, 64) % 16
    a, b = percept.state_hash(g.astype(np.uint8)), percept.state_hash(g.astype(np.int8).copy())
    assert a == b and len(a) == 16
    g2 = g.astype(np.uint8)
    g2[5, 5] ^= 1
    assert percept.state_hash(g2) != a
    s1 = percept.segment(g.astype(np.uint8))
    percept._shape_cache.clear()
    s2 = percept.segment(g.astype(np.uint8))
    assert [n.color_shape_hash for n in s1.nodes] == [n.color_shape_hash for n in s2.nodes]


def test_frame_diff_synthetic():
    before = np.zeros((20, 20), np.uint8)
    place(before, L_SHAPE, 2, 2, 4)       # will move
    place(before, L_SHAPE, 10, 10, 6)     # will rotate in place
    before[15:17, 2:4] = 8                # will recolour
    before[2, 15] = 9                     # will disappear
    before[12:14, 2:4] = 11               # will grow
    after = np.zeros_like(before)
    place(after, L_SHAPE, 2, 5, 4)
    place(after, np.rot90(L_SHAPE, -1), 10, 10, 6)
    after[15:17, 2:4] = 12
    after[18, 18] = 3                     # appears
    after[12:15, 2:4] = 11
    d = percept.frame_diff(before, after)
    assert d["changed_cell_count"] > 0
    assert [m["delta"] for m in d["moved"]] == [[0, 3]]
    assert len(d["rotated"]) == 1 and d["rotated"][0]["degrees"] in (90, 270)
    assert d["changed_color"] == [{"bbox": [15, 2, 16, 3], "from": L[8], "to": L[12],
                                   "id_before": d["changed_color"][0]["id_before"],
                                   "id_after": d["changed_color"][0]["id_after"]}]
    assert [r["pixels"] for r in d["resized"]] == [[4, 6]]
    assert [a["bbox"] for a in d["appeared"]] == [[18, 18, 18, 18]]
    assert [a["bbox"] for a in d["disappeared"]] == [[2, 15, 2, 15]]
    assert percept.frame_diff(before, before)["changed_cell_count"] == 0


def test_changed_masks():
    a = np.zeros((64, 64), np.uint8)
    b = a.copy()
    b[1, 30] = 3
    assert percept.changed_masks(a, b) == (True, False)
    b[30, 30] = 3
    assert percept.changed_masks(a, b) == (True, True)
    assert percept.changed_masks(a, a) == (False, False)


def test_animation_helpers_synthetic():
    f0 = np.zeros((6, 6), np.uint8)
    f1 = f0.copy(); f1[1, 1] = 2
    f2 = f0.copy(); f2[2, 2] = 2           # (1,1) changed back: transient
    chain = percept.collapse_frames(f0, [f1, f1, f2, f2])
    assert len(chain) == 3
    count, bbox = percept.transient_cells(chain)
    assert count == 1 and bbox == [1, 1, 1, 1]
    tl = percept.animation_timeline(chain)
    assert tl["frames"] == 3 and tl["transient_pixels"] == 1
    assert f"1,1:{L[0]}>{L[2]}" in tl["text"] and f"1,1:{L[2]}>{L[0]}" in tl["text"]
    big = f0.copy(); big[:, :] = 5
    tl = percept.animation_timeline([f0, big], max_cells_per_step=24)
    assert tl["steps"][0]["bbox"] == [0, 0, 5, 5] and "cells" not in tl["steps"][0]


def test_render_sizes():
    g = np.random.default_rng(0).integers(0, 16, (64, 64)).astype(np.uint8)
    img = Image.open(io.BytesIO(percept.render_png(g, scale=10)))
    assert img.size == (640, 640) and img.mode == "RGB"
    assert img.getpixel((5, 5)) == tuple(percept.PALETTE[g[0, 0]])
    h = g.copy(); h[10, 20] = (h[10, 20] + 1) % 16
    diff = Image.open(io.BytesIO(percept.render_diff_png(g, h, scale=4)))
    assert diff.size == (256, 256)
    assert diff.getpixel((20 * 4, 10 * 4)) == tuple(percept.PALETTE[h[10, 20]])
    assert diff.getpixel((0, 0)) == tuple(percept.DIM_RGB)
    lab = Image.open(io.BytesIO(percept.render_png_with_grid_labels(g, scale=10)))
    assert lab.size == (670, 670)
    text = percept.ascii(g)
    assert len(text.splitlines()) == 64 and all(len(line) == 64 for line in text.splitlines())
    assert percept.crop(g, -2, 60, 3, 70).shape == (4, 4)


# ---- real frames

@pytest.fixture(scope="module")
def tu93_steps(tmp_path_factory):
    env = Env.offline("tu93-0768757b", ENVS, recordings_dir=tmp_path_factory.mktemp("rec"))
    out = []
    before = env.observation.grid
    for a in ["UP", "DOWN", "LEFT", "RIGHT"] * 3:
        obs = env.step(Action(a))
        out.append((before, obs))
        before = obs.grid
    return out


def test_segment_speed_real_frames(tu93_steps):
    grids = [o.grid for _, o in tu93_steps]
    percept.segment(grids[0])  # warm the shape cache, as in a real game
    worst = 0.0
    for g in grids:
        best = 1.0
        for _ in range(3):  # best of 3: the machine may be busy with other work
            t = time.perf_counter()
            seg = percept.segment(g)
            best = min(best, time.perf_counter() - t)
        worst = max(worst, best)
        assert sum(n.pixels for n in seg.nodes) == 64 * 64
    assert worst < 0.02


def test_segment_speed_noise():
    g = np.random.default_rng(1).integers(0, 16, (64, 64)).astype(np.uint8)
    t = time.perf_counter()
    seg = percept.segment(g)
    assert time.perf_counter() - t < 0.1 and len(seg.nodes) > 3000


def test_real_animation_and_diff(tu93_steps):
    animated = [(b, o) for b, o in tu93_steps if len(o.frames) > 1]
    assert animated, "tu93 moves should animate"
    before, obs = animated[0]
    chain = percept.collapse_frames(before, obs.frames)
    assert len(chain) >= 3 and np.array_equal(chain[0], before)
    tl = percept.animation_timeline(chain)
    assert tl["frames"] == len(chain) and tl["text"].count("\n") == len(chain) - 2
    count, bbox = percept.transient_cells(chain)
    assert count >= 0 and (bbox is None) == (count == 0)
    d = percept.frame_diff(before, obs.grid)
    assert d["changed_cell_count"] == int((before != obs.grid).sum())
    assert d["moved"] or d["rotated"] or d["resized"] or d["appeared"] or d["changed_color"]


def test_transient_mask():
    f0 = np.zeros((4, 4), np.uint8)
    f1 = f0.copy(); f1[0, 0] = 1
    m = percept.transient_mask([f0, f1, f0])
    assert m.dtype == bool and m.shape == (4, 4) and m.sum() == 1 and m[0, 0]
    assert not percept.transient_mask([f0, f1]).any()
