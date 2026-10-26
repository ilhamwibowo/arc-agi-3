"""Perception over 64x64 grids: segmentation, ascii, PNG rendering, frame diffs, animation summaries.

Pure numpy + Pillow + stdlib. Grids are uint8 arrays (rows, cols) with values 0..15.
Coordinates are always (row, col); bboxes are [r0, c0, r1, c1] inclusive.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .types import COLOR_HEX, COLOR_LETTERS, COLOR_NAMES  # noqa: F401 (names re-exported)

# ARC-AGI-3 palette (types.COLOR_HEX, same as arc_agi.rendering.COLOR_MAP): 0 white, 5 black.
PALETTE = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in COLOR_HEX], dtype=np.uint8)
DIM_RGB = np.array((0x14, 0x1B, 0x3A), dtype=np.uint8)  # navy used for unchanged cells in diffs


def _h(data: bytes, size: int = 6) -> str:
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def state_hash(grid: np.ndarray) -> str:
    """Stable 16-hex hash of a grid's shape and contents."""
    g = np.ascontiguousarray(grid, dtype=np.uint8)
    return _h(bytes(str(g.shape), "ascii") + g.tobytes(), 8)


# ---------------------------------------------------------------- segmentation

@dataclass(repr=False)
class Node:
    id: int
    color: int
    letter: str
    pixels: int
    bbox: list[int]                     # [r0, c0, r1, c1]
    cells: list[tuple[int, int]]
    shape_hash: str                     # rotation-invariant (0/90/180/270), colour-blind
    pose_hash: str                      # rotation-sensitive, colour-blind
    color_shape_hash: str               # rotation-invariant, includes colour
    rotation: Optional[int]             # clockwise degrees from canonical pose; None if 4-fold
    rotational_symmetry: int            # 1, 2 or 4
    children: list[int] = field(default_factory=list)
    parent: Optional[int] = None

    @property
    def center(self) -> tuple[float, float]:
        r0, c0, r1, c1 = self.bbox
        return ((r0 + r1) / 2, (c0 + c1) / 2)

    @property
    def hash(self) -> str:
        """Same as color_shape_hash: equal for same-colour objects of the same shape in any rotation."""
        return self.color_shape_hash

    # Dict-style access so model code like node['color'] / 'bbox' in node / dict(node) works.
    def keys(self) -> list[str]:
        return list(NODE_KEYS)

    def __getitem__(self, key: str):
        if key not in NODE_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key) if key in NODE_KEYS else default

    def __contains__(self, key: object) -> bool:
        return key in NODE_KEYS

    def __iter__(self):
        return iter(NODE_KEYS)

    def items(self) -> list[tuple[str, object]]:
        return [(k, getattr(self, k)) for k in NODE_KEYS]

    def __repr__(self) -> str:
        return (f"Node(id={self.id}, color={self.color} {self.letter}, bbox={self.bbox}, "
                f"pixels={self.pixels}, hash={self.hash[:4]})")

    def to_dict(self, with_cells: bool = False) -> dict:
        d = {k: getattr(self, k) for k in NODE_KEYS if k != "cells"}
        if with_cells:
            d["cells"] = list(self.cells)
        return d


NODE_KEYS = ("id", "color", "letter", "pixels", "bbox", "cells", "hash", "shape_hash", "pose_hash",
             "color_shape_hash", "rotation", "rotational_symmetry", "children", "parent", "center")


@dataclass(repr=False)
class Segmentation:
    nodes: list[Node]
    adjacency: list[tuple[int, int]]    # (a, b) with a < b, edge-sharing components
    labels: np.ndarray                  # component id per cell

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, i: int) -> Node:
        return self.nodes[i]

    def __repr__(self) -> str:
        shown = [repr(n) for n in self.nodes[:60]]
        if len(self.nodes) > 60:
            shown.append(f"... {len(self.nodes) - 60} more")
        return f"Segmentation({len(self.nodes)} objects, {len(self.adjacency)} adjacent pairs)\n" \
            + "\n".join(shown)

    def node_at(self, row: int, col: int) -> Node:
        return self.nodes[int(self.labels[row, col])]

    def by_color(self, color: int) -> list[Node]:
        return [n for n in self.nodes if n.color == color]

    def neighbors(self, node_id: int) -> list[int]:
        return sorted({b if a == node_id else a for a, b in self.adjacency if node_id in (a, b)})


def _rot_cw(cells: tuple, h: int, w: int) -> tuple[tuple, int, int]:
    """Rotate normalized cells 90 degrees clockwise: (r, c) -> (c, h-1-r)."""
    return tuple(sorted((c, h - 1 - r) for r, c in cells)), w, h


_shape_cache: dict[tuple, tuple[str, str, Optional[int], int, bytes]] = {}


def _shape_info(cells: tuple, h: int, w: int) -> tuple[str, str, Optional[int], int, bytes]:
    """(shape_hash, pose_hash, rotation, symmetry, canonical_key) for normalized sorted cells."""
    key = (h, w, cells)
    hit = _shape_cache.get(key)
    if hit is not None:
        return hit
    poses = [(cells, h, w)]
    for _ in range(3):
        poses.append(_rot_cw(*poses[-1]))
    encoded = [repr((ph, pw, pc)).encode() for pc, ph, pw in poses]
    # poses[k] = pose rotated k*90 cw. Canonical = smallest encoding; the pose is the canonical
    # rotated by (4-k)*90 cw where k is the first index reaching the canonical.
    canon = min(encoded)
    k = encoded.index(canon)
    symmetry = sum(1 for e in encoded if e == encoded[0])
    rotation = None if symmetry == 4 else ((4 - k) % 4) * 90 % (360 // symmetry)
    info = (_h(canon), _h(encoded[0]), rotation, symmetry, canon)
    if len(_shape_cache) > 50000:
        _shape_cache.clear()
    _shape_cache[key] = info
    return info


def _label(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """4-connected same-colour labelling in raster order (top-most, then left-most first)."""
    rows, cols = grid.shape
    g = grid.tolist()
    labels = [[-1] * cols for _ in range(rows)]
    comps: list[list[tuple[int, int]]] = []
    for r in range(rows):
        lr = labels[r]
        for c in range(cols):
            if lr[c] != -1:
                continue
            cid = len(comps)
            color = g[r][c]
            lr[c] = cid
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                if y > 0 and labels[y - 1][x] == -1 and g[y - 1][x] == color:
                    labels[y - 1][x] = cid
                    stack.append((y - 1, x))
                if y + 1 < rows and labels[y + 1][x] == -1 and g[y + 1][x] == color:
                    labels[y + 1][x] = cid
                    stack.append((y + 1, x))
                if x > 0 and labels[y][x - 1] == -1 and g[y][x - 1] == color:
                    labels[y][x - 1] = cid
                    stack.append((y, x - 1))
                if x + 1 < cols and labels[y][x + 1] == -1 and g[y][x + 1] == color:
                    labels[y][x + 1] = cid
                    stack.append((y, x + 1))
            cells.sort()
            comps.append(cells)
    return np.array(labels, dtype=np.int32), comps


def _adjacency(labels: np.ndarray) -> list[tuple[int, int]]:
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = a != b
        pairs.append(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], axis=1))
    allp = np.unique(np.concatenate(pairs), axis=0) if pairs else np.empty((0, 2), int)
    return [(int(x), int(y)) for x, y in allp]


def _enclosed(labels: np.ndarray, node: Node) -> set[int]:
    """Ids of components lying entirely in holes of `node` (not reachable from outside it)."""
    r0, c0, r1, c1 = node.bbox
    sub = labels[r0:r1 + 1, c0:c1 + 1]
    h, w = sub.shape
    wall = (sub == node.id).tolist()
    seen = [[False] * w for _ in range(h)]
    stack = [(r, c) for r in range(h) for c in (0, w - 1)] + \
            [(r, c) for c in range(w) for r in (0, h - 1)]
    while stack:
        r, c = stack.pop()
        if seen[r][c] or wall[r][c]:
            continue
        seen[r][c] = True
        for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= y < h and 0 <= x < w and not seen[y][x] and not wall[y][x]:
                stack.append((y, x))
    inside = np.array(seen, dtype=bool) | np.array(wall, dtype=bool)
    hole_ids = set(np.unique(sub[~inside]).tolist())
    outside_ids = set(np.unique(sub[np.array(seen, dtype=bool)]).tolist())
    return hole_ids - outside_ids - {node.id}


def segment(grid: np.ndarray) -> Segmentation:
    """4-connected same-colour components with shape hashes, containment and adjacency."""
    grid = np.asarray(grid)
    labels, comps = _label(grid)
    nodes = []
    for cid, cells in enumerate(comps):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        norm = tuple((r - r0, c - c0) for r, c in cells)
        shape, pose, rot, sym, canon = _shape_info(norm, r1 - r0 + 1, c1 - c0 + 1)
        color = int(grid[cells[0]])
        nodes.append(Node(id=cid, color=color, letter=COLOR_LETTERS[color], pixels=len(cells),
                          bbox=[r0, c0, r1, c1], cells=cells, shape_hash=shape, pose_hash=pose,
                          color_shape_hash=_h(bytes([color]) + canon), rotation=rot,
                          rotational_symmetry=sym))
    _containment(labels, nodes)
    return Segmentation(nodes=nodes, adjacency=_adjacency(labels), labels=labels)


def _containment(labels: np.ndarray, nodes: list[Node]) -> None:
    # Only components whose bbox has a strict interior holding another bbox can enclose anything.
    enclosed: dict[int, set[int]] = {}
    boxes = np.array([n.bbox for n in nodes], dtype=np.int32).reshape(-1, 4)
    for n in nodes:
        r0, c0, r1, c1 = n.bbox
        if r1 - r0 < 2 or c1 - c0 < 2 or n.pixels < 4:
            continue
        if not np.any((boxes[:, 0] > r0) & (boxes[:, 1] > c0)
                      & (boxes[:, 2] < r1) & (boxes[:, 3] < c1)):
            continue
        inner = _enclosed(labels, n)
        if inner:
            enclosed[n.id] = inner
    # Direct children: enclosed by n but not by another component that n encloses.
    for nid, inner in enclosed.items():
        nested = set()
        for m in inner:
            nested |= enclosed.get(m, set())
        nodes[nid].children = sorted(inner - nested)
        for ch in nodes[nid].children:
            nodes[ch].parent = nid


# ---------------------------------------------------------------- text / images

def ascii(grid: np.ndarray) -> str:
    """One letter per cell (types.COLOR_LETTERS), one line per row."""
    return "\n".join("".join(COLOR_LETTERS[v] for v in row) for row in np.asarray(grid).tolist())


def crop(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Inclusive crop, clipped to the grid."""
    rows, cols = grid.shape
    return grid[max(0, r0):min(rows, r1 + 1), max(0, c0):min(cols, c1 + 1)].copy()


def _png(rgb: np.ndarray, scale: int) -> bytes:
    img = Image.fromarray(np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def to_rgb(grid: np.ndarray) -> np.ndarray:
    return PALETTE[np.asarray(grid, dtype=np.uint8) & 15]


def render_png(grid: np.ndarray, scale: int = 10) -> bytes:
    """Nearest-neighbour upscaled PNG in the ARC-AGI-3 palette."""
    return _png(to_rgb(grid), scale)


def render_diff_png(before: np.ndarray, after: np.ndarray, scale: int = 10) -> bytes:
    """Changed cells in their new (true) colour; unchanged cells dimmed to navy."""
    rgb = to_rgb(after).copy()
    rgb[np.asarray(before) == np.asarray(after)] = DIM_RGB
    return _png(rgb, scale)


def render_png_with_grid_labels(grid: np.ndarray, scale: int = 10, every: int = 8) -> bytes:
    """PNG with a label margin: row/col indices every `every` cells and faint grid lines."""
    from PIL import ImageDraw
    rows, cols = grid.shape
    margin = 3 * scale
    rgb = np.repeat(np.repeat(to_rgb(grid), scale, axis=0), scale, axis=1)
    img = Image.new("RGB", (cols * scale + margin, rows * scale + margin), (255, 255, 255))
    img.paste(Image.fromarray(rgb, "RGB"), (margin, margin))
    draw = ImageDraw.Draw(img)
    for i in range(0, max(rows, cols), every):
        if i < cols:
            x = margin + i * scale
            draw.line([(x, margin), (x, margin + rows * scale)], fill=(128, 128, 128))
            draw.text((x + 1, 1), str(i), fill=(0, 0, 0))
        if i < rows:
            y = margin + i * scale
            draw.line([(margin, y), (margin + cols * scale, y)], fill=(128, 128, 128))
            draw.text((1, y + 1), str(i), fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- diffs

def changed_masks(before: np.ndarray, after: np.ndarray, border: int = 4) -> tuple[bool, bool]:
    """(board_changed, gameplay_changed); gameplay ignores the outer `border` cells (HUD)."""
    diff = np.asarray(before) != np.asarray(after)
    board = bool(diff.any())
    inner = diff[border:diff.shape[0] - border, border:diff.shape[1] - border] if border else diff
    return board, bool(inner.any())


def _touched(seg: Segmentation, changed: np.ndarray) -> list[Node]:
    rows, cols = changed.shape
    out = []
    for n in seg.nodes:
        r0, c0, r1, c1 = n.bbox
        if changed[max(0, r0 - 1):min(rows, r1 + 2), max(0, c0 - 1):min(cols, c1 + 2)].any():
            out.append(n)
    return out


def _dist(a: Node, b: Node) -> float:
    (ar, ac), (br, bc) = a.center, b.center
    return abs(ar - br) + abs(ac - bc)


def _greedy_match(before: list[Node], after: list[Node]) -> list[tuple[Node, Node]]:
    """Deterministic greedy assignment by ascending centre distance, ties by ids."""
    cand = sorted((_dist(a, b), a.id, b.id, i, j)
                  for i, a in enumerate(before) for j, b in enumerate(after))
    used_a, used_b, pairs = set(), set(), []
    for _, _, _, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((before[i], after[j]))
    return pairs


def _box(n: Node) -> dict:
    return {"id": n.id, "color": n.letter, "bbox": n.bbox, "pixels": n.pixels}


def _overlap(a: list[int], b: list[int]) -> int:
    h = min(a[2], b[2]) - max(a[0], b[0]) + 1
    w = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return max(0, h) * max(0, w)


def _area(b: list[int]) -> int:
    return (b[2] - b[0] + 1) * (b[3] - b[1] + 1)


def frame_diff(before: np.ndarray, after: np.ndarray, seg_before: Optional[Segmentation] = None,
               seg_after: Optional[Segmentation] = None, max_group_match: int = 40,
               background_fraction: float = 0.25) -> dict:
    """Object-level diff between two grids.

    Heuristic, and correspondences can be ambiguous:
    - Only components whose bbox (dilated by 1) touches a changed cell are considered.
    - Components are matched within groups of equal colour+rotation-invariant shape by greedy
      nearest bbox centre (Manhattan), not an optimal assignment; groups larger than
      `max_group_match` on either side are not matched and land in appeared/disappeared.
    - A component that merges with / splits from a same-colour neighbour changes shape, so it
      shows up as resized (if bboxes overlap >= half the smaller one) or appeared/disappeared.
    - Components covering >= `background_fraction` of the grid are treated as background and
      ignored (their outline changes whenever anything on them moves).
    - rotated = pose changed and centre moved <= 2; a rotation with a larger move is `moved`
      with `rotated_by`. Rotation of 2-fold symmetric shapes is mod 180.
    """
    before, after = np.asarray(before), np.asarray(after)
    changed = before != after
    out = {"changed_cell_count": int(changed.sum()), "moved": [], "rotated": [], "appeared": [],
           "disappeared": [], "changed_color": [], "resized": []}
    if not out["changed_cell_count"]:
        return out
    sb = seg_before or segment(before)
    sa = seg_after or segment(after)
    big = background_fraction * changed.size
    tb = [n for n in _touched(sb, changed) if n.pixels < big]
    ta = [n for n in _touched(sa, changed) if n.pixels < big]
    groups: dict[str, tuple[list[Node], list[Node]]] = {}
    for n in tb:
        groups.setdefault(n.color_shape_hash, ([], []))[0].append(n)
    for n in ta:
        groups.setdefault(n.color_shape_hash, ([], []))[1].append(n)
    left_b, left_a = [], []
    for key in sorted(groups):
        gb, ga = groups[key]
        if len(gb) > max_group_match or len(ga) > max_group_match:
            left_b += gb
            left_a += ga
            continue
        pairs = _greedy_match(gb, ga)
        mb = {a.id for a, _ in pairs}
        ma = {b.id for _, b in pairs}
        left_b += [n for n in gb if n.id not in mb]
        left_a += [n for n in ga if n.id not in ma]
        for a, b in pairs:
            _classify(a, b, out)
    _pair_leftovers(left_b, left_a, out)
    return out


def _classify(a: Node, b: Node, out: dict) -> None:
    rot = 0
    if a.rotation is not None and b.rotation is not None:
        rot = (b.rotation - a.rotation) % (360 // a.rotational_symmetry)
    dr, dc = b.bbox[0] - a.bbox[0], b.bbox[1] - a.bbox[1]
    rec = {"color": a.letter, "from": a.bbox, "to": b.bbox, "id_before": a.id, "id_after": b.id}
    if rot and _dist(a, b) <= 2:
        out["rotated"].append({**rec, "degrees": rot})
    elif rot or dr or dc:
        out["moved"].append({**rec, "delta": [dr, dc], **({"rotated_by": rot} if rot else {})})


def _pair_leftovers(left_b: list[Node], left_a: list[Node], out: dict) -> None:
    # Phase A: same shape at the same top-left with a different colour => recolour.
    used_b, used_a = set(), set()
    for a in left_b:
        for b in left_a:
            if b.id in used_a or a.id in used_b:
                continue
            if a.bbox[:2] == b.bbox[:2] and a.pose_hash == b.pose_hash and a.color != b.color:
                out["changed_color"].append({"bbox": a.bbox, "from": a.letter, "to": b.letter,
                                             "id_before": a.id, "id_after": b.id})
                used_b.add(a.id)
                used_a.add(b.id)
    # Phase B: same colour, bboxes overlapping >= half of the smaller => resize/reshape.
    cand = []
    for a in left_b:
        for b in left_a:
            if a.id in used_b or b.id in used_a or a.color != b.color:
                continue
            ov = _overlap(a.bbox, b.bbox)
            if ov and ov >= 0.5 * min(_area(a.bbox), _area(b.bbox)):
                cand.append((-ov, abs(a.pixels - b.pixels), a.id, b.id, a, b))
    for _, _, _, _, a, b in sorted(cand, key=lambda t: t[:4]):
        if a.id in used_b or b.id in used_a:
            continue
        used_b.add(a.id)
        used_a.add(b.id)
        if a.bbox == b.bbox and a.pixels == b.pixels:
            continue  # same outline and size: only interior holes moved, already reported
        out["resized"].append({"color": a.letter, "from": a.bbox, "to": b.bbox,
                               "pixels": [a.pixels, b.pixels], "id_before": a.id,
                               "id_after": b.id})
    out["disappeared"] = [_box(n) for n in left_b if n.id not in used_b]
    out["appeared"] = [_box(n) for n in left_a if n.id not in used_a]


# ---------------------------------------------------------------- animation

def collapse_frames(before: Optional[np.ndarray], frames: list[np.ndarray]) -> list[np.ndarray]:
    """Prepend the pre-action board and drop adjacent duplicates (non-adjacent repeats kept)."""
    chain = ([np.asarray(before)] if before is not None else []) + [np.asarray(f) for f in frames]
    out: list[np.ndarray] = []
    for f in chain:
        if not out or not np.array_equal(out[-1], f):
            out.append(f)
    return out


def transient_cells(frames: list[np.ndarray]) -> tuple[int, Optional[list[int]]]:
    """Cells whose value changes 2+ times along the chain: (count, bbox or None)."""
    mask = transient_mask(frames)
    count = int(mask.sum())
    if not count:
        return 0, None
    rr, cc = np.nonzero(mask)
    return count, [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]


def transient_mask(frames: list[np.ndarray]) -> np.ndarray:
    """Boolean grid of cells whose value changes 2+ times along the chain."""
    stack = np.stack([np.asarray(f) for f in frames])
    if len(stack) < 3:
        return np.zeros(stack.shape[1:], dtype=bool)
    return (stack[1:] != stack[:-1]).sum(axis=0) >= 2


def animation_timeline(frames: list[np.ndarray], max_cells_per_step: int = 24,
                       max_total: int = 80) -> dict:
    """Compact per-step text diff of a collapsed frame chain.

    Each step lists changed cells as 'r,c:X>Y' (colour letters); steps with more changes than
    `max_cells_per_step` list a bbox and count instead. Listing stops after `max_total` cells.
    """
    frames = [np.asarray(f) for f in frames]
    count, bbox = transient_cells(frames)
    steps, lines, total = [], [], 0
    for i in range(1, len(frames)):
        diff = np.argwhere(frames[i] != frames[i - 1])
        n = len(diff)
        rec = {"step": i, "changed": n}
        if n and total < max_total and n <= max_cells_per_step:
            take = diff[:max_total - total]
            total += len(take)
            rec["cells"] = " ".join(
                f"{r},{c}:{COLOR_LETTERS[frames[i - 1][r, c]]}>{COLOR_LETTERS[frames[i][r, c]]}"
                for r, c in take.tolist())
        elif n:
            rr, cc = diff[:, 0], diff[:, 1]
            rec["bbox"] = [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())]
        steps.append(rec)
        detail = rec.get("cells") or (f"bbox {rec['bbox']}" if "bbox" in rec else "")
        lines.append(f"frame {i}: {n} cells changed {detail}".rstrip())
    return {"frames": len(frames), "transient_pixels": count, "transient_bbox": bbox,
            "steps": steps, "text": "\n".join(lines)}
