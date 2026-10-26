# env + percept: verified toolkit facts

Checked on 2026-10-27 against arc-agi 0.9.8 / arcengine 0.9.3 with the 25 public games in
`data/environments` (offline mode). Tests: `tests/test_env.py`, `tests/test_percept.py`.

## Frames
- Every step returns `FrameDataRaw.frame`: a list of 64x64 `int8` grids, one per engine tick
  (`ARCBaseGame.perform_action` renders a frame each time `step()` runs until the action
  completes). The last frame is the settled board.
- Observed frames per action: ls20 1 (6 on some), tu93 1 or 8, ft09 clicks 1 or 5, g50t up to 25.
  Adjacent frames can be identical; `Env` drops adjacent duplicates and casts to `uint8`.
- All values are 0..15. The palette is the ARC-AGI-3 one (`arc_agi.rendering.COLOR_MAP`), not
  ARC-1: 0 white, 5 black, 8 red, 9 blue, 14 green, etc. `percept.PALETTE` copies it.
- Competition (remote) responses carry the same list in JSON; the toolkit converts to int8 arrays.

## Actions and coordinates
- `available_actions` is a list of ids: 1..4 = UP/DOWN/LEFT/RIGHT, 5 = SPACE, 6 = MOUSE,
  7 = UNDO. RESET (0) is never listed but always accepted.
- MOUSE = ACTION6 with `data={'x': col, 'y': row}` in display coordinates 0..63 (pydantic
  `ComplexAction` validates 0..63). Verified: clicks on ft09 change cells around (row=y, col=x)
  and never around the transposed point. The game maps display to its own grid via
  `camera.display_to_grid`, so scaled/letterboxed levels still take display coordinates.
- ACTION7 (UNDO) is available in 6 public games at level 1: ar25, bp35, lf52, sb26, sk48, su15.
- Click-only games (only MOUSE): ft09, lp85, r11l, s5i5, tn36, vc33. su15 = MOUSE+UNDO.

## Levels
- `win_levels` (total levels) and `levels_completed` arrive in every response, from the first
  reset on (public games: 6..10 levels).
- A level-up happens inside the action: the frames of that step end on the next level (from the engine code; no level is solved in the tests yet).

## RESET and terminal states
- RESET mid-level restarts the current level and keeps `levels_completed` (verified: board equals
  the level start).
- Trap: the engine does a FULL reset (levels_completed -> 0, `full_reset=True`) when no action has
  been taken since the current level started, i.e. two RESETs in a row, or RESET right after a
  level-up. `Env` refuses such a RESET by default (`safe_reset=True`) when levels would be lost.
- Env var `ONLY_RESET_LEVELS=true` makes every RESET a level reset (except after WIN). `Arena`
  always sets it before creating the Arcade. It is read by the engine process, so it only matters
  for local engines; in competition the gateway runs the engine and its setting is unknown, so the
  `Env` guard stays on in every mode (a refused RESET would only have returned the same board).
- After GAME_OVER or WIN any non-RESET action returns an empty frame list and (engine bug)
  `levels_completed=0`. `Env` keeps the previous board and level count for that response.
- GAME_OVER is reachable by random play (ls20 after ~130 moves, tu93 ~50); RESET recovers it.

## Competition mode
- `Arcade(operation_mode=COMPETITION, arc_base_url=..., arc_api_key=...)`; the env var
  `OPERATION_MODE=competition` overrides any constructor mode. Games come from
  `arcade.available_environments` (fetched from `/api/games`). `make()` builds a
  `RemoteEnvironmentWrapper` that POSTs to `/api/cmd/ACTIONn` with `x`/`y` keys.
- `Arena(mode="offline"|"competition", environments_dir=..., base_url=..., api_key=...,
  recordings_dir=...)`; `arena.game_ids()`, `arena.make(game_id) -> Env` (thread-safe across
  games; one Env per thread), `arena.open_scorecard()/close_scorecard()`.
- Scorecards: `open_scorecard()` posts `competition_mode: true`; `Arena` opens one and passes it to
  every `make()`, and `close_scorecard()` closes it once at the end.
- Hidden state (`ARCBaseGame._get_hidden_state`) is never read; trace records (`Env.log_dict`)
  contain the action, state, level counts, frame count and a grid hash only.

## percept notes
- `segment` labels 4-connected same-colour components in raster order. `shape_hash` is invariant
  to 0/90/180/270 rotation (not reflection) and colour; `color_shape_hash` adds colour;
  `rotation` is clockwise degrees from the canonical pose (mod 180 for 2-fold shapes, None for
  4-fold). `children` are directly enclosed components (holes of a node and what sits in them).
  Timing: 3-6 ms on public-game frames, ~20 ms on 64x64 uniform noise.
- `frame_diff` ignores components covering >= 25% of the grid (background) and matches by greedy
  nearest centre within colour+shape groups (no scipy). Ambiguous cases are documented in its
  docstring.
