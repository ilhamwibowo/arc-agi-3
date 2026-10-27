# Fix review 5: items from trace-review-run5

All tests pass: `.venv/bin/python -m pytest -q tests/` gives 146 passed (131 before). Seven
commits on main, one per item plus a follow-up on item 3. New tests are in
tests/test_fix_review5.py; one test each in tests/test_fix_review3.py, tests/test_sandbox.py and
tests/test_prompts.py was updated for the new behaviour. Not touched: runner.py, kaggle_main.py,
kaggle_serve.py, llm.py, context.py, notebook.py, types.py.

Prompt growth over the run #5 system prompt (8803 chars): +404 chars, guarded by
`test_output_cap_and_short_burst_line` (< 500).

## 1. RESET inside a batch (fix 3 / bug 1) - f3cd11b

- sandbox.py:513-519: the terminal latch is set only for `level_completed`, `game_over`,
  `run_complete` and `level_reset`. A model RESET is an ordinary executed action: the batch
  carries on with the actions after it, `stop_reason` stays None and nothing is skipped. The
  module docstring says so.
- The `reset_noop` guard (session.py) is unchanged: a RESET with no move since the level start
  or the last RESET is still refused for free, so `action([RESET, ...])` at a fresh board costs
  nothing and continues with the rest.
- wm record: every executed RESET is still a restart boundary (`Record.restarts`), so a batch
  `UP, UP, RESET, DOWN` records three transitions with `restarts == [2]`.
- Prompt, RESET line (prompts.py:53):

  ```
  -         "dozens of actions; RESET costs one.")
  +         "dozens of actions; RESET costs one. The start board is deterministic, so actions may "
  +         "follow RESET in the same `action([...])` call.")
  ```
- Tests: `test_reset_then_replay_in_one_call` (ls20: `action(['RESET','UP','UP'])` executes 3),
  `test_reset_restarts_level_continues_batch_and_marks_wm_record` (rewritten from the fix-3
  test: 4 executed, 0 skipped, the board after the RESET equals the start, a second RESET right
  after the DOWN is accepted and the next one is a no-op).

## 2. wm fixes (bug 2, bug 3 / fix 6) - 13b4953

- wm.py:857: `untried_actions` compares `view_hash` (HUD blanked) instead of `grid_hash`.
- wm.py:426 `_grid_arg(x)`: None = current board, a Record, a `{'grid': ...}` state, anything
  with `.grid` (FrameView), or an array.
- wm.py:439 `click_candidates(grid=None)`.
- wm.py:779 `induce_motion(grid=None, record=None)`: the first positional argument may be a
  grid, a frame or a Record (old calls `induce_motion(rec)` still work); the grid only changes
  where avatar candidates are located.
- wm.py:986 `_flex_pred`: every `goal_menu` predicate is wrapped so `pred()`, `pred(grid)`,
  `pred(current_frame)` and `pred(state)` all work. `wm.plan(goal=pred)` passes a state, as
  before.
- Prompt, WORLD_MODEL last line (prompts.py:111):

  ```
  -- Free: `wm.induce_motion()`, `wm.untried_actions()`, `wm.transition_graph()`."""
  +- Free (no model): `wm.induce_motion()`, `wm.untried_actions()`, `wm.transition_graph()`, `wm.click_candidates()` (one MOUSE per object), `wm.goal_menu()` -> [(name, pred)]. Grid arguments are optional (default current board; a frame works)."""
  ```
  The WORLD_MODEL size cap in tests/test_prompts.py went from 1610 to 1770 chars (the section is
  1763).
- Not done (not in the brief): `induce_motion` still returns `{'actions': {}}` for a click
  game; the opener's unclicked-objects line covers that case instead.
- Tests: `test_untried_actions_ignores_the_hud_bar`,
  `test_click_candidates_goal_menu_and_induce_motion_accept_grid_frame_or_nothing`,
  `test_prompt_lists_the_free_helpers_with_their_real_signatures`.

## 3. Opener evidence lines (fix 2 + fix 4) - 644bce2, 25dfd4b

Data side, in wm.py (pure, from the Record the sandbox already builds):

- wm.py:884 `unclicked_objects(record, max_items=8)`: objects on the current board (non-
  background, outside the HUD band, biggest first) whose (colour, shape) was never the
  component under a MOUSE click on this level. Identity is colour+shape, not position, so a
  toggled copy counts as tried. A click on background counts as nothing clicked. Returns
  `{'objects': [{color, size, row, col}], 'total'}`.
- wm.py:905 `blocked_moves(record, max_grids=40)`: per direction, the number of recorded moves
  that moved nothing (`induce_motion.blocked_cases`) and the colours of the cells the avatar
  would have entered (the consensus (dy, dx) applied to the avatar's footprint; -1 = board
  edge), plus the colours successful moves did enter (their real offsets). The avatar is the
  piece of the avatar colour whose shape was seen moving (follow-up commit: picking the largest
  component of that colour gave "entered edge" on ls20). Colours are read from the last 40
  cases of each kind; counts cover the whole level.
- wm.py:941 `novelty(record, window=20)`: actions this level, distinct `view_hash` states seen
  (level start included), how many of the last `window` actions produced a never-seen state,
  and the most frequent no-op action with its count.
- wm.py:1071 `_moved_components` now flood-fills only from changed cells
  (`_components_touching`, wm.py:1131) instead of segmenting both full boards. Same result;
  `induce_motion` on a 300-move ls20 record went from 2.4 s to about 0.5 s. The whole
  `evidence()` call costs 0.6 s at 300 keyboard actions and under 60 ms on click games.
- sandbox.py:326 `PythonTool.evidence()`: the three parts, each in its own try/except (a part
  that fails is reported as `<key>_error` and left out of the opener).
- agent.py:205 `Agent._evidence()` calls `tool.evidence()` when the tool has one (FakeTool
  does not) and passes it to `prompts.build_opener(evidence=...)`. A failure is traced as
  `evidence_error` and the turn goes on.
- prompts.py:230 `evidence_lines(evidence, valid_actions)` renders, in this order, after the
  budget line and before "Current state":
  - `Objects never clicked on this level: G 8px at (30,27), D 16px at (26,44) (+9 more).`
    only when MOUSE is valid and the list is non-empty; at most 8 objects.
  - `Blocked moves: LEFT refused 4x (target cells colour D), UP refused 1x (target cells colour
    edge); successful moves entered B, S, R.` only when at least one move was refused.
  - `This level: 97 actions, 31 distinct states; last 20 actions reached 2 new states;
    MOUSE(58,46) was a no-op 15 times.` only when the level has actions. With fewer than 20
    actions the window is the actual count ("last 6 actions reached 5 new states").
- Prompt, STRATEGY (prompts.py:84):

  ```
  -- The game is deterministic: a sequence that failed once will fail again."""
  +- The game is deterministic: a sequence that failed once will fail again.
  +- Before 20 actions of trial and error on a level, click each unclicked object once. If the last 20 actions reached fewer than 5 new states, stop probing variants and test something not yet examined."""
  ```
- Tests: `test_unclicked_objects_use_colour_and_shape_identity`,
  `test_blocked_moves_report_target_and_entered_colours`,
  `test_novelty_counts_states_by_view_hash`, `test_evidence_lines_format_and_skip_when_empty`,
  `test_sandbox_evidence_and_agent_opener_on_fake_session` (sandbox FakeSession through
  PythonTool, then an Agent with a tool whose `evidence()` is faked, checking the line lands in
  the user message).

## 4. Cheaper cuts (fix 1) - 33b1b79

- agent.py:52 `AgentConfig.output_tokens = 6_000`.
- Prompt, TOOL last line (prompts.py:76):

  ```
  -- Call the tool as often as you like. When unsure, run code rather than reasoning longer: code answers in a second what thinking only guesses."""
  +- Call the tool as often as you like. Think in short bursts: when a hypothesis can be checked with code, call the tool instead of reasoning further."""
  ```
- Reasoning budget: llm.py sends only `chat_template_kwargs: {"enable_thinking": ...}`
  (llm.py:163) and kaggle_serve.py passes `--default-chat-template-kwargs
  {"preserve_thinking": true}` (kaggle_serve.py:235). No per-call thinking budget is passed
  anywhere, and none was added (cannot be tested locally).
- Test: `test_output_cap_and_short_burst_line` (also the < 500 chars prompt-growth guard).

## 5. Helper retention (fix 5 / bug 5) - 67ca439

- sandbox.py:677 `_freeze_deps`: for each fresh function, every module-scope name it reads
  (function body, nested scopes and default-value expressions, sandbox.py:725
  `_default_names`) that is neither provided, a builtin, an import nor another function is
  looked up in the snippet's globals and snapshotted into `entry['bound']`.
- sandbox.py:697 `_snapshot`: plain data only - None, bool, int, float, numpy scalars, str,
  bytes, dict, list, tuple, set, frozenset (recursively, tuples stay tuples) and numpy arrays
  (copied, made read-only). Size is approximated (array `nbytes`, string length, 8 per number,
  per-item overhead) and the total per function is capped at `SNAPSHOT_LIMIT` = 64 KB
  (sandbox.py:75).
- sandbox.py:608 `_restore` puts the bound values back into the call's globals under their
  names before re-executing the def. The function reads them as globals, so the snippet can
  read them too; rebinding such a name in a later snippet affects that call only, and a
  default such as `d=DELTA` is bound when the function is restored. Documented in the
  docstring and the prompt.
- Refusals are one line each: `uses BIG (over 64 KB of data with the other frozen values);
  keep it in memory[...] and read it from there`; `uses f (a function, which cannot be frozen);
  pass it as an argument or keep plain data in memory[...]`; `uses NOPE which is not available
  later; pass it as an argument or import it at top level` (unchanged). Defaults that read a
  provided name (`def f(g=current_frame)`) are refused with `default value reads current_frame,
  which changes every call; compute it inside the function` (sandbox.py:747); the old
  "defaults must be literals" check is gone.
- `source(name)` prints `# frozen at definition: K, WALLS` above the source (sandbox.py:395).
- Prompt, TOOL second line (prompts.py:72):

  ```
  -- Variables reset between calls, but top-level functions you define are kept and can be called later (no decorators/annotations; pass data as arguments); `source('name')` prints one.
  +- Variables reset between calls, but top-level functions you define are kept and can be called later (no decorators/annotations; top-level values they read are frozen copies, up to 64 KB); `source('name')` prints one.
  ```
- tests/test_sandbox.py `test_retained_functions` updated: `uses_k` (reads K) and `nonliteral`
  (default `len`) are now retained; the redefinition case uses `b=current_frame`.
- Tests: `test_retained_functions_freeze_snippet_level_data`,
  `test_retention_refuses_unfreezable_or_oversized_values_with_a_reason`.

## 6. Slot handover (bug 4) - b2896f8

- scheduler.py:181 `PriorityGate.should_yield(game, hysteresis=1.2)`: under the gate lock,
  True when `game` holds a slot and some other waiting game's priority exceeds `game`'s by more
  than the factor (an unstarted waiter has infinite priority and always wins). False when the
  game holds nothing or nobody outranks it.
- agent.py:371: at a trim the slot is released only if `_should_yield()` is true; the trim
  trace event has a `released` flag. A gate without `should_yield` (older fakes) keeps the old
  always-release rule.
- agent.py:302 `_rotate_slot()`, called at the start of every `play_turn` (agent.py:354):
  if the slot has been held for `AgentConfig.slot_rotate_seconds` (900 s, agent.py:72) and
  `should_yield` is true, release it and trace `slot_rotate`; the next `_call_llm` re-acquires
  through the normal priority queue. `_slot_since` is set on the first successful acquire and
  cleared on release (agent.py:287-293). `Agent(clock=...)` injects the clock for tests.
- Thread-safety: `should_yield` reads `_active` and `_waiting` under `self._cond`, like the
  other gate methods; `_slot_since` is per agent, that is per game thread.
- Tests: `test_should_yield_needs_a_waiter_above_the_hysteresis` (real gate, threads: a waiter
  within 1.2x does not trigger, one above does, an unstarted one does),
  `test_agent_releases_at_trim_only_when_outranked_and_rotates_after_15_minutes` (fake gate and
  clock: no release at a trim when not outranked, release and re-acquire when outranked, no
  rotation at 100 s, rotation at 1000 s only when outranked, hold timer restarts),
  `test_gate_without_should_yield_keeps_the_old_release_at_trim`.

## Deviations from the brief and open points

- Item 2: the WORLD_MODEL section grew by 163 chars; the fix-3 size cap test was raised to 1770.
- Item 3: nothing in the trace data contradicted the brief. The blocked-moves colours are
  read from the most recent 40 cases of each kind (a 300-move level otherwise cost 2.8 s per
  turn); the counts cover every case.
- Item 5: frozen values are visible to the snippet as globals (they are restored into the
  call's namespace, which is the function's globals). A truly private copy would need a
  separate globals dict per function and then `current_frame` etc. would go stale inside it.
- Item 6: `should_yield` compares against the published formula as it stands; the decay with
  `tokens_on_level` that the review notes is unchanged, so a game that thought hard still
  ranks low once it has released. The 15-minute rotation plus the hysteresis is the only
  change.
- docs/trace-review-run5.md is untracked in git; left for the lead.
