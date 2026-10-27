# Fix review 3: items from trace-review-run3-weak

All tests pass: `.venv/bin/python -m pytest -q tests/` gives 126 passed. Nothing is committed.
Not touched: agent.py, runner.py, kaggle_main.py, notebook.py. New tests are in
tests/test_fix_review3.py; tests/test_prompts.py was rewritten for the new example.

## 1. RESET exposed

- `GameConfig.expose_reset` now defaults to True (types.py:114). `build_system_prompt` defaults
  to True as well (prompts.py), so its default matches the config. agent.py already passes the
  config value.
- The engine never lists RESET in `available_actions`, so `GameSession.valid_actions` appends it
  when exposed and the game is not won (session.py:57).
- **Guard, beyond the spec.** A RESET with no non-RESET action since the level started or since
  the last RESET is refused with `reset_noop` and costs nothing (session.py:76). It covers the
  "second RESET" case. It also covers a RESET at a fresh level 1, which Env.safe_reset does not
  refuse and which would otherwise spend an action on a full reset. A RESET at GAME_OVER and the
  automatic RESET are not affected. The counter is `since_restart` (session.py:37, 105).
- A RESET ends the batch: `stop_reason='reset'`, and the terminal latch is set (sandbox.py:459).
  The model has to look at the start board.
- **wm record.** RESET transitions are not dynamics. Each one now adds a restart boundary,
  `Record.restarts` (wm.py:50, sandbox.py:375).
  - The moves before a RESET stay in the same level's record.
  - `verify` re-inits at a boundary even when the boards happen to match.
  - `current_state` replays only the segment after the last RESET (wm.py:393), so the planner
    starts from the post-RESET board.
  - The automatic RESET after a game over adds a boundary too.
- **Prompt.** The ACTIONS RESET text is the requested rule (prompts.py:53). It adds "Use it when
  the position is lost or the move budget left cannot cover your plan."
  - `describe_result` shows "level restarted from its start board".
  - The session message reads "RESET: the level restarted from its start board; completed
    levels are kept".
- **Not verified here:** "completed levels are kept" on a real level-up. I relied on your local
  check and Env.safe_reset. The test covers a mid-level RESET on ls20, the board restore and the
  no-op refusal.

## 2. Budget-bar readout

- `percept.band_runs` (percept.py:560) returns the longest run per (band, color) over the 4
  border bands. Rows are scanned for the top and bottom bands, columns for the side bands.
- `percept.BudgetTracker` (percept.py:580) is fed by `GameSession._track_budget`
  (session.py:129) after every executed action. It resets at game start, level completion,
  level_reset, game over, RESET and UNDO. UNDO is included because it may hand cells back.
- **Deviation from the spec, deliberate.** wa30 and g50t, the two weakest games, lose one bar
  cell every 2-4 actions, not every action. A strict "k per action over the last 3 actions" rule
  never fires on them. The rule implemented:
  - since the last reset, the run never grew;
  - every change was the same shrink k;
  - there were at least 3 shrinks.
  - The estimate is `left = run / k * (actions / shrinks)`. The description reads either
    "shrinks 1/action" or "shrinks 1 per ~3.0 actions".
  - ls20 is exact: 39 left after 3 moves, counting down to 0 at move 42, then the 43rd resets.
- **Coverage over 40 random moves on all 25 public games.**
  - Silent: ft09, lp85, s5i5, sb26, sk48, tu93, vc33.
  - Fires on the rest, all on a band run that shrank monotonically for 40 moves with no growth.
  - Fractional rates make early estimates noisy, for example 81 then 101 on cd82 between
    consecutive moves.
  - If a game refunds blocked moves, the averaged rate overestimates. That is the safe direction.
- `ActionResult` gained `budget_left`, `budget_bar` and `skipped` at the end (types.py:91-93).
- **Where the model sees it.**
  - The opener gets `prompts.budget_line` (prompts.py:213) before "Current state", from the last
    executed result: "Move budget: ~N actions left on this level (bar on the bottom edge: ...).
    Plan length must fit; if it cannot, RESET first."
  - The `[actions]` line ends with "| move budget ~N left".
  - `per_action[i]['budget_left']` is in the call summary.
- **Gap.** A turn with no actions passes empty `last_results`, so that turn's opener has no budget
  line. Passing the last known value would need an agent.py change.

## 3. wm.check / wm.plan_abstract

- `wm.check(read, step, all_levels=False)` (wm.py:463) replays the recorded transitions:
  `step(read(before), action)` against `read(after)`. Each transition is checked on its own,
  not chained, and it costs no game action.
  - Non-ongoing moves are skipped, because their `after` is not a normal board.
  - It returns `ok`, `accuracy`, `checked/correct/skipped`, `by_action` ('c/n'),
    `first_mismatch` (level, index, action, before, predicted, actual, each repr ≤160 chars, or
    `error`), up to 3 `mismatches`, `unseen_actions` and a one-line `summary`.
  - Model exceptions become mismatches. Equality handles numpy inside states (`same_state`,
    wm.py:880).
- `wm.plan_abstract(read, step, goal, actions=None, max_nodes=20000)` (wm.py:522) is a small BFS.
  I did not reuse `plan()`, because that one is tied to the registered grid model.
  - States may be unhashable: tuples, dicts, lists or arrays are keyed canonically.
  - `step` gets a deep copy, so in-place mutation is safe.
  - Actions default to the valid keyboard actions, plus one click per object when MOUSE is
    available.
  - It returns `found`, `actions`, `plan`, `nodes`, `reason`.
- **The action form passed to `step`.** It is 'UP' or the tuple ('MOUSE', row, col).
  `sandbox.parse_action` now accepts that tuple (sandbox.py:190), so `action(p['actions'])`
  works directly.
- **Added `wm.level_start`** (wm.py:458), the level's first board. My first example built the key
  map from the current board. The test caught that this misses a key that was already picked up.
  Static maps have to come from the level start, and the example now does that.
- **WORLD_MODEL rewritten** (prompts.py:100). check and plan_abstract lead in 4 lines, followed
  by a 13-line example (block position plus key state; step moves unless a wall blocks). The
  full simulator is 2 lines and the free helpers 1.
  - The section is 1600 chars. That is about 457 tokens by the repo's 3.5 chars/token estimate.
    The code-heavy text should tokenize below that.
  - The example is longer than the requested 6 lines. A key pickup needs the static maps line
    and the plan line.
  - The system prompt is about 2490 estimated tokens, under the 2800 test cap.
- **Tests.**
  - tests/test_prompts.py executes the example against an independent ground truth. It expects
    6/6 predicted, and the plan must end on the door holding the key.
  - It also checks mismatch reporting, unseen actions, error capture, unhashable states, the
    in-place mutating step and the no-path reason.
  - tests/test_fix_review3.py runs check and plan_abstract on ls20 through the sandbox. A
    wall-free model mispredicts exactly the blocked LEFT. The planned DOWN DOWN is executed for
    real and reaches the goal.

## 4. Trims

- `ContextManager.trim` (context.py:180) first calls `strip_old_reasoning` (context.py:165). It
  removes reasoning from assistant messages before the last `keep_reasoning_turns=6` blocks.
  Blocks are dropped only if the total is still above budget - drain, as before.
- **Return value change.** `trim` returns stripped plus dropped message counts, so any non-zero
  value still means "the prefix changed" for agent.py's slot release and trace. `last_trim` has
  the split. tests/test_agent.py now reads `last_trim['dropped']`.
- `trimmed_since_last_user` (context.py:80) is set only when blocks were dropped. agent.py
  clears it after adding the notice. `add_user` does not, so a nudge cannot swallow it.
- `prompts.trim_notice(memory_summary, retained)` returns "Trim notice: older turns were dropped.
  Your memory dict still holds: ... Retained functions: ...", followed by a newline.
- `PythonTool.memory_summary(limit=400)` (sandbox.py:276) lists `'key': repr` pairs. Each repr is
  capped at 60 chars and the whole string at the limit.
- **For your agent.py wiring.** `build_opener` returns a list of parts when images are attached.
  In that case prepend the notice as a `text_part` instead of using string `+`.

## 5. Stop reasons

- When a batch stops, `stop_reason` and `skipped` are set on the result that caused it
  (sandbox.py:477). That is the last executed one, or the refused one.
- **Mutated in place, not copied.** The same object is `history[i].result`, so the opener, the
  `[actions]` line, the trace `stops` field and `history[i].stop_reason` all agree. Nothing else
  holds these objects in a way that conflicts.
- The `[actions]` line ends with "| stopped: no_op, 2 requested action(s) not run".
- `describe_result` and the opener's "Guard stop" line skip stop reasons that only restate the
  effect: game_over, level_completed, level_reset, run_complete and reset.
- Tested for no_op on ls20, max_actions_per_call and reset.

## 6. Small items

- `collections`, `itertools` and `math` are bound in every namespace and listed in PROVIDED, so
  retained functions may use them (sandbox.py:55). The TOOL prompt line says so.
- `source(name)` prints a retained function with its imports, or lists what is retained
  (sandbox.py:334).
- `TransitionView` has `.step` (engine step after the action), `.level_reset` and a live
  `.stop_reason`. `.result` has `level_reset`. The RUNTIME prompt line lists them.
- **Neutral LEVEL RESET wording.**
  - Session message: "LEVEL RESET: the board returned to this level's start position (a fatal
    move, an exhausted move budget, or a deliberate restart mechanic)".
  - The opener's LEVEL_RESET paragraph (prompts.py:150) says to check `history[-1]` and to carry
    on when the restart was a game mechanic.
  - The SIGNALS line and `describe_result` drop "life probably lost".

## Skipped

- Scroll-aware diffs, as instructed.
