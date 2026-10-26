# arc3 — working conventions

Goal: an independent ARC-AGI-3 agent for the Kaggle competition (deadline Nov 2 2026 23:59 UTC).
Read PLAN.md first. Reference material is under reference/ (read-only; never copy harness code from
reference/public_notebooks or the Duck tree — re-implement from the briefs).

- Python 3.12 only; `.venv/bin/python`; run tests with `.venv/bin/python -m pytest -q tests/`.
- Dependencies must be installable OFFLINE on Kaggle from the competition wheelhouse
  (reference/kaggle-runtime-lock.json: arc-agi 0.9.8, arcengine 0.9.3, numpy, pillow, requests, ...).
  Do not add dependencies outside that list without flagging it.
- Shared contracts live in src/arc3/types.py. Change them only by appending; tell the lead.
- The model NEVER sees: game id, environment source, hidden state, human baselines, scorecards.
  Agents receive only Observations and their own history.
- Every module ships with pytest tests that run on CPU with no model, using the 25 public games in
  data/environments through arc3.env (offline mode).
- Logging: each game run writes runs/<run>/<game>/trace.jsonl (every action, frames hashed,
  tool calls, model calls) so failures are diagnosable without a GPU.
- Keep code plain and readable: module docstring, small functions, no cleverness. Comments explain
  why, not what.
- Nothing here touches ../arc-harness or ../decker.
