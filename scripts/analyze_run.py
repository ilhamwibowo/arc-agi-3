#!/usr/bin/env python3
"""Summarise a run directory of per-game trace.jsonl / summary.json files.

Usage: analyze_run.py RUN_DIR [--html out.html] [--json out.json]
Everything is derived from the trace; summary.json (when present) only fills gaps, because Kaggle
output can be partial. Every field is optional.
"""
import argparse
import html
import json
import sys
from pathlib import Path


def read_trace(path: Path) -> list[dict]:
    events = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:      # a truncated last line is normal after a crash
                pass
    return events


def read_summary(path: Path) -> dict:
    try:
        s = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(s, dict):
        return {}
    res = s.get("result")           # runner.py nests the agent stats under "result"
    return {**s, **res} if isinstance(res, dict) else s


def analyze_game(name: str, events: list[dict], summary: dict) -> dict:
    g = {"game": name, "scored_actions": 0, "resets": 0, "turns": 0, "model_calls": 0,
         "tool_calls": 0, "tool_errors": 0, "model_errors": 0, "generated_tokens": 0,
         "prompt_tokens": 0, "cached_tokens": 0, "model_seconds": 0.0, "level_actions": [],
         "timeline": [], "tools": [], "model_stats": [], "stop_reasons": {}}
    level, win, end = 0, None, {}
    for e in events:
        ev, kind = e.get("event"), e.get("kind")
        if ev == "start":
            win = e.get("win_levels", win)
            level = e.get("levels", 0) or 0
        elif ev == "action":
            if e.get("automatic"):
                g["resets"] += 1
            else:
                g["scored_actions"] += 1
            idx = g["scored_actions"]
            new = e.get("levels", level) or 0
            died = e.get("state") == "GAME_OVER"
            if new > level:
                g["level_actions"].append(idx)
            if new != level or died or e.get("automatic"):
                g["timeline"].append({"step": e.get("step"), "action_index": idx, "level": new,
                                      "note": "level up" if new > level else
                                      "death" if died else "reset" if e.get("automatic") else "level change"})
            level = new
        elif kind == "model_call":
            u = e.get("usage") or {}
            g["model_calls"] += 1
            g["model_seconds"] += e.get("seconds") or 0
            g["model_stats"].append({"seconds": e.get("seconds"), "finish": e.get("finish_reason"),
                                     "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                                     "cached": u.get("cached_tokens"), "reasoning_chars": e.get("reasoning_chars"),
                                     "content": e.get("content")})
        elif kind == "model_error":
            g["model_errors"] += 1
        elif kind == "tool_call":
            g["tool_calls"] += 1
            g["tool_errors"] += bool(e.get("error"))
            for r in e.get("stops") or []:
                g["stop_reasons"][r] = g["stop_reasons"].get(r, 0) + 1
            g["tools"].append({k: e.get(k) for k in ("code", "stdout", "error", "actions", "stops", "seconds")})
        elif kind == "turn":
            g["turns"] += 1
            g["generated_tokens"] += e.get("generated_tokens") or 0
        elif kind == "game_end":
            end = e
    # Prompt/cached tokens: the per-call usage is authoritative; fall back to totals.
    g["prompt_tokens"] = sum(m["prompt"] or 0 for m in g["model_stats"]) or end.get("prompt_tokens") \
        or summary.get("prompt_tokens") or 0
    g["cached_tokens"] = sum(m["cached"] or 0 for m in g["model_stats"]) or end.get("cached_tokens") \
        or summary.get("cached_tokens") or 0
    for k in ("turns", "model_calls", "tool_calls", "generated_tokens"):
        g[k] = g[k] or end.get(k) or summary.get(k) or 0
    if not g["scored_actions"]:
        g["scored_actions"] = end.get("actions") or summary.get("actions") or 0
    g["resets"] = g["resets"] or end.get("resets") or summary.get("resets") or 0
    g["levels_completed"] = max(level, end.get("levels_completed") or 0,
                                summary.get("levels_completed") or 0)
    g["win_levels"] = win or end.get("win_levels") or summary.get("win_levels") or summary.get("levels_total")
    g["cached_fraction"] = g["cached_tokens"] / g["prompt_tokens"] if g["prompt_tokens"] else None
    g["mean_model_seconds"] = g["model_seconds"] / len(g["model_stats"]) if g["model_stats"] else None
    prev, per = 0, []
    for i in g["level_actions"]:
        per.append(i - prev)
        prev = i
    g["actions_per_level"] = per
    g["end_reason"] = end.get("end_reason") or summary.get("end_reason") or summary.get("error") \
        or summary.get("status") or ("no game_end (crashed or partial)" if events else "no trace")
    g["seconds"] = end.get("seconds") or summary.get("seconds")
    return g


def analyze_run(run: Path) -> dict:
    games = []
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        if (d / "trace.jsonl").exists() or (d / "summary.json").exists():
            games.append(analyze_game(d.name, read_trace(d / "trace.jsonl"), read_summary(d / "summary.json")))
    sums = ("levels_completed", "win_levels", "scored_actions", "resets", "turns", "model_calls",
            "tool_calls", "tool_errors", "generated_tokens", "prompt_tokens", "cached_tokens")
    total = {k: sum(g[k] or 0 for g in games) for k in sums}
    total["cached_fraction"] = total["cached_tokens"] / total["prompt_tokens"] if total["prompt_tokens"] else None
    secs = sum(g["model_seconds"] for g in games)
    n = sum(len(g["model_stats"]) for g in games)
    total["mean_model_seconds"] = secs / n if n else None
    return {"run": str(run), "games": games, "total": total}


COLS = [("game", "game"), ("levels", None), ("acts", "scored_actions"), ("resets", "resets"),
        ("turns", "turns"), ("mcalls", "model_calls"), ("tools", "tool_calls"), ("terr", "tool_errors"),
        ("gen_tok", "generated_tokens"), ("prompt_tok", "prompt_tokens"), ("cached", None),
        ("s/call", None), ("acts/level @idx", None), ("end", "end_reason")]


def fmt_row(g: dict, total: bool = False) -> list[str]:
    lv = f"{g['levels_completed']}/{g['win_levels'] if g['win_levels'] is not None else '?'}"
    cf = "-" if g["cached_fraction"] is None else f"{g['cached_fraction']:.0%}"
    ms = "-" if g["mean_model_seconds"] is None else f"{g['mean_model_seconds']:.1f}"
    if total:
        apl = ""
    else:
        apl = " ".join(f"{a}@{i}" for a, i in zip(g["actions_per_level"], g["level_actions"])) or "-"
    row = [g.get("game", "TOTAL"), lv]
    row += [str(g[k]) for k in ("scored_actions", "resets", "turns", "model_calls", "tool_calls",
                                "tool_errors", "generated_tokens", "prompt_tokens")]
    return row + [cf, ms, apl, str(g.get("end_reason", ""))]


def text_table(report: dict) -> str:
    rows = [[h for h, _ in COLS]] + [fmt_row(g) for g in report["games"]]
    rows.append(fmt_row({**report["total"], "game": "TOTAL"}, total=True))
    w = [max(len(r[i]) for r in rows) for i in range(len(COLS))]
    lines = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    lines.insert(1, "  ".join("-" * x for x in w))
    lines.insert(-1, lines[1])
    return "\n".join(lines)


def html_table(report: dict) -> str:
    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>"
    rows = [tr([h for h, _ in COLS], "th")] + [tr(fmt_row(g)) for g in report["games"]]
    rows.append(tr(fmt_row({**report["total"], "game": "TOTAL"}, total=True)))
    return "<table>" + "".join(rows) + "</table>"


def html_game(g: dict) -> str:
    e = html.escape
    out = [f"<h2>{e(g['game'])} &mdash; {g['levels_completed']}/{g['win_levels']} levels, "
           f"{e(str(g['end_reason']))}</h2>"]
    out.append("<h3>Timeline</h3><ul>" + "".join(
        f"<li>step {t['step']} (action #{t['action_index']}): {t['note']}, level {t['level']}</li>"
        for t in g["timeline"]) + "</ul>" if g["timeline"] else "<p>No level changes, deaths or resets.</p>")
    if g["stop_reasons"]:
        out.append("<p>Stop reasons: " + e(", ".join(f"{k} x{v}" for k, v in g["stop_reasons"].items())) + "</p>")
    out.append("<details><summary>Model calls (%d)</summary><table>" % len(g["model_stats"]))
    out.append("<tr><th>#</th><th>sec</th><th>finish</th><th>prompt</th><th>cached</th><th>completion</th>"
               "<th>reasoning chars</th><th>content</th></tr>")
    for i, m in enumerate(g["model_stats"], 1):
        out.append("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in (
            i, m["seconds"], m["finish"], m["prompt"], m["cached"], m["completion"],
            m["reasoning_chars"], (m["content"] or "")[:200])) + "</tr>")
    out.append("</table></details>")
    out.append("<details><summary>Tool calls (%d, %d errors)</summary>" % (g["tool_calls"], g["tool_errors"]))
    for i, t in enumerate(g["tools"], 1):
        bad = " class=bad" if t["error"] else ""
        out.append(f"<details{bad}><summary>#{i}: {len(t['actions'] or [])} actions"
                   f"{', ERROR' if t['error'] else ''}</summary><pre>{e(t['code'] or '(no code)')}</pre>")
        if t["stdout"]:
            out.append(f"<p>stdout</p><pre>{e(t['stdout'][:1500])}</pre>")
        if t["error"]:
            out.append(f"<p>error</p><pre>{e(str(t['error'])[:1500])}</pre>")
        out.append(f"<p>actions: {e(' '.join(t['actions'] or []) or '-')}; "
                   f"stops: {e(', '.join(map(str, t['stops'] or [])) or '-')}</p></details>")
    out.append("</details>")
    return "\n".join(out)


def html_report(report: dict) -> str:
    css = ("body{font:14px sans-serif;margin:2em;max-width:1200px}table{border-collapse:collapse}"
           "td,th{border:1px solid #bbb;padding:2px 6px;text-align:left}pre{background:#f4f4f4;padding:6px;"
           "overflow:auto;max-height:300px}details{margin:4px 0 4px 1em}.bad>summary{color:#b00}")
    return (f"<!doctype html><meta charset=utf-8><title>Run report</title><style>{css}</style>"
            f"<h1>{html.escape(report['run'])}</h1>{html_table(report)}"
            + "\n".join(html_game(g) for g in report["games"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--html", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if not a.run_dir.is_dir():
        print(f"not a directory: {a.run_dir}", file=sys.stderr)
        return 1
    report = analyze_run(a.run_dir)
    print(text_table(report))
    if a.html:
        a.html.write_text(html_report(report))
    if a.json:
        a.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
