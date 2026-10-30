"""Notebook builder: valid JSON, parseable cells, embedded source round-trips, metadata refs."""
import ast
import json
import sys
from pathlib import Path

from arc3 import notebook

SRC = Path(notebook.__file__).resolve().parent


def test_build_writes_valid_notebook_and_metadata(tmp_path):
    out = notebook.build("nbtest", out_root=tmp_path, offline_subset=["ls20"])
    nb = json.loads((out / "nbtest.ipynb").read_text())
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    assert meta["code_file"] == "nbtest.ipynb" and meta["is_private"] is True
    assert meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaRtxPro6000"
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert meta["dataset_sources"] == ["dfranzen/pennyroyal-v253"]
    assert len(meta["model_sources"]) == 2
    setup = nb["cells"][1]["source"]
    assert "OFFLINE_SUBSET = ['ls20']" in setup and "KAGGLE_IS_COMPETITION_RERUN" in setup


def test_embedded_source_cell_unpacks_identical_tree(tmp_path):
    nb = notebook.build_notebook()
    cell = next(c["source"] for c in nb["cells"] if "ARC3_TAR_B64" in c["source"])
    dest = tmp_path / "work"
    dest.mkdir()
    saved_path = list(sys.path)
    try:
        exec(compile(cell, "source-cell", "exec"), {"ARC3_ROOT": str(dest)})
    finally:
        sys.path[:] = saved_path
    expected = {p.relative_to(SRC) for p in SRC.rglob("*.py") if "__pycache__" not in p.parts}
    got = {p.relative_to(dest / "arc3") for p in (dest / "arc3").rglob("*.py")}
    assert expected == got
    for rel in expected:
        assert (dest / "arc3" / rel).read_bytes() == (SRC / rel).read_bytes()


def test_tar_is_deterministic():
    assert notebook.source_tar_b64() == notebook.source_tar_b64()


def test_cells_in_order_and_server_always_stopped():
    sources = [c["source"] for c in notebook.build_notebook()["cells"] if c["cell_type"] == "code"]
    headers = [s.splitlines()[0] for s in sources]
    assert [h.split(".")[0] for h in headers] == [f"# {i}" for i in range(9)]
    assert "stop_server" in sources[-1]
    assert "--no-index" in sources[2]


def test_smoke_no_server_flag_never_applies_in_competition(tmp_path):
    """The no-server save-run shortcut must be ANDed with `not TRUE_SUBMISSION` in the config cell,
    guard both the server launch and the play call, and default to off."""
    from arc3.notebook import build_notebook
    nb = build_notebook(offline_subset=1, per_game_cap_seconds=180, smoke_no_server=True)
    src = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                    for c in nb["cells"])
    assert "SMOKE_NO_SERVER = True and not TRUE_SUBMISSION" in src
    assert "if SMOKE_NO_SERVER:\n    print(\"SMOKE_NO_SERVER: model server not started" in src
    assert "if SMOKE_NO_SERVER:\n        raise SystemExit" in src
    nb2 = build_notebook(offline_subset=1, per_game_cap_seconds=180)
    src2 = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                     for c in nb2["cells"])
    assert "SMOKE_NO_SERVER = False and not TRUE_SUBMISSION" in src2
