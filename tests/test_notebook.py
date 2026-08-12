"""Regression checks for the end-user Colab training workflow."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "train_colab.ipynb"


def _load_notebook() -> dict:
    """Load the committed notebook as JSON without requiring Jupyter."""
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_colab_conflicts_are_removed_before_training_dependencies() -> None:
    """Keep Colab's unused Gradio and torchao packages out of dependency resolution."""
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    cleanup = "!pip uninstall -qy torchao gradio gradio-client"
    install = "!pip install -q -r requirements-train.txt"

    assert cleanup in source
    assert install in source
    assert source.index(cleanup) < source.index(install)


def test_notebook_has_no_saved_runtime_state() -> None:
    """Prevent credentials, logs, and stale failures from leaking through outputs."""
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
