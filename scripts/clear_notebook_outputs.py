#!/usr/bin/env python3
"""Remove stale execution output from a notebook before it is committed."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    args = parser.parse_args()

    path = Path(args.notebook)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            # Outputs from one machine are misleading on another machine and
            # can accidentally preserve stack traces or environment details.
            cell["execution_count"] = None
            cell["outputs"] = []

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
