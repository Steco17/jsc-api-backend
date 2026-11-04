"""Privacy-aware append-only storage for human-reviewed translation pairs."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ContributionStorageFullError(RuntimeError):
    """Raised before untrusted writes can grow the contribution file indefinitely."""


class ContributionRepository:
    """Persist bounded JSONL rows with restrictive filesystem permissions."""

    def __init__(
        self,
        data_directory: Path,
        *,
        max_file_bytes: int,
        store_contributor_names: bool,
    ) -> None:
        self._data_directory = data_directory
        self._destination = data_directory / "contributed.jsonl"
        self._max_file_bytes = max_file_bytes
        self._store_contributor_names = store_contributor_names
        self._write_lock = threading.Lock()

    @property
    def destination(self) -> Path:
        return self._destination

    def save(self, values: Mapping[str, Any]) -> None:
        """Append one normalized row without retaining names unless explicitly enabled."""

        row = dict(values)
        if not self._store_contributor_names:
            row.pop("contributor", None)
        row["timestamp"] = datetime.now(UTC).isoformat()
        encoded_row = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )

        with self._write_lock:
            self._data_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            current_size = self._destination.stat().st_size if self._destination.exists() else 0
            if current_size + len(encoded_row) > self._max_file_bytes:
                raise ContributionStorageFullError("Contribution storage has reached its limit")

            descriptor = os.open(
                self._destination,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "ab", closefd=False) as handle:
                    handle.write(encoded_row)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)

            try:
                self._destination.chmod(0o600)
            except OSError:
                # Windows does not implement POSIX file modes. Container deployments do.
                pass
