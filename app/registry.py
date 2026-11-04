"""Language registry loading, validation, and capability views."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

ACTIVE_STATUSES = {"pretrained", "fine_tuned"}
VALID_STATUSES = {"planned", "data_ready", "pretrained", "fine_tuned"}
LANGUAGE_CODE_PATTERN = re.compile(r"^[a-z0-9]{2,8}_[A-Z][a-z]{3}$")
MAX_REGISTRY_BYTES = 1_048_576


class RegistryError(ValueError):
    """Raised when the language registry is malformed or inconsistent."""


@dataclass(frozen=True)
class LanguageRegistry:
    """Validated immutable language metadata and derived capability views."""

    entries: Mapping[str, Mapping[str, Any]]
    active_languages: Mapping[str, str]
    contribution_languages: Mapping[str, str]
    pretrained_languages: frozenset[str]
    fine_tuned_languages: frozenset[str]

    def base_directions(self) -> set[tuple[str, str]]:
        """Return ordered pairs supported by the untouched multilingual model."""

        return {
            (source, target)
            for source in self.pretrained_languages
            for target in self.pretrained_languages
            if source != target
        }

    def validate_pair(self, source: str, target: str, *, contribution: bool) -> None:
        """Require two distinct codes from the selected public registry view."""

        allowed = self.contribution_languages if contribution else self.active_languages
        if source not in allowed or target not in allowed:
            raise RegistryError("Unknown or unavailable language code")
        if source == target:
            raise RegistryError("Source and target languages must differ")

    def catalog(self) -> list[dict[str, Any]]:
        """Return the public metadata contract without internal registry notes."""

        return [
            {
                "code": code,
                "iso_code": code.split("_", maxsplit=1)[0],
                "name": info["name"],
                "script": code.split("_", maxsplit=1)[1],
                "status": info["status"],
                "featured": info.get("featured", False),
            }
            for code, info in self.entries.items()
        ]


def _validated_entry(code: str, raw_info: object) -> Mapping[str, Any]:
    if not LANGUAGE_CODE_PATTERN.fullmatch(code):
        raise RegistryError(f"Invalid language code in registry: {code!r}")
    if not isinstance(raw_info, dict):
        raise RegistryError(f"Registry entry for {code} must be an object")

    name = raw_info.get("name")
    status = raw_info.get("status")
    featured = raw_info.get("featured", False)
    note = raw_info.get("note")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise RegistryError(f"Registry entry for {code} has an invalid name")
    if status not in VALID_STATUSES:
        raise RegistryError(f"Registry entry for {code} has an invalid status")
    if not isinstance(featured, bool):
        raise RegistryError(f"Registry entry for {code} has an invalid featured flag")
    if note is not None and (not isinstance(note, str) or len(note) > 1000):
        raise RegistryError(f"Registry entry for {code} has an invalid note")

    return MappingProxyType(
        {
            "name": name.strip(),
            "status": status,
            "featured": featured,
            **({"note": note.strip()} if isinstance(note, str) else {}),
        }
    )


def load_language_registry(path: Path) -> LanguageRegistry:
    """Load a bounded JSON file and return validated immutable registry views."""

    if not path.is_file():
        raise RegistryError(f"Language registry was not found: {path}")
    if path.stat().st_size > MAX_REGISTRY_BYTES:
        raise RegistryError("Language registry is unexpectedly large")
    try:
        raw_registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryError("Language registry could not be read") from error
    if not isinstance(raw_registry, dict) or not raw_registry:
        raise RegistryError("Language registry must be a non-empty object")

    entries = {code: _validated_entry(code, info) for code, info in raw_registry.items()}
    contribution_languages = {code: info["name"] for code, info in entries.items()}
    active_languages = {
        code: info["name"] for code, info in entries.items() if info["status"] in ACTIVE_STATUSES
    }
    pretrained = frozenset(
        code for code, info in entries.items() if info["status"] == "pretrained"
    )
    fine_tuned = frozenset(
        code for code, info in entries.items() if info["status"] == "fine_tuned"
    )
    return LanguageRegistry(
        entries=MappingProxyType(entries),
        active_languages=MappingProxyType(active_languages),
        contribution_languages=MappingProxyType(contribution_languages),
        pretrained_languages=pretrained,
        fine_tuned_languages=fine_tuned,
    )
