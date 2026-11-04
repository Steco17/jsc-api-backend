"""Thread-safe model lifecycle and translation orchestration."""

from __future__ import annotations

import json
import threading
from typing import Any

from app.config import Settings
from app.registry import LanguageRegistry

MAX_MANIFEST_BYTES = 1_048_576
MAX_MANIFEST_DIRECTIONS = 10_000


class ModelNotReadyError(RuntimeError):
    """Raised when model-backed translation is requested before startup completes."""


class UnsupportedDirectionError(ValueError):
    """Raised when the deployed model does not prove support for a requested pair."""


class TranslationService:
    """Own model state and prevent mutable tokenizer races between requests."""

    def __init__(self, settings: Settings, registry: LanguageRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._translator: Any | None = None
        self._tokenizer: Any | None = None
        self._supported_directions = registry.base_directions()
        self._tokenizer_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    @property
    def supported_directions(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._supported_directions)

    def _load_training_manifest(self) -> dict[str, Any] | None:
        path = self.settings.tokenizer_dir / "training_manifest.json"
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise RuntimeError("Training manifest is unexpectedly large")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Training manifest could not be read") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise RuntimeError("Unsupported training manifest schema")
        directions = manifest.get("trained_directions")
        if not isinstance(directions, list):
            raise RuntimeError("Training manifest has no trained_directions")
        if len(directions) > MAX_MANIFEST_DIRECTIONS:
            raise RuntimeError("Training manifest contains too many directions")
        return manifest

    def _validated_manifest_directions(self, manifest: dict[str, Any]) -> set[tuple[str, str]]:
        directions: set[tuple[str, str]] = set()
        for entry in manifest["trained_directions"]:
            if not isinstance(entry, list) or len(entry) != 2:
                raise RuntimeError("Training manifest contains an invalid direction")
            source, target = entry
            if not isinstance(source, str) or not isinstance(target, str):
                raise RuntimeError("Training manifest direction codes must be strings")
            unknown = {source, target} - self.registry.entries.keys()
            if unknown:
                raise RuntimeError("Training manifest references unknown languages")
            if source != target:
                directions.add((source, target))
        return directions

    def load_model(self) -> None:
        """Load local model assets and verify registry, tokenizer, and manifest agreement."""

        if not self.settings.model_dir.is_dir():
            raise RuntimeError("CTranslate2 model directory was not found")
        if not self.settings.tokenizer_dir.is_dir():
            raise RuntimeError("Tokenizer directory was not found")

        import ctranslate2
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
            fix_mistral_regex=False,
        )
        translator = ctranslate2.Translator(
            str(self.settings.model_dir),
            device="cpu",
            compute_type="int8",
        )
        manifest = self._load_training_manifest()

        if self.registry.fine_tuned_languages and manifest is None:
            raise RuntimeError(
                "Fine-tuned registry entries require a training manifest"
            )

        directions = self.registry.base_directions()
        if manifest:
            directions |= self._validated_manifest_directions(manifest)
            manifest_languages = set(manifest.get("languages", []))
            missing = self.registry.fine_tuned_languages - manifest_languages
            if missing:
                raise RuntimeError("Fine-tuned languages are missing from the model manifest")

        for code in self.registry.active_languages:
            if tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id:
                raise RuntimeError("Tokenizer is missing an active language token")

        self._tokenizer = tokenizer
        self._translator = translator
        self._supported_directions = {
            pair
            for pair in directions
            if pair[0] in self.registry.active_languages
            and pair[1] in self.registry.active_languages
        }

    def unload_model(self) -> None:
        self._translator = None
        self._tokenizer = None

    def ensure_ready(self) -> None:
        if not self.model_loaded:
            raise ModelNotReadyError("Translation model is not loaded")

    def translate(self, text: str, source: str, target: str) -> tuple[str, str]:
        """Return translated text and a stable status for the configured mode."""

        if self.settings.translation_mode == "placeholder":
            self.registry.validate_pair(source, target, contribution=True)
            return "", "model_pending"

        self.ensure_ready()
        self.registry.validate_pair(source, target, contribution=False)
        if (source, target) not in self._supported_directions:
            raise UnsupportedDirectionError("Translation direction is not enabled")

        with self._tokenizer_lock:
            self._tokenizer.src_lang = source
            tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))

        results = self._translator.translate_batch(
            [tokens],
            target_prefix=[[target]],
            beam_size=4,
            max_decoding_length=512,
        )
        output_tokens = results[0].hypotheses[0]
        if output_tokens and output_tokens[0] == target:
            output_tokens = output_tokens[1:]

        with self._tokenizer_lock:
            translated_text = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(output_tokens),
                skip_special_tokens=True,
            )
        return translated_text, "translated"

    def validate_contribution_pair(self, source: str, target: str) -> None:
        self.registry.validate_pair(source, target, contribution=True)
