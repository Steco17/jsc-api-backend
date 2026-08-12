"""FastAPI service for the Cameroon languages translation model.

The converted CTranslate2 model is shared by all requests.  Translation
directions are loaded from the model's ``training_manifest.json`` instead of
being inferred from a language status flag.  This prevents an English-only
fine-tune from accidentally advertising unsupported French or local-language
pairs.
"""

import json
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model_ct2"))
TOKENIZER_DIR = Path(os.getenv("TOKENIZER_DIR", "model_out/merged"))
API_KEY = os.getenv("API_KEY")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

LANGUAGES_FILE = Path(__file__).resolve().parent.parent / "data" / "languages.json"
with LANGUAGES_FILE.open(encoding="utf-8") as registry_handle:
    REGISTRY = json.load(registry_handle)

# Translation dropdowns contain only languages already represented by the
# deployed model.  Contribution forms use the complete registry below because
# collecting data is most important before a language becomes deployable.
ACTIVE_STATUSES = {"pretrained", "fine_tuned"}
LANGUAGES = {
    code: info["name"] for code, info in REGISTRY.items() if info["status"] in ACTIVE_STATUSES
}
CONTRIBUTION_LANGUAGES = {code: info["name"] for code, info in REGISTRY.items()}
PRETRAINED_LANGUAGES = {code for code, info in REGISTRY.items() if info["status"] == "pretrained"}
FINE_TUNED_LANGUAGES = {code for code, info in REGISTRY.items() if info["status"] == "fine_tuned"}


def base_directions() -> set[tuple[str, str]]:
    """Return all ordered pairs native to the untouched NLLB base model."""

    return {
        (source, target)
        for source in PRETRAINED_LANGUAGES
        for target in PRETRAINED_LANGUAGES
        if source != target
    }


_translator = None
_tokenizer = None
_supported_directions = base_directions()

# NLLB stores src_lang as mutable tokenizer state.  FastAPI executes normal
# functions in a thread pool, so two requests can otherwise encode with each
# other's source language.  Translation itself remains outside this lock.
_tokenizer_lock = threading.Lock()
_write_lock = threading.Lock()


def _load_training_manifest(tokenizer_dir: Path) -> dict | None:
    """Load and structurally validate a model's training manifest."""

    path = tokenizer_dir / "training_manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported training manifest schema in {path}")
    if not isinstance(manifest.get("trained_directions"), list):
        raise RuntimeError(f"training manifest has no trained_directions: {path}")
    return manifest


def _validated_manifest_directions(manifest: dict) -> set[tuple[str, str]]:
    """Convert manifest pairs to tuples and reject unknown registry codes."""

    directions: set[tuple[str, str]] = set()
    for entry in manifest["trained_directions"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise RuntimeError(f"Invalid trained direction in manifest: {entry!r}")
        source, target = entry
        unknown = {source, target} - REGISTRY.keys()
        if unknown:
            raise RuntimeError(f"Training manifest references unknown languages: {sorted(unknown)}")
        if source != target:
            directions.add((source, target))
    return directions


def load_model() -> None:
    """Load model assets and verify registry, tokenizer, and manifest agreement."""

    global _translator, _tokenizer, _supported_directions

    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"CTranslate2 model directory not found: {MODEL_DIR}. "
            "Convert or mount the trained model before starting the API."
        )
    if not TOKENIZER_DIR.is_dir():
        raise RuntimeError(
            f"Tokenizer directory not found: {TOKENIZER_DIR}. "
            "Mount model_out/merged before starting the API."
        )

    import ctranslate2
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    translator = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8")
    manifest = _load_training_manifest(TOKENIZER_DIR)

    if FINE_TUNED_LANGUAGES and manifest is None:
        raise RuntimeError(
            "The language registry contains fine_tuned entries, but the model "
            "has no training_manifest.json. Refusing to guess its capabilities."
        )

    directions = base_directions()
    if manifest:
        directions |= _validated_manifest_directions(manifest)

        manifest_languages = set(manifest.get("languages", []))
        missing_from_manifest = FINE_TUNED_LANGUAGES - manifest_languages
        if missing_from_manifest:
            raise RuntimeError(
                "Fine-tuned registry languages missing from model manifest: "
                f"{sorted(missing_from_manifest)}"
            )

    # A registry status can be changed by hand, but the tokenizer is the final
    # proof that a language marker exists in the deployed model artifact.
    for code in LANGUAGES:
        if tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer does not contain active language token {code}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _tokenizer = tokenizer
    _translator = translator
    _supported_directions = {
        pair for pair in directions if pair[0] in LANGUAGES and pair[1] in LANGUAGES
    }


def unload_model() -> None:
    """Release runtime objects during application shutdown and tests."""

    global _translator, _tokenizer
    _translator = None
    _tokenizer = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bind heavyweight model loading to the ASGI application lifecycle."""

    load_model()
    try:
        yield
    finally:
        unload_model()


app = FastAPI(
    title="Cameroon Languages Translation API",
    version="0.2.0",
    lifespan=lifespan,
)


def check_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require X-API-Key when API_KEY is configured by the deployment."""

    if API_KEY and (x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "Invalid or missing X-API-Key")


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    src_lang: str
    tgt_lang: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject strings that satisfy min_length using whitespace only."""

        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class ContributeRequest(BaseModel):
    src: str = Field(min_length=1, max_length=2000)
    tgt: str = Field(min_length=1, max_length=2000)
    src_lang: str
    tgt_lang: str
    contributor: str | None = Field(default=None, max_length=100)

    @field_validator("src", "tgt")
    @classmethod
    def pair_text_must_not_be_blank(cls, value: str) -> str:
        """Normalize surrounding whitespace and reject empty contributions."""

        value = value.strip()
        if not value:
            raise ValueError("translation text must not be blank")
        return value


def validate_known_languages(src_lang: str, tgt_lang: str, allowed: dict[str, str]) -> None:
    """Validate two distinct language codes against a selected registry view."""

    for code in (src_lang, tgt_lang):
        if code not in allowed:
            raise HTTPException(
                422,
                f"Unknown language code '{code}'. Supported: {sorted(allowed)}",
            )
    if src_lang == tgt_lang:
        raise HTTPException(422, "src_lang and tgt_lang must differ")


def validate_translation_direction(src_lang: str, tgt_lang: str) -> None:
    """Require a direction proven by base pretraining or the model manifest."""

    validate_known_languages(src_lang, tgt_lang, LANGUAGES)
    if (src_lang, tgt_lang) not in _supported_directions:
        raise HTTPException(
            422,
            f"Translation direction {src_lang} -> {tgt_lang} is not enabled "
            "for the deployed model.",
        )


def ensure_model_ready() -> None:
    """Return a service-level error instead of an attribute error in edge cases."""

    if _translator is None or _tokenizer is None:
        raise HTTPException(503, "Translation model is not loaded")


@app.get("/health")
def health():
    """Report process liveness and model readiness separately."""

    loaded = _translator is not None and _tokenizer is not None
    return {
        "status": "ready" if loaded else "starting",
        "model_loaded": loaded,
        "enabled_directions": len(_supported_directions),
    }


@app.get("/ready")
def ready():
    """Provide a readiness probe that returns 503 until inference is possible."""

    ensure_model_ready()
    return {"status": "ready"}


@app.get("/languages")
def languages():
    """List languages enabled for translation by registry status."""

    return LANGUAGES


@app.get("/contribution-languages")
def contribution_languages():
    """List every language accepted by the human-review data pipeline."""

    return CONTRIBUTION_LANGUAGES


@app.get("/directions")
def directions():
    """List exact source and target pairs enabled by the deployed model."""

    return {
        "directions": [
            {"src_lang": source, "tgt_lang": target}
            for source, target in sorted(_supported_directions)
        ]
    }


@app.post("/translate", dependencies=[Depends(check_key)])
def translate(request: TranslateRequest):
    """Translate one text using a manifest-approved language direction."""

    ensure_model_ready()
    validate_translation_direction(request.src_lang, request.tgt_lang)

    with _tokenizer_lock:
        _tokenizer.src_lang = request.src_lang
        tokens = _tokenizer.convert_ids_to_tokens(_tokenizer.encode(request.text))

    results = _translator.translate_batch(
        [tokens],
        target_prefix=[[request.tgt_lang]],
        beam_size=4,
        max_decoding_length=512,
    )
    output_tokens = results[0].hypotheses[0]
    if output_tokens and output_tokens[0] == request.tgt_lang:
        output_tokens = output_tokens[1:]

    with _tokenizer_lock:
        translation = _tokenizer.decode(
            _tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )

    return {
        "translation": translation,
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang,
    }


@app.post("/contribute", dependencies=[Depends(check_key)])
def contribute(request: ContributeRequest):
    """Append a reviewable sentence pair for any registered project language."""

    validate_known_languages(request.src_lang, request.tgt_lang, CONTRIBUTION_LANGUAGES)
    row = request.model_dump()
    row["timestamp"] = datetime.now(UTC).isoformat()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / "contributed.jsonl"
    with _write_lock, destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "status": "saved",
        "message": "Merci / Thank you! Your pair will be reviewed before training.",
    }
