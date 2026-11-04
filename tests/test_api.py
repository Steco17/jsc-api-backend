"""HTTP, abuse-control, and concurrency regression tests for the API."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.application import create_app
from app.config import Settings
from app.contributions import ContributionRepository
from app.registry import LanguageRegistry, load_language_registry
from app.translation import TranslationService

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "languages.json"
CONTRIBUTION_KEY = "c" * 32


class EchoTokenizer:
    """Small tokenizer double that exposes source-language race conditions."""

    src_lang = "eng_Latn"
    unk_token_id = -1

    def encode(self, text):
        selected_language = self.src_lang
        time.sleep(0.002)
        return [selected_language, text]

    def convert_ids_to_tokens(self, values):
        return values

    def convert_tokens_to_ids(self, values):
        if isinstance(values, str):
            return 1
        return values

    def decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return "|".join(values)


class EchoTranslator:
    """CTranslate2-compatible double that echoes source tokens after the prefix."""

    def translate_batch(self, batches, target_prefix, **_kwargs):
        hypothesis = target_prefix[0] + batches[0]
        return [SimpleNamespace(hypotheses=[hypothesis])]


def settings_for(tmp_path: Path, **environment_overrides: str) -> Settings:
    environment = {
        "APP_ENV": "test",
        "TRANSLATION_MODE": "model",
        "DATA_DIR": str(tmp_path),
        "DATABASE_PATH": str(tmp_path / "jsc.sqlite3"),
        "LANGUAGES_FILE": str(REGISTRY_PATH),
        "ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
        "CORS_ORIGINS": "http://localhost:5173",
        "CONTRIBUTION_API_KEY": CONTRIBUTION_KEY,
    }
    environment.update(environment_overrides)
    return Settings.from_environment(environment)


def application_for(
    tmp_path: Path,
    *,
    registry: LanguageRegistry | None = None,
    preload_model: bool = True,
    **environment_overrides: str,
):
    settings = settings_for(tmp_path, **environment_overrides)
    language_registry = registry or load_language_registry(settings.languages_file)
    service = TranslationService(settings, language_registry)
    if preload_model:
        service._tokenizer = EchoTokenizer()
        service._translator = EchoTranslator()
        service._supported_directions = language_registry.base_directions()
    repository = ContributionRepository(
        settings.data_dir,
        max_file_bytes=settings.contribution_max_file_bytes,
        store_contributor_names=settings.store_contributor_names,
    )
    application = create_app(
        settings,
        registry=language_registry,
        translation_service=service,
        contribution_repository=repository,
    )
    return TestClient(application), service, repository


@pytest.fixture
def client(tmp_path):
    api_client, _service, _repository = application_for(tmp_path)
    return api_client


def test_contribution_is_key_protected_and_drops_names_by_default(client, tmp_path) -> None:
    response = client.post(
        "/contribute",
        headers={"X-API-Key": CONTRIBUTION_KEY},
        json={
            "src": "Good morning",
            "tgt": "Idi na mbua",
            "src_lang": "eng_Latn",
            "tgt_lang": "dua_Latn",
            "contributor": "Private Name",
        },
    )

    assert response.status_code == 200
    saved = json.loads((tmp_path / "contributed.jsonl").read_text(encoding="utf-8"))
    assert saved["tgt_lang"] == "dua_Latn"
    assert saved["timestamp"].endswith("+00:00")
    assert "contributor" not in saved


def test_contribution_is_disabled_without_a_secret(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        CONTRIBUTION_API_KEY="",
    )
    response = client.post(
        "/contribute",
        json={
            "src": "Good morning",
            "tgt": "Bonjour",
            "src_lang": "eng_Latn",
            "tgt_lang": "fra_Latn",
        },
    )
    assert response.status_code == 503
    assert not (tmp_path / "contributed.jsonl").exists()


def test_translation_requires_a_manifest_approved_direction(client) -> None:
    service = client.app.state.translation_service
    service._supported_directions = set()
    response = client.post(
        "/translate",
        json={"text": "Bonjour", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Translation direction is not enabled"


def test_readiness_returns_503_without_a_model(tmp_path) -> None:
    client, _service, _repository = application_for(tmp_path, preload_model=False)
    response = client.get("/ready")
    assert response.status_code == 503


def test_placeholder_runtime_starts_without_model(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        preload_model=False,
        TRANSLATION_MODE="placeholder",
    )
    with client:
        health = client.get("/health")
        readiness = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "translation_mode": "placeholder",
        "model_loaded": False,
        "enabled_directions": 0,
    }
    assert readiness.status_code == 200
    assert readiness.json()["model_loaded"] is False


def test_placeholder_translation_returns_stable_empty_response(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        preload_model=False,
        TRANSLATION_MODE="placeholder",
    )
    response = client.post(
        "/translate",
        json={"text": "Good morning", "src_lang": "eng_Latn", "tgt_lang": "ewo_Latn"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "translation": "",
        "src_lang": "eng_Latn",
        "tgt_lang": "ewo_Latn",
        "status": "model_pending",
    }


def test_placeholder_languages_include_african_training_targets(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        preload_model=False,
        TRANSLATION_MODE="placeholder",
    )
    response = client.get("/languages")
    assert response.status_code == 200
    assert response.json()["ewo_Latn"] == "Ewondo"
    assert response.json()["dua_Latn"] == "Duala"


def test_language_catalog_exposes_only_public_metadata(client) -> None:
    response = client.get("/language-catalog")
    assert response.status_code == 200
    catalog = {entry["code"]: entry for entry in response.json()["languages"]}
    assert catalog["wes_Latn"] == {
        "code": "wes_Latn",
        "iso_code": "wes",
        "name": "Cameroon Pidgin English",
        "script": "Latn",
        "status": "planned",
        "featured": True,
    }
    assert "note" not in catalog["wes_Latn"]


def test_translation_api_key_is_enforced_when_configured(tmp_path) -> None:
    translation_key = "t" * 32
    client, _service, _repository = application_for(
        tmp_path,
        TRANSLATION_API_KEY=translation_key,
    )
    payload = {"text": "Good morning", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}

    assert client.post("/translate", json=payload).status_code == 401
    response = client.post(
        "/translate",
        json=payload,
        headers={"X-API-Key": translation_key},
    )
    assert response.status_code == 200


def test_concurrent_translations_keep_their_source_language(client) -> None:
    requests = [
        ("eng_Latn", "fra_Latn") if index % 2 else ("fra_Latn", "eng_Latn")
        for index in range(30)
    ]

    def translate(pair):
        source, target = pair
        response = client.post(
            "/translate",
            json={"text": "sample", "src_lang": source, "tgt_lang": target},
        )
        assert response.status_code == 200
        return source, response.json()["translation"]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(translate, requests))

    for expected_source, translation in results:
        assert translation.startswith(f"{expected_source}|")


def test_unknown_fields_and_oversized_bodies_are_rejected(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        MAX_REQUEST_BODY_BYTES="4096",
    )
    unexpected = client.post(
        "/translate",
        json={
            "text": "sample",
            "src_lang": "eng_Latn",
            "tgt_lang": "fra_Latn",
            "unexpected": "value",
        },
    )
    oversized = client.post(
        "/translate",
        content="x" * 5000,
        headers={"Content-Type": "application/json"},
    )
    assert unexpected.status_code == 422
    assert oversized.status_code == 413


def test_rate_limit_and_security_headers_are_enforced(tmp_path) -> None:
    client, _service, _repository = application_for(
        tmp_path,
        TRANSLATION_RATE_LIMIT="2",
    )
    payload = {"text": "sample", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"}
    assert client.post("/translate", json=payload).status_code == 200
    assert client.post("/translate", json=payload).status_code == 200
    limited = client.post("/translate", json=payload)
    assert limited.status_code == 429
    assert limited.headers["x-content-type-options"] == "nosniff"
    assert limited.headers["cache-control"] == "no-store"
    assert limited.headers["x-frame-options"] == "DENY"
    assert limited.headers["x-permitted-cross-domain-policies"] == "none"


def test_development_docs_have_a_restricted_compatible_csp(tmp_path) -> None:
    client, _service, _repository = application_for(tmp_path, ENABLE_API_DOCS="true")

    response = client.get("/docs")

    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in policy
    assert "https://cdn.jsdelivr.net" in policy


def test_untrusted_host_is_rejected(client) -> None:
    response = client.get("/health", headers={"Host": "attacker.invalid"})
    assert response.status_code == 400


def test_manifest_rejects_unknown_language_codes(tmp_path) -> None:
    settings = settings_for(tmp_path)
    registry = load_language_registry(settings.languages_file)
    service = TranslationService(settings, registry)
    with pytest.raises(RuntimeError, match="unknown languages"):
        service._validated_manifest_directions(
            {"trained_directions": [["eng_Latn", "unknown_Latn"]]}
        )


def test_model_startup_loads_manifest_directions(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model_ct2"
    tokenizer_dir = tmp_path / "merged"
    model_dir.mkdir()
    tokenizer_dir.mkdir()
    (tokenizer_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "languages": ["eng_Latn", "ewo_Latn"],
                "trained_directions": [
                    ["eng_Latn", "ewo_Latn"],
                    ["ewo_Latn", "eng_Latn"],
                ],
            }
        ),
        encoding="utf-8",
    )

    import ctranslate2
    import transformers

    base_registry = load_language_registry(REGISTRY_PATH)
    active_languages = MappingProxyType(
        {**base_registry.active_languages, "ewo_Latn": "Ewondo"}
    )
    registry = replace(
        base_registry,
        active_languages=active_languages,
        fine_tuned_languages=frozenset({"ewo_Latn"}),
    )
    settings = settings_for(
        tmp_path,
        MODEL_DIR=str(model_dir),
        TOKENIZER_DIR=str(tokenizer_dir),
    )
    tokenizer = EchoTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        ctranslate2,
        "Translator",
        lambda *_args, **_kwargs: EchoTranslator(),
    )

    service = TranslationService(settings, registry)
    service.load_model()

    assert ("eng_Latn", "ewo_Latn") in service.supported_directions
    assert ("ewo_Latn", "eng_Latn") in service.supported_directions
    assert service.model_loaded
    service.unload_model()
