"""HTTP and concurrency regression tests for the translation API."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main


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


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Provide an HTTP client with lightweight in-memory inference doubles."""

    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "_tokenizer", EchoTokenizer())
    monkeypatch.setattr(main, "_translator", EchoTranslator())
    monkeypatch.setattr(main, "_supported_directions", main.base_directions())
    return TestClient(main.app)


def test_contribution_accepts_a_planned_language(client, tmp_path) -> None:
    response = client.post(
        "/contribute",
        json={
            "src": "Good morning",
            "tgt": "Idi na mbua",
            "src_lang": "eng_Latn",
            "tgt_lang": "dua_Latn",
            "contributor": "steco",
        },
    )

    assert response.status_code == 200
    saved = json.loads((tmp_path / "contributed.jsonl").read_text(encoding="utf-8"))
    assert saved["tgt_lang"] == "dua_Latn"
    assert saved["timestamp"].endswith("+00:00")


def test_translation_requires_a_manifest_approved_direction(client, monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "LANGUAGES",
        {**main.LANGUAGES, "ewo_Latn": "Ewondo"},
    )
    response = client.post(
        "/translate",
        json={
            "text": "Bonjour",
            "src_lang": "fra_Latn",
            "tgt_lang": "ewo_Latn",
        },
    )

    assert response.status_code == 422
    assert "not enabled" in response.json()["detail"]


def test_readiness_returns_503_without_a_model(monkeypatch) -> None:
    monkeypatch.setattr(main, "_translator", None)
    monkeypatch.setattr(main, "_tokenizer", None)
    response = TestClient(main.app).get("/ready")
    assert response.status_code == 503


def test_api_key_is_enforced_when_configured(client, monkeypatch) -> None:
    monkeypatch.setattr(main, "API_KEY", "secret-value")
    payload = {
        "text": "Good morning",
        "src_lang": "eng_Latn",
        "tgt_lang": "fra_Latn",
    }

    assert client.post("/translate", json=payload).status_code == 401
    response = client.post(
        "/translate",
        json=payload,
        headers={"X-API-Key": "secret-value"},
    )
    assert response.status_code == 200


def test_concurrent_translations_keep_their_source_language(client) -> None:
    requests = [
        ("eng_Latn", "fra_Latn") if index % 2 else ("fra_Latn", "eng_Latn") for index in range(30)
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


def test_manifest_rejects_unknown_language_codes() -> None:
    with pytest.raises(RuntimeError, match="unknown languages"):
        main._validated_manifest_directions({"trained_directions": [["eng_Latn", "unknown_Latn"]]})


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

    tokenizer = EchoTokenizer()
    monkeypatch.setattr(main, "MODEL_DIR", model_dir)
    monkeypatch.setattr(main, "TOKENIZER_DIR", tokenizer_dir)
    monkeypatch.setattr(main, "DATA_DIR", tmp_path / "contributions")
    monkeypatch.setattr(main, "FINE_TUNED_LANGUAGES", {"ewo_Latn"})
    monkeypatch.setattr(
        main,
        "LANGUAGES",
        {**main.LANGUAGES, "ewo_Latn": "Ewondo"},
    )
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

    main.load_model()

    assert ("eng_Latn", "ewo_Latn") in main._supported_directions
    assert ("ewo_Latn", "eng_Latn") in main._supported_directions
    assert main._translator is not None
    main.unload_model()
