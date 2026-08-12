"""Focused tests for fine-tuning setup that do not require model weights."""

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict
from peft import LoraConfig

from scripts import finetune


class TokenizerDouble:
    """Minimal tokenizer that records safe special-token extension behavior."""

    unk_token_id = 0

    def __init__(self):
        self.mapping = {"eng_Latn": 1, "fra_Latn": 2}
        self.replace_flag = None

    def convert_tokens_to_ids(self, token):
        return self.mapping.get(token, self.unk_token_id)

    def add_special_tokens(self, payload, replace_additional_special_tokens):
        self.replace_flag = replace_additional_special_tokens
        for token in payload["additional_special_tokens"]:
            self.mapping[token] = len(self.mapping) + 1
        return len(payload["additional_special_tokens"])

    def __len__(self):
        return len(self.mapping) + 1


class ModelDouble:
    """Minimal model that records embedding resize arguments."""

    resized = None

    def resize_token_embeddings(self, size, mean_resizing):
        self.resized = (size, mean_resizing)


def test_pinned_peft_accepts_selective_trainable_token_indices() -> None:
    config = LoraConfig(
        r=16,
        target_modules=["q_proj"],
        trainable_token_indices=[256206, 256207],
        task_type="SEQ_2_SEQ_LM",
    )

    assert config.trainable_token_indices == [256206, 256207]
    assert config.modules_to_save is None


def test_unknown_languages_extend_special_tokens_without_replacing_nllb_codes() -> None:
    tokenizer = TokenizerDouble()
    model = ModelDouble()

    token_ids = finetune.add_unknown_language_tokens(
        tokenizer,
        model,
        ["eng_Latn", "ewo_Latn", "fub_Arab"],
    )

    assert tokenizer.replace_flag is False
    assert model.resized == (5, False)
    assert token_ids == [3, 4]


def test_dataset_metadata_reports_observed_directions() -> None:
    rows = {
        "src": ["one", "two"],
        "tgt": ["un", "deux"],
        "src_lang": ["eng_Latn", "fra_Latn"],
        "tgt_lang": ["fra_Latn", "eng_Latn"],
    }
    dataset = DatasetDict(train=Dataset.from_dict(rows), dev=Dataset.from_dict(rows))

    languages, directions = finetune.dataset_metadata(dataset)

    assert languages == ["eng_Latn", "fra_Latn"]
    assert directions == [
        ["eng_Latn", "fra_Latn"],
        ["fra_Latn", "eng_Latn"],
    ]


def test_checkpoint_resume_ignores_incomplete_latest_directory(tmp_path) -> None:
    run_config = {"fingerprint": "same"}
    (tmp_path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    complete = tmp_path / "checkpoint-100"
    incomplete = tmp_path / "checkpoint-200"
    complete.mkdir()
    incomplete.mkdir()
    (complete / "trainer_state.json").write_text("{}", encoding="utf-8")

    selected = finetune.compatible_checkpoint(tmp_path, run_config, "auto")

    assert selected == str(complete)


def test_checkpoint_resume_rejects_changed_run_fingerprint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_config.json").write_text(json.dumps({"fingerprint": "old"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different data or training settings"):
        finetune.compatible_checkpoint(
            tmp_path,
            {"fingerprint": "new"},
            "auto",
        )


def test_non_resuming_run_rejects_an_output_with_checkpoints(tmp_path) -> None:
    (tmp_path / "checkpoint-100").mkdir()

    with pytest.raises(RuntimeError, match="already contains checkpoints"):
        finetune.compatible_checkpoint(
            tmp_path,
            {"fingerprint": "new"},
            "never",
        )


def test_run_fingerprint_survives_regenerated_file_timestamps(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("same bytes\n", encoding="utf-8")
    dev.write_text("dev bytes\n", encoding="utf-8")
    args = SimpleNamespace(
        train=str(train),
        dev=str(dev),
        lr=1e-4,
        batch=4,
        grad_accum=16,
        epochs=4,
        max_steps=-1,
        eval_steps=500,
        max_train_samples=0,
        max_dev_samples=0,
        seed=42,
    )
    first = finetune.build_run_config(args, ["eng_Latn"], [])

    train.touch()
    dev.touch()
    second = finetune.build_run_config(args, ["eng_Latn"], [])

    assert first["fingerprint"] == second["fingerprint"]

    train.write_text("new! bytes\n", encoding="utf-8")
    third = finetune.build_run_config(args, ["eng_Latn"], [])
    assert third["dataset_files"][0]["bytes"] == first["dataset_files"][0]["bytes"]
    assert third["fingerprint"] != first["fingerprint"]


def test_run_config_rejects_a_stale_dataset_manifest(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text("train\n", encoding="utf-8")
    dev.write_text("dev\n", encoding="utf-8")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"sha256": {"train": "wrong", "dev": "wrong"}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        train=str(train),
        dev=str(dev),
        lr=1e-4,
        batch=4,
        grad_accum=16,
        epochs=4,
        max_steps=-1,
        eval_steps=500,
        max_train_samples=0,
        max_dev_samples=0,
        seed=42,
    )

    with pytest.raises(RuntimeError, match="does not match dataset_manifest"):
        finetune.build_run_config(args, ["eng_Latn"], [])
