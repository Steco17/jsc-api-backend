#!/usr/bin/env python3
"""Fine-tune NLLB-200 with LoRA and selective language-token training.

This script is designed for a CUDA GPU such as a Colab T4.  It automatically
detects language codes that NLLB does not know, adds those tokens, and trains
only the new token rows instead of the complete 256k-token embedding table.
That distinction saves several gigabytes of optimizer and gradient memory.

The output includes ``merged/training_manifest.json``.  The API reads this
manifest so it exposes only directions that were actually present in the
training data.
"""

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE = "facebook/nllb-200-distilled-600M"
MAX_LEN = 256
REQUIRED_COLUMNS = {"src", "tgt", "src_lang", "tgt_lang"}


def dataset_metadata(dataset: DatasetDict) -> tuple[list[str], list[list[str]]]:
    """Validate the dataset and return its languages and observed directions."""

    for split in ("train", "dev"):
        if split not in dataset:
            raise ValueError(f"dataset is missing the '{split}' split")
        missing = REQUIRED_COLUMNS - set(dataset[split].column_names)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{split} split is missing columns: {names}")
        if len(dataset[split]) == 0:
            raise ValueError(f"{split} split is empty")

    # Arrow columns are iterated without materializing a list of 1.7 million
    # dictionaries, keeping this validation pass memory-efficient.
    directions = {
        (src, tgt)
        for src, tgt in zip(
            dataset["train"]["src_lang"], dataset["train"]["tgt_lang"], strict=False
        )
    }
    languages = sorted({code for pair in directions for code in pair})
    return languages, [list(pair) for pair in sorted(directions)]


def resolve_precision(requested: str) -> str:
    """Choose BF16 when the GPU supports its wider numeric range."""

    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested, but this CUDA GPU does not support it")
    return requested


def build_run_config(args, languages: list[str], directions: list[list[str]]) -> dict:
    """Create the compatibility record used to validate checkpoint resumes."""

    dataset_manifest_path = Path(args.train).resolve().parent / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.is_file()
        else {}
    )
    manifest_hashes = dataset_manifest.get("sha256", {})

    dataset_files = []
    for split, path_string in (("train", args.train), ("dev", args.dev)):
        path = Path(path_string).resolve()
        stat = path.stat()
        with path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        expected_checksum = manifest_hashes.get(split)
        if expected_checksum and checksum != expected_checksum:
            raise RuntimeError(
                f"{path.name} does not match dataset_manifest.json. "
                "Regenerate and validate prepared data before training."
            )
        dataset_files.append(
            {
                # Content hashes allow regenerated files to resume while still
                # detecting a same-size correction to any training sentence.
                "name": path.name,
                "bytes": stat.st_size,
                "sha256": checksum,
            }
        )

    config = {
        "base_model": BASE,
        "languages": languages,
        "trained_directions": directions,
        "target_languages": sorted(args.target_langs),
        "dataset_files": dataset_files,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "gradient_accumulation": args.grad_accum,
        "precision": args.precision,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "eval_steps": args.eval_steps,
        "max_train_samples": args.max_train_samples,
        "max_dev_samples": args.max_dev_samples,
        "seed": args.seed,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config


def compatible_checkpoint(out_dir: Path, run_config: dict, resume: str) -> str | None:
    """Return the newest complete compatible checkpoint, if one exists.

    A disconnected Colab write can leave a numbered checkpoint directory that
    lacks ``trainer_state.json``.  Such a directory is ignored rather than
    being handed to Trainer as a valid resume point.
    """

    checkpoints = sorted(
        out_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        reverse=True,
    )
    if resume == "never" and checkpoints:
        raise RuntimeError(
            f"{out_dir} already contains checkpoints. "
            "Use a new --out directory for a non-resuming run."
        )
    if not checkpoints:
        return None

    config_path = out_dir / "run_config.json"
    if not config_path.exists():
        raise RuntimeError(
            f"{out_dir} contains checkpoints without run_config.json. "
            "Use a new --out directory or pass --resume never."
        )
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    if previous.get("fingerprint") != run_config["fingerprint"]:
        raise RuntimeError(
            "Existing checkpoints were created with different data or training "
            "settings. Use a new --out directory to avoid corrupting the run."
        )

    for checkpoint in checkpoints:
        if (checkpoint / "trainer_state.json").is_file():
            return str(checkpoint)
        print(f"Ignoring incomplete checkpoint: {checkpoint}")
    return None


def add_unknown_language_tokens(tokenizer, model, languages: list[str]) -> list[int]:
    """Register unknown language codes and return only their new token IDs."""

    unknown = [
        code
        for code in languages
        if tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id
    ]
    if not unknown:
        return []

    # Replacing additional_special_tokens would remove NLLB's existing language
    # codes from the special-token registry.  Extending the registry preserves
    # all 200 original codes and makes the new codes safe during decoding.
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": unknown},
        replace_additional_special_tokens=False,
    )
    if added != len(unknown):
        raise RuntimeError(f"expected to add {len(unknown)} language tokens, but added {added}")

    # mean_resizing=False gives genuinely new language markers independent
    # random rows.  Only these rows are made trainable by PEFT below.
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    token_ids = [tokenizer.convert_tokens_to_ids(code) for code in unknown]
    print(f"Added {len(unknown)} language tokens: {', '.join(unknown)}")
    return token_ids


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    """Tokenize mixed language pairs efficiently and without state leakage."""

    def preprocess_batch(batch: dict) -> dict:
        size = len(batch["src"])
        encoded_rows: list[dict | None] = [None] * size
        by_direction: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, pair in enumerate(zip(batch["src_lang"], batch["tgt_lang"], strict=False)):
            by_direction[pair].append(index)

        # NLLB stores src_lang and tgt_lang as mutable tokenizer state.  Each
        # homogeneous sub-batch is processed completely before that state is
        # changed for the next direction.
        for (src_lang, tgt_lang), indices in by_direction.items():
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            sources = [batch["src"][index] for index in indices]
            targets = [batch["tgt"][index] for index in indices]
            encoded = tokenizer(
                sources,
                text_target=targets,
                truncation=True,
                max_length=max_length,
            )
            for offset, original_index in enumerate(indices):
                encoded_rows[original_index] = {
                    key: values[offset] for key, values in encoded.items()
                }

        if any(row is None for row in encoded_rows):
            raise RuntimeError("tokenization left an input row unprocessed")
        keys = encoded_rows[0].keys()
        return {key: [row[key] for row in encoded_rows] for key in keys}

    return dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=512,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing multilingual translation pairs",
    )


def filter_target_languages(
    dataset: DatasetDict,
    target_languages: list[str],
) -> DatasetDict:
    """Keep pairs involving selected languages and reject unknown selections."""

    if not target_languages:
        return dataset

    requested = set(target_languages)
    available = set(dataset["train"]["src_lang"]) | set(dataset["train"]["tgt_lang"])
    missing = requested - available
    if missing:
        raise ValueError(f"target languages are absent from the training data: {sorted(missing)}")

    filtered = dataset.filter(
        lambda row: row["src_lang"] in requested or row["tgt_lang"] in requested,
        desc=f"Selecting pairs for {', '.join(sorted(requested))}",
    )
    for split in ("train", "dev"):
        if len(filtered[split]) == 0:
            raise ValueError(f"target-language selection left the '{split}' split empty")
    return filtered


def maybe_limit_samples(dataset: DatasetDict, train_limit: int, dev_limit: int):
    """Return deterministic prefixes for a quick end-to-end smoke run."""

    if train_limit:
        dataset["train"] = dataset["train"].select(range(min(train_limit, len(dataset["train"]))))
    if dev_limit:
        dataset["dev"] = dataset["dev"].select(range(min(dev_limit, len(dataset["dev"]))))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", default="model_out")
    parser.add_argument(
        "--new-langs",
        nargs="*",
        default=[],
        help="optional extra codes; unknown codes in the data are detected automatically",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=1,
        help="one pass is the safe default for this multi-million-row corpus",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--eval-batch",
        type=int,
        default=0,
        help="evaluation batch size; zero reuses --batch",
    )
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--empty-cache-steps",
        type=int,
        default=0,
        help="release unused CUDA cache every N optimizer steps; zero disables it",
    )
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help="auto prefers BF16 on supported GPUs and otherwise uses FP16",
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument(
        "--target-langs",
        nargs="+",
        default=[],
        help="train only pairs involving these language codes",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LEN,
        help="maximum source and target token length; 128 reduces local GPU memory",
    )
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Use a Colab T4 or another CUDA training host.")
    args.eval_batch = args.eval_batch or args.batch
    if min(args.batch, args.eval_batch, args.grad_accum, args.eval_steps) < 1:
        parser.error("batch, eval-batch, grad-accum, and eval-steps must be positive")
    if args.max_length < 1:
        parser.error("max-length must be positive")
    if min(args.max_train_samples, args.max_dev_samples) < 0:
        parser.error("sample limits cannot be negative")
    if args.empty_cache_steps < 0 or args.dataloader_workers < 0:
        parser.error("empty-cache-steps and dataloader-workers cannot be negative")

    gpu = torch.cuda.get_device_properties(0)
    args.precision = resolve_precision(args.precision)
    print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GiB)")
    print(f"Training precision: {args.precision.upper()}")
    print(
        f"Physical batch: {args.batch}; gradient accumulation: {args.grad_accum}; "
        f"effective batch: {args.batch * args.grad_accum}"
    )

    dataset = load_dataset("json", data_files={"train": args.train, "dev": args.dev})
    dataset = filter_target_languages(dataset, args.target_langs)
    dataset = maybe_limit_samples(dataset, args.max_train_samples, args.max_dev_samples)
    languages, directions = dataset_metadata(dataset)
    languages = sorted(set(languages) | set(args.new_langs))
    run_config = build_run_config(args, languages, directions)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = compatible_checkpoint(out_dir, run_config, args.resume)
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # NLLB uses SentencePiece, so the optional Mistral regex migration does
    # not apply. Saving this explicit choice also keeps local reloads quiet.
    tokenizer = AutoTokenizer.from_pretrained(BASE, fix_mistral_regex=False)
    model_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    new_token_ids = add_unknown_language_tokens(tokenizer, model, languages)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        trainable_token_indices=new_token_ids or None,
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    model.print_trainable_parameters()

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable / total > 0.10:
        raise RuntimeError(
            f"{trainable / total:.1%} of the model is trainable. "
            "Expected selective token training plus LoRA to stay below 10%."
        )

    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    updates_per_epoch = math.ceil(len(tokenized["train"]) / (args.batch * args.grad_accum))
    planned_steps = (
        args.max_steps if args.max_steps > 0 else max(1, math.ceil(updates_per_epoch * args.epochs))
    )
    effective_eval_steps = min(args.eval_steps, planned_steps)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_empty_cache_steps=args.empty_cache_steps or None,
        eval_strategy="steps",
        eval_steps=effective_eval_steps,
        save_strategy="steps",
        save_steps=effective_eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=min(50, effective_eval_steps),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_workers,
        save_safetensors=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gradient_norms = [
        float(entry["grad_norm"])
        for entry in trainer.state.log_history
        if "grad_norm" in entry
    ]
    finite_gradient_steps = sum(math.isfinite(norm) for norm in gradient_norms)
    print(f"Peak GPU memory allocated: {peak_allocated / 1024**3:.2f} GiB")
    print(f"Peak GPU memory reserved: {peak_reserved / 1024**3:.2f} GiB")

    runtime_metrics = {
        "schema_version": 1,
        "gpu": gpu.name,
        "gpu_total_memory_bytes": gpu.total_memory,
        "precision": args.precision,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "gradient_norms": gradient_norms,
        "finite_gradient_steps": finite_gradient_steps,
        "train_metrics": train_result.metrics,
    }
    (out_dir / "runtime_metrics.json").write_text(
        json.dumps(runtime_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if gradient_norms and finite_gradient_steps == 0:
        raise RuntimeError(
            "Every logged gradient norm was non-finite, so optimizer updates may have been "
            "skipped. Use --precision bf16 on a supported GPU or lower the learning rate."
        )

    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged_dir = out_dir / "merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    shutil.copy2(out_dir / "runtime_metrics.json", merged_dir / "runtime_metrics.json")

    manifest = {
        "schema_version": 1,
        "base_model": BASE,
        "languages": languages,
        "trained_directions": directions,
        "new_language_token_ids": new_token_ids,
        "run_fingerprint": run_config["fingerprint"],
    }
    (merged_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_manifest = Path(args.train).parent / "dataset_manifest.json"
    if source_manifest.is_file():
        shutil.copy2(source_manifest, merged_dir / "dataset_manifest.json")

    print(f"Merged model saved to {merged_dir}")
    print(
        "Next: ct2-transformers-converter --model",
        merged_dir,
        "--output_dir model_ct2 --quantization int8",
    )


if __name__ == "__main__":
    main()
