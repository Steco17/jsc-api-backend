# Training Guide

This guide explains the corrected data preparation, Colab training, evaluation, and deployment workflow.

## 1. Input formats

The canonical JSONL format is:

```json
{"src":"Good morning","tgt":"Local translation","src_lang":"eng_Latn","tgt_lang":"ewo_Latn","group_id":"optional-source-group"}
```

`group_id` identifies rows that must never be separated across train, development, and test.
For Bible corpora, use `verse_key` as the group.

CSV input may contain language columns directly:

```csv
group_id,src,tgt,src_lang,tgt_lang
MAT.1.1,Source text,Target text,eng_Latn,ewo_Latn
```

It may also use one fixed pair supplied on the command line:

```powershell
python scripts/prepare_data.py data/translations.csv `
  --csv `
  --src-col en `
  --tgt-col ewo `
  --group-col verse_key `
  --src-lang eng_Latn `
  --tgt-lang ewo_Latn `
  --add-reverse `
  -o data/prepared
```

Malformed JSON and missing required CSV columns now stop preparation with a precise error.
Silently training on a partially parsed corpus is not allowed.

## 2. Leakage-safe splitting

Individual rows are not safe split units when several rows represent the same source passage.
The preparation script assigns the complete group with SHA-256 using the group identifier and seed.

This approach provides four guarantees:

1. Every variant of a `verse_key` stays in one split.
2. A reverse translation stays beside its original row.
3. The same `verse_key` receives the same split in every language file.
4. Regenerating with the same seed produces the same assignment.

When no explicit group exists, a direction-independent identifier is derived from the complete sentence pair.

The default split is 95 percent training, 2.5 percent development, and 2.5 percent test.
Fractions are applied as stable hash thresholds, so exact row percentages can vary slightly.

## 3. Script-specific language targets

The raw Fulfulde file contains both Latin and Arabic script.
`scripts/prepare_all.py` runs that file twice with explicit script filters.

Latin rows receive `fub_Latn`.
Arabic rows receive `fub_Arab`.

This preserves both data sources while giving users deterministic control over requested output script.

## 4. Prepare the complete corpus

Run:

```powershell
python scripts/prepare_all.py data -o data/prepared --add-reverse
```

`--add-reverse` is required when both English-to-local and local-to-English service directions are intended.
It roughly doubles the row count, but it does not create split leakage.

The batch command creates per-language files under `data/prepared/_by_lang` and these combined files:

```text
data/prepared/train.jsonl
data/prepared/dev.jsonl
data/prepared/test.jsonl
data/prepared/dataset_manifest.json
```

The dataset manifest records target languages, directions, row counts, the grouping method, and SHA-256 checksums for every generated split.

## 5. Inspect data quality before GPU use

At minimum, verify:

- Every source text expresses the same meaning as its target.
- `group_id` represents a real shared source unit.
- Script labels match actual Unicode script.
- Text is natural sentence-level language rather than isolated dictionary entries.
- Duplicates, untranslated rows, and extreme length mismatches were removed.
- Native speakers reviewed random samples from every language and script.
- The religious domain represented by the Bible corpus matches or is supplemented for the intended product domain.

Large row counts cannot compensate for systematic alignment or labeling errors.

## 6. Reproducible dependencies

Install:

```powershell
pip install -r requirements-train.txt
```

Transformers, Datasets, PEFT, Accelerate, SentencePiece, SacreBLEU, and CTranslate2 are pinned.
PyTorch remains in a bounded major-version range because Colab supplies a CUDA-specific build.

The Colab notebook removes the unused preinstalled `torchao` package before importing PEFT.
This prevents the optional quantization backend from breaking normal LoRA initialization.

## 7. New language tokens and LoRA

The training script reads every source and target code from the dataset.
It asks the tokenizer whether each code already exists.
Unknown codes are added automatically.

The complete NLLB embedding table is not made trainable.
PEFT `trainable_token_indices` updates only the new language-token rows.
LoRA adapters update attention projections and feed-forward layers.

This retains the intended parameter-efficient behavior and avoids optimizer state for hundreds of millions of unrelated embedding values.

## 8. GPU memory controls

The recommended T4 settings are:

```powershell
python scripts/finetune.py `
  --train data/prepared/train.jsonl `
  --dev data/prepared/dev.jsonl `
  --out model_out `
  --epochs 1 `
  --batch 4 `
  --grad-accum 16 `
  --eval-steps 500
```

The model is loaded in float16.
Gradient checkpointing reduces activation memory.
A physical batch of four leaves memory headroom for long sentences.
Sixteen accumulation steps retain an effective batch size of 64.
One epoch is the initial limit because it already covers more than 3.3 million prepared examples.
Extend training only after development and test evidence shows that another pass is beneficial.

## 9. Required smoke run

Before the full run, exercise the complete lifecycle with a small deterministic subset:

```powershell
python scripts/finetune.py `
  --train data/prepared/train.jsonl `
  --dev data/prepared/dev.jsonl `
  --out model_smoke `
  --epochs 1 `
  --batch 2 `
  --grad-accum 1 `
  --eval-steps 1 `
  --max-steps 2 `
  --max-train-samples 256 `
  --max-dev-samples 64 `
  --resume never
```

This detects dependency, tokenizer, LoRA, CUDA, evaluation, checkpoint, merge, and manifest failures before the expensive run.

## 10. Safe checkpoint resume

`model_out/run_config.json` records a fingerprint of important data and training properties.
Automatic resume is accepted only when the fingerprint matches.

A checkpoint directory without `trainer_state.json` is treated as an interrupted write and ignored.
Changing languages, data sizes, directions, base model, batch, accumulation, or learning rate requires a new output directory.

This prevents an old Drive checkpoint from being silently reused for a different experiment.

## 11. Model output

Successful training creates:

```text
model_out/merged/
  config.json
  model.safetensors
  tokenizer files
  training_manifest.json
  dataset_manifest.json
```

`training_manifest.json` records the model base, all dataset languages, observed directions, new token IDs, and run fingerprint.
The API uses this file as deployment capability evidence.

## 12. Evaluation and deployment

Convert the exact merged model:

```powershell
ct2-transformers-converter `
  --model model_out/merged `
  --output_dir model_ct2 `
  --quantization int8 `
  --force
```

Evaluate:

```powershell
python scripts/evaluate.py `
  --test data/prepared/test.jsonl `
  --ct2 model_ct2 `
  --tokenizer model_out/merged
```

Do not expose a language merely because training completed.
Compare each direction with the base model and obtain native-speaker ratings on unseen sentences.
After approval, update that language to `fine_tuned` in `data/languages.json` and redeploy the matching model, tokenizer, and manifest together.

## 13. Colab-specific workflow

Use `notebooks/train_colab.ipynb` only in Google Colab.
The notebook validates Colab, repository access, pinned versions, GPU memory, prepared group integrity, and smoke training before the full run.

Use `Runtime > Run all` so no prerequisite cell is skipped.
If a session disconnects, run all cells again and the full training command will resume the newest compatible complete checkpoint.
