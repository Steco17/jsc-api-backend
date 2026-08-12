# Cameroon Languages Translation API

This project fine-tunes NLLB-200 for Cameroonian languages and serves the result through a FastAPI and CTranslate2 API.
It also collects reviewed community translation pairs for later training runs.

The repository contains source data and training code, but it does not contain trained model weights.
The API becomes ready only after `model_out/merged` and `model_ct2` have been produced or mounted.

## What the application provides

- `POST /translate` translates a direction approved by the deployed model manifest.
- `POST /contribute` accepts sentence pairs for every language in `data/languages.json`, including planned languages.
- `GET /languages` lists languages currently enabled for translation.
- `GET /contribution-languages` lists all languages accepted by the data collection pipeline.
- `GET /directions` lists the exact source and target pairs supported by the deployed model.
- `GET /health` reports process health and model-loading state.
- `GET /ready` returns HTTP 503 until translation is ready.

## Important correctness rules

### Dataset groups stay together

The raw corpora contain several translations of the same Bible `verse_key`.
Splitting those variants independently would leak the same verse into training and evaluation.

`scripts/prepare_data.py` assigns each group through a stable SHA-256 hash.
The same `verse_key` therefore receives the same split across all language files.
Reverse-direction rows retain the same `group_id`, so they cannot cross splits either.

### Fulfulde scripts are separate targets

The Fulfulde CSV contains both Latin and Arabic writing.
Preparation emits those rows as `fub_Latn` and `fub_Arab` instead of teaching one token to represent both scripts.

### Translation capability comes from the model

Language status alone does not prove that a model learned a direction.
`scripts/finetune.py` writes `training_manifest.json` beside the merged tokenizer.
The API reads that manifest and refuses directions that were not present during training.

### Only new language rows are trained

NLLB has an embedding table with more than 256,000 tokens.
Training the complete table for a few new language markers consumes several gigabytes of unnecessary optimizer state.
The training script uses PEFT selective token training so only newly added language rows and LoRA adapters receive gradients.

## Language registry

`data/languages.json` is authoritative.
Each entry has one of these statuses:

- `pretrained` means the base NLLB model already knows the language.
- `data_ready` means cleaned training data exists but no deployed model has passed evaluation.
- `fine_tuned` means the deployed model and its manifest contain the language.
- `planned` means contributions are welcome but training data is not ready.

Do not change a language to `fine_tuned` until its directions pass automatic and native-speaker evaluation.

## Local development

The project requires Python 3.11 or 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
ruff check .
```

The dependencies are pinned so the same commit does not silently acquire an incompatible future Transformers or PEFT release.

## Prepare all training data

Run preparation from any working directory.
The scripts resolve their own paths and no longer depend on the shell being at the repository root.

```powershell
python scripts/prepare_all.py data -o data/prepared --add-reverse
```

The command produces:

```text
data/prepared/train.jsonl
data/prepared/dev.jsonl
data/prepared/test.jsonl
data/prepared/dataset_manifest.json
```

Each JSONL row contains `src`, `tgt`, `src_lang`, `tgt_lang`, and `group_id`.
The manifest records row counts, target languages, grouping policy, observed training directions, and SHA-256 checksums for every split.

## Run training in Google Colab

Open `notebooks/train_colab.ipynb` with its Colab badge.
Do not run the notebook in local Jupyter or VS Code because it intentionally uses `google.colab` and Google Drive.

Before selecting `Runtime > Run all`:

1. Select a T4 GPU or a larger CUDA GPU.
2. Add a Colab secret named `GITHUB_TOKEN`.
3. Give the fine-grained token read-only Contents access to this private repository.
4. Enable notebook access for the secret.

The notebook performs these checks in order:

1. It proves that the runtime is Colab and mounts Drive.
2. It clones or fast-forwards the repository without saving the token in Git configuration.
3. It installs the pinned compatibility set and removes the unused conflicting `torchao` package.
4. It verifies CUDA and at least 14 GiB of GPU memory.
5. It regenerates leakage-safe bidirectional data.
6. It validates target coverage and split group disjointness.
7. It runs a two-step end-to-end smoke training.
8. It starts or safely resumes the full Drive-backed training run.

Unknown NLLB language codes are detected from the dataset automatically.
The notebook does not maintain a duplicate hardcoded language list.

## Manual training command

The equivalent full command is:

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

The physical batch is deliberately small enough for a T4.
Gradient accumulation retains an effective batch size of 64.
Gradient checkpointing further reduces activation memory.
One epoch already processes more than 3.3 million prepared rows.
Run additional epochs only when held-out results show continued improvement.

Checkpoint resume is automatic only when the dataset sizes, languages, directions, base model, and important hyperparameters match `run_config.json`.
An incomplete checkpoint is ignored.
An incompatible checkpoint causes a clear error instead of silently corrupting a run.

## Evaluate before deployment

Convert the merged model:

```powershell
ct2-transformers-converter `
  --model model_out/merged `
  --output_dir model_ct2 `
  --quantization int8 `
  --force
```

Run held-out evaluation:

```powershell
python scripts/evaluate.py `
  --test data/prepared/test.jsonl `
  --ct2 model_ct2 `
  --tokenizer model_out/merged
```

Review every direction independently.
Compare the fine-tuned result with the original NLLB model and ask native speakers to evaluate unseen samples.
Only then change approved registry entries from `data_ready` to `fine_tuned`.

## Run the API

```powershell
$env:API_KEY = "replace-with-a-secret"
$env:MODEL_DIR = "model_ct2"
$env:TOKENIZER_DIR = "model_out/merged"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

Example translation for an enabled direction:

```powershell
curl.exe -X POST http://localhost:8000/translate `
  -H "Content-Type: application/json" `
  -H "X-API-Key: replace-with-a-secret" `
  -d '{"text":"Good morning","src_lang":"eng_Latn","tgt_lang":"fra_Latn"}'
```

Example contribution for a planned language:

```powershell
curl.exe -X POST http://localhost:8000/contribute `
  -H "Content-Type: application/json" `
  -H "X-API-Key: replace-with-a-secret" `
  -d '{"src":"Good morning","tgt":"Idi na mbua","src_lang":"eng_Latn","tgt_lang":"dua_Latn","contributor":"steco"}'
```

## Docker

The image includes application code and the language registry but not model weights.
Run it with read-only model mounts and a writable contribution directory.

```powershell
docker build -t cameroon-translate .
docker run --rm -p 8000:8000 `
  -e API_KEY=replace-with-a-secret `
  -v ${PWD}/model_ct2:/srv/model_ct2:ro `
  -v ${PWD}/model_out/merged:/srv/model_out/merged:ro `
  -v ${PWD}/data:/srv/data `
  cameroon-translate
```

The container runs as an unprivileged user.
Its health check calls `/ready`, so an absent or invalid model keeps the container unhealthy.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model_ct2` | CTranslate2 model directory |
| `TOKENIZER_DIR` | `model_out/merged` | Merged tokenizer and training manifest directory |
| `API_KEY` | unset | Optional `X-API-Key` value |
| `DATA_DIR` | `data` | Writable directory for reviewed contributions |

## License

NLLB-200 weights use a non-commercial license.
Use this model only where that license is acceptable, or replace the base model before commercial deployment.
