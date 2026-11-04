# Cameroon Languages Translation API

This project fine-tunes NLLB-200 for Cameroonian languages and serves the result through a FastAPI and CTranslate2 API.
It also collects reviewed community translation pairs for later training runs.

The repository contains source data and training code, but it does not contain trained model weights.
The API defaults to a placeholder runtime that starts without model files and returns an intentional empty translation.
Set `TRANSLATION_MODE=model` after `model_out/merged` and `model_ct2` have been produced or mounted.

## What the application provides

- `POST /translate` uses one stable request and response contract in placeholder and model modes.
- `POST /contribute` accepts sentence pairs only when a contribution API key is configured and supplied.
- `GET /languages` lists languages currently enabled for translation.
- `GET /language-catalog` lists public registry metadata and honest readiness states.
- `GET /contribution-languages` lists all languages accepted by the data collection pipeline.
- `GET /directions` lists the exact source and target pairs supported by the deployed model.
- `GET /health` reports process health and model-loading state.
- `GET /ready` reports the configured runtime as ready and returns HTTP 503 only when model mode cannot serve inference.

## Application structure

- `app/application.py` assembles FastAPI middleware, routes, and dependencies.
- `app/config.py` validates immutable runtime settings and secrets.
- `app/environment.py` loads only allowlisted values from the mounted settings file.
- `app/registry.py` owns validated language metadata and supported pairs.
- `app/translation.py` owns model loading, readiness, and thread-safe inference.
- `app/contributions.py` owns privacy-aware append-only contribution storage.
- `app/security.py` owns API-key guards, rate limiting, body limits, and response headers.
- `app/schemas.py` owns strict request and response contracts.
- `scripts/` contains administrator seeding and offline preparation, training, conversion, and evaluation tools.

Production serving dependencies, development tools, and training dependencies are intentionally separated into `requirements.txt`, `requirements-dev.txt`, and `requirements-train.txt`.
Training libraries are not installed in the API image unless they are also required for local model inference.

The placeholder response is:

```json
{
  "translation": "",
  "src_lang": "eng_Latn",
  "tgt_lang": "ewo_Latn",
  "status": "model_pending"
}
```

Model mode uses the same shape with a populated `translation` and `status` set to `translated`.

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

## Train on a Windows PC

Local training is verified on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB of VRAM.
Create the pinned CUDA environment and run the required smoke gate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_local.ps1 -Rebuild
.\scripts\train_local.ps1 -Mode smoke
```

Then run a direction-balanced pilot for one target:

```powershell
.\scripts\train_local.ps1 `
  -Mode pilot `
  -TargetLanguage ewo_Latn `
  -SamplesPerDirection 500
```

The setup requires Python 3.11 or 3.12, `uv`, a recent NVIDIA driver, and PowerShell.
The execution-policy change applies only to the current PowerShell window.
See [Local Windows Training Guide](docs/LOCAL_TRAINING_GUIDE.md) for modes, measured VRAM, checkpoint recovery, full-corpus training, conversion, and evaluation.

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

Run the placeholder API during application development:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The placeholder runtime exposes every registered language to the frontend and does not load model files.

Run the trained model runtime after model artifacts are available:

```powershell
$env:TRANSLATION_API_KEY = "use-a-random-secret-with-at-least-32-characters"
$env:TRANSLATION_MODE = "model"
$env:MODEL_DIR = "model_ct2"
$env:TOKENIZER_DIR = "model_out/merged"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

Example translation for an enabled direction:

```powershell
curl.exe -X POST http://localhost:8000/translate `
  -H "Content-Type: application/json" `
  -H "X-API-Key: use-a-random-secret-with-at-least-32-characters" `
  -d '{"text":"Good morning","src_lang":"eng_Latn","tgt_lang":"fra_Latn"}'
```

Example contribution for a planned language:

```powershell
curl.exe -X POST http://localhost:8000/contribute `
  -H "Content-Type: application/json" `
  -H "X-API-Key: use-a-random-secret-with-at-least-32-characters" `
  -d '{"src":"Good morning","tgt":"Idi na mbua","src_lang":"eng_Latn","tgt_lang":"dua_Latn","contributor":"steco"}'
```

## Docker

Docker Compose builds and runs the Vue frontend, Nginx proxy and FastAPI backend together.
The backend seeds one administrator from `jsc_api_backend/.env` before it starts accepting requests.
Compose mounts that file read-only and does not copy password or API-key values into the container configuration environment.

Create the private environment file:

```powershell
Copy-Item .env.example .env
```

Replace every admin placeholder in `.env` with the real administrator information.
The seed rejects example email domains, weak passwords and unchanged placeholder passwords.

Start the complete application from this backend directory:

```powershell
docker compose up --build
```

When both services are healthy, Docker prints the frontend, backend API and API documentation links in the startup output.
The default links are `http://localhost:8080`, `http://localhost:8001` and `http://localhost:8001/docs`.
When starting in detached mode, use `docker compose logs ready` to display the links.
Set `JSC_FRONTEND_PORT` or `JSC_BACKEND_PORT` before running Compose when those host ports are unavailable.

The `jsc-data` named volume stores `jsc.sqlite3` and contributed translation pairs across container restarts.
The admin seed is idempotent, so restarting does not create duplicate administrators.
Changing admin details or the password in `.env` updates the existing row with the same normalized email.
Passwords are stored only as salted PBKDF2 hashes.

Run the seed manually when needed:

```powershell
docker compose run --rm backend python -m scripts.seed_admin
```

The seed prepares the administrator record for the future authentication layer.
This project does not yet expose an admin login endpoint or administration interface.

The backend image includes application code and the language registry but not model weights.
Add model-mode settings and read-only model mounts to `compose.yaml` after trained artifacts are available.

Both containers use read-only root filesystems, drop every Linux capability, set `no-new-privileges`, and run as unprivileged users.
Published ports bind to `127.0.0.1` by default.
Its health check calls `/ready`, so an absent or invalid model keeps the container unhealthy.

Review [SECURITY.md](SECURITY.md) before any public deployment.

## Render development deployment

The repository includes `render.yaml`, which creates the backend and frontend as separate Docker web services in Frankfurt.
Both services deploy from their GitHub `dev` branches only after their repository CI check passes.
The frontend proxies `/api` to the backend over Render's private network, while the browser keeps using a same-origin API path.

The backend is deliberately pinned to `TRANSLATION_MODE=placeholder` in the Blueprint.
It starts without model artifacts and returns `status: "model_pending"` until training and evaluation are complete.
Do not switch the Render service to model mode until approved model and tokenizer artifacts are mounted and every required language direction has passed evaluation.

See [Render Development Deployment](docs/RENDER_DEPLOYMENT.md) for setup, verification, and rollback instructions.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model_ct2` | CTranslate2 model directory |
| `TOKENIZER_DIR` | `model_out/merged` | Merged tokenizer and training manifest directory |
| `APP_ENV` | `development` | Runtime policy profile; production tightens defaults |
| `ALLOWED_HOSTS` | local and internal hosts | Explicit Host header allowlist |
| `TRANSLATION_API_KEY` | unset | Optional `X-API-Key` for translation requests |
| `CONTRIBUTION_API_KEY` | unset | Required to enable contribution writes |
| `API_KEY` | unset | Legacy fallback for both endpoint keys |
| `DATA_DIR` | `data` | Writable directory for reviewed contributions |
| `DATABASE_PATH` | `<DATA_DIR>/jsc.sqlite3` | SQLite database containing the seeded administrator |
| `LANGUAGES_FILE` | repository registry | Language registry location |
| `TRANSLATION_MODE` | `placeholder` | Use `placeholder` without weights or `model` for trained inference |
| `CORS_ORIGINS` | local Vite origins | Comma-separated browser origins allowed to call the API directly |
| `ENABLE_API_DOCS` | off in production | Expose interactive OpenAPI documentation |
| `EXPOSE_HEALTH_DETAILS` | off in production | Include mode and model details in health responses |
| `MAX_REQUEST_BODY_BYTES` | `16384` | Maximum API request body size |
| `STORE_CONTRIBUTOR_NAMES` | `false` | Store contributor names as personal data |
| `ADMIN_EMAIL` | required | Normalized administrator email address |
| `ADMIN_PASSWORD` | required | Strong administrator password that is hashed before storage |
| `ADMIN_FIRST_NAME` | required | Administrator first name |
| `ADMIN_LAST_NAME` | required | Administrator last name |
| `ADMIN_PHONE` | unset | Optional administrator phone number |
| `ADMIN_JOB_TITLE` | unset | Optional administrator job title |
| `ADMIN_ORGANIZATION` | unset | Optional administrator organization |

## License

NLLB-200 weights use a non-commercial license.
Use this model only where that license is acceptable, or replace the base model before commercial deployment.
