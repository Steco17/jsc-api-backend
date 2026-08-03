# Cameroon Languages Translation API

A REST API that translates between **French / English** and **Cameroonian languages** (29 languages with data ready, including Ewondo, Fulfulde, and Ghomala — see below), powered by a model **you own and fine-tuned yourself** (NLLB-200 + LoRA). It also **collects new sentence pairs** from users to continuously improve the model.

## Features

- `POST /translate` — translate text between any supported language pair
- `POST /contribute` — community members submit new sentence pairs (your data-collection pipeline)
- `GET /languages` — list supported languages
- `GET /health` — service/model status
- CPU-friendly serving via **CTranslate2 int8** (no GPU needed in production)
- Optional API-key auth (`API_KEY` env var)

## Supported languages

The authoritative list is `data/languages.json` (code -> name + status:
`pretrained` = native to NLLB-200, `data_ready` = cleaned training data on
hand but not trained yet, `fine_tuned` = trained and live in the deployed
model, `planned` = targeted but no data yet). `app/main.py` loads its
`LANGUAGES` dict from this file, exposing only `pretrained`/`fine_tuned`
entries. Currently live (usable today, no training needed):

| Code | Language |
|---|---|
| `fra_Latn` | French |
| `eng_Latn` | English |
| `fuv_Latn` | Fulani |

No language has been fine-tuned into a deployed model yet. 29 Cameroonian
languages — including Ewondo (`ewo_Latn`), Fulfulde (`fub_Latn`), and Ghomala
(`bbj_Latn`) — have cleaned data ready in `data/prepared/` awaiting the first
training run (see `docs/TRAINING_GUIDE.md`). Duala (`dua_Latn`) is
deliberately out of scope for now. Flip a language's status to `"fine_tuned"`
in `data/languages.json` once it's trained and merged into the deployed
model — no other file needs editing.

## Project structure

```
cameroon-translation-api/
├── README.md               # this file
├── ROADMAP.md              # full step-by-step build plan
├── TASKS.md                # checklist before the API is production-ready
├── requirements.txt        # serving dependencies (CPU)
├── requirements-train.txt  # training dependencies (GPU / Colab)
├── Dockerfile
├── data/                   # training data (JSONL) + contributed.jsonl
├── scripts/
│   ├── prepare_data.py     # clean, dedupe, split data
│   ├── finetune.py         # fine-tune NLLB-200 with LoRA (GPU)
│   └── evaluate.py         # BLEU / chrF++ scores
└── app/
    └── main.py             # FastAPI service
```

## Quick start

### 1. Prepare data (CPU, local)

Put your parallel data in `data/` as JSONL (`{"src", "tgt", "src_lang", "tgt_lang"}`) or TSV, then:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py data/*.jsonl -o data/prepared --add-reverse
```

### 2. Fine-tune (GPU — free Google Colab works)

Upload `scripts/finetune.py` + `data/prepared/` to Colab (GPU runtime):

```bash
pip install -r requirements-train.txt
python scripts/finetune.py --train data/prepared/train.jsonl \
    --dev data/prepared/dev.jsonl --new-langs dua_Latn --out model_out
```

Download `model_out/merged/` when done — these weights are yours.

### 3. Convert for CPU serving

```bash
ct2-transformers-converter --model model_out/merged \
    --output_dir model_ct2 --quantization int8
```

### 4. Evaluate

```bash
python scripts/evaluate.py --test data/prepared/test.jsonl \
    --ct2 model_ct2 --tokenizer model_out/merged
```

### 5. Run the API

```bash
export API_KEY=change-me          # optional but recommended
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

### Example requests

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" -H "X-API-Key: change-me" \
  -d '{"text": "Bonjour, comment allez-vous ?", "src_lang": "fra_Latn", "tgt_lang": "ewo_Latn"}'

curl -X POST http://localhost:8000/contribute \
  -H "Content-Type: application/json" -H "X-API-Key: change-me" \
  -d '{"src": "Good morning", "tgt": "Idi na mbua", "src_lang": "eng_Latn", "tgt_lang": "dua_Latn", "contributor": "steco"}'
```

## Docker

```bash
docker build -t cameroon-translate .
docker run -p 8000:8000 -e API_KEY=change-me \
  -v $(pwd)/model_ct2:/srv/model_ct2 \
  -v $(pwd)/model_out/merged:/srv/model_out/merged \
  -v $(pwd)/data:/srv/data \
  cameroon-translate
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model_ct2` | CTranslate2 model path |
| `TOKENIZER_DIR` | `model_out/merged` | HF tokenizer path |
| `API_KEY` | unset | If set, requests need `X-API-Key` header |
| `DATA_DIR` | `data` | Where contributed pairs are appended |

## Improvement loop

Contributed pairs (`data/contributed.jsonl`) → human review → merge into training data → re-run fine-tuning → reconvert → redeploy. Repeat monthly.

## Docs

- `ROADMAP.md` — the full build plan, phase by phase
- `TASKS.md` — what remains before production
- `CODE_DOCUMENTATION.docx` — every function explained for developers

## License note

NLLB-200 weights are CC-BY-NC (non-commercial). Fine for research/community use; for a commercial product, switch the base model (e.g. MADLAD-400) or obtain licensing.
