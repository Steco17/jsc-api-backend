# Cameroon Languages Translation API — Roadmap

Goal: an API you fully own that translates French/English <-> Cameroonian languages (29 languages with data ready, including Ewondo and Fulfulde; Duala deliberately deferred), powered by a model you fine-tuned yourself.

**Key decision — base model:** don't train from scratch (needs millions of sentence pairs + weeks of GPU). Fine-tune **NLLB-200-distilled-600M** (Meta, open weights, CC-BY-NC — use MADLAD-400 or train-from-scratch later if you need commercial licensing). NLLB already knows French and English, and Fulani (`fuv_Latn`) — a different language from this project's Fulfulde. This project's Fulfulde is the Adamawa variety (`fub_Latn`), which NLLB does not know. Ewondo, Fulfulde, and every other Cameroonian language in this project are added as new language tokens via `--new-langs`. You download the weights, fine-tune them, and keep them — total control.

**Key constraint — you have CPU only:**
- Fine-tuning a 600M model on CPU is impractical. Use **free Google Colab / Kaggle GPU** (T4) to fine-tune, then download the weights. The model is still 100% yours.
- Serving on CPU is fine: convert to **CTranslate2 int8** → ~4x smaller, fast CPU inference.

---

## Phase 1 — Data (week 1–2)

1. Inventory your existing parallel data. Target format: one JSONL row per pair:
   `{"src": "Bonjour, comment vas-tu ?", "tgt": "Mbolo, one mvoe?", "src_lang": "fra_Latn", "tgt_lang": "ewo_Latn"}`
2. Augment with public corpora:
   - JW300 / Bible corpora (Ewondo, Fulfulde and 27 other Cameroonian languages available; Duala not currently in scope)
   - FLORES-200 has no entries for any of this project's Cameroonian languages (including Fulfulde, which is the Adamawa variety `fub_Latn` here, not FLORES-200's Nigerian `fuv_Latn`) — not usable as an extra test set for this project
   - MAFAND-MT and masakhane.io community datasets
3. Clean: run `scripts/prepare_data.py` — dedupes, drops empty/too-long pairs, normalizes unicode, splits train/dev/test (95/2.5/2.5).
4. Add reverse-direction rows (tgt→src) so one model translates both ways.
5. Minimum viable: ~5–10k pairs per language for a noticeable fine-tune; 50k+ for good quality.

## Phase 2 — Fine-tuning (week 2–4)

1. Open `scripts/finetune.py` in Colab (GPU runtime), upload your prepared JSONL files.
2. It loads `facebook/nllb-200-distilled-600M`, adds any new language tokens (e.g. `dua_Latn`) and resizes embeddings.
3. Uses LoRA (peft) — trains ~1% of parameters, fits on a free T4, merges back into full weights at the end.
4. Starting hyperparameters: lr 1e-4, effective batch 64 (16 x grad-accum 4), 3–5 epochs, early stopping on dev loss.
5. Download `model_out/merged/` — these weights are yours. Version them (Git LFS or private HF repo).

## Phase 3 — Evaluation (ongoing)

1. `scripts/evaluate.py` computes **chrF++ and BLEU** per direction on the held-out test set (sacrebleu).
2. Compare against base NLLB as the baseline; keep a results log per run.
3. Human check: native speakers rate 50–100 random outputs per language. chrF > 40 is usable; > 50 is decent.

## Phase 4 — CPU-optimized serving

1. Convert merged model:
   `ct2-transformers-converter --model model_out/merged --output_dir model_ct2 --quantization int8`
2. `app/main.py` (FastAPI) loads the CT2 model once at startup and serves:
   - `POST /translate` — {text, src_lang, tgt_lang} → translation
   - `POST /contribute` — accepts new sentence pairs from users (your data-collection pipeline; rows accumulate in `data/contributed.jsonl` and feed the next fine-tune)
   - `GET /languages`, `GET /health`
3. Run locally: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Phase 5 — Deployment & iteration loop

1. Dockerize (Dockerfile included) → deploy on any CPU VPS (8GB RAM handles the int8 600M model; ~€5–10/mo on Hetzner/Contabo).
2. Set the `API_KEY` env var before going public (auth is stubbed in `app/main.py`).
3. **Flywheel:** contributed pairs → human review → merge into training data → re-fine-tune monthly → reconvert → redeploy. This loop is how quality grows for low-resource languages.

---

## Project layout

```
cameroon-translation-api/
├── ROADMAP.md
├── requirements.txt          # serving deps (CPU)
├── requirements-train.txt    # training deps (Colab GPU)
├── Dockerfile
├── data/                     # your JSONL data goes here
├── scripts/
│   ├── prepare_data.py       # clean + split
│   ├── finetune.py           # NLLB + LoRA fine-tune (run on GPU)
│   └── evaluate.py           # BLEU / chrF++
└── app/
    └── main.py               # FastAPI service
```

## Language codes

| Language | Code | In NLLB already? |
|---|---|---|
| French | fra_Latn | yes |
| English | eng_Latn | yes |
| Fulani | fuv_Latn | yes — a different language from this project's Fulfulde; not currently in this project's training data |
| Fulfulde (Adamawa) | fub_Latn | no — added as new token by finetune.py |
| Ewondo | ewo_Latn | no — added as new token by finetune.py |
| Duala | dua_Latn | not currently in scope — no data collected yet |

Full list of all languages this project targets, with data-readiness status, lives in `data/languages.json`.
