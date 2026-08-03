"""Cameroon Languages Translation API - FastAPI service.

ARCHITECTURE
------------
- The fine-tuned model (see scripts/finetune.py) is converted to CTranslate2
  int8 format, which is ~4x smaller and fast on plain CPUs - no GPU needed
  in production.
- The model and tokenizer are loaded ONCE at startup and kept in memory;
  every request reuses them (loading per-request would take seconds).
- /contribute implements the data-collection side of the project: users
  submit new sentence pairs which are appended to data/contributed.jsonl
  and, after human review, feed the next fine-tuning round.

CONFIGURATION (environment variables)
-------------------------------------
  MODEL_DIR      path to the CTranslate2 model      (default: model_ct2)
  TOKENIZER_DIR  path to the HF tokenizer           (default: model_out/merged)
  API_KEY        if set, requests must send header  X-API-Key: <key>
  DATA_DIR       where contributed pairs are stored (default: data)

RUN
---
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  Interactive docs: http://localhost:8000/docs
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration - read once at import time.
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "model_ct2")
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "model_out/merged")
API_KEY = os.getenv("API_KEY")            # None => auth disabled (dev mode)
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Supported languages: loaded from data/languages.json, which tracks every
# language this project targets. Only "pretrained" (native to NLLB-200) and
# "fine_tuned" (already trained into the deployed model) entries are exposed
# here - "data_ready" (cleaned data on hand, not trained yet) and "planned"
# (no data yet) languages exist in the registry for the training pipeline to
# reference but aren't usable until their status flips to "fine_tuned".
LANGUAGES_FILE = Path(__file__).resolve().parent.parent / "data" / "languages.json"
with open(LANGUAGES_FILE, encoding="utf-8") as _f:
    _REGISTRY = json.load(_f)
LANGUAGES = {code: info["name"] for code, info in _REGISTRY.items()
             if info["status"] in ("pretrained", "fine_tuned")}

app = FastAPI(title="Cameroon Languages Translation API", version="0.1.0")

# Globals populated at startup. Module-level so all requests share them.
_translator = None       # ctranslate2.Translator instance
_tokenizer = None        # HuggingFace tokenizer
_write_lock = threading.Lock()   # serializes appends to contributed.jsonl
                                 # (uvicorn may run multiple threads)


def check_key(x_api_key: str | None = Header(default=None)):
    """FastAPI dependency: reject the request if API_KEY is set and the
    client did not send a matching X-API-Key header.

    If the API_KEY env var is unset, auth is disabled (useful during
    development) - set it before exposing the API publicly.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


@app.on_event("startup")
def load_model():
    """Load model + tokenizer once when the server starts.

    Imports are done here (not at module top) so the app module can be
    imported for testing without the heavy ML dependencies installed.
    """
    global _translator, _tokenizer
    import ctranslate2
    from transformers import AutoTokenizer

    _translator = ctranslate2.Translator(MODEL_DIR, device="cpu",
                                         compute_type="int8")
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request schemas (pydantic validates types and length limits for us).
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000,
                      description="Text to translate")
    src_lang: str                       # e.g. "fra_Latn"
    tgt_lang: str                       # e.g. "ewo_Latn"


class ContributeRequest(BaseModel):
    src: str = Field(min_length=1, max_length=2000)
    tgt: str = Field(min_length=1, max_length=2000)
    src_lang: str
    tgt_lang: str
    contributor: str | None = None      # optional name/handle for credit


def validate_langs(src_lang: str, tgt_lang: str):
    """Shared validation: both codes must be supported and different."""
    for code in (src_lang, tgt_lang):
        if code not in LANGUAGES:
            raise HTTPException(422, f"Unknown language code '{code}'. "
                                     f"Supported: {list(LANGUAGES)}")
    if src_lang == tgt_lang:
        raise HTTPException(422, "src_lang and tgt_lang must differ")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe for monitoring/load balancers."""
    return {"status": "ok", "model_loaded": _translator is not None}


@app.get("/languages")
def languages():
    """List supported language codes and names (for client dropdowns)."""
    return LANGUAGES


@app.post("/translate", dependencies=[Depends(check_key)])
def translate(req: TranslateRequest):
    """Translate text between two supported languages.

    Pipeline:
      1. Tokenize the source text. Setting tokenizer.src_lang first makes
         the tokenizer prepend the source-language token (NLLB convention).
      2. translate_batch with target_prefix=[[tgt_lang]] forces the decoder
         to begin with the target-language token - that is how the model
         knows which language to produce.
      3. Strip that language token from the output and decode back to text.
    """
    validate_langs(req.src_lang, req.tgt_lang)

    _tokenizer.src_lang = req.src_lang
    tokens = _tokenizer.convert_ids_to_tokens(_tokenizer.encode(req.text))

    results = _translator.translate_batch(
        [tokens],
        target_prefix=[[req.tgt_lang]],
        beam_size=4,               # quality/speed trade-off (1 = greedy/fastest)
        max_decoding_length=512,
    )

    out_tokens = results[0].hypotheses[0]
    if out_tokens and out_tokens[0] == req.tgt_lang:
        out_tokens = out_tokens[1:]        # drop the language marker token

    text = _tokenizer.decode(
        _tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)

    return {"translation": text, "src_lang": req.src_lang,
            "tgt_lang": req.tgt_lang}


@app.post("/contribute", dependencies=[Depends(check_key)])
def contribute(req: ContributeRequest):
    """Collect a new sentence pair from the community.

    Rows are appended to data/contributed.jsonl with a UTC timestamp.
    They are NOT used automatically - review them (quality/spam) and merge
    the good ones into the training data for the next fine-tuning round.
    The lock prevents interleaved writes when multiple requests arrive
    at the same time.
    """
    validate_langs(req.src_lang, req.tgt_lang)

    row = req.model_dump()
    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    with _write_lock, open(DATA_DIR / "contributed.jsonl", "a",
                           encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"status": "saved", "message": "Merci / Thank you! Your pair will "
            "be reviewed and used in the next training round."}
