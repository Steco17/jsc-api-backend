# Tasks Before the API Is Ready for Use

Work through these in order.
Check off as you go.

## 0. Your Cameroonian-languages source data

- [ ] Add your JSON file to `data/` (e.g. `data/cameroon_languages.json`)
- [ ] Confirm its structure: is it word/phrase-level (a dictionary) or full parallel sentences? This determines how much rework is needed before training.
- [ ] List every distinct Cameroonian language it contains
- [ ] For each language, decide its NLLB language code: `fuv_Latn` is Fulani, a different language NLLB happens to know natively - not this project's Fulfulde. This project's Fulfulde is the Adamawa variety, coded `fub_Latn`, which NLLB does not know; it and every other Cameroonian language (like `ewo_Latn` for Ewondo) are new tokens added via `--new-langs`
- [ ] Write a converter script (`scripts/convert_json_to_pairs.py`) that turns the JSON into the `{"src", "tgt", "src_lang", "tgt_lang"}` JSONL format `scripts/prepare_data.py` expects
- [ ] If the source is word/phrase pairs rather than sentences, decide whether to: (a) train a lexicon-augmented model anyway, (b) generate template sentences around the entries, or (c) treat it as a supplementary dictionary and prioritize sourcing real sentence-level data separately — quality depends heavily on which

## 1. Data

- [ ] Run the converter from step 0, producing raw JSONL
- [ ] Augment with public corpora: JW300/Bible (Ewondo, Fulfulde `fub_Latn`, and other Cameroonian languages; Duala out of scope for now), MAFAND-MT, masakhane.io datasets
- [ ] FLORES-200 has no entries for this project's Fulfulde (`fub_Latn`, Adamawa) or any other Cameroonian language here - it's not usable as an extra test set for this project
- [ ] Run `scripts/prepare_data.py ... --add-reverse` → train/dev/test splits
- [ ] Verify counts: aim for ≥5-10k pairs per language (50k+ for good quality)
- [ ] Spot-check 50 random pairs per language with a native speaker for alignment errors

## 2. Training (Google Colab GPU)

- [ ] Create a Colab notebook with GPU runtime; `pip install -r requirements-train.txt`
- [ ] Upload `scripts/finetune.py` and `data/prepared/`
- [ ] Run fine-tuning with `--new-langs` listing every language from step 0 that NLLB doesn't already know; watch dev loss decrease
- [ ] Download `model_out/merged/` to your machine
- [ ] Back up the weights (Git LFS or private Hugging Face repo)

## 3. Evaluation

- [ ] Convert model: `ct2-transformers-converter --model model_out/merged --output_dir model_ct2 --quantization int8`
- [ ] Run `scripts/evaluate.py` — record BLEU/chrF++ per direction, for every language pair
- [ ] Also evaluate the *base* NLLB model — confirm your fine-tune beats it
- [ ] Human evaluation: native speakers rate 50-100 outputs per language
- [ ] Quality gate: chrF++ > 40 in every direction you plan to expose (drop/hide weak directions)

## 4. Make the API support every language generically

- [ ] Replace the hardcoded `LANGUAGES` dict in `app/main.py` with one loaded from a config/data file, so the set of supported languages comes from what was actually trained rather than being hand-edited in code
- [ ] Confirm `GET /languages` reflects the full list after each retrain
- [ ] Since `POST /translate` already takes `src_lang`/`tgt_lang` generically, any-to-any translation among supported languages works automatically once the `LANGUAGES` map is complete — no endpoint logic changes needed
- [ ] Add a startup check that fails loudly if a language in the config has no matching token in the loaded tokenizer (catches config/model drift)

## 5. API hardening

- [ ] Set `API_KEY` and test rejection of unauthenticated requests
- [ ] Add rate limiting (e.g. `slowapi`) to prevent abuse
- [ ] Add CORS middleware if a web frontend will call the API
- [ ] Add request logging (latency, language pair, status) — no storing of user text unless consented
- [ ] Load test on your target machine: measure requests/sec and p95 latency; batch requests if needed
- [ ] Write basic tests (pytest + httpx): `/health`, `/languages`, `/translate` happy path + bad language code, `/contribute`
- [ ] Add input sanitation review for `/contribute` (length caps exist; consider profanity/spam filter)

## 6. Deployment

- [ ] Build Docker image; run locally with mounted model volumes
- [ ] Provision a CPU VPS with >= 8 GB RAM
- [ ] Deploy container; set env vars; mount persistent volume for `data/contributed.jsonl`
- [ ] Put behind HTTPS (Caddy/Nginx + Let's Encrypt)
- [ ] Set up uptime monitoring and disk-space alerts
- [ ] Backup strategy for `contributed.jsonl` (it's future training gold)

## 7. Post-launch loop (recurring)

- [ ] Review contributed pairs weekly; approve/reject
- [ ] Merge approved pairs into training data monthly
- [ ] Re-fine-tune, re-evaluate (must beat previous scores), reconvert, redeploy
- [ ] Track score history per release, per language pair

## 8. Legal / licensing

- [ ] Confirm usage is non-commercial (NLLB weights are CC-BY-NC) or switch base model
- [ ] Add terms for contributors (their submissions may be used for training)
- [ ] Confirm you have rights to use the content of your Cameroonian-languages JSON file for training (e.g. if scraped or from a third party)
