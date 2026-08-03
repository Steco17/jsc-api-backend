# Training Guide for CSV Translation Data

This guide explains how to train the Cameroon Languages Translation API when your translations are stored in CSV files.

The complete workflow is:

```text
CSV files
  -> clean and validate translations
  -> convert to JSONL
  -> split into train, development, and test sets
  -> fine-tune the model on a GPU
  -> evaluate translation quality
  -> convert the trained model for CPU inference
  -> start the translation API
```

## 1. Structure the CSV data

The recommended format contains one translation pair per row:

```csv
src,tgt,src_lang,tgt_lang
Bonjour,Mbolo,fra_Latn,ewo_Latn
Comment allez-vous ?,O ne mvoe?,fra_Latn,ewo_Latn
Good morning,Idi na mbua,eng_Latn,dua_Latn
```

The required columns are:

- `src`: The original sentence.
- `tgt`: The correct translation.
- `src_lang`: The language code of the original sentence.
- `tgt_lang`: The language code of the translation.

`data/languages.json` is the authoritative list of every language code this
project targets, tagged `pretrained` (native to NLLB-200), `data_ready`
(cleaned data on hand, not trained yet), `fine_tuned` (trained and live in
the deployed model), or `planned` (targeted, no data yet). Pretrained,
usable without any training:

| Language | Code |
|---|---|
| French | `fra_Latn` |
| English | `eng_Latn` |
| Fulani | `fuv_Latn` |

Note `fuv_Latn` (Fulani) is a different language from this project's
Fulfulde, which is the Adamawa variety, `fub_Latn` - NLLB does not know it.

29 Cameroonian languages, including Ewondo (`ewo_Latn`), Fulfulde
(`fub_Latn`), and Ghomala (`bbj_Latn`), are `data_ready`: cleaned data
already sits in `data/prepared/`, waiting for the first training run. Duala
(`dua_Latn`) is `planned` but deliberately out of scope for now - no data
collected. Flip a language's status to `"fine_tuned"` once it's trained and
merged into the deployed model.

A separate CSV file can also be used for each language pair:

```csv
french,ewondo
Bonjour,Mbolo
Merci,Akiba
```

This simpler format must be converted and assigned language codes before training.

## 2. Check data quality

The model learns from the information in the CSV, including any mistakes.

Before training, confirm that:

- Every source sentence matches its translation.
- Spelling and punctuation are consistent.
- Every row has the correct language codes.
- Empty rows are removed.
- Duplicate translations are removed.
- Languages are not accidentally mixed within a column.
- The CSV is saved using UTF-8 encoding.
- Sentences are natural translations instead of incomplete dictionary glosses.

Ask a native speaker to review random samples from every language pair.

## 3. Collect enough data

These quantities are approximate:

- Fewer than 1,000 pairs are suitable mainly for experimentation.
- Between 5,000 and 10,000 pairs can produce a noticeable improvement.
- Between 20,000 and 50,000 pairs provide a useful starting point.
- More than 50,000 clean pairs per language provide much better potential.

Clean and correctly aligned data is more valuable than a larger collection of incorrect data.

Avoid filling the dataset with thousands of nearly identical religious, legal, or dictionary sentences unless that is the intended translation domain.

## 4. Convert CSV to JSONL and split

`scripts/prepare_data.py` reads CSV files directly with the `--csv` flag, so
conversion and splitting happen in one command.

If the CSV has `src`/`tgt` columns and no `src_lang`/`tgt_lang` columns, pass
the language codes explicitly:

```powershell
python scripts/prepare_data.py data/translations.csv `
  --csv --src-lang fra_Latn --tgt-lang ewo_Latn `
  -o data/prepared
```

If the columns are named differently, use `--src-col`/`--tgt-col`:

```powershell
python scripts/prepare_data.py data/translations.csv `
  --csv --src-col en --tgt-col bbj `
  --src-lang eng_Latn --tgt-lang bbj_Latn `
  -o data/prepared
```

If the CSV already has `src_lang`/`tgt_lang` columns, those are used per-row
and `--src-lang`/`--tgt-lang` can be omitted.

Add `--add-reverse` to teach both directions from the same command; see the
splitting caution below before using it for a serious training run.

The command creates:

```text
data/prepared/train.jsonl
data/prepared/dev.jsonl
data/prepared/test.jsonl
```

Each file has a specific purpose:

- `train.jsonl` teaches the model.
- `dev.jsonl` measures progress during training and supports early stopping.
- `test.jsonl` measures final translation quality on examples the model has not seen.

### Important splitting caution

The current preparation script creates reverse rows before randomly splitting the data.

One direction of a sentence pair can therefore appear in training while its reverse appears in testing.

This is data leakage and can make evaluation scores look better than the model's real performance.

The preparation pipeline should keep an original pair and its reverse in the same split before a serious training run.

The pipeline should also stratify splits by language pair so that smaller languages receive enough development and test examples.

## 5. Train on a GPU

Do not train this model on a normal CPU because the NLLB base model contains approximately 600 million parameters.

Suitable training environments include:

- Google Colab with a GPU runtime.
- Kaggle Notebooks with a GPU runtime.
- A rented NVIDIA GPU server.
- A local NVIDIA GPU with sufficient memory.

Install the training dependencies:

```powershell
pip install -r requirements-train.txt
```

NLLB-200 natively knows French, English, and Fulani (`fuv_Latn`) only. Every
other Cameroonian language in this project — including Ewondo and this
project's Fulfulde (`fub_Latn`, the Adamawa variety, not Fulani) — needs
`--new-langs`, listing every target-language code that appears in your
training data. For the current 29-language `data_ready` set (see
`data/languages.json`):

```powershell
python scripts/finetune.py `
  --train data/prepared/train.jsonl `
  --dev data/prepared/dev.jsonl `
  --new-langs agq_Latn ags_Latn azo_Latn bbj_Latn bbk_Latn bfd_Latn bmo_Latn bri_Latn bsq_Latn bum_Latn bwt_Latn etu_Latn ewo_Latn fub_Latn gya_Latn ksf_Latn lem_Latn lmp_Latn lns_Latn mcu_Latn mgo_Latn muy_Latn nge_Latn oku_Latn pny_Latn tui_Latn vut_Latn xmg_Latn yat_Latn `
  --out model_out `
  --epochs 4 `
  --batch 8 `
  --grad-accum 8
```

If your training data is ONLY French/English/Fulani pairs, omit
`--new-langs` entirely:

```powershell
python scripts/finetune.py `
  --train data/prepared/train.jsonl `
  --dev data/prepared/dev.jsonl `
  --out model_out `
  --epochs 4 `
  --batch 8 `
  --grad-accum 8
```

The combination of `--batch 8` and `--grad-accum 8` gives an effective batch size of 64 while reducing GPU memory usage.

Training produces the following directory:

```text
model_out/merged/
```

This directory contains the fine-tuned model and tokenizer.

## 6. Understand the training process

The fine-tuning script performs these operations:

1. It downloads `facebook/nllb-200-distilled-600M`.
2. It loads the training and development datasets.
3. It adds one new, randomly-initialized language token per code passed to
   `--new-langs` and resizes the embedding matrix to fit them.
4. It attaches LoRA adapters to the model.
5. It trains the adapters using the translation pairs.
6. It monitors development loss during training.
7. It stops early if development loss stops improving.
8. It merges the trained adapters into the base model.
9. It saves the result in `model_out/merged`.

LoRA reduces the number of parameters that must be trained, which makes fine-tuning practical on a smaller GPU.

## 7. Convert the model for CPU inference

Convert the merged model to an int8 CTranslate2 model:

```powershell
ct2-transformers-converter `
  --model model_out/merged `
  --output_dir model_ct2 `
  --quantization int8
```

The API loads the converted model from `model_ct2` and uses the tokenizer from `model_out/merged`.

## 8. Evaluate the model

Run the evaluation script:

```powershell
python scripts/evaluate.py `
  --test data/prepared/test.jsonl `
  --ct2 model_ct2 `
  --tokenizer model_out/merged
```

Example output:

```text
fra_Latn -> ewo_Latn (250 rows): BLEU 22.4 | chrF++ 43.8
ewo_Latn -> fra_Latn (250 rows): BLEU 27.1 | chrF++ 49.2
```

Use chrF++ as the main automatic metric for these low-resource languages.

Automatic scores are not sufficient by themselves.

Native speakers should also evaluate at least 50 to 100 translations for every direction that will be made available to users.

Compare the fine-tuned model with the original NLLB model to confirm that training produced a real improvement.

## 9. Run the API

After converting the model, start the service:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at:

```text
http://localhost:8000/docs
```

A translation request looks like this:

```json
{
  "text": "Bonjour, comment allez-vous ?",
  "src_lang": "fra_Latn",
  "tgt_lang": "ewo_Latn"
}
```

## 10. Recommended order of work

Before spending time or money on a full GPU training run:

1. Fix reverse-pair leakage between training and testing.
2. Add splitting that preserves representation for each language pair.
3. Import a small CSV dataset.
4. Run the complete pipeline as a smoke test.
5. Review the resulting translations with native speakers.
6. Run full training only after the smoke test succeeds.

## Licensing note

NLLB-200 uses a non-commercial license.

It is suitable for research and community use, but a commercial service will require a commercially compatible base model or separate licensing permission.
