# How to Know Whether Training Went Well

Successful training cannot be determined from training loss alone.

Use validation loss, test metrics, comparison with the original model, and human evaluation together.

## 1. Check validation loss

During training, look for log entries similar to:

```text
loss: 2.41
eval_loss: 2.18
```

A healthy training run normally shows both training loss and validation loss decreasing:

```text
Training loss:   3.2 -> 2.4 -> 1.8
Validation loss: 3.3 -> 2.6 -> 2.2
```

This indicates that the model is learning patterns that also work on data excluded from training.

### Warning signs

Investigate the run if:

- Training loss decreases while validation loss increases.
- Validation loss never improves.
- Loss becomes `NaN` or infinite.
- Training stops unexpectedly.
- Loss remains almost unchanged.
- Validation loss varies dramatically between evaluations.

When training loss improves while validation loss gets worse, the model is probably overfitting.

Possible responses include using fewer epochs, adding more clean data, increasing regularization, or lowering the learning rate.

## 2. Measure quality on the test set

Convert the trained model to CTranslate2 before running the existing evaluation script.

```powershell
ct2-transformers-converter `
  --model model_out/merged `
  --output_dir model_ct2 `
  --quantization int8
```

Run the evaluation:

```powershell
python scripts/evaluate.py `
  --test data/prepared/test.jsonl `
  --ct2 model_ct2 `
  --tokenizer model_out/merged
```

The script reports BLEU and chrF++ for every translation direction.

Example output:

```text
fra_Latn -> ewo_Latn (250 rows): BLEU 22.4 | chrF++ 43.8
ewo_Latn -> fra_Latn (250 rows): BLEU 27.1 | chrF++ 49.2
```

Use chrF++ as the primary automatic metric for these low-resource languages.

### Rough chrF++ interpretation

| chrF++ | Rough interpretation |
|---:|---|
| Below 20 | Usually poor |
| 20 to 30 | Limited quality |
| 30 to 40 | Understandable in some cases |
| 40 to 50 | Potentially usable |
| Above 50 | Promising |

These ranges are guidelines rather than universal quality guarantees.

Scores depend on the language, writing system, dataset, sentence length, and translation domain.

## 3. Evaluate every direction separately

Quality is not normally equal in both directions.

Review every direction that the API will expose:

```text
fra_Latn -> ewo_Latn
ewo_Latn -> fra_Latn
eng_Latn -> dua_Latn
dua_Latn -> eng_Latn
```

Do not approve a language pair only because the average score across all directions is acceptable.

Each exposed direction should pass its own quality checks.

## 4. Compare with the original NLLB model

Evaluate the original NLLB model and the fine-tuned model using exactly the same clean test set.

Training was beneficial only if the fine-tuned model consistently performs better.

Example:

| Model | French to Ewondo chrF++ |
|---|---:|
| Original NLLB | 35.2 |
| Fine-tuned model | 46.8 |

This example indicates a meaningful improvement.

A difference smaller than one point may be caused by normal evaluation variation.

Keep a results table for every training run so that a new model is never deployed merely because it is newer.

## 5. Ask native speakers to evaluate translations

Automatic metrics cannot reliably determine whether a translation sounds natural or preserves the intended meaning.

Ask native speakers to review at least 50 to 100 unseen sentences for every translation direction.

Reviewers should consider:

- Correct meaning.
- Grammar.
- Natural wording.
- Spelling and punctuation.
- Missing information.
- Added or invented information.
- Use of the correct language and dialect.

A simple rating scale is:

| Rating | Meaning |
|---:|---|
| 1 | Completely incorrect |
| 2 | Mostly incorrect |
| 3 | Understandable but flawed |
| 4 | Correct with minor issues |
| 5 | Fully correct and natural |

A promising model should receive mostly ratings of 4 or 5.

Save the reviewer ratings and comments so recurring error patterns can guide the next dataset revision.

## 6. Test realistic sentences manually

Use sentences that were not present in the training CSV.

Include:

- Short sentences.
- Long sentences.
- Questions.
- Negation.
- Numbers and dates.
- Names and places.
- Formal and informal language.
- Common spelling mistakes.
- Sentences from the intended real-world domain.

Watch for:

- Repeated words or phrases.
- Invented content.
- Missing details.
- Untranslated source text.
- Output in the wrong language.
- Broken names, numbers, or punctuation.
- Fluent output that changes the original meaning.

## 7. Prevent misleading evaluation results

The test set must contain genuinely unseen sentence pairs.

The preparation script now assigns complete source groups with a stable SHA-256 hash.

Every row sharing a Bible `verse_key`, including textual variants in other language files, receives the same split.

Reverse-direction rows retain that group identifier and cannot cross into another split.

Keep `group_id` available in evaluation artifacts so the train, development, and test group sets can be audited as disjoint.

New data importers must provide a meaningful source-document group when near-duplicates or multiple translations share one underlying passage.

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the complete data preparation and training workflow.

## 8. Suggested acceptance criteria

Consider a translation direction ready for limited testing when:

- Validation loss improved without later increasing substantially.
- The final model outperformed the original NLLB model on a clean test set.
- The test set contains no training data or reversed-pair leakage.
- chrF++ reached an acceptable level for the intended use.
- Native speakers rated most translations as correct and natural.
- The model handles realistic sentences without frequent severe errors.
- No major regression occurred in another supported language direction.

Higher standards are required for medical, legal, financial, emergency, or other high-risk translations.

## Final success check

The strongest evidence of successful training is:

```text
Validation loss improved
+ the fine-tuned model beat original NLLB
+ the model performed well on a clean and unseen test set
+ native speakers approved the translations
```

Do not judge training success from training loss alone.
