# Cameroon Languages Translation API Roadmap

Last updated: 2026-08-12

The goal is a model-owned translation API for English and Cameroonian languages.
French and other directions must remain unavailable unless the deployed model has training and evaluation evidence for them.

## Current status

| Area | Status | Evidence or remaining gate |
| --- | --- | --- |
| Engineering foundation | Complete | Data preparation, training safeguards, API validation, concurrency protection, and regression tests are implemented. |
| Prepared corpus | Complete | The corrected grouped, bidirectional corpus has been regenerated and audited. |
| Local automated verification | Complete | All 19 tests pass with warnings treated as errors, and lint and compilation checks pass. |
| Colab GPU smoke training | Next | Run the notebook on a T4-class GPU and record memory and trainable-parameter metrics. |
| Full model training | Pending | Complete the first controlled one-epoch run and preserve all artifacts. |
| Model evaluation | Pending | Compare every trained direction with base NLLB and complete native-speaker review. |
| Deployment candidate | Pending | Requires evaluation approval, CTranslate2 conversion, and real-model HTTP testing. |
| Production release | Pending | Requires performance, security, storage, monitoring, and licensing gates. |

The release is currently blocked by GPU training and model-quality evidence, not by the prepared-data pipeline.
Docker image validation is also pending because a Docker daemon was not available in the local verification environment.

## Verified corpus baseline

| Split | Rows | Source groups |
| --- | ---: | ---: |
| Train | 3,375,218 | 29,485 |
| Development | 91,638 | 803 |
| Test | 92,330 | 793 |
| Total | 3,559,186 | 31,081 |

- The corpus contains 60 bidirectional training directions across 30 language-script targets.
- Stable SHA-256 group assignment keeps related verse variants and their reverse rows in the same split.
- Every accepted forward pair has a reverse pair.
- Latin and Arabic Fulfulde rows use separate language-script tokens.
- The generated dataset manifest records row counts, directions, and file checksums.
- Automated validation found no source-group leakage between train, development, and test splits.

## Completed engineering foundation

- [x] Separate pretrained, data-ready, fine-tuned, and planned languages in the registry.
- [x] Clean, validate, deduplicate, and group source passages during preparation.
- [x] Prevent grouped passages and reverse rows from leaking across splits.
- [x] Represent Latin and Arabic Fulfulde with separate language-script tokens.
- [x] Fine-tune with LoRA while training only newly added language-token rows.
- [x] Detect unknown language tokens automatically before training.
- [x] Validate repository access, dependencies, CUDA memory, data integrity, and smoke training in Colab.
- [x] Reject incompatible checkpoint resumes and ignore incomplete checkpoint directories.
- [x] Record exact trained directions, configuration, and artifacts in manifests.
- [x] Validate the registry, tokenizer, and training manifest together during API startup.
- [x] Protect mutable tokenizer state from concurrent translation requests.
- [x] Accept contributions for planned and data-ready languages without advertising unsupported translation directions.
- [x] Add API-key support, readiness reporting, and actionable startup failures.
- [x] Cover the discovered data and API regressions with automated tests.

## Phase 1: Corpus quality and governance

- [x] Regenerate `data/prepared` with the corrected grouped splitter and reverse directions.
- [x] Run the automated integrity audit over every prepared split.
- [ ] Review at least 50 random pairs for every language-script target with native speakers.
- [ ] Record dialect, orthography, ambiguity, and rejection reasons during review.
- [ ] Measure how strongly the Bible domain differs from intended user requests.
- [ ] Add conversational, educational, administrative, and locally relevant sentence data.
- [ ] Record corpus source, license, script, dialect, and human-review status.

Exit gate: every enabled language-script target has documented provenance and a representative native-speaker quality sample.

## Phase 2: First controlled GPU training run

- [ ] Open `notebooks/train_colab.ipynb` in Colab with a T4-class GPU and run all cells from a clean runtime.
- [ ] Complete the two-step smoke training before starting the full run.
- [ ] Record trainable parameter count, trainable percentage, peak GPU memory, and smoke loss.
- [ ] Start the first full run with one epoch, batch size 4, and gradient accumulation 16.
- [ ] Monitor training and development loss at every evaluation interval.
- [ ] Resume interrupted work only when the saved run fingerprint matches the current configuration and data.
- [ ] Preserve the final merged model, tokenizer, checkpoints, run configuration, training manifest, and dataset manifest together.
- [ ] Save the complete Colab log and the exact source revision used for the run.

Exit gate: a reproducible merged model exists, its manifests match the prepared corpus, and the training run completes without memory or numerical failures.

## Phase 3: Establish evaluation gates

- [ ] Convert the merged model to CTranslate2 int8.
- [ ] Validate that the converted model and tokenizer contain every trained language token.
- [ ] Compute BLEU and chrF++ for every trained direction separately.
- [ ] Evaluate the untouched base NLLB model on the same held-out source groups.
- [ ] Require the fine-tuned model to beat the base model by a meaningful, documented margin.
- [ ] Ask native speakers to rate 50 to 100 unseen translations per direction.
- [ ] Reject directions with wrong-language output, wrong script, severe meaning loss, or frequent hallucination.
- [ ] Store automatic scores, human ratings, failure examples, configuration, and checksums beside the model release.

Exit gate: every advertised direction passes both automatic comparison and native-speaker review.

## Phase 4: Approve a deployment candidate

- [ ] Change only evaluated language entries from `data_ready` to `fine_tuned`.
- [ ] Confirm every fine-tuned code exists in `training_manifest.json` and the tokenizer.
- [ ] Confirm `/directions` lists only evaluated source and target pairs.
- [ ] Build the production Docker image from a clean checkout.
- [ ] Run the complete HTTP suite against the real CTranslate2 model.
- [ ] Verify startup failure, readiness, authentication, validation, and contribution behavior in the container.
- [ ] Measure p50, p95, and p99 latency on the target CPU host.
- [ ] Load test concurrent translation requests and contribution writes.

Exit gate: the container passes functional, concurrency, and performance tests with the exact candidate model artifact.

## Phase 5: Production hardening

- [ ] Require `API_KEY` in the deployment environment or integrate the product identity provider.
- [ ] Add distributed rate limiting before horizontal scaling.
- [ ] Configure HTTPS, structured request logging, and privacy-safe metrics.
- [ ] Keep user text out of logs unless explicit consent and retention rules exist.
- [ ] Mount contributions on durable storage with tested backups and recovery.
- [ ] Alert on readiness failures, latency regression, disk usage, and malformed contribution spikes.
- [ ] Document deployment, rollback, model replacement, incident response, and data-retention procedures.

Exit gate: production operations can detect, contain, and recover from service or model failures without losing accepted contribution data.

## Phase 6: Improvement loop

- [ ] Review contributed rows with language experts.
- [ ] Preserve rejected rows separately with review reasons.
- [ ] Merge approved groups into the source corpus.
- [ ] Regenerate deterministic splits without moving existing source groups when possible.
- [ ] Train a new candidate and compare it with the current production model.
- [ ] Deploy only when every enabled direction retains or improves its quality gate.

## Locked technical decisions

- The API must not claim arbitrary any-to-any translation support.
- The first controlled training run uses one epoch so quality and overfitting can be measured before increasing compute.
- Checkpoint resume is allowed only for an identical run fingerprint.
- The model, tokenizer, dataset manifest, training manifest, evaluation report, and checksums form one release unit.
- Unsupported or unevaluated directions stay hidden even when their language exists in the contribution registry.

## Licensing constraint

NLLB-200 is restricted to non-commercial use under its model license.
A commercial deployment requires a compatible base model or separate permission before release.

## Immediate next action

Run the complete Colab notebook on a T4-class GPU, capture the smoke metrics, and continue to the one-epoch run only if the smoke gate passes.
