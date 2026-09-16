# Local Windows Training Guide

This runbook is the canonical procedure for training the translation model on a Windows PC with an NVIDIA GPU.
Use [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for data preparation and model-design details.
Use [TRAINING_EVALUATION_GUIDE.md](TRAINING_EVALUATION_GUIDE.md) before approving any trained direction.

## Verified local baseline

The local workflow was exercised end to end on this hardware and software baseline:

| Item | Verified value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4 GiB |
| PyTorch | 2.7.0+cu128 with CUDA 12.8 |
| Training precision | BF16 |
| Smoke directions | 60 |
| Trainable parameters | 8,681,472, or 1.39 percent |
| Peak allocated GPU memory | 2.27 GiB |
| Peak reserved GPU memory | 2.79 GiB |
| Numeric check | Two finite gradient norms |
| Artifact check | CTranslate2 int8 conversion and CPU inference passed |

The smoke model was deliberately discarded after verification.
Its translation scores are not quality evidence because it trained for only two optimizer steps.

## Prerequisites

Install these before running the repository scripts:

- Windows PowerShell 5.1 or PowerShell 7.
- Python 3.11 or 3.12.
- `uv` on `PATH`.
- A recent NVIDIA driver with `nvidia-smi` on `PATH`.
- At least 4 GiB of NVIDIA GPU memory.
- Enough free disk space for the environment, caches, checkpoints, and merged model.

Keep the laptop connected to power and prevent Windows from sleeping during a long run.

## One-time environment setup

Open PowerShell in the repository root.
Allow repository scripts only for the current terminal, then rebuild the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_local.ps1 -Rebuild
```

The setup script performs these actions:

1. Creates `.venv` with Python 3.11 or 3.12.
2. Installs the official CUDA 12.8 PyTorch wheel.
3. Installs the pinned serving, training, and development dependencies.
4. Confirms that PyTorch can use the NVIDIA GPU.
5. Runs the complete test suite unless `-SkipTests` is supplied.

The ignored `.uv-cache` and `.hf-cache` directories make later setup and model loading faster.
Do not delete them unless disk space is needed or a clean download is required.

## Training modes

`scripts/train_local.ps1` exposes four explicit modes:

| Mode | Data used | Intended purpose | Default output |
| --- | --- | --- | --- |
| `smoke` | Two train and one development example per direction | Verify the complete training lifecycle | Temporary directory under `model_smoke` |
| `pilot` | Balanced sample for selected targets | Measure learning behavior before a long run | `model_out/pilot-<targets>` |
| `language` | Every prepared row for selected targets | Train a candidate for selected targets | `model_out/local-<targets>` |
| `all` | Every prepared row for all 30 targets | Train one all-target candidate | `model_out/local-all` |

The 4 GiB profile uses a physical batch of 1, an evaluation batch of 1, a 128-token limit, gradient checkpointing, and zero data-loader workers.
Precision mode `auto` chooses BF16 on supported GPUs and otherwise uses FP16.
The training script refuses to merge a model when every logged gradient norm is non-finite.

## Step 1: run the required smoke gate

Run:

```powershell
.\scripts\train_local.ps1 -Mode smoke
```

The runner builds a direction-balanced subset containing all 60 prepared directions.
It verifies dependency loading, token creation, LoRA setup, CUDA execution, evaluation, checkpointing, finite gradients, model merging, and manifest creation.
Successful smoke artifacts are removed automatically.
Use `-KeepSmoke` only when artifacts are needed for debugging.

## Step 2: run one balanced pilot

Choose one target from `data/languages.json` with status `data_ready`.
For example, run an Ewondo pilot with 500 examples in each direction:

```powershell
.\scripts\train_local.ps1 `
  -Mode pilot `
  -TargetLanguage ewo_Latn `
  -SamplesPerDirection 500
```

The pilot sampler uses deterministic reservoir sampling so it does not train only on the beginning of a corpus.
The example command creates 1,000 training rows, split evenly between English to Ewondo and Ewondo to English.
It also creates up to 25 development examples per direction.

Multiple targets may be supplied when they must share one model:

```powershell
.\scripts\train_local.ps1 `
  -Mode pilot `
  -TargetLanguage ewo_Latn,fub_Latn `
  -SamplesPerDirection 500
```

Start with one target because failure analysis and quality evaluation are much clearer.

## Step 3: inspect pilot evidence

Read the recorded runtime metrics:

```powershell
Get-Content model_out\pilot-ewo_Latn\runtime_metrics.json
```

Confirm all of these before continuing:

- `finite_gradient_steps` is greater than zero.
- `train_loss` is finite.
- Development loss is finite and improves during a longer pilot.
- Peak reserved GPU memory remains below total GPU memory with useful headroom.
- The run created `merged/training_manifest.json`.
- The manifest lists only the directions intended for this candidate.

Do not approve or deploy a pilot model.
A pilot proves that the configuration learns without numerical or memory failure, not that translations are useful.

## Step 4: train the selected target corpus

After the pilot passes, train every prepared row for that target:

```powershell
.\scripts\train_local.ps1 -Mode language -TargetLanguage ewo_Latn
```

Long runs validate the complete prepared corpus before training begins.
Compatible checkpoints resume automatically when the same command is run again.
Use `-SkipValidation` only when the exact prepared files were validated immediately beforehand.

Training all targets is available:

```powershell
.\scripts\train_local.ps1 -Mode all
```

One all-target epoch contains 3,375,218 training rows and can take months on a 4 GiB laptop GPU.
Use a larger CUDA GPU or Colab when one model must cover every target.

## Checkpoint and output rules

Every persistent run writes these records:

```text
model_out/<run>/
  checkpoint-*/
  run_config.json
  runtime_metrics.json
  merged/
    config.json
    model.safetensors
    tokenizer files
    dataset_manifest.json
    runtime_metrics.json
    training_manifest.json
```

Resume is allowed only when the dataset checksums, languages, directions, precision, sequence length, and important hyperparameters match.
Use a new `-OutDir` when changing any training property.
Do not copy a checkpoint into another run directory.

## Convert and evaluate a candidate

Replace the paths below with the completed run directory:

```powershell
ct2-transformers-converter `
  --model model_out\local-ewo_Latn\merged `
  --output_dir model_ct2\local-ewo_Latn `
  --quantization int8 `
  --force

.\.venv\Scripts\python.exe scripts\evaluate.py `
  --test data\prepared\test.jsonl `
  --ct2 model_ct2\local-ewo_Latn `
  --tokenizer model_out\local-ewo_Latn\merged
```

Evaluate each direction independently against the untouched base model.
Native speakers must review unseen examples before a registry status changes to `fine_tuned`.

## Failure recovery

### CUDA is unavailable

Run `nvidia-smi`, then rerun `scripts/setup_local.ps1`.
Confirm that setup reports a CUDA-enabled PyTorch build and the expected GPU name.

### Training runs out of GPU memory

Close applications using the discrete GPU and retry the identical command.
The local profile already uses batch size 1, so do not reduce the sequence limit without recording the quality trade-off in a new run configuration.

### Gradient norms are non-finite

Use precision mode `auto` or `bf16` on a supported GPU.
If the problem remains, lower the learning rate and use a new output directory.
The failed run must not be converted or deployed.

### Resume is rejected

Read the error and compare `run_config.json` with the intended command.
Start a new output directory when data or settings changed.
Do not bypass the fingerprint check.
