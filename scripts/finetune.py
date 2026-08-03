#!/usr/bin/env python3
"""Fine-tune NLLB-200-distilled-600M on Cameroonian language pairs using LoRA.

*** RUN THIS ON A GPU. A free Google Colab T4 is enough. Not viable on CPU. ***

HOW IT WORKS (high level)
-------------------------
1. Load Meta's NLLB-200 (600M distilled) - a multilingual translation model
   that already understands 200 languages including French, English, and
   Fulani (fuv_Latn - a different language from this project's Fulfulde).
   It does NOT know Ewondo, this project's Fulfulde (fub_Latn, Adamawa), or
   any other Cameroonian language in this project.
2. For languages NLLB does NOT know (e.g. Ewondo, Fulfulde), add a brand-new
   language token to the tokenizer and grow the model's embedding matrix
   to match.
3. Attach LoRA adapters (peft): instead of updating all 600M weights, we
   train small low-rank matrices injected into the attention/FFN layers
   (~1% of parameters). This is what makes training fit on a free T4.
4. Train with HuggingFace Seq2SeqTrainer, early-stopping on dev loss.
5. Merge the LoRA adapters back into the base weights and save a normal,
   standalone model folder that you fully own.

USAGE
-----
  python finetune.py --train data/prepared/train.jsonl \
      --dev data/prepared/dev.jsonl --new-langs dua_Latn --out model_out

OUTPUT
------
  model_out/merged/   full merged weights + tokenizer. Next step:
  ct2-transformers-converter --model model_out/merged \
      --output_dir model_ct2 --quantization int8
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

BASE = "facebook/nllb-200-distilled-600M"   # base checkpoint we fine-tune
MAX_LEN = 256                               # max tokens per sentence side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.jsonl from prepare_data.py")
    ap.add_argument("--dev", required=True, help="dev.jsonl (used for early stopping)")
    ap.add_argument("--out", default="model_out", help="output directory")
    ap.add_argument("--new-langs", nargs="*", default=[],
                    help="language codes NOT already in NLLB, e.g. dua_Latn")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="1e-4 is a good LoRA default; full fine-tune would use ~5e-5")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="effective batch = batch * grad_accum = 64")
    ap.add_argument("--eval-steps", type=int, default=500,
                    help="lower this for small/smoke-test runs - a run with "
                         "fewer optimizer steps than eval-steps never "
                         "evaluates or checkpoints, which breaks "
                         "load_best_model_at_end")
    ap.add_argument("--no-load-best", action="store_true",
                    help="skip reloading the best checkpoint after training. "
                         "That reload briefly holds two copies of the model "
                         "in GPU memory and can OOM on small local GPUs; "
                         "not needed on a real GPU like a Colab T4")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load base model + tokenizer.
    # ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)

    # ------------------------------------------------------------------
    # 2. Register new language tokens (e.g. Duala) if requested.
    #    NLLB marks languages with special tokens like 'fra_Latn'. A new
    #    language needs its own token, and the embedding matrix must be
    #    resized to make room for it.
    # ------------------------------------------------------------------
    if args.new_langs:
        added = tok.add_special_tokens(
            {"additional_special_tokens": list(args.new_langs)})
        if added:
            model.resize_token_embeddings(len(tok))
            # New rows in the embedding matrix are randomly initialized by
            # resize_token_embeddings() - no NLLB-200 language is a close
            # enough relative of these languages to justify a warm start.

    # ------------------------------------------------------------------
    # 3. Attach LoRA adapters.
    #    target_modules = attention projections + feed-forward layers.
    #    modules_to_save: when we added new tokens, the embedding table must
    #    be trained fully (not via LoRA) so the new tokens actually learn
    #    something. NLLB ties embed_tokens and lm_head (tie_word_embeddings)
    #    so listing only "embed_tokens" here also updates the output head -
    #    listing both makes PEFT wrap each separately, which breaks the tie
    #    and silently duplicates the entire (huge, ~256k-vocab) embedding
    #    matrix in memory.
    # ------------------------------------------------------------------
    lora = LoraConfig(
        r=16,                # rank of the low-rank update matrices
        lora_alpha=32,       # scaling factor (alpha/r = effective LR multiplier)
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        modules_to_save=["embed_tokens"] if args.new_langs else None,
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # sanity check: should be ~1-5%

    # ------------------------------------------------------------------
    # 4. Load and tokenize the dataset.
    # ------------------------------------------------------------------
    ds = load_dataset("json", data_files={"train": args.train, "dev": args.dev})

    def preprocess(ex):
        """Convert one {"src","tgt","src_lang","tgt_lang"} row into model inputs.

        NLLB convention:
          - setting tok.src_lang prepends the source-language token to inputs
          - setting tok.tgt_lang makes text_target labels start with the
            target-language token, which teaches the model WHICH language
            to produce.
        """
        tok.src_lang = ex["src_lang"]
        enc = tok(ex["src"], truncation=True, max_length=MAX_LEN)
        tok.tgt_lang = ex["tgt_lang"]
        labels = tok(text_target=ex["tgt"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = labels["input_ids"]
        return enc

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)

    # ------------------------------------------------------------------
    # 5. Training configuration.
    #    - fp16 on GPU halves memory use
    #    - eval every --eval-steps steps; keep the best checkpoint by dev loss
    #    - EarlyStoppingCallback stops after 3 evals without improvement
    # ------------------------------------------------------------------
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=not args.no_load_best,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",    # disable wandb/tensorboard auto-logging
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        # The collator pads each batch dynamically to its longest sequence
        # (cheaper than padding everything to MAX_LEN).
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Resume from the latest checkpoint under --out if one exists (e.g. a
    # Colab session that disconnected mid-run) instead of starting over.
    # save_steps/save_total_limit above already write these periodically.
    has_checkpoint = any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=has_checkpoint or None)

    # ------------------------------------------------------------------
    # 6. Merge LoRA adapters into the base weights => a plain standalone
    #    model. After this the model has NO peft dependency and can be
    #    converted with CTranslate2 for CPU serving.
    # ------------------------------------------------------------------
    merged = model.merge_and_unload()
    out = Path(args.out) / "merged"
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Merged model saved to {out}")
    print("Next: ct2-transformers-converter --model", out,
          "--output_dir model_ct2 --quantization int8")


if __name__ == "__main__":
    main()
