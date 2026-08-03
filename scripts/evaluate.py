#!/usr/bin/env python3
"""Evaluate translation quality on the held-out test set.

METRICS
-------
- BLEU   : classic MT metric, counts n-gram overlap with the reference.
- chrF++ : character-level F-score (+ word bigrams). MORE RELIABLE than BLEU
           for low-resource and morphologically rich languages, so treat
           chrF++ as your primary number.

Scores are reported PER DIRECTION (e.g. fra_Latn->ewo_Latn separately from
ewo_Latn->fra_Latn) because quality is rarely symmetric.

Runs on CPU using the CTranslate2 int8 model - i.e. it measures the EXACT
model your API will serve, not the pre-conversion weights.

USAGE
-----
  python evaluate.py --test data/prepared/test.jsonl \
      --ct2 model_ct2 --tokenizer model_out/merged
"""
import argparse
import json
from collections import defaultdict

import sacrebleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="test.jsonl from prepare_data.py")
    ap.add_argument("--ct2", default="model_ct2", help="CTranslate2 model dir")
    ap.add_argument("--tokenizer", default="model_out/merged",
                    help="HF tokenizer dir (saved next to the merged model)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per direction for a quick smoke test")
    args = ap.parse_args()

    # Imports are inside main() so `--help` works without the heavy deps.
    import ctranslate2
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    translator = ctranslate2.Translator(args.ct2, device="cpu",
                                        compute_type="int8")

    # ------------------------------------------------------------------
    # Group test rows by (src_lang, tgt_lang) so each direction gets its
    # own score.
    # ------------------------------------------------------------------
    by_dir = defaultdict(list)
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_dir[(r["src_lang"], r["tgt_lang"])].append(r)

    for (src_lang, tgt_lang), rows in by_dir.items():
        if args.limit:
            rows = rows[:args.limit]

        # Tokenize all source sentences. CTranslate2 wants token STRINGS,
        # not ids, hence encode -> convert_ids_to_tokens.
        tok.src_lang = src_lang
        batches = [tok.convert_ids_to_tokens(tok.encode(r["src"])) for r in rows]

        # target_prefix forces the decoder to start with the target-language
        # token - this is how NLLB knows which language to output.
        results = translator.translate_batch(
            batches, target_prefix=[[tgt_lang]] * len(batches), beam_size=4)

        # Decode hypotheses, stripping the leading language token.
        hyps = []
        for res in results:
            toks = res.hypotheses[0]
            if toks and toks[0] == tgt_lang:
                toks = toks[1:]
            hyps.append(tok.decode(tok.convert_tokens_to_ids(toks),
                                   skip_special_tokens=True))

        # sacrebleu expects refs as a list of reference LISTS (we have one
        # reference per sentence, so one inner list).
        refs = [[r["tgt"] for r in rows]]
        bleu = sacrebleu.corpus_bleu(hyps, refs)
        chrf = sacrebleu.corpus_chrf(hyps, refs, word_order=2)  # word_order=2 => chrF++
        print(f"{src_lang} -> {tgt_lang} ({len(rows)} rows): "
              f"BLEU {bleu.score:.1f} | chrF++ {chrf.score:.1f}")


if __name__ == "__main__":
    main()
