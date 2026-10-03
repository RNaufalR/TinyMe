#!/usr/bin/env python3
"""Evidence probe: is a numeric span token-identical across contexts?

Motivation (measured 2026-10-03, branch ``arena/01a0ff12-tinyme``)
----------------------------------------------------------------
The released model reproduced memorised training templates (60/60 first-30-char
matches on the first 60 supervised training records) yet copied the operand out
of the prompt in **0/10** of its own training ``tool/compute`` records, and the
independent A-H suite scored ``task_completion 0.00``.

The diagnosis is tokenisation, not attention.  Under tok-v2 the byte-level BPE
merged digit runs *with their context*, so the same literal produced different
token sequences in the user turn and in the tool-call argument:

    'Compute exactly: 4837 * 962 + 71'  -> ['Compute',' exactly',':',' 48','37',' *',...]
    '"expression": "4837 * 962 + 71"'   -> ['"','expression','":',' "','48','3','7',...]

Copying therefore required *re-tokenising* the span (a learned merge-boundary
translation) instead of copying token ids, which a 2.5 M-parameter model with
~300 supervised ``compute`` examples never learns.

tok-v3 inserts ``pre_tokenizers.Digits(individual_digits=True)`` ahead of the
byte-level pre-tokenizer, so a digit is always its own pre-token: a numeric span
is token-identical everywhere and copying becomes an induction head task.

This probe measures the property on a tokenizer trained from scratch both ways
on the same corpus, so the comparison is not confounded by vocabulary size.

Run:  python docs/audit_evidence/tokenizer_context_copy_probe.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer, pre_tokenizers, trainers  # noqa: E402

from src.tokenizer.bpe import build_tokenizer  # noqa: E402

CONTEXTS = [
    ("user turn", "Compute exactly: 4837 * 962 + 71", "4837"),
    ("json argument", '"expression": "4837 * 962 + 71"', "4837"),
    ("user turn", "Read the stored source FACT-988002 about population of jakarta.", "988002"),
    ("json argument", '"source_id": "FACT-988002"', "988002"),
    ("user turn", "What is the population of bandung?", "bandung"),
    ("json argument", '"query": "population of bandung"', "bandung"),
]


def _demo_corpus() -> list[str]:
    rng = random.Random(0)
    out: list[str] = []
    for _ in range(3000):
        a, b, c = rng.randint(37, 9800), rng.randint(11, 90), rng.randint(3, 60)
        expr = f"{a} * {b} + {c}"
        out.append(f"Compute exactly: {expr}")
        out.append(json.dumps({"name": "compute", "arguments": {"expression": expr}}, sort_keys=True))
        sid = f"FACT-{rng.randint(0, 0xFFFFFF):06X}"
        out.append(f"Read the stored source {sid} about population of jakarta.")
        out.append(json.dumps({"name": "fetch", "arguments": {"source_id": sid}}, sort_keys=True))
    return out


def train(individual_digits: bool) -> Tokenizer:
    tok = build_tokenizer(4096, individual_digits=individual_digits)
    tok.train_from_iterator(_demo_corpus(), trainer=trainers.BpeTrainer(
        vocab_size=4096, min_frequency=1, special_tokens=["<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()), length=3000 * 4)
    return tok


def digit_pieces(tok: Tokenizer, text: str, needle: str) -> list[str]:
    """The token pieces that spell ``needle`` (first occurrence)."""
    ids = tok.encode(text).ids
    pieces = [tok.decode([i]) for i in ids]
    joined = "".join(pieces)
    start = joined.find(needle)
    assert start >= 0, (text, needle, joined)
    end = start + len(needle)
    out, pos = [], 0
    for piece in pieces:
        nxt = pos + len(piece)
        if pos < end and nxt > start:
            out.append(piece)
        pos = nxt
    return out


def main() -> int:
    report: dict = {"probes": [], "verdict": {}}
    for label, individual in (("tok-v2-style (context merges)", False),
                              ("tok-v3 (individual digits)", True)):
        tok = train(individual)
        print("=" * 78)
        print(label)
        print("=" * 78)
        ok = 0
        for i in range(0, len(CONTEXTS), 2):
            (_, text_a, needle), (_, text_b, _n) = CONTEXTS[i], CONTEXTS[i + 1]
            pa = digit_pieces(tok, text_a, needle)
            pb = digit_pieces(tok, text_b, needle)
            same = pa == pb
            ok += same
            print(f"  {needle:8s} user={pa} arg={pb}  copy-safe={same}")
            report["probes"].append({"tokenizer": label, "needle": needle,
                                     "user_pieces": pa, "argument_pieces": pb, "copy_safe": same})
        verdict = f"{ok}/{len(CONTEXTS) // 2} numeric spans are token-identical across contexts"
        report["verdict"][label] = verdict
        print(f"  -> {verdict}\n")
    # ------------------------------------------------ the frozen shipped tokenizer
    frozen = ROOT / "datasets" / "versions" / "dataset_v5" / "tokenizer.json"
    if frozen.exists():
        from src.data.dataset_api import load_tokenizer

        tok = load_tokenizer("dataset_v5")
        print("=" * 78)
        print("frozen dataset_v5 tokenizer (tok-v3)")
        print("=" * 78)
        ok = 0
        for i in range(0, len(CONTEXTS), 2):
            (_, text_a, needle), (_, text_b, _n) = CONTEXTS[i], CONTEXTS[i + 1]
            pa = [tok.decode([tid], skip_special_tokens=False) for tid in tok.encode_ids(text_a)]
            pb = [tok.decode([tid], skip_special_tokens=False) for tid in tok.encode_ids(text_b)]
            da = [x for x in pa if x.strip().isdigit()]
            db = [x for x in pb if x.strip().isdigit()]
            same = da == db
            ok += same
            print(f"  {needle:8s} user_digits={da} arg_digits={db}  copy-safe={same}")
            report["probes"].append({"tokenizer": "dataset_v5/tokenizer.json", "needle": needle,
                                     "user_pieces": [x for x in pa if x.strip()],
                                     "argument_pieces": [x for x in pb if x.strip()],
                                     "copy_safe": same})
        verdict = f"{ok}/{len(CONTEXTS) // 2} numeric spans are token-identical across contexts"
        report["verdict"]["dataset_v5/tokenizer.json (tok-v3)"] = verdict
        print(f"  -> {verdict}\n")

    dest = Path(__file__).with_suffix(".out.json")
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("summary")
    for k, v in report["verdict"].items():
        print(f"  {k}: {v}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
