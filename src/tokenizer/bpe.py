"""Custom byte-level BPE tokenizer built on the Rust `tokenizers` library.

Special tokens carry the structural grammar used across every task type:
<|system|>, <|user|>, <|thought|>, <|answer|>, <|assistant|>, <|code|>, <|endcode|>.
Byte-level pre-tokenization guarantees a 0% unknown-token rate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing

from ..utils.io_utils import REPO_ROOT, write_json

logger = logging.getLogger("tinyme.tokenizer")

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|system|>", "<|user|>", "<|thought|>", "<|answer|>",
    "<|assistant|>", "<|code|>", "<|endcode|>", "<|endoftext|>",
]
TOKENIZER_VERSION = "tok-v1"


class TinyMeTokenizer:
    """Thin wrapper that adds ids/length accounting and report generation."""

    def __init__(self, tok: Tokenizer, name: str = TOKENIZER_VERSION):
        self.tok = tok
        self.name = name

    # ---------------------------------------------------------------- basics
    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False):
        return self.tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TinyMeTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # -------------------------------------------------------------- metrics
    def measure(self, texts: list[str]) -> dict[str, Any]:
        """Token efficiency, UNK rate, and per-domain breakdown."""
        import statistics as st

        n_tokens, n_chars = [], []
        unk_id = self.tok.token_to_id("<|unk|>")
        unk_count = 0
        for t in texts:
            enc = self.tok.encode(t, add_special_tokens=False)
            n_tokens.append(len(enc.ids))
            n_chars.append(len(t))
            if unk_id is not None:
                unk_count += sum(1 for i in enc.ids if i == unk_id)
        total_tokens = sum(n_tokens)
        return {
            "vocab_size": self.vocab_size,
            "samples": len(texts),
            "total_tokens": total_tokens,
            "total_chars": sum(n_chars),
            "avg_tokens_per_sample": round(total_tokens / max(len(texts), 1), 3),
            "avg_chars_per_token": round(sum(n_chars) / max(total_tokens, 1), 4),
            "chars_per_token": round(sum(n_chars) / max(total_tokens, 1), 4),
            "median_tokens_per_sample": st.median(n_tokens) if n_tokens else 0,
            "p95_tokens_per_sample": sorted(n_tokens)[int(0.95 * len(n_tokens))] if n_tokens else 0,
            "max_tokens_per_sample": max(n_tokens) if n_tokens else 0,
            "unk_token_count": unk_count,
            "unk_rate": round(unk_count / max(total_tokens, 1), 6),
        }


def build_tokenizer(vocab_size: int = 4096) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int = 4096,
                    min_frequency: int = 2) -> TinyMeTokenizer:
    """Train a byte-level BPE tokenizer on the corpus."""
    texts = list(texts)
    tok = build_tokenizer(vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    # Post-processor: automatically wrap sequences with <|bos|>/<|eos|>.
    tok.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
        special_tokens=[
            ("<|bos|>", tok.token_to_id("<|bos|>")),
            ("<|eos|>", tok.token_to_id("<|eos|>")),
        ],
    )
    logger.info("trained BPE tokenizer: vocab_size=%d on %d texts", tok.get_vocab_size(), len(texts))
    return TinyMeTokenizer(tok)


def tokenizer_report(tokenizer: TinyMeTokenizer, corpora: dict[str, list[str]],
                     file_size_bytes: int = 0, vocab_candidates: list[int] | None = None) -> dict[str, Any]:
    """Build the TOKENIZER_REPORT.md payload."""
    report: dict[str, Any] = {
        "tokenizer_version": tokenizer.name,
        "algorithm": "byte-level BPE (Rust `tokenizers`), ByteLevel pre-tokenizer, ByteLevel decoder",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "file_size_bytes": file_size_bytes,
        "domains": {},
    }
    for domain, texts in corpora.items():
        report["domains"][domain] = tokenizer.measure(texts)
    return report


def write_tokenizer_report(report: dict[str, Any], path: str | Path) -> None:
    """Render the report as markdown."""
    lines = [
        "# TOKENIZER REPORT",
        "",
        f"- **Algorithm:** {report['algorithm']}",
        f"- **Tokenizer version:** {report['tokenizer_version']}",
        f"- **Vocabulary size:** {report['vocab_size']}",
        f"- **Serialized `tokenizer.json` size:** {report['file_size_bytes']:,} bytes "
        f"({report['file_size_bytes'] / 1024:.1f} KiB) — **counts toward the <50 MB artifact**",
        f"- **Special tokens:** {', '.join(report['special_tokens'])}",
        "",
        "## Per-domain efficiency",
        "",
        "| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, m in report["domains"].items():
        lines.append(
            f"| {domain} | {m['samples']} | {m['total_tokens']} | {m['avg_tokens_per_sample']} "
            f"| {m['chars_per_token']} | {m['unk_rate']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A high chars/token ratio on code means fewer tokens are needed to represent a line of "
        "Python, which directly increases the effective context available to a <50 MB model.",
        "- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, "
        "so no input can ever be out-of-vocabulary.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_vocab_reference(tokenizer: TinyMeTokenizer, path: str | Path) -> None:
    vocab = tokenizer.tok.get_vocab()
    write_json(vocab, path)
