# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (Rust `tokenizers`), ByteLevel pre-tokenizer, ByteLevel decoder
- **Tokenizer version:** tok-v1
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 265,896 bytes (259.7 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|system|>, <|user|>, <|thought|>, <|answer|>, <|assistant|>, <|code|>, <|endcode|>, <|endoftext|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all | 937 | 135545 | 144.658 | 3.7768 | 0.000000 |
| code | 427 | 105416 | 246.876 | 3.4593 | 0.000000 |
| math_logic | 375 | 21434 | 57.157 | 3.7396 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
