# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder
- **Tokenizer version:** tok-v3
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 266,659 bytes (260.4 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|thought|>, <|answer|>, <|code|>, <|endcode|>, <|tool_call|>, <|end_tool_call|>, <|tool_result|>, <|end_tool_result|>, <|final|>, <|endtool_call|>, <|endtool_result|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all_train | 14787 | 3262472 | 220.631 | 3.1996 | 0.000000 |
| code | 2000 | 351030 | 175.515 | 3.0842 | 0.000000 |
| prose | 144 | 34034 | 236.347 | 3.0167 | 0.000000 |
| math | 518 | 44955 | 86.786 | 3.1179 | 0.000000 |
| tool_call_json | 1 | 17 | 17.0 | 3.9412 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
