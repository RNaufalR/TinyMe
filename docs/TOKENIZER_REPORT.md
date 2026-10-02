# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder
- **Tokenizer version:** tok-v2
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 265,717 bytes (259.5 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|thought|>, <|answer|>, <|code|>, <|endcode|>, <|tool_call|>, <|end_tool_call|>, <|tool_result|>, <|end_tool_result|>, <|final|>, <|endtool_call|>, <|endtool_result|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all_train | 4227 | 729319 | 172.538 | 3.6778 | 0.000000 |
| code | 2000 | 452044 | 226.022 | 3.6651 | 0.000000 |
| prose | 144 | 32760 | 227.5 | 3.134 | 0.000000 |
| math | 1277 | 86188 | 67.493 | 3.9037 | 0.000000 |
| tool_call_json | 1 | 17 | 17.0 | 3.9412 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
