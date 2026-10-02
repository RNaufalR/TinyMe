# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder
- **Tokenizer version:** tok-v2
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 267,051 bytes (260.8 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|thought|>, <|answer|>, <|code|>, <|endcode|>, <|tool_call|>, <|end_tool_call|>, <|tool_result|>, <|end_tool_result|>, <|final|>, <|endtool_call|>, <|endtool_result|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all_train | 1680 | 432773 | 257.603 | 3.7569 | 0.000000 |
| code | 1487 | 394109 | 265.036 | 3.8148 | 0.000000 |
| prose | 144 | 32074 | 222.736 | 3.2011 | 0.000000 |
| math | 28 | 2312 | 82.571 | 3.0722 | 0.000000 |
| tool_call_json | 1 | 21 | 21.0 | 3.1905 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
