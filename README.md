# TinyMe

TinyMe is a **small language model paired with an external tool runtime**: a typed
tool protocol, a local retrieval index and an isolated code-execution sandbox. It
is deliberately *not* presented as a frontier model, and nothing in this
repository claims frontier capability. It is a reproducible study of how far a
few million parameters get on verified data when the surrounding runtime - not
raw memorisation - has to carry factual work.

The project was rebuilt end-to-end during a corrective audit (2026-10-02) after
the first baseline (`EXP-001`) was found to be invalid as a held-out baseline:
its "validation" split was a slice of the training file, the evaluation never
executed generated code, and preprocessing destroyed code indentation. That
audit, the evidence probes and the fixes are recorded in
[`docs/CORRECTIVE_AUDIT.md`](docs/CORRECTIVE_AUDIT.md); `EXP-001` artifacts are
kept as **HISTORICAL / INVALID-AS-HELDOUT-BASELINE** and are never resumed.

## What is actually here

| Layer | What it does | Where |
| :--- | :--- | :--- |
| Model | decoder-only transformer, RoPE, RMSNorm, SwiGLU, tied embeddings | `src/model/` |
| Data | content-aware preprocessing, segmented records, 4-stage dedup, stratified splits, one sequence builder | `src/data/`, `data_sources/` |
| Training | gradient accumulation, task-aware loss masking, padding mask, atomic checkpoints, resume, deterministic sampler, RNG sync | `src/training/` |
| Tools | `search`, `fetch`, `compute`, `code`, `files` behind strict protocol tokens | `src/tools/`, `src/agent/` |
| Sandbox | network-namespace isolation, rlimits (CPU/mem/file/process), scrubbed env, bounded output, workspace confinement | `src/sandbox/` |
| Evaluation | held-out only, executable code metrics, tool selection/arguments/execution, grounding, quantization comparison | `src/evaluation/`, `scripts/evaluate.py` |
| Release | fp32/fp16/int8/int4 weights, tokenizer, config, manifest, checksums, standalone entry point | `release/`, `scripts/package_model.py` |

### Architectures (measured, not estimated)

| Name | Params | FP32 bytes | Context | Role |
| :--- | ---: | ---: | ---: | :--- |
| `nano` | 2,557,632 | 10,230,528 (9.76 MiB) | 512 (primary seq_len 256) | default on this 2-vCPU sandbox |
| `base` | 8,933,440 | 35,733,760 (34.1 MiB) | 512 | GPU profile |
| `medium` | 17,308,032 | 69,232,128 (66.0 MiB) | 512 | scaling reference only |

Parameter counts are produced by `count_parameters()` on real initialised
weights and cross-checked against the analytical estimate in
`estimate_sizes()`; the release manifest repeats the measurement from the saved
file.

## Data (dataset_v2)

* Sources: Python standard library (PSF-2.0), three curated HuggingFace sets
  (MIT / CC-BY-4.0 / CDLA-Sharing-1.0, 15 records each, fetched once and cached
  because `huggingface.co` is unreachable from the build sandbox) and verified
  synthetic generators (arithmetic, word problems, sequences, boolean,
  deduction, algorithm traces, code generation/repair/explanation, instruction
  following, tool use).
* Every synthetic record is verified before emission: reference solutions run
  under subprocess tests, arithmetic is recomputed, instance examples are
  produced by executing the reference implementation, and duplicate instances
  are dropped and *reported* rather than re-emitted to inflate density.
* Splits: **train 4227 / validation 741 / test 810 / challenge 60** at
  `seq_len=256`, grouped by template family (`group_id`) so generators cannot
  straddle splits. Contamination check (exact, normalised, MinHash, code
  shingles, template overlap): **PASS**.
* Active target tokens: 588,816 (train). Padding ratio and masked-token
  accounting are reported per stage by `scripts/prepare_data_v2.py`.
* Tool-use trajectories use the **exact runtime protocol**: the same argument
  names the registry validates and the same JSON result envelopes the runtime
  produces (333/333 generated calls validated against `ToolRegistry`).

Reports: [`docs/DATA_QUALITY_REPORT.md`](docs/DATA_QUALITY_REPORT.md),
[`docs/SPLIT_SUMMARY.md`](docs/SPLIT_SUMMARY.md),
[`docs/TOKENIZER_REPORT.md`](docs/TOKENIZER_REPORT.md),
[`docs/LICENSE_POLICY.md`](docs/LICENSE_POLICY.md),
[`docs/DATA_PROVENANCE.json`](docs/DATA_PROVENANCE.json).

## Tool protocol

The model speaks a strict, verifiable protocol instead of free-form text:

```
<|tool_call|>{"name": "search", "arguments": {"query": "...", "k": 3}}<|end_tool_call|>
<|tool_result|>{"ok": true, "name": "search", "result": {...}, "duration_s": 0.0004}<|end_tool_result|>
<|final|>answer text with [SOURCE-ID] citations<|eos|>
```

* `src/agent/tool_registry.py` validates names, required arguments, types,
  ranges, lengths and choices before execution; unknown tools and unknown
  arguments are refused with a structured error.
* Tool results are **context only** - the loss mask never trains on them.
* The evidence engine (`src/agent/evidence.py`) refuses to invent citations: a
  citation must correspond to a source id that appeared in a tool result.

## Sandbox

Code execution is separated from network retrieval. Measured capabilities on
this host (`detect_isolation(force=True)`): `bwrap` unavailable, user/mount/net
namespaces available, rlimits available - honest label **`namespace(net+mount)`**,
never "container" or "VM".

Recorded escape suite (`docs/audit_evidence/sandbox_escape_suite.out.txt`,
12 cases): network egress blocked, host home tree hidden, secret env vars absent,
writes to the home tree blocked, fork bomb capped (`BlockingIOError`), CPU burn
killed (`SIGXCPU`), memory bomb `MemoryError` (RLIMIT_AS), file flood
`OSError: File too large` (RLIMIT_FSIZE), output truncated at the configured
byte cap. Two cases are *allowed by design* and labelled as such: child
processes inside the cap, and `/etc/passwd` being readable (shared `/etc`,
non-secret, needed by the loader). See
[`docs/SANDBOX.md`](docs/SANDBOX.md).

## Quick start

```bash
pip install numpy jax optax safetensors tokenizers pyyaml pytest psutil requests pypdf

# 1. rebuild the dataset (writes datasets/versions/dataset_v2 + docs reports)
python scripts/prepare_data_v2.py --version dataset_v2 --scale 2.5 --seq-len 256

# 2. train (stage A) - fresh ids only, never resume EXP-001
python scripts/train.py --experiment EXP-002-CORRECTED-NANO --stage pretrain \
    --arch nano --dataset dataset_v2 --seq-len 256 --micro-batch 8 --grad-accum 4 \
    --max-steps 400 --lr 6e-4 --dtype float32

# 3. evaluate on held-out splits (executable code + tool metrics)
python scripts/evaluate.py --experiment EXP-002-CORRECTED-NANO --checkpoint best \
    --dataset dataset_v2 --split test --variants fp32,fp16,int8,int4

# 4. package the release (writes release/ + checksums + reports)
python scripts/package_model.py --experiment EXP-002-CORRECTED-NANO --checkpoint best

# 5. tests
python -m pytest tests/ -q
```

## Running a released model without the training environment

```bash
cd release
pip install numpy safetensors tokenizers
python inference.py --prompt "What is 144 / 12?" --model model_fp32.safetensors
```

`release/inference.py` imports only `numpy`, `safetensors` and `tokenizers`; it
re-implements the forward pass in NumPy and dequantizes int8/int4 weights on
load. Its logits are verified to match the repository engine exactly
(`max|Δ| = 0.0` on the packaged fp32 file).

## Results and honest status

Held-out measurements, training curves, quantization deltas and the
comparability rules live in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md),
[`docs/TRAINING_REPORT.md`](docs/TRAINING_REPORT.md),
[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) and
[`FINAL_REPORT.md`](FINAL_REPORT.md). Comparisons across a changed tokenizer,
dataset fingerprint or evaluation protocol are marked **NOT COMPARABLE** instead
of being reported as improvements.

Known limitations, stated up front:

* a ~2.6M-parameter model cannot store broad world knowledge; factual answers
  are expected to come from `search`/`fetch` with citations;
* the corpus is small by LLM standards (hundreds of thousands of active target
  tokens), so most capabilities are demonstrated, not mastered;
* int4 weights are lossy (measured round-trip error is reported in
  `docs/MODEL_COMPARISON.md`, not hidden);
* the sandbox shares `/etc` and does not hide other unprivileged host paths
  outside the home tree; it is honest about that rather than claiming stronger
  isolation than it has.

## Repository layout

```
src/{model,data,tokenizer,training,inference,quantization,sandbox,agent,tools,evaluation,utils}/
data_sources/            # stdlib + curated HF + verified synthetic generators
scripts/                 # prepare_data_v2, train, evaluate, package_model, quantize, build_retrieval_index
datasets/                # versions/<v>/ (jsonl splits + tokenizer + manifest), processed shards, retrieval index
tests/                   # 27 modules - correctness, masking, parity, determinism, sandbox, tools, eval
docs/                    # audit, quality, tokenizer, split, licence, sandbox, training reports
release/                 # packaged weights + reports (regenerate with scripts/package_model.py)
```
