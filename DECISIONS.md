# ENGINEERING & ARCHITECTURAL DECISIONS (`DECISIONS.md`)

## DEC-001: Compute & Autodiff Backend Selection
- **Context:** Sandbox has 2 vCPUs (Intel Xeon Ice Lake @ 2.60GHz with AVX-512 & VNNI), 3.8 GiB RAM, no GPU, and `download.pytorch.org/whl/cpu` is blocked by the sandbox TLS firewall while standard PyPI `torch` downloads >2.5 GB of unused CUDA packages.
- **Decision:** Use **JAX 0.10.2 + XLA CPU (`jaxlib 0.10.2`) + Optax 0.2.8 + Safetensors 0.8.0 + NumPy 2.4.6** for training, and support both **JAX JIT** and **standalone pure NumPy + Safetensors** for inference and quantization.
- **Rationale:** XLA compiles the entire Transformer forward, backward, and AdamW step into fused AVX-512 native machine code with minimal memory overhead, while `.safetensors` + pure NumPy guarantees the final `release/` artifact can run anywhere without heavy dependencies.
- **Status:** `VERIFIED`

## DEC-002: Model Architecture (`TinyMe-Reasoner` Decoder-Only Transformer)
- **Context:** Must satisfy `FINAL_MODEL_SIZE < 50 MB` (target 10–42 MB across FP32/FP16/INT8/INT4) while maximizing computational reasoning, code understanding, and algorithmic problem solving per byte.
- **Decision:** Use a modern **Llama-3 / Qwen-2.5 style Pre-Norm Decoder-Only Transformer** with:
  - Rotary Position Embeddings (**RoPE**, $\theta = 10000.0$)
  - **RMSNorm** (root-mean-square layer normalization)
  - **SwiGLU** gated feed-forward networks
  - **Grouped-Query / Multi-Head Causal Attention**
  - **Weight Tying** between input token embeddings and output LM head (`tie_word_embeddings = True`), saving `vocab_size * d_model` parameters (~2.0M–4.1M parameters) and improving representation alignment on compact corpora.
- **Status:** `VERIFIED`

## DEC-003: Dual-Path External Data Ingestion Strategy
- **Context:** Sandbox TLS filter permits `api.github.com`, `github.com`, and `pypi.org`, while blocking direct socket connections to `huggingface.co`. However, the environment provides `fetch_page` which reaches `datasets-server.huggingface.co` and `huggingface.co`.
- **Decision:**
  1. Ingest real Hugging Face dataset slices (`roneneldan/TinyStories` [CDLA-Sharing-1.0], `openai/gsm8k` [MIT], `google-research-datasets/mbpp` [CC-BY-4.0]) via `datasets-server.huggingface.co` into `datasets/raw/huggingface/` and record their exact URLs, licenses, and hashes in `DATA_PROVENANCE.json`.
  2. Ingest real GitHub repositories and datasets (`TheAlgorithms/Python` [MIT], `openai/human-eval` [MIT], `keon/algorithms` [MIT]) directly via `api.github.com`.
  3. Extract local Python 3.11 standard library algorithms & documentation (`PSF-2.0`).
  4. Generate execution-verified synthetic curriculum data across all 9 categories (`A` through `I`) using deterministic problem/solution/validator pipelines.
- **Status:** `VERIFIED`

## DEC-004: Tool-protocol single source of truth (2026-10-02 corrective audit)
- **Context:** The synthetic corpus advertised `search.max_results`, `fetch.url` and `code.source`/`code.tests`, while `ToolRegistry` validates `query`/`k`, `source_id` and `code`. Every trajectory the model learned was therefore *rejected* by the runtime (`unknown argument`), and the model also never saw the real result envelope (`{"ok", "name", "result", "duration_s"}`).
- **Decision:** `data_sources/synthetic_v2.TOOL_SPECS` mirrors the runtime registry exactly, the mock search/fetch/compute/code providers emit the runtime's JSON envelope, mock source ids are read from the shipped retrieval index, and `tests/test_tool_protocol.py` fails the build if the two ever drift again. The partially trained run on the drifted corpus was stopped and its artifacts kept as `EXP-002-PRE-PROTOCOL`.
- **Rationale:** Training on arguments the runtime refuses can never yield working tool use; measuring tool-argument accuracy on such a model would be theatre.
- **Status:** `VERIFIED` (333/333 generated calls pass `ToolRegistry.validate_arguments`, 0 drift)

## DEC-005: Restart policy for invalidated runs
- **Context:** Three EXP-002 attempts were invalidated mid-run (duplicate-inflated corpus, category-starving random split, tool-argument drift). The audit forbids resuming invalid experiments and forbids overwriting history.
- **Decision:** Every invalidated run is stopped, renamed to a descriptive historical id (`EXP-002-PREP-VALIDATION`, `EXP-002-PRE-STRATIFIED`, `EXP-002-PRE-PROTOCOL`) with its log kept, and recorded in `EXPERIMENT_LOG.jsonl` with an explicit `status`/`reason`. A fresh `EXP-002-CORRECTED-NANO` run is started from scratch (no resume) each time the data contract changes.
- **Status:** `VERIFIED`

## DEC-006: Sandbox honesty over marketing
- **Context:** No `bwrap`, no PID namespace: isolation is `namespace(net+mount)` plus rlimits. Claiming "container-grade" isolation would be false.
- **Decision:** Report `detected_summary()["capabilities"]["level"]` verbatim in docs, publish the 12-case escape suite with a per-case verdict and expectation, and document residual risks (`/etc` shared, other same-uid processes visible, sandbox root on host fs) instead of hiding them.
- **Status:** `VERIFIED` (see `docs/SANDBOX.md`, `docs/audit_evidence/sandbox_escape_suite.out.txt`)

## DEC-007: Loss is not capability — the tool-SFT acceptance rule (2026-10-02)

- **Context:** `EXP-004-TOOL-SFT-V2` reached val_ppl 5.41 (better than its predecessor) and still completed **0 of 20** tool tasks. A reviewer optimising on the loss curve would have shipped it.
- **Decision:** A checkpoint is accepted only when `scripts/evaluate_tools.py` reports non-zero `task_completion`, `multi_step_success`, `error_recovery_success` and `citation_validity`. Loss/ppl is reported for reproducibility but is never an acceptance criterion (`docs/TRAINING_REPORT.md` §22 record).
- **Consequence:** All three SFT iterations are published as **FAILED** with their measurements, and `release/` ships the best-measured checkpoint while `FINAL_REPORT.md` states plainly that capability is not achieved.
- **Status:** `VERIFIED` (decision enforced; see the capability results table in `docs/AUDIT_MATRIX.md`)

## DEC-008: A repeated target sentence is a defect, not a style choice (2026-10-02)

- **Context:** The corpus contained one code-repair thought sentence 382 times with **1 distinct value** — 5.7 % of all target segments. The model learned it as a high-probability continuation and emitted it *in place of* task content (including as the final answer to arithmetic prompts).
- **Decision:** Every template family must draw its non-task text from a pool (`data_sources/synthetic_v2.py::_THOUGHTS`, 6 phrasings per family) and the text itself must be justified before it enters the loss. Target-text repetition is treated like data contamination: measured, reported, and fixed at the data level.
- **Rationale:** In a 0.72 M-token corpus, one string at 5.7 % frequency outranks the compositional knowledge the run is supposed to learn.
- **Status:** `VERIFIED` (distinct thoughts 1 → 6 per family; `dataset_v3` rev4 rebuilt, contamination PASS)

## DEC-009: Quantisation selection by measured functional regression (2026-10-02)

- **Context:** Raw round-trip error says fp16 is near-lossless and int4 is 450× worse, but the audit asks for *functional* regression, and this model's fp32 reference is already failing.
- **Decision:** Publish all four variants with both the numeric round-trip error and the functional suites (tool + per-domain). Designate **fp32 as the reference**; advertise int4 explicitly as "smallest, measurably degraded" (tool syntax 0.84 → 0.40, execution 0.50 → 0.00) instead of implying it is equivalent.
- **Status:** `VERIFIED` (`docs/QUANTIZATION_REPORT.md`, `release/model_comparison.md`)

## DEC-010: Portability and capability are never merged into one claim (2026-10-02)

- **Context:** The §14 clean-room gate passed — `python3 inference.py` ran with empty `PYTHONPATH` and no repo — while producing the wrong answer (`-799` for a straightforward arithmetic prompt).
- **Decision:** Report §14 as *portability* evidence, and report the correctness of the generated text separately under §4/§2C. A passing gate is never presented as capability.
- **Status:** `VERIFIED` (`FINAL_REPORT.md` §3 and §7.5)
