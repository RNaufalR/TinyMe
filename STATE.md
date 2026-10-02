# PROJECT STATE — TASK OBSERVER

**Last Updated:** 2026-10-02 (corrective-audit execution, session branch `arena/01a0fa8e-tinyme`)

- **CURRENT TASK:** corrective audit execution — P0 correctness layer **complete and test-covered**;
  P1 data engine complete; P1 tool runtime + sandbox implemented; canonical retraining (EXP-002) in progress.
- **CURRENT STATE:**
  - `docs/CORRECTIVE_AUDIT.md` records every defect with executed-probe evidence (`docs/audit_evidence/*.out.txt`).
    Headline confirmed defects: RoPE vs explicit rotation max|Δ| 4.076; EXP-001 "held-out" ppl 2.455 derived
    from `train.jsonl[:32]`; `eval.jsonl` had 0 `token_ids`; trainer applied the optimizer once per *microbatch*
    (no gradient accumulation); checkpoints carried no optimizer/RNG/sampler state; dedup LSH bucket loop was dead
    code; shard writer re-wrote the whole `.npy` per append; `run_sandboxed_python` was a bare `subprocess.run(cwd="/tmp")`.
  - **P0 fixes implemented + covered by tests:** content-aware preprocessing (code indentation, fences, control
    tokens, AST validation on the *target*, malformed code = hard reject), structured segmented record contract
    (`SYSTEM/USER/ASSISTANT/THOUGHT/TOOL_CALL/TOOL_RESULT/FINAL/CODE`) with `loss_mask` and `IGNORE_INDEX=-100`,
    task-aware loss masking (tool results context-only), padding mask (pad_id → zero loss, active/masked/padding
    reported), genuine train/validation/test/challenge split on `group_id` with contamination PASS/FAIL gates,
    trainer reads real `validation` and refuses `train.jsonl`, tokenizer trained on the train split only with a
    startup vocab assertion, one authoritative sequence builder (`seq_len` 256 primary, target-aware truncation),
    true gradient accumulation (params frozen across microbatches + numerical-equivalence test), deterministic
    sampler with persisted state, atomic checkpoint save/load/verify with fingerprints, scheduler/optimizer state
    resume, RNG sync (python/numpy/JAX/data/synthetic), RoPE + cached-vs-full parity tests, NaN/Inf guards and
    grad-norm logging, `compute_dtype` that changes real arithmetic and is recorded in the run config.
  - **Data engine:** `scripts/prepare_data_v2.py` builds `dataset_v2` end-to-end (ingest → license → preprocess →
    quality → dedup → group split → tokenizer(train only) → pack/shard → manifest + docs). Latest build:
    **train 3062 / validation 259 / test 259 / challenge 60**, 6757 kept records (2 hard-rejected malformed code
    targets), dedup 6757 → 3700, **585 187 active train target tokens**, padding ratio 0.161,
    **contamination PASS** on all split pairs, 19 special tokens (audit-mandated `tool_call`/`tool_result`/`final`
    control tokens with `end_tool_call`/`end_tool_result`).
  - **Tool runtime + sandbox:** `src/agent/{protocol,tool_registry,router,planner,executor,evidence}.py` and
    `src/tools/{search,fetch,compute,code,files,retrieval,context}.py`. Isolation is auto-detected and labelled
    honestly: this host supports user+mount+net namespaces (`unshare -Urnm`), so executed code has **no network**
    (verified: `socket.create_connection` → OSError), the host home tree is a tmpfs, the environment is scrubbed,
    CPU/memory/file/output/fd rlimits apply and the process cap is enforced with `prlimit` inside the namespace
    (verified: fork bomb stops after `max_processes` children). Offline BM25 retrieval index (408 documents,
    181 KB corpus) with provenance on every passage.
  - **Test suite:** 139 tests pass (`python3 -m pytest tests/ -q`), covering tokenizer, preprocessing, sequences,
    loss/padding masks, gradient accumulation, checkpoint/resume, sampler, RoPE, KV-cache parity, dedup,
    split contamination, dataset API, quality filters, model shapes, dtypes, numpy/JAX parity, inference,
    numerical stability, RNG determinism, tool protocol, tools, sandbox and agent loop.
- **BLOCKERS:**
  - `huggingface.co` / `raw.githubusercontent.com` remain TLS-blocked from the sandbox; the HF corpus is the
    locally cached slice already in the repository, and every synthetic record is independently verified.
- **LAST VERIFIED RESULT:** `python3 -m pytest tests/ -q` → **139 passed**; `dataset_v2` manifest →
  contamination PASS, malformed_code_rate 0.0; `run_python` escape probes → network blocked, `/home` hidden.
- **NEXT ACTION:** finish the canonical EXP-002 run, then EXP-003 (base feasibility pilot), EXP-004 (tool/instruction
  SFT), EXP-005 (quantization), wire `scripts/{evaluate,package_model,quantize}.py` + `release/`, write
  `docs/{TRAINING_REPORT,MODEL_COMPARISON}.md` and `FINAL_REPORT.md`, rewrite `README.md`, and push to
  `arena/01a0f722-tinnyme` (via pull request from the session branch).
