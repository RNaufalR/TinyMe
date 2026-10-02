# PROJECT STATE — TASK OBSERVER

**Last Updated:** 2026-10-02 (audit execution + capability iteration, session branch `arena/01a0fc10-tinyme`, content-identical to `arena/01a0fa8e-tinyme` @ `8446a03`)

- **CURRENT TASK:** `TinyMeTask.md` §1–§29 execution and audit. All 29 sections are
  implemented, executed and reported in `docs/AUDIT_MATRIX.md`; the capability
  sections are reported as **FAILED** with measurements (see §4 below).
- **CURRENT STATE:**
  - **Corrected training (fresh initialisation, no resume of invalidated runs):**
    `EXP-002-CORRECTED-NANO` 400 steps, val_ppl 122.02, rerun byte-identical on all
    400 logged steps. Then three tool-SFT iterations:
    `EXP-004-TOOL-SFT` (320 steps, ppl 7.21) → `EXP-004-TOOL-SFT-V2` (700 steps,
    ppl 5.41) → `EXP-004-TOOL-SFT-V3` (600 steps, **ppl 4.62**). Base feasibility
    pilot `EXP-003-BASE-PILOT` (25 steps, 8,933,440 params, 1,000.8 tok/s).
  - **Capability verdict: FAILED.** §4 tool suite on the best checkpoint:
    `task_completion` 0.00, `multi_step_success` 0.00, `error_recovery_success`
    0.00, `citation_validity` 0.00, `grounded_final_answer` 0.30,
    `execution_success` 0.50, `tool_syntax_validity` 0.84, `argument_validity`
    1.00. Test split (2,158 samples) mean accuracy **0.0571**; challenge split
    mean accuracy **0.1000**. Verdict recorded per §22 (never accepted on loss).
  - **Root cause, measured:** one repeated target sentence (382 copies, 1 distinct
    value, 5.7 % of all target segments) taught a memorisation prior; fixed at the
    data level (`_THOUGHTS` pool, 6 phrasings/family → `dataset_v3` **rev4**,
    7,220 train records, 718,077 active tokens, contamination PASS). The remaining
    constraint is corpus scale: 718,077 tokens = **1.40 %** of the 20N budget
    (`docs/CAPACITY_ANALYSIS.md`).
  - **Release:** `release/` rebuilt from `EXP-004-TOOL-SFT-V3/best` — 13 files,
    19,688,665 B (18.78 MiB); variants fp32 10,233,776 / fp16 5,118,480 /
    int8 2,600,976 / int4 1,451,272 B; `< 50 MB` true for all four.
  - **Clean-room gate:** `/tmp/cleanrel2` with `env -i`, empty `PYTHONPATH` and
    only release files → `python3 inference.py` loads 2,557,632 params and
    generates. Portability VERIFIED; the answer itself is wrong (capability
    FAILED), and the two are reported separately.
  - **Sandbox:** 18-case escape suite, 18 PASS, 16 blocked-or-contained, label
    `namespace(net+mount+pid)`.
  - **Tests:** `python3 -m pytest tests/ -q` → **273 passed** (33 modules).
- **BLOCKERS:**
  - `huggingface.co` / `raw.githubusercontent.com` remain TLS-blocked from the
    sandbox; the HF corpus is the locally cached slice in the repository and
    every synthetic record is independently verified.
  - Corpus scale (0.72 M active tokens vs 51.2 M needed for 20N) — a data
    problem, not a compute problem; Base would repeat the same failure.
- **LAST VERIFIED RESULT:** `pytest tests/ -q` → 273 passed; RoPE vs independent
  rotation reference 4.287e-07 (tolerance 1e-5); NumPy↔JAX parity 1.53e-05;
  release clean-room gate loads and generates; contamination PASS.
- **NEXT ACTION:** none required for the audit — all 29 sections are closed with
  a status. Any future work must first raise the corpus above ~6N active tokens
  with genuinely diverse multi-step supervision (`docs/CAPACITY_ANALYSIS.md` §5).
