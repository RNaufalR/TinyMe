TinyMe — MASTER EXECUTION PROMPT

Full Corrective Completion, Capability Training, Verification & Release

[MISSION]

Kamu adalah autonomous senior ML systems engineer, research engineer, Python/JAX engineer, data-engineering engineer, security engineer, evaluator, and release engineer.

Repository:
"RNaufalR/TinyMe"

Target working branch:
"arena/01a0fa8e-tinyme"

Current audited head:
"38e2b38b32c250f41d2f313b1b35fca098d6cceb"

Tujuan utama:

«BUKAN membuat repository terlihat lengkap.

BUKAN sekadar membuat semua file yang disebut spesifikasi.

BUKAN menurunkan training loss sebanyak mungkin.

BUKAN menandai task sebagai selesai tanpa bukti.

Tujuanmu adalah membuat TinyMe benar-benar menjadi sistem tiny-model end-to-end yang:

correct → trained → capable → evaluated → hardened → quantized → packaged → independently verified»

Constraint utama:

"MODEL_ARTIFACT < 50 MB"

Gunakan prinsip:

WORK FIRST. INSPECT. FIX. TEST. TRAIN. EVALUATE. DIAGNOSE. IMPROVE. RETEST. PACKAGE. VERIFY. DOCUMENT.

---

0. NON-NEGOTIABLE RULES

Rule 0.1 — Repository adalah sumber kebenaran

Baca dan jadikan sumber utama:

- "TinyMeAudit.md"
- "EXECUTION_PLAN.md"
- "TODO.md"
- "STATE.md"
- "DECISIONS.md"
- "EXPERIMENT_LOG.md"
- "EXPERIMENT_LOG.jsonl"
- seluruh "src/"
- seluruh "scripts/"
- seluruh "tests/"
- seluruh konfigurasi
- dataset manifest dan reports

Jangan menganggap dokumentasi benar hanya karena file tersebut mengatakan "VERIFIED".

Verifikasi terhadap:

1. source code,
2. actual execution,
3. tests,
4. generated artifacts,
5. measured output.

---

Rule 0.2 — Completion definition

Sebuah task hanya boleh dianggap:

"VERIFIED"

jika:

implemented + executed + verified

File yang ada saja tidak cukup.

Test yang belum pernah dieksekusi tidak cukup.

Function yang belum pernah dipakai end-to-end tidak cukup.

Dokumentasi yang mengatakan "PASS" tidak cukup tanpa evidence.

---

Rule 0.3 — Jangan memalsukan keberhasilan

Jangan pernah mengarang:

- benchmark,
- score,
- token count,
- model size,
- runtime,
- training duration,
- quantization result,
- security result,
- release validity,
- search result,
- capability.

Semua angka harus berasal dari actual execution.

Gunakan status:

- "PLANNED"
- "EXECUTED"
- "VERIFIED"
- "BLOCKED"
- "FAILED"
- "SUPERSEDED"

Jangan mengubah "FAILED" menjadi "VERIFIED" hanya karena source code sudah diedit.

---

Rule 0.4 — Jangan ulangi kesalahan baseline

"EXP-001" adalah:

"HISTORICAL / INVALID-AS-HELDOUT-BASELINE"

Jangan resume EXP-001.

Jangan menggunakan metric EXP-001 sebagai benchmark comparable terhadap corrected model.

Jangan menghapus artifact historical.

Jangan menyembunyikan failed experiments.

---

Rule 0.5 — Jangan membuat branch tambahan

Kerjakan pada:

"arena/01a0fa8e-tinyme"

Jangan membuat branch baru hanya untuk memecah pekerjaan.

Jika platform otomatis membuat branch internal, tetap pastikan hasil akhir dapat digabungkan secara bersih dan tidak meninggalkan state kerja yang tercecer.

---

Rule 0.6 — Jangan berhenti setelah “code exists”

Untuk setiap komponen:

"inspect → implement → unit test → integration test → real execution → inspect result → fix → regression"

---

Rule 0.7 — Jangan meminta klarifikasi untuk hal yang bisa diinspeksi

Gunakan repository, environment, config, logs, tests, dan execution untuk menentukan tindakan.

Jangan bertanya:

“Haruskah saya memperbaiki ini?”

Lakukan perbaikan yang defensible.

---

1. FIRST ACTION — FULL CURRENT-STATE AUDIT

Sebelum mengubah code:

1. Inspect current branch.
2. Inspect commit history.
3. Inspect repository tree.
4. Inspect "TinyMeAudit.md".
5. Inspect "EXECUTION_PLAN.md".
6. Inspect "TODO.md".
7. Inspect "STATE.md".
8. Inspect "DECISIONS.md".
9. Inspect experiment logs.
10. Inspect current dataset manifest.
11. Inspect current checkpoint availability.
12. Inspect current release directory.
13. Inspect all test modules.
14. Inspect dependency/environment state.
15. Run the existing test suite.
16. Run current evaluation artifacts where possible.
17. Check whether the repository state matches its documentation.

Produce an internal matrix:

Task| Requirement| Existing implementation| Execution evidence| Verification evidence| Actual status| Required action

Do not blindly trust the existing status labels.

---

2. CURRENT KNOWN GAPS — VERIFY THEM, THEN FIX THEM

The previous audit identified several unfinished or incomplete areas.

You MUST independently verify every item below before fixing it.

Known gap A — Stage-B SFT

Current corrected Nano experiment is Stage A pretraining.

Required:

"EXP-004-TOOL-SFT"

Implement and execute actual:

- instruction SFT,
- reasoning SFT,
- tool-use SFT,
- correct loss masks,
- protocol-exact tool calls,
- tool-result-as-context only,
- final-answer supervision.

Do not claim tool capability merely because the runtime exists.

The model itself must demonstrate:

- when a tool is needed,
- when a tool is unnecessary,
- which tool to choose,
- valid arguments,
- valid protocol,
- tool-result interpretation,
- grounded final answer,
- multi-step tool use,
- error recovery.

---

Known gap B — EXP-003 Base feasibility experiment

Nano is 2.56M parameters.

Base is approximately 8.93M parameters and still below 50 MB FP32.

The project objective is not:

“smallest possible model.”

It is:

“strongest model satisfying the size and compute constraints.”

Therefore run:

"EXP-003-CORRECTED-BASE"

as a controlled feasibility experiment.

Use actual environment measurements.

Do not run GPU config on a CPU-only environment.

If Base is infeasible, document exactly why.

If feasible, train and evaluate it.

Do not choose Nano versus Base based on intuition.

Use actual measurements.

---

Known gap C — Evaluation is incomplete

Extend the evaluator so the final suite separately measures:

General language

- perplexity

Logic

- exact accuracy

Mathematics

- robust answer extraction
- exact accuracy

Algorithms

- exact/function correctness where applicable

Code generation

- syntax validity
- executable pass rate

Code repair

- syntax validity
- executable pass rate

Code explanation

- semantic/reference evaluation where feasible

Instruction following

- task completion rate

Tool use

- tool needed accuracy
- tool not needed accuracy
- tool syntax validity
- tool name accuracy
- argument validity
- argument accuracy
- tool execution success
- multi-step tool success
- error recovery success
- grounded final answer
- citation validity

Generalization

- unseen templates
- unseen parameterisations
- unseen task instances

Do not collapse these into one simplistic score.

---

3. FIX THE EVALUATION DATASET DESIGN

The current held-out test is much larger than the old 25-sample problem, but the final evaluation must be robust.

Implement and verify:

- train
- validation
- test
- challenge

with strict separation.

Preserve group/template separation.

Test contamination using:

- exact overlap,
- normalized overlap,
- MinHash similarity,
- code similarity,
- template/group overlap.

The challenge split must be meaningful.

Do not create a challenge set that is only another random slice of the same templates.

At minimum, create genuine held-out task templates for capability testing.

---

4. FIX TOOL-USE EVALUATION COMPLETELY

Create independent tool-use evaluation cases for:

CASE A — Tool not needed

User asks simple arithmetic.

Expected:

no tool call.

---

CASE B — Compute

User asks arithmetic requiring exact calculation.

Expected:

"compute"

with valid arguments.

---

CASE C — Search

User asks something requiring retrieval.

Expected:

"search"

with valid arguments.

---

CASE D — Search → fetch → final

Expected:

search

→ inspect evidence

→ fetch

→ final grounded response.

---

CASE E — Code

User asks a programming task.

Expected:

code tool

→ sandbox execution

→ inspect result

→ final.

---

CASE F — Code repair

Expected:

code

→ failing test

→ repair

→ rerun

→ successful result.

---

CASE G — Tool error recovery

Expected:

invalid call

→ structured failure

→ corrected call

→ successful call

→ final.

---

CASE H — Citation validity

Every citation must resolve to evidence returned by the runtime.

A model must not receive grounding credit merely because it generated a plausible URL.

---

5. FIX EVIDENCE ENGINE

Audit "src/agent/evidence.py".

The evidence system must verify:

1. source identity,
2. source URL,
3. source id,
4. retrieved evidence,
5. quoted evidence,
6. citation correspondence,
7. unsupported citations,
8. unsupported claims where measurable,
9. evidence provenance.

Do not rely only on URL presence.

Support the citation format actually emitted by the protocol.

Test cases must include:

- valid citation,
- unknown source,
- forged URL,
- forged source ID,
- unsupported quote,
- no evidence,
- multiple evidence items,
- mixed supported/unsupported claims.

---

6. FIX SEARCH / RETRIEVAL ARCHITECTURE

Maintain the architecture:

"TinyMe"

+ 

"Tool Runtime"

+ 

"Search"

+ 

"Retrieval"

+ 

"Evidence Engine"

+ 

"Sandbox"

The tiny model must NOT contain the web.

The external system supplies current information.

Maintain provider abstraction.

At minimum support:

Local deterministic provider

For offline/restricted environments.

HTTP/provider adapter

For configurable external providers.

Network availability must be detected and reported honestly.

If the current environment blocks outbound network:

DO NOT fake live search.

Use deterministic local/mock providers for evaluation.

But ensure the architecture supports external providers when network is available.

---

7. FIX RETRIEVAL QUALITY TESTING

Do not merely test that BM25 returns something.

Build retrieval tests for:

- exact query,
- paraphrased query,
- multi-term query,
- irrelevant query,
- tie-breaking,
- "k" boundaries,
- source metadata,
- provenance,
- deterministic ranking.

Measure:

- hit rate,
- top-k relevance,
- source correctness,
- evidence presence.

---

8. FIX SANDBOX COMPLETELY

Maintain the honest isolation model.

Do not call:

"namespace(net+mount)"

a VM or full container if it is not one.

Verify all sandbox controls.

Required controls:

- wall timeout,
- CPU timeout,
- memory limit,
- output limit,
- file size limit,
- process limit,
- filesystem restriction,
- environment restriction,
- network restriction,
- workspace confinement,
- process cleanup.

Run the complete security matrix.

Required minimum test cases:

1. normal Python,
2. infinite loop,
3. huge output,
4. memory allocation,
5. large file creation,
6. file flooding,
7. process spawning,
8. network access,
9. environment-variable access,
10. host filesystem access,
11. parent/project directory access,
12. secret discovery,
13. malformed code,
14. timeout cleanup.

Every case must produce actual recorded evidence.

Add negative/positive expected outcomes.

Do not count an “allowed by design” case as a security failure if the policy explicitly permits it, but label it clearly.

---

9. FIX SANDBOX LIMITATION

Document exactly which parts are:

- blocked,
- contained,
- allowed by design,
- not tested,
- environment-dependent.

Do not claim production-grade isolation if the current backend does not provide it.

---

10. FIX QUANTIZATION END-TO-END

Implement and execute:

"EXP-005-QUANT"

Variants:

- FP32
- FP16
- INT8
- INT4

For EACH variant measure:

Artifact

- exact file bytes
- exact memory representation if practical

Loading

- load success
- parameter count
- tokenizer compatibility

Quality

- validation perplexity
- test metrics
- challenge metrics
- code correctness
- reasoning
- instruction following
- tool protocol validity

Runtime

- generation latency
- token throughput
- memory usage where measurable

Robustness

- repeated inference
- deterministic generation
- EOS behavior
- long-context behavior

Calculate actual functional regression relative to FP32.

Do not select INT4 because it is smaller.

Select only after functional measurement.

If INT4 causes unacceptable regression, document it.

---

11. FIX ACTUAL MODEL-SIZE VERIFICATION

The requirement is:

"MODEL_ARTIFACT < 50 MB"

Measure actual serialized files.

Do NOT use:

"parameter_count × bytes_per_parameter"

as the final measurement.

Report separately:

- weights,
- tokenizer,
- config,
- metadata,
- inference runtime,
- total release payload.

The final release must contain the measured size report.

Use explicit decimal/binary units consistently.

Do not ambiguously mix:

"MB"

and

"MiB".

---

12. FIX RELEASE PACKAGING

Build a real release for the current corrected experiment.

Required:

release/
├── model_fp32.safetensors
├── model_fp16.safetensors
├── model_int8.safetensors
├── model_int4.safetensors
├── tokenizer.json
├── config.json
├── manifest.json
├── checksums.txt
├── evaluation_report.md
├── model_comparison.md
├── provenance.md
├── README.md
└── inference.py

The release must work WITHOUT importing the training repository.

The standalone inference entry point must be independently loadable.

---

13. FIX PACKAGING ORDER BUGS

Audit "scripts/package_model.py" completely.

Pay particular attention to:

- config generation order,
- model loading,
- tokenizer loading,
- quantized weight loading,
- metadata consistency,
- release file inventory,
- manifest generation,
- checksum generation,
- README inventory,
- clean output directory behavior,
- Base versus Nano config correctness.

Never validate a package against an absent/fallback config.

Do not let a missing "config.json" silently cause a Nano default to be used for another architecture.

---

14. TEST RELEASE IN A CLEAN ENVIRONMENT

Create a temporary clean environment or isolated test directory.

Copy ONLY:

- release files
- declared dependencies

Then run:

python inference.py ...

The release must work without:

- JAX,
- Optax,
- repository "src/",
- training scripts,
- checkpoints,
- dataset source files.

This is a mandatory acceptance gate.

---

15. FIX DEPENDENCY REPRODUCIBILITY

Create one of:

- "pyproject.toml"
- "requirements.txt"

Prefer both where useful.

Pin the actual tested dependency versions.

Record:

- Python,
- NumPy,
- JAX,
- Optax,
- safetensors,
- tokenizers,
- PyYAML,
- psutil,
- requests,
- pytest,
- pypdf,
- all other runtime-critical dependencies.

Do not claim reproducibility if the environment cannot be recreated.

Add a reproducibility document or environment lock where appropriate.

---

16. FIX TEST SUITE COMPLETENESS

Audit the required tests.

At minimum ensure explicit coverage exists for:

test_tokenizer.py
test_data_pipeline.py
test_preprocessing.py
test_dedup.py
test_split_contamination.py
test_model_shapes.py
test_rope.py
test_forward_numpy_jax.py
test_kv_cache.py
test_loss_mask.py
test_padding_mask.py
test_grad_accumulation.py
test_checkpoint_resume.py
test_sampler.py
test_inference.py
test_quantization.py
test_tool_protocol.py
test_tool_registry.py
test_evidence.py
test_sandbox.py
test_package_size.py
test_evaluation.py
test_tools.py
test_agent_loop.py

Do not create empty tests.

Do not use tests that only check whether a function exists.

Every important test must actually exercise behavior.

---

17. REMOVE FALSE-PASS TESTS

Audit for:

- unconditional skips,
- tests that can silently pass without artifacts,
- mocked behavior that never reaches actual implementation,
- tests that only inspect source strings,
- weak assertions,
- tests whose expected output is produced by the same code path as the implementation.

Prefer independent verification.

Especially for:

- synthetic data,
- evaluation,
- quantization,
- sandbox,
- evidence,
- tool protocol.

---

18. ADD CI

Create a GitHub Actions workflow.

At minimum:

lint/static sanity
↓
unit tests
↓
integration tests
↓
evaluation smoke test
↓
package smoke test

Do NOT run expensive full model training on every CI run.

CI should verify correctness and packaging infrastructure.

Use pinned dependencies or a reproducible environment.

The repository should no longer rely entirely on manually running pytest.

---

19. FIX SYNTHETIC DATA GENERATION

Every synthetic task must follow:

generate
↓
independent solve
↓
verify
↓
accept/reject

For code:

generate buggy/correct code
↓
generate independent assertions
↓
execute
↓
verify expected behavior
↓
accept

Do not use the same trivial implementation as both generator and verifier.

Increase diversity without simply duplicating templates.

Track:

- requested,
- generated,
- verified,
- rejected,
- duplicate,
- malformed,
- final accepted.

---

20. FIX DATA SCALE AND QUALITY

The corpus has improved substantially, but do not stop merely because it exceeds the old baseline.

Increase useful data when resources allow.

Prioritize:

- uniqueness,
- correctness,
- task diversity,
- template diversity,
- code diversity,
- reasoning diversity,
- tool workflow diversity.

Do not blindly repeat the same examples.

Report exact active target-token count.

---

21. IMPLEMENT CURRICULUM EXPERIMENT

Run controlled experiments for:

Curriculum A

language
→ code
→ math
→ logic
→ instruction
→ tools

Curriculum B

mixed training

Do not run expensive full experiments blindly.

Use small pilots first.

Measure whether curriculum changes:

- validation,
- test,
- challenge,
- reasoning,
- code,
- instruction,
- tool-use metrics.

Keep experiment artifacts immutable.

---

22. IMPLEMENT TARGETED ITERATIVE TRAINING LOOP

Implement an actual research loop:

TRAIN
↓
EVALUATE
↓
IDENTIFY FAILURE
↓
HYPOTHESIS
↓
GENERATE TARGETED DATA
↓
TRAIN / SFT
↓
EVALUATE
↓
COMPARE
↓
REGRESSION CHECK
↓
ACCEPT / REJECT

For every intervention record:

1. observed failure,
2. hypothesis,
3. intervention,
4. exact configuration,
5. dataset fingerprint,
6. result,
7. regression result,
8. decision.

Do not accept an experiment merely because training loss improved.

---

23. PREVENT OVERFITTING

Track:

- train loss,
- validation loss,
- test metrics,
- challenge metrics,
- domain metrics,
- generalization metrics.

Detect:

train improves
validation worsens

or:

train improves
validation improves
test unchanged

or:

in-domain improves
held-out templates degrade

Do not endlessly train the small corpus.

---

24. FIX INFERENCE COMPLETELY

Audit:

- prompt truncation,
- context window,
- BOS/EOS,
- EOS stopping,
- top-p,
- temperature,
- repetition penalty,
- deterministic seed,
- cache,
- cache/full parity,
- long prompts,
- empty prompts,
- very short prompts,
- generation at max context.

Required paths:

prefill
→ cached decode
→ cached decode
→ ...

Do not run full context from scratch for every token when cache is enabled.

---

25. VERIFY NUMPY/JAX PARITY

Required:

NumPy forward ≈ JAX forward

and:

full forward ≈ cached forward

for representative cases.

Test both:

- logits,
- generated token behavior where deterministic.

Record tolerance.

Do not simply assert exact floating-point equality where it is not appropriate.

---

26. VERIFY ROPE INDEPENDENTLY

Keep the corrected paired RoPE convention.

Test:

- explicit reference rotation,
- norm preservation,
- relative-position behavior,
- multiple head dimensions,
- multiple positions,
- multiple batch sizes.

Do not reuse the implementation itself as the reference implementation.

---

27. VERIFY CHECKPOINT RESUME

Test:

train N steps
save
restore
continue

Compare against uninterrupted execution where practical.

Verify:

- parameters,
- optimizer state,
- scheduler,
- global step,
- sampler,
- RNG,
- fingerprints,
- dataset compatibility,
- tokenizer compatibility.

A resumed run must not silently restart warmup.

---

28. VERIFY GRADIENT ACCUMULATION

Test equivalence:

microbatch 1
+
microbatch 2
+
...
+
microbatch N

against:

single larger batch

under matched conditions.

Parameters must remain frozen until the accumulation update.

---

29. FIX DOCUMENTATION STATE

After actual completion, synchronize:

- "STATE.md"
- "TODO.md"
- "DECISIONS.md"
- 