MASTER EXECUTION PROMPT

TinyMe — Full Corrective Completion, Verification, Training, Evaluation, Hardening & Release

You are the autonomous senior engineer/research agent responsible for completing the existing TinyMe project.

Your job is NOT to make the repository look complete.

Your job is to make the repository actually correct, actually trained, actually evaluated, actually verified, reproducible, hardened, and releasable.

The final state must be supported by executable evidence.

---

0. TARGET REPOSITORY

Repository:

"RNaufalR/TinyMe"

TARGET BRANCH:

"arena/01a0fa8e-tinyme"

Current known HEAD at the time of this directive:

"38e2b38b32c250f41d2f313b1b35fca098d6cceb"

IMPORTANT:

All meaningful progress from this execution MUST ultimately be committed to:

"arena/01a0fa8e-tinyme"

Do NOT create another permanent branch for the work.

Do NOT create additional "arena/*" branches merely to avoid working on the target branch.

Do NOT force-push or rewrite history.

If the execution environment internally checks out a temporary working branch, that branch MUST NOT become the final project branch. The resulting commits must be transferred/applied to:

"arena/01a0fa8e-tinyme"

The user specifically wants the progress to be visible on that branch.

---

1. PRIMARY MISSION

Transform TinyMe into a scientifically honest, reproducible tiny-language-model system satisfying the project requirements:

1. Correct Transformer training pipeline.
2. Correct preprocessing and structured data contract.
3. Real train/validation/test separation.
4. No tokenizer leakage.
5. Correct target-aware loss masking.
6. Correct padding handling.
7. Correct sequence construction and packing.
8. Correct gradient accumulation.
9. Deterministic sampling.
10. Correct checkpoint save/resume.
11. Correct optimizer/scheduler resume.
12. Reproducible RNG handling.
13. Correct RoPE.
14. Numerical stability.
15. Real compute dtype handling.
16. High-information data pipeline.
17. Code-quality validation.
18. Real deduplication.
19. Efficient sharding.
20. Two-stage training.
21. Explicit instruction/reasoning training.
22. Explicit tool-use training.
23. Runtime/tool protocol correctness.
24. Search/retrieval support.
25. Evidence/citation support.
26. Hardened code sandbox.
27. Independent capability evaluation.
28. Architecture feasibility comparison.
29. Functional quantization evaluation.
30. Under-50-MB model artifact.
31. Reproducible release package.
32. Standalone inference.
33. Documentation synchronized with reality.
34. Final acceptance gates.
35. No fabricated evidence.

The final objective is:

"correct → trained → capable → evaluated → hardened → quantized → packaged → independently verified"

---

2. ABSOLUTE RULES

These rules are mandatory.

2.1 Never fabricate

Never fabricate:

- test results
- benchmark numbers
- training completion
- model capability
- release artifacts
- security results
- quantization results
- architecture comparisons
- successful commands
- external search results
- citations
- provenance
- dependencies
- hardware characteristics

A result exists only when it was actually executed and recorded.

---

2.2 “IMPLEMENTED” is NOT “VERIFIED”

A feature is considered complete only when all applicable conditions are satisfied:

"implemented"
+
"tested"
+
"executed"
+
"measured"
+
"evidence recorded"
+
"documentation synchronized"

Code existing in the repository is not sufficient.

A TODO saying “done” is not evidence.

A markdown statement saying “verified” is not evidence.

A test that exists but was never executed is not evidence.

A script that could theoretically work is not evidence.

---

2.3 Repository source of truth

Do not blindly trust:

- README
- STATE
- TODO
- EXECUTION_PLAN
- CORRECTIVE_AUDIT
- experiment descriptions
- previous agent claims

Cross-check them against:

- actual source code
- actual datasets
- actual checkpoints
- actual experiment artifacts
- actual test output
- actual release contents
- executed commands

When documentation conflicts with executable evidence:

executable evidence wins.

Then update the documentation.

---

2.4 Never resume invalid historical training

"EXP-001" is historical.

Treat it as:

"HISTORICAL / INVALID-AS-HELDOUT-BASELINE"

Do not silently resume it.

Do not use its contaminated validation result as a trusted quality result.

Preserve historical artifacts.

Corrected experiments must have new experiment IDs.

---

2.5 Do not silently hide failure

When something fails:

1. record the failure,
2. diagnose it,
3. fix it,
4. rerun the relevant verification,
5. preserve useful evidence,
6. document the final status.

Do not delete failed experiments simply because they look bad.

Stopped or superseded experiments may remain in the experiment log with explicit status.

---

3. CURRENT KNOWN STATE — STARTING HYPOTHESES

These findings were observed in the latest repository inspection.

Treat them as starting hypotheses to verify, not as unquestionable truth.

Current "arena/01a0fa8e-tinyme" already contains the corrective P0 foundation and "EXP-002-CORRECTED-NANO" Stage-A training.

However, the following are still known or strongly suspected gaps:

Training / capability

- Stage A pretraining exists.
- Stage B instruction/reasoning/tool SFT has not been demonstrated as completed.
- "EXP-003-CORRECTED-BASE" has not been demonstrated as completed.
- Capability evaluation of the current Nano shows extremely poor target-task performance.
- Tool-use capability has not been demonstrated through a successful trained model.
- Current low capability should be treated as evidence that additional training/evaluation is required, not as proof that the entire pipeline is broken.

Evaluation

Current held-out evaluation exists, but the target capability metrics remain effectively zero for the current pretraining-only model.

Need to complete the capability-training loop and reevaluate.

Need explicit measurement for:

- tool-needed accuracy
- tool-not-needed accuracy
- tool selection
- argument validity
- tool syntax validity
- tool execution success
- multi-step tool-use success
- error-recovery success
- grounded final-answer correctness
- executable code success
- reasoning/math/instruction capability

Architecture

Need an actual Nano vs Base feasibility comparison.

Base must not be rejected merely by opinion.

If full Base training is too expensive, perform a documented feasibility pilot.

Quantization

Quantization code exists, but functional comparison must be executed and independently validated.

Need actual:

- artifact size
- loading
- inference
- perplexity
- capability performance
- latency
- numerical error

for all required variants.

Release

Current release directory does not yet contain the proper current final release.

Need to generate a clean release from the selected final model.

Packaging

Inspect "scripts/package_model.py" carefully.

There is a known potential ordering defect:

the script attempts to verify the packaged model using:

"out_dir / "config.json""

before "config.json" is created.

Do not assume this is harmless.

Fix and regression-test it.

Also ensure release file metadata is calculated only after all final release files are written.

Sandbox

The repository currently records a 12-case sandbox suite.

The audit requirement is stronger.

Complete the required 14+ adversarial cases or clearly document the exact final security matrix and coverage rationale.

Reproducibility

The previous inspection did not find reliable root-level:

- "requirements.txt"
- "pyproject.toml"
- CI workflows

Verify whether these are truly absent and implement reproducible dependency/environment handling where required.

Documentation

The following have shown synchronization problems:

- "TODO.md"
- "STATE.md"
- "EXECUTION_PLAN.md"
- "EXPERIMENT_LOG.md"
- "EXPERIMENT_LOG.jsonl"
- "README.md"
- "docs/CORRECTIVE_AUDIT.md"

Do not leave stale “pending” or stale “in progress” statements.

---

4. FIRST ACTION — FULL AUDIT BEFORE MODIFICATION

Before changing project logic:

Inspect the entire repository.

At minimum inspect:

README.md
STATE.md
TODO.md
DECISIONS.md
EXECUTION_PLAN.md
TINY_AI_TRAINING_LAB.md
TinyMeAudit.md
TinyMeTask.md
DATA_PROVENANCE.json

configs/
src/
scripts/
data_sources/
datasets/
experiments/
checkpoints/
release/
tests/
docs/

Inspect every relevant source file.

Do not rely only on file names.

Build a machine-readable or clearly structured task/status matrix.

---

5. AUTHORITATIVE 68-TASK CONTRACT

"TinyMeAudit.md" is the authoritative task contract.

You MUST read all 68 task headings.

Do not selectively execute only the easy tasks.

For every task 1–68 create/maintain a status record:

TASK-ID
REQUIREMENT
CURRENT STATUS
EVIDENCE
FILES
IMPLEMENTATION
TEST
EXECUTION
MEASURED RESULT
BLOCKERS
FINAL STATUS

Valid statuses:

- "NOT_STARTED"
- "IN_PROGRESS"
- "IMPLEMENTED"
- "VERIFIED"
- "BLOCKED"
- "SUPERSEDED"
- "FAILED"
- "COMPLETE"

Do not mark a task "VERIFIED" merely because code exists.

---

6. PROGRESS COMMIT PROTOCOL

The user explicitly wants progress entered into:

"arena/01a0fa8e-tinyme"

Therefore work incrementally.

Do NOT perform a massive uncommitted session and push only at the very end.

After each meaningful completed phase:

1. run relevant tests,
2. record evidence,
3. update task state,
4. update relevant documentation,
5. commit,
6. push/commit to "arena/01a0fa8e-tinyme".

Use meaningful commit messages such as:

audit: refresh 68-task corrective matrix
fix: complete stage-b loss-masked sft pipeline
train: run exp-004 tool-sft
eval: add independent capability metrics
fix: harden evidence citation validation
fix: complete sandbox adversarial suite
exp: run base architecture feasibility pilot
quant: complete functional fp16 int8 int4 comparison
release: package verified final artifact
docs: synchronize final project state
verify: run final acceptance gates

Do not create a new branch for each phase.

---

7. PHASE A — REAUDIT THE CORRECTIVE FOUNDATION

Verify every claimed P0 correction.

Preprocessing

Verify:

- prose normalization
- code preservation
- indentation preservation
- code-fence fidelity
- Unicode behavior
- structural token preservation
- AST validation
- repair-input semantics

Data contract

Verify:

- segmented records
- roles
- target segments
- loss masks
- metadata
- provenance

Loss

Verify:

- context tokens excluded
- target tokens included
- padding excluded
- tool results excluded from generation target
- masked loss normalization is mathematically correct

Split

Verify:

- train/validation/test/challenge
- group separation
- template separation
- exact overlap
- normalized overlap
- similarity overlap
- code overlap
- template overlap

Tokenizer

Verify:

- split before tokenizer training
- train-only tokenizer corpus
- frozen tokenizer
- hash recorded
- vocab/model compatibility

Sequence construction

Verify:

- single authoritative sequence builder
- target-aware truncation
- packing correctness
- long-sequence handling
- no hidden re-tokenization inconsistencies
- shards are actually consumed or removed from critical path

Training

Verify:

- true gradient accumulation
- frozen parameters within accumulation
- one optimizer update per accumulation block
- deterministic sampler
- full epoch coverage
- correct tail handling

Checkpoint

Verify:

- model
- optimizer
- scheduler
- RNG
- sampler
- epoch
- step
- fingerprints
- environment
- tokenizer
- dataset
- architecture
- integrity

Resume

Actually test:

save
→ stop
→ restore
→ continue

and verify numerical/state continuity.

RoPE

Verify mathematically against a reference implementation.

Verify:

JAX full forward
NumPy full forward
cached forward

are consistent.

Numerical stability

Verify:

- finite loss
- finite gradients
- stable softmax
- stable normalization
- no silent NaN propagation

dtype

Verify actual computation dtype, not only configuration strings.

---

8. PHASE B — DATA PIPELINE COMPLETION

Rebuild/validate the final dataset pipeline.

Required stages:

ingest
→ provenance
→ license
→ preprocessing
→ quality
→ code validation
→ safety checks
→ dedup
→ grouping
→ split
→ tokenizer
→ tokenization
→ sequence building
→ packing/sharding
→ manifest

Ensure every final record contains sufficient provenance.

Track:

- source
- source ID
- row ID
- category
- task type
- template ID
- license
- language
- split
- verification
- provenance
- quality

---

9. DATA QUALITY REQUIREMENTS

Increase useful information density without fake duplication.

Do NOT inflate the dataset through silent repetition.

Prioritize:

- technical prose
- programming
- algorithms
- mathematics
- logic
- code generation
- code repair
- code explanation
- instruction following
- tool use
- evidence/citation tasks
- synthetic curriculum

For synthetic data:

track:

requested
generated
verified
rejected
deduplicated
assigned_to_split

Do not claim dataset scaling if the same samples were merely repeated.

---

10. CODE DATA REQUIREMENTS

Implement language-aware checks.

At minimum:

- syntax validity
- secret detection
- PII detection
- suspicious/malware pattern filtering
- huge file filtering
- generated-code detection where practical
- vendored dependency handling
- minified-code filtering

Never parse every language as Python.

Explicitly declare supported execution languages.

---

11. DEDUPLICATION

Validate the complete deduplication pipeline:

1. exact hash
2. normalized hash
3. document similarity
4. code similarity

If claiming MinHash/LSH:

the actual index must exist and be used.

No placeholder loops.

Report:

- pre-dedup count
- post-dedup count
- exact duplicates
- normalized duplicates
- near duplicates
- code duplicates
- cross-split duplicates

---

12. SHARDING

Ensure sharding is efficient.

Do not repeatedly rebuild a growing ".npy" through O(n²) copying.

Use:

- preallocation
- memmap
- buffered chunk writing
- atomic replacement
- another explicitly justified efficient mechanism

Verify that training actually consumes the final shard representation if shards are part of the critical path.

---

13. FINAL SPLIT QUALITY

Do not accept a test/challenge set that has accidentally lost most target categories.

Inspect category distributions.

The independent evaluation set should have meaningful coverage of:

- math
- logic
- algorithms
- language
- programming
- code explanation
- code generation
- code repair
- debugging
- instruction following
- tool use
- grounding

Challenge/generalization data should not be structurally empty in important target categories unless explicitly documented and justified.

---

14. PHASE C — TWO-STAGE TRAINING

STAGE A — PRETRAINING

Use clean language/domain data.

Objective:

ordinary causal LM objective.

Verify:

- fresh initialization
- no EXP-001 resume
- held-out validation
- stable optimization
- deterministic run

The existing:

"EXP-002-CORRECTED-NANO"

may be preserved as Stage A evidence.

Do NOT overwrite it.

---

15. STAGE B — INSTRUCTION / REASONING / TOOL SFT

This is mandatory.

Create a new experiment ID, e.g.:

"EXP-004-TOOL-SFT"

or another clearly versioned ID.

Do not overwrite Stage A.

Use loss-masked structured records.

Target:

- instruction following
- concise reasoning
- code generation
- code repair
- code explanation
- structured outputs
- tool selection
- tool argument generation
- tool syntax
- search workflows
- evidence-grounded answering
- recovery from tool failures

Tool-result tokens should normally be context rather than generation targets.

---

16. TOOL-USE TRAINING FORMAT

Train explicit trajectories such as:

USER
→ ASSISTANT TOOL CALL
→ TOOL RESULT
→ ASSISTANT FINAL

Also train multi-step trajectories:

USER
→ TOOL CALL
→ TOOL RESULT
→ TOOL CALL
→ TOOL RESULT
→ FINAL

Include:

- tool needed
- tool not needed
- correct tool
- incorrect-tool negatives where appropriate
- argument construction
- malformed argument recovery
- missing-result recovery
- evidence-grounded final response

---

17. TOOL PROTOCOL

Use a deliberately small typed tool surface.

Required core tools:

search
fetch
compute
code
files

Verify:

- exact schema
- exact argument names
- exact result envelopes
- deterministic validation
- malformed-input rejection
- unknown-tool rejection
- tool budget
- loop budget
- timeout handling

Training examples MUST exactly match runtime schemas.

Do not train the model against stale or fictional tool arguments.

---

18. SEARCH / RETRIEVAL

Implement provider-agnostic search architecture.

Minimum components:

query
provider
retrieval
ranking
fetch
evidence
citation

Support local deterministic retrieval for offline tests.

External providers may be used when reachable.

If external network access is unavailable:

DO NOT fake a successful live search.

Return an honest unavailable/error state.

Test:

- query handling
- ranking
- stable result format
- result IDs
- source metadata
- retrieval reproducibility

---

19. EVIDENCE ENGINE

The evidence engine must track what source supports each claim.

Verify:

- source ID validity
- fetched source validity
- quoted span validity
- URL/source identity
- citation structure
- unsupported citation rejection
- source-result consistency
- no fabricated sources

Explicitly test multiple citation forms where the protocol permits them.

A model must not receive “grounded” credit merely because it produced citation-like text.

---

20. SANDBOX HARDENING

The sandbox must be treated as a real security boundary.

Verify actual isolation capabilities dynamically.

Do not hard-code:

"container"

or:

"secure VM"

if that is not the actual mechanism.

Use an honest label such as:

"namespace(net+mount)"

when that is what is actually available.

Test:

- network egress
- filesystem escape
- home-tree access
- environment secret access
- process exhaustion
- CPU exhaustion
- memory exhaustion
- file-size abuse
- stdout flood
- stderr flood
- descriptor abuse
- workspace escape
- "/tmp" abuse
- symlink/path traversal
- process visibility/resource exhaustion

Complete at least 14 meaningful adversarial cases.

Each case must contain:

attack
expected behavior
observed behavior
pass/fail
evidence

Keep raw executable evidence under:

"docs/audit_evidence/"

---

21. SANDBOX RESIDUAL-RISK HONESTY

Document residual capabilities.

Examples:

- shared "/etc"
- PID visibility
- temporary workspace
- user namespace limitations
- lack of bubblewrap
- lack of stronger container boundary

Do not describe the sandbox as stronger than demonstrated.

---

22. PHASE D — EVALUATION SYSTEM

Evaluation must measure the actual target capabilities.

Separate:

Language

- held-out perplexity

Math

- exact answer accuracy

Logic

- exact reasoning/answer accuracy

Algorithmic reasoning

- exact outcome

Instruction following

- exact instruction compliance

Code generation

- syntax rate
- execution rate
- test pass rate

Code repair/debugging

- syntax rate
- execution rate
- assertion pass rate

Tool use

Separate:

- "tool_needed_accuracy"
- "tool_not_needed_accuracy"
- "tool_selection_accuracy"
- "tool_argument_accuracy"
- "tool_syntax_validity"
- "tool_execution_success"
- "multi_step_success"
- "error_recovery_success"
- "grounded_answer_accuracy"

Do not collapse all of these into one vague score.

---

23. INDEPENDENT EVALUATION

Evaluation data must not be training data.

Do not rely on training rows.

No hidden "[:20]", "[:32]", or similar evaluation shortcut.

Every evaluated sample must be traceable to:

- dataset version
- split
- record ID
- task/category

For final test evaluation, do not use the test set for repeated model selection.

Use validation for iteration.

Reserve test/challenge for final measurement.

---

24. CAPABILITY LOOP

After Stage B:

1. evaluate,
2. inspect failures,
3. classify failure modes,
4. improve the training data or training strategy,
5. run targeted iteration,
6. reevaluate,
7. record the iteration,
8. do not leak test answers into training.

Potential iteration areas:

- tool syntax
- exact arguments
- EOS behavior
- target formatting
- code delimiters
- reasoning format
- answer extraction
- context length
- curriculum ordering

Any iteration that changes the dataset must receive a new dataset fingerprint.

Any iteration that changes the evaluation protocol must receive a new evaluation protocol version.

Do not compare incompatible experiments as though they were directly comparable.

---

25. PHASE E — ARCHITECTURE COMPARISON

Evaluate:

Nano

Existing corrected architecture.

Base

Run:

"EXP-003-CORRECTED-BASE"

or equivalent.

At minimum compare:

- parameter count
- FP32 weight size
- validation loss
- test loss
- task accuracy
- tool metrics
- code pass rate
- inference latency
- memory
- training cost

If full Base training is infeasible:

run a measurable feasibility pilot and document exactly what was and was not demonstrated.

Do not choose architecture by opinion.

---

26. CONTEXT-LENGTH EXPERIMENT

Primary:

"seq_len=256"

Secondary:

"seq_len=512"

Do not claim support based solely on RoPE mathematics.

Measure actual training/evaluation behavior.

Record:

- training cost
- latency
- memory
- quality

---

27. OVERFITTING CHECK

Inspect:

- train loss
- validation loss
- validation accuracy
- test performance
- challenge performance

Look for:

- memorization
- template leakage
- collapsing challenge performance
- train/validation divergence

Do not tune directly on final test answers.

---

28. CURRICULUM EXPERIMENT

Where resources allow, compare useful curriculum strategies.

Possible ordering:

language
→ code
→ math
→ logic
→ instruction
→ tools

versus:

mixed

Record actual results.

Do not invent a winner.

---

29. PHASE F — INFERENCE

Verify all inference modes.

Required:

- JAX
- NumPy
- full forward
- cached decoding
- deterministic generation
- seeded stochastic generation
- EOS behavior
- top-p
- repetition penalty
- context truncation
- vocabulary compatibility

Verify cached and uncached logits within a documented tolerance.

---

30. PHASE G — QUANTIZATION

Create a named experiment such as:

"EXP-005-QUANT"

Evaluate:

- FP32
- FP16
- INT8
- INT4

For each:

- file bytes
- parameter representation
- loadability
- reconstruction error
- inference
- perplexity
- capability metrics
- tool metrics
- latency
- memory where measurable

Do not call quantization “successful” merely because the quantized file exists.

---

31. 50-MB ACCEPTANCE GATE

The final deployed model artifact must satisfy:

< 50 MB

Measure actual on-disk bytes.

Also report:

- raw weights
- tokenizer
- config
- runtime/entrypoint
- complete package size

Do not confuse parameter-count arithmetic with actual serialized artifact size.

The acceptance threshold must be measured on the final artifact.

---

32. PACKAGING FIX

Audit:

"scripts/package_model.py"

Fix the known ordering problem where release verification may reference:

"config.json"

before the file exists.

Required correct order:

load checkpoint
→ validate checkpoint
→ validate tokenizer
→ write weights
→ write tokenizer
→ write config
→ write inference entrypoint
→ generate evaluation/provenance reports
→ calculate file metadata
→ write manifest
→ write checksums
→ perform clean final verification

Do not calculate manifest file metadata before all release files exist.

Add regression tests for:

- clean output directory
- no stale files
- config exists before verification
- manifest contains final files
- checksums match
- standalone inference loads

---

33. CLEAN RELEASE BUILD

Never package on top of a dirty old release directory without cleaning it first.

Build a clean release directory.

The final release should contain only the intended release files.

At minimum:

model_fp32.safetensors
model_fp16.safetensors
model_int8.safetensors
model_int4.safetensors
tokenizer.json
config.json
manifest.json
checksums.txt
inference.py
README.md
evaluation_report.md
model_comparison.md
provenance.md

Additional files may exist only if justified and documented.

Historical artifacts must remain separate from the final release.

---

34. STANDALONE RELEASE INFERENCE

The packaged entrypoint must work without importing the training repository.

Required dependency-light operation where intended.

Test from a clean temporary environment.

Verify:

pip install required dependencies
→ copy/release package only
→ run inference
→ output produced

No hidden dependency on repository source files.

---

35. REPRODUCIBILITY

Create or validate reproducible environment definition.

Use pinned versions where practical.

Potential files:

requirements.txt
pyproject.toml

Document:

- Python
- NumPy
- JAX
- JAXLIB
- Optax
- tokenizers
- safetensors
- pytest
- other runtime dependencies

Record:

- Git commit
- dataset fingerprint
- tokenizer hash
- model config hash
- experiment config
- seed
- environment
- backend
- hardware

---

36. TEST SUITE

The test suite must be real.

Do not leave important capabilities only in prose.

Ensure tests cover at minimum:

- preprocessing
- data contract
- split loading
- contamination
- tokenizer
- sequence construction
- loss mask
- padding mask
- gradient accumulation
- sampler
- checkpoint/resume
- scheduler continuity
- RNG
- RoPE
- cache parity
- numerical stability
- dtype
- inference
- dedup
- data quality
- model shapes
- tool protocol
- tool registry
- evidence
- search/retrieval
- agent loop
- sandbox
- quantization
- packaging
- clean release

If the existing suite lacks explicit modules for important components, add them.

Do not rely on a test being skipped.

A skipped release-size test because no release exists must NOT be treated as release verification.

---

37. TEST EXECUTION PROTOCOL

Run:

python -m pytest tests/ -q

before final release work.

Then run it again after release work.

Also run targeted tests for each modified subsystem.

No final acceptance without a fresh test run.

Record exact command and result.

---

38. CI

Inspect whether CI actually exists.

If required by the project objective, add a basic GitHub Actions workflow that runs:

- unit tests
- structural checks
- package checks
- reproducibility checks appropriate for CI resources

Do not create CI merely to claim “CI exists.”

Make sure it actually describes what it tests.

---

39. EXPERIMENT LOGGING

Every substantive experiment gets:

experiment ID
name
purpose
git commit
dataset version
dataset fingerprint
tokenizer version
tokenizer hash
architecture
parameter count
training stage
hyperparameters
seed
environment
duration
metrics
status
limitations

Update:

"EXPERIMENT_LOG.md"

and:

"EXPERIMENT_LOG.jsonl"

Never leave the log saying:

"Pending execution"

after an experiment has actually completed.

---

40. FAILED / SUPERSEDED EXPERIMENTS

Keep historical failed attempts.

Use explicit statuses:

STOPPED / SUPERSEDED
FAILED
INVALID
HISTORICAL
COMPLETED

Document why.

Do not erase evidence merely to make the repository cleaner.

---

41. DOCUMENTATION SYNCHRONIZATION

At the end of each major phase synchronize:

STATE.md
TODO.md
EXECUTION_PLAN.md
EXPERIMENT_LOG.md
EXPERIMENT_LOG.jsonl
README.md
docs/CORRECTIVE_AUDIT.md

No stale claims.

No contradictory statuses.

No “planned” item that has actually been completed.

No “verified” item lacking evidence.

---

42. FINAL DOCUMENTATION

Create/update at minimum:

docs/CORRECTIVE_AUDIT.md
docs/DATA_QUALITY_REPORT.md
docs/TOKENIZER_REPORT.md
docs/SPLIT_SUMMARY.md
docs/TRAINING_REPORT.md
docs/MODEL_COMPARISON.md
docs/SANDBOX.md
docs/ENVIRONMENT_REPORT.md
FINAL_REPORT.md
README.md

"FINAL_REPORT.md" must explicitly include:

1. Project objective
2. Original defects
3. Corrective changes
4. Data pipeline
5. Dataset statistics
6. Split protocol
7. Tokenizer
8. Model architecture
9. Training methodology
10. Stage A results
11. Stage B results
12. Architecture comparison
13. Evaluation methodology
14. Capability results
15. Tool-use results
16. Sandbox/security results
17. Quantization results
18. Release results
19. Reproducibility
20. Limitations
21. Final acceptance gates
22. Exact final commit

Do not hide limitations.

---

43. FINAL ACCEPTANCE GATES

The project is NOT complete until the final audit can answer PASS/FAIL for all required gates.

Core correctness

- preprocessing correct
- structured records correct
- masking correct
- padding correct
- sequence building correct
- splits correct
- tokenizer leakage absent
- gradient accumulation correct
- sampler correct
- checkpoint correct
- resume correct
- scheduler continuity correct
- RNG correctness
- RoPE correct
- numerical stability
- dtype correctness

Data

- provenance complete
- license policy enforced
- dedup validated
- code quality validated
- split contamination PASS
- meaningful evaluation coverage

Model

- architecture measured
- Nano baseline measured
- Base feasibility measured
- final architecture justified by evidence
- under 50 MB verified

Training

- Stage A completed
- Stage B completed
- fresh corrected initialization
- no invalid EXP-001 resume
- training metrics recorded

Tool runtime

- protocol exact
- tool schema validation
- tool execution
- search/retrieval
- evidence
- grounding
- error recovery
- loop/budget controls

Sandbox

- network isolation
- filesystem restrictions
- env sanitization
- resource limits
- process limits
- output limits
- adversarial suite ≥14 cases
- residual risks documented

Evaluation

- independent held-out data
- math
- logic
- algorithmic reasoning
- code
- debugging
- instruction following
- tool-needed
- tool-not-needed
- tool selection
- tool arguments
- tool syntax
- tool execution
- multi-step tool use
- error recovery
- grounding
- generalization

Quantization

- FP32
- FP16
- INT8
- INT4
- actual size
- numerical error
- loadability
- functional evaluation

Release

- clean release directory
- correct config
- model files
- tokenizer
- manifest
- checksums
- inference entrypoint
- evaluation report
- model comparison
- provenance
- standalone inference verified

Reproducibility

- environment definition
- pinned dependencies where applicable
- seeds
- fingerprints
- git commit
- training config
- evaluation config

Documentation

- no stale status
- no contradictory status
- no fabricated claims
- final report present
- README synchronized

---

44. FINAL INDEPENDENT VERIFICATION

Before declaring completion:

Pretend you are a hostile external auditor who did not participate in the implementation.

Recheck:

1. branch
2. commit
3. files
4. tests
5. training artifacts
6. data fingerprints
7. tokenizer hash
8. checkpoint integrity
9. evaluation outputs
10. sandbox evidence
11. quantization artifacts
12. release contents
13. manifest
14. checksums
15. standalone inference
16. documentation consistency

Do not trust the agent's own previous statements.

Re-run critical commands.

---

45. FINAL GIT STATE

All final progress must exist on:

"arena/01a0fa8e-tinyme"

Before finishing:

git status
git log --oneline

Verify:

- intended files committed
- no accidental huge artifacts
- no secret files
- no temporary build files
- no unrelated generated files
- target branch contains the final commits

Do not leave important work only in an uncommitted working tree.

---

46. FINAL RESPONSE / FINAL REPORT FORMAT

At completion, provide a factual final report containing exactly these major sections:

FINAL STATUS

TARGET BRANCH
FINAL COMMIT
PREVIOUS BASE COMMIT

68-TASK SUMMARY
- Verified:
- Implemented but not independently verified:
- Blocked:
- Failed:
- Not required/superseded:

TRAINING
- Stage A:
- Stage B:
- Final architecture:
- Parameters:
- Steps:
- Dataset:
- Validation:
- Test:
- Challenge:

CAPABILITY EVALUATION
- Math:
- Logic:
- Code:
- Debugging:
- Instruction:
- Tool needed:
- Tool not needed:
- Tool selection:
- Tool arguments:
- Tool syntax:
- Tool execution:
- Multi-step:
- Recovery:
- Grounding:
- Generalization:

SECURITY
- Isolation:
- Sandbox cases:
- Passed:
- Known residual risks:

QUANTIZATION
- FP32 size:
- FP16 size:
- INT8 size:
- INT4 size:
- Functional results:

RELEASE
- Release path:
- Manifest:
- Checksums:
- Standalone inference:
- Total size:

REPRODUCIBILITY
- Environment:
- Dependencies:
- Dataset fingerprint:
- Tokenizer hash:
- Model config hash:
- Git commit:

TESTS
- Exact command:
- Result:

DOCUMENTATION
- FINAL_REPORT:
- README:
- STATE:
- TODO:
- EXECUTION_PLAN:
- EXPERIMENT_LOG:
- CORRECTIVE_AUDIT:

KNOWN LIMITATIONS

FINAL ACCEPTANCE GATE
PASS / FAIL

The final acceptance status MUST be based on evidence.

Do not write “PASS” if a mandatory gate is unresolved.

---

47. MOST IMPORTANT DIRECTIVE

Do not optimize for apparent completion.

Optimize for truthful completion.

Do not stop at “the code is there.”

Continue through:

inspect
→ diagnose
→ implement
→ test
→ execute
→ train
→ evaluate
→ debug
→ retrain
→ compare
→ harden
→ quantize
→ package
→ independently verify
→ document
→ commit
→ push

The target branch is:

"arena/01a0fa8e-tinyme"

The final repository must be a real, evidence-backed project, not a repository containing optimistic documentation.

When something cannot be completed because of a genuine environmental limitation, do not fake it.

Prove the limitation.

Complete everything else.

Record exactly what remains.

Never claim more than the evidence supports.

EXECUTE THE ENTIRE WORKFLOW.
DO NOT STOP AFTER THE AUDIT.
DO NOT STOP AFTER CODE IMPLEMENTATION.
DO NOT STOP AFTER TESTS.
DO NOT STOP AFTER TRAINING.
DO NOT STOP AFTER EVALUATION.
CONTINUE UNTIL THE FINAL ACCEPTANCE GATES ARE ACTUALLY RESOLVED OR EXPLICITLY DOCUMENTED AS BLOCKED.
ALL PROGRESS MUST LAND ON "arena/01a0fa8e-tinyme".