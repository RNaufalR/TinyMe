"""Independent, domain-specific evaluation (corrective audit §19/§20).

Design rules:

* evaluation reads the **real held-out splits** (``test`` and ``challenge``)
  written by ``scripts/prepare_data_v2.py``; no ``train.jsonl`` row is ever used;
* prompts are rebuilt from the structured segments of each record, so text
  markers inside code or answers can never leak the target;
* **no sample cap**: every record in the domain is evaluated unless the caller
  explicitly passes ``max_samples``; the denominator is always reported;
* code correctness is measured by *executing* generated code against the
  record's real assertions inside :mod:`src.sandbox.runner` (network-disabled);
* tool use is measured on syntax, tool selection, argument match, execution and
  answer correctness — never on a single blended score.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agent.protocol import (FINAL, TOOL_CALL, ProtocolError, parse_model_output,
                              render_tool_result)
from ..agent.tool_registry import ToolRegistry
from ..data.dataset_api import load_split_records
from ..data.sequence import ROLE_TOKENS
from ..sandbox.runner import run_python
from ..sandbox.policy import SandboxPolicy

logger = logging.getLogger("tinyme.eval")

DOMAINS = ["language", "logic", "math", "algorithmic_reasoning", "code", "debugging",
           "code_generation", "instruction_following", "tool_selection", "tool_arguments",
           "tool_syntax", "tool_execution", "grounding", "generalization"]

DEFAULT_SANDBOX = SandboxPolicy(wall_timeout_s=10, cpu_timeout_s=5, memory_mb=512,
                                max_output_bytes=8192, network=False)


# --------------------------------------------------------------------- utils
def render_prompt(record: dict) -> str:
    """Render every non-target segment as the model-facing prompt."""
    parts: list[str] = []
    for seg in record.get("segments", []):
        if seg.get("target"):
            continue
        role = seg.get("role", "user")
        parts.append(ROLE_TOKENS.get(role, "<|user|>") + "\n" + seg.get("text", ""))
    return "\n".join(parts).strip() + "\n" + ROLE_TOKENS.get("assistant", "<|assistant|>") + "\n"


def render_target(record: dict) -> str:
    return "\n".join(seg.get("text", "") for seg in record.get("segments", []) if seg.get("target")).strip()


def extract_final(text: str) -> str:
    try:
        parsed = parse_model_output(text)
        if parsed.final:
            return parsed.final.strip()
    except ProtocolError:
        pass
    match = re.search(re.escape(FINAL) + r"\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    for marker in ("<|answer|>", "<|assistant|>", "<|code|>"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text.strip()


def extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    code = re.search(r"<\|code\|>\s*\n?(.*?)(?:<\|endcode\|>|\Z)", text, re.DOTALL)
    if code:
        return code.group(1).strip()
    if extract_final(text) != text.strip():
        return extract_final(text)
    return text.strip()


def normalize(text: Any) -> str:
    return re.sub(r"[\s,]+", "", str(text).strip().lower()).rstrip(".")


def execute_candidate(code: str, tests: list[str], policy: SandboxPolicy = DEFAULT_SANDBOX) -> dict:
    script = code + "\n\n" + "\n".join(tests) + "\n" if tests else code
    result = run_python(script, policy=policy)
    return {"ok": result.ok, "exit_code": result.exit_code, "timed_out": result.timed_out,
            "stdout": result.stdout[-2000:], "stderr": result.stderr[-1500:],
            "isolation_level": result.isolation.get("level"),
            "duration_s": result.duration_s}


# ------------------------------------------------------------------- results
@dataclass
class EvaluationResult:
    model: str
    dataset_version: str
    split: str
    domains: dict[str, dict] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0
    environment: dict = field(default_factory=dict)

    @property
    def overall(self) -> dict:
        scored = {k: v for k, v in self.domains.items()
                  if v.get("accuracy") is not None and v.get("samples")}
        return {"domains_evaluated": len(scored),
                "mean_accuracy": round(sum(v["accuracy"] for v in scored.values()) / len(scored), 4)
                if scored else None,
                "total_samples": sum(v.get("samples", 0) for v in self.domains.values())}

    def to_dict(self) -> dict:
        return {"model": self.model, "dataset_version": self.dataset_version, "split": self.split,
                "overall": self.overall, "domains": self.domains, "examples": self.examples,
                "started_at": self.started_at, "duration_s": round(self.duration_s, 2),
                "environment": self.environment}


# -------------------------------------------------------------------- suites
def build_suite(version: str, split: str = "test", *, max_samples: int | None = None) -> list[dict]:
    """Load held-out records for one split (never a training split)."""
    if split == "train":
        raise ValueError("evaluation must use held-out splits; 'train' is refused")
    records = load_split_records(version, split)
    if max_samples:
        records = records[: max(1, int(max_samples))]
    return records


def _domain_for(record: dict) -> str | None:
    cat = record.get("category", "")
    return {"math": "math", "logic": "logic", "algorithm": "algorithmic_reasoning",
            "code_gen": "code_generation", "code_repair": "debugging",
            "programming": "code", "instruction": "instruction_following",
            "tool_use": "tool_use", "code_explain": None, "language": None}.get(cat)


# ------------------------------------------------------------------ evaluate
def evaluate_model(generate_fn: Callable[..., str], records: list[dict], *,
                   model_name: str = "unknown", dataset_version: str = "unknown",
                   split: str = "test", max_new_tokens: int = 96,
                   policy: SandboxPolicy = DEFAULT_SANDBOX,
                   registry: ToolRegistry | None = None,
                   perplexity_fn: Callable[[str], float] | None = None,
                   examples_per_domain: int = 4) -> EvaluationResult:
    started = time.time()
    result = EvaluationResult(model=model_name, dataset_version=dataset_version, split=split,
                              started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    registry = registry or ToolRegistry.default()
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(_domain_for(record) or "other", []).append(record)

    # ------------------------------------------------------------ language
    language = buckets.get("language", [])
    if language and perplexity_fn is not None:
        ppls = []
        for record in language:
            try:
                ppls.append(float(perplexity_fn(record["text"])))
            except Exception as exc:  # pragma: no cover - diagnostic path
                logger.warning("perplexity failed: %s", exc)
        result.domains["language"] = {
            "samples": len(language), "metric": "bits_per_byte", "perplexity": round(
                sum(ppls) / len(ppls), 4) if ppls else None, "accuracy": None,
            "note": "perplexity on held-out text (reported separately, not mixed into accuracy)"}

    def exact_domain(domain: str, category: str) -> None:
        rows = buckets.get(category, [])
        if not rows:
            return
        correct, answered = 0, 0
        examples = []
        for record in rows:
            prompt = render_prompt(record)
            output = generate_fn(prompt, max_new_tokens=max_new_tokens)
            got, want = normalize(extract_final(output)), normalize(record.get("answer", ""))
            answered += 1 if got else 0
            hit = bool(got) and got == want
            correct += int(hit)
            if len(examples) < examples_per_domain:
                examples.append({"prompt_tail": prompt[-220:], "generation": output[:400],
                                 "reference": str(record.get("answer", ""))[:200], "correct": hit})
        result.domains[domain] = {"samples": len(rows), "metric": "accuracy", "correct": correct,
                                  "answered": answered, "accuracy": round(correct / len(rows), 4),
                                  "examples": examples}

    exact_domain("math", "math")
    exact_domain("logic", "logic")
    exact_domain("algorithmic_reasoning", "algorithm")
    exact_domain("instruction_following", "instruction")

    # ---------------------------------------------------------------- code
    def code_domain(domain: str, category: str) -> None:
        rows = buckets.get(category, [])
        if not rows:
            return
        passed, executed, parseable = 0, 0, 0
        examples = []
        for record in rows:
            prompt = render_prompt(record)
            output = generate_fn(prompt, max_new_tokens=max_new_tokens)
            code = extract_code(output)
            syntax_ok = False
            if code:
                check = run_python(f"import ast\nast.parse({json.dumps(code)})", policy=policy)
                syntax_ok = check.ok
                parseable += int(syntax_ok)
            tests = record.get("tests") or []
            run = execute_candidate(code, tests, policy) if (code and syntax_ok and tests) else None
            if run is not None:
                executed += 1
                passed += int(run["ok"])
            if len(examples) < examples_per_domain:
                examples.append({"prompt_tail": prompt[-200:], "generated_code": code[:500],
                                 "syntax_ok": syntax_ok,
                                 "execution": {k: v for k, v in (run or {}).items() if k != "stdout"},
                                 "reference_final": extract_final(render_target(record))[:200],
                                 "passed": bool(run and run["ok"])})
        result.domains[domain] = {
            "samples": len(rows), "metric": "execution_pass_rate", "parseable": parseable,
            "executed": executed, "passed": passed,
            "pass_rate_of_executed": round(passed / executed, 4) if executed else None,
            "accuracy": round(passed / len(rows), 4),
            "examples": examples}

    code_domain("code_generation", "code_gen")
    code_domain("debugging", "code_repair")
    code_domain("code", "programming")

    # ------------------------------------------------------------ tool use
    tool_rows = buckets.get("tool_use", [])
    if tool_rows:
        syntax_ok_n = select_ok, args_ok, exec_ok, answer_ok = 0, 0, 0, 0, 0
        examples = []
        for record in tool_rows:
            prompt = render_prompt(record)
            output = generate_fn(prompt, max_new_tokens=max_new_tokens)
            expected_tool = record.get("expected_tool") or _expected_tool(record)
            try:
                parsed = parse_model_output(output)
                valid = bool(parsed.final) or bool(parsed.tool_calls)
                syntax_ok_n += int(valid)
            except ProtocolError as exc:
                valid, parsed = False, None
                if len(examples) < examples_per_domain:
                    examples.append({"record_id": record.get("record_id"), "protocol_error": exc.reason,
                                     "generation": output[:300]})
                continue
            named_ok = args_match = executed = None
            if expected_tool in (None, "none"):
                named_ok = parsed.final is not None
                if named_ok:
                    select_ok += 1
            elif parsed.tool_calls:
                call = parsed.tool_calls[0]
                named_ok = call.name == expected_tool
                select_ok += int(named_ok)
                reference_call = _reference_call(record)
                args_match = _args_match(call.arguments, reference_call.get("arguments", {}))
                args_ok += int(bool(args_match))
                tool_result = registry.call(call.name, call.arguments)
                executed = tool_result.ok
                exec_ok += int(executed)
                if valid:
                    follow_prompt = (prompt + output + "\n" + render_tool_result(tool_result.to_dict())
                                     + "\n<|assistant|>\n")
                    follow = generate_fn(follow_prompt, max_new_tokens=max_new_tokens)
                    answer_ok += int(normalize(extract_final(follow)) == normalize(record.get("answer", "")))
            else:
                named_ok = False
            if len(examples) < examples_per_domain:
                examples.append({"record_id": record.get("record_id"), "expected_tool": expected_tool,
                                 "generation": output[:300], "tool_name_ok": named_ok,
                                 "arguments_ok": args_match, "executed_ok": executed})
        n = len(tool_rows)
        result.domains["tool_syntax"] = {"samples": n, "metric": "syntax_rate",
                                         "valid": syntax_ok_n, "accuracy": round(syntax_ok_n / n, 4)}
        result.domains["tool_selection"] = {"samples": n, "metric": "tool_name_accuracy",
                                            "correct": select_ok, "accuracy": round(select_ok / n, 4)}
        result.domains["tool_arguments"] = {"samples": n, "metric": "argument_match_accuracy",
                                            "correct": args_ok, "accuracy": round(args_ok / n, 4)}
        result.domains["tool_execution"] = {"samples": n, "metric": "tool_call_success_rate",
                                            "succeeded": exec_ok, "accuracy": round(exec_ok / n, 4)}
        result.domains["grounding"] = {"samples": n, "metric": "grounded_answer_accuracy",
                                       "correct": answer_ok, "accuracy": round(answer_ok / n, 4),
                                       "note": "answer after feeding the real tool result back"}
        result.examples.extend(examples[:examples_per_domain])

    # -------------------------------------------------------------- overall
    if split == "challenge" or buckets.get("challenge"):
        generalization = {k: v for k, v in result.domains.items() if v.get("accuracy") is not None}
        if generalization:
            result.domains["generalization"] = {
                "samples": sum(v["samples"] for v in generalization.values()),
                "metric": "mean_accuracy_on_held_out_templates",
                "accuracy": round(sum(v["accuracy"] for v in generalization.values()) / len(generalization), 4),
                "inherits": sorted(generalization)}
    result.duration_s = time.time() - started
    return result


def _expected_tool(record: dict) -> str | None:
    for seg in record.get("segments", []):
        if seg.get("role") == "tool_call":
            try:
                return json.loads(seg["text"].split(TOOL_CALL, 1)[-1]).get("name")
            except (json.JSONDecodeError, AttributeError):
                return None
    return None


def _reference_call(record: dict) -> dict:
    for seg in record.get("segments", []):
        if seg.get("role") == "tool_call":
            body = seg["text"].split(TOOL_CALL, 1)[-1]
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
    return {}


def _args_match(got: dict, want: dict) -> bool:
    """Numeric-tolerant argument comparison (1 vs 1.0 must not count as a miss)."""

    def _norm(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 6)
        if isinstance(value, str):
            try:
                return round(float(value), 6)
            except ValueError:
                return re.sub(r"\s+", " ", value.strip().lower())
        return json.dumps(value, sort_keys=True)

    if not want:
        return not got
    return all(key in got and _norm(got[key]) == _norm(want[key]) for key in want)


# ------------------------------------------------------------------- report
def write_evaluation_report(results: list[EvaluationResult], path: str | Path,
                            *, title: str = "Evaluation report") -> Path:
    path = Path(path)
    lines = [f"# {title}", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- Models: {', '.join(r.model for r in results)}",
             f"- Data: held-out `test`/`challenge` splits only (no training rows)", ""]
    lines += ["| Domain | " + " | ".join(r.model for r in results) + " |",
              "| :--- | " + " | ".join("---:" for _ in results) + " |"]
    for domain in DOMAINS:
        row = [domain]
        for r in results:
            d = r.domains.get(domain)
            if not d:
                row.append("—")
            elif d.get("accuracy") is None:
                row.append(f"ppl={d.get('perplexity')}")
            else:
                row.append(f"{d['accuracy']:.3f} ({d.get('passed', d.get('correct', d.get('valid', 0)))}/{d.get('samples', 0)})")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Notes", "",
              "- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.",
              "- Code domains execute the generated program together with the record's real assertions",
              "  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass",
              "  `ast.parse`, `executed` counts runs that actually reached the assertions.",
              "- Tool domains are reported separately: syntax, tool-name accuracy, argument match,",
              "  tool execution success and the answer given after the real result is fed back.",
              "- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
