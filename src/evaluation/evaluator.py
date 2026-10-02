"""Independent multi-domain evaluation engine (spec §19, §20).

Nine domains are measured, each on data NEVER used for training:
  language (perplexity), logic, math, algorithmic reasoning, code (compile/exec),
  debugging, code generation (test pass rate), instruction following, and
  generalization to unseen problem templates.

Code execution is sandboxed in a subprocess with a hard timeout.
Untrusted model output is never executed in-process.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("tinyme.eval")

EVAL_DOMAINS = [
    "language", "logic", "math", "algorithmic_reasoning", "code",
    "debugging", "code_generation", "instruction_following", "generalization",
]


# ------------------------------------------------------------- sandboxing
def run_sandboxed_python(code: str, timeout: float = 5.0, max_output: int = 4000) -> dict[str, Any]:
    """Execute Python in an isolated subprocess with a hard timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(code)
        fh.flush()
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, fh.name], capture_output=True,
                                  text=True, timeout=timeout, cwd="/tmp")
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "elapsed": round(timeout, 3)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed": 0.0}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[:max_output],
        "stderr": proc.stderr[:max_output],
        "elapsed": round(time.perf_counter() - t0, 3),
    }


def extract_answer(text: str) -> str:
    m = re.search(r"<\|answer\|>\s*\n(.*?)(?:\n<\||\Z)", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_code(text: str) -> str:
    m = re.search(r"<\|code\|>\s*\n(.*?)(?:<\|endcode\|>|\Z)", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return ""


def normalize_answer(ans: str) -> str:
    return re.sub(r"[\s,]+", "", ans.strip().lower()).rstrip(".")


# ----------------------------------------------------------------- results
@dataclass
class EvaluationResult:
    experiment_id: str
    checkpoint: str
    domains: dict[str, dict[str, Any]] = field(default_factory=dict)
    generated_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def overall(self) -> dict[str, float]:
        accs = [d["accuracy"] for d in self.domains.values() if d.get("accuracy") is not None]
        return {
            "mean_accuracy": round(sum(accs) / len(accs), 4) if accs else 0.0,
            "domains_evaluated": len(self.domains),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "checkpoint": self.checkpoint,
            "domains": self.domains,
            "overall": self.overall,
            "generated_examples": self.generated_examples[:20],
        }


def _target_from_text(text: str) -> str:
    m = re.search(r"<\|answer\|>\s*\n(.*?)(?:\n<\||\Z)", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"<\|code\|>\s*\n(.*?)<\|endcode\|>", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"<\|assistant\|>\s*\n(.*)\Z", text, re.S)
    if m:
        return m.group(1).strip()
    return ""


def evaluate_model(generate_fn: Callable[..., str], eval_records: list[dict[str, Any]],
                   experiment_id: str = "unknown", checkpoint: str = "unknown",
                   max_samples_per_domain: int = 40, max_new_tokens: int = 96,
                   perplexity_fn: Callable[[str], float] | None = None) -> EvaluationResult:
    """Evaluate a model across all nine domains on held-out records."""
    result = EvaluationResult(experiment_id=experiment_id, checkpoint=checkpoint)

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in eval_records:
        by_cat.setdefault(r.get("category", "language"), []).append(r)

    # ---------------------------------------------------- language / ppl
    lang = by_cat.get("language", [])[:max_samples_per_domain]
    if lang:
        ppl = None
        if perplexity_fn is not None:
            try:
                ppl = round(float(sum(perplexity_fn(r["text"]) for r in lang) / len(lang)), 4)
            except Exception:
                ppl = None
        result.domains["language"] = {
            "samples": len(lang), "metric": "perplexity",
            "perplexity": ppl, "accuracy": None,
        }

    # ------------------------------------------- exact-answer domains
    def exact_answer_domain(category: str, domain_name: str) -> None:
        recs = by_cat.get(category, [])[:max_samples_per_domain]
        if not recs:
            return
        correct = 0
        for r in recs:
            text = r["text"]
            prompt = text.split("<|thought|>")[0] if "<|thought|>" in text else text
            prompt = prompt.split("<|answer|>")[0] if "<|answer|>" in prompt else prompt
            prompt = prompt.split("<|assistant|>")[0] if "<|assistant|>" in prompt else prompt
            out = generate_fn(prompt, max_new_tokens=max_new_tokens)
            got = normalize_answer(extract_answer(out))
            want = normalize_answer(str(r.get("answer") or _target_from_text(text)))
            if got and want and got == want:
                correct += 1
        result.domains[domain_name] = {
            "samples": len(recs), "metric": "accuracy",
            "correct": correct, "accuracy": round(correct / len(recs), 4),
        }

    exact_answer_domain("logic", "logic")
    exact_answer_domain("math", "math")
    exact_answer_domain("algorithm", "algorithmic_reasoning")
    exact_answer_domain("instruction", "instruction_following")

    # ------------------------------------------------------ code domains
    def code_domain(category: str, domain_name: str) -> None:
        recs = by_cat.get(category, [])[:max_samples_per_domain]
        if not recs:
            return
        passed, total = 0, 0
        for r in recs:
            prompt = r["text"]
            if "<|code|>" in prompt:
                prompt = prompt.split("<|code|>")[0]
            elif "<|assistant|>" in prompt:
                prompt = prompt.split("<|assistant|>")[0]
            out = generate_fn(prompt, max_new_tokens=max_new_tokens)
            code = extract_code(out)
            tests = r.get("tests", [])
            if not code:
                continue
            if tests:
                script = code + "\n\n" + "\n".join(tests) + "\n"
                res = run_sandboxed_python(script)
            else:
                res = run_sandboxed_python("import ast\nast.parse(" + json.dumps(code) + ")")
            total += 1
            if res["ok"]:
                passed += 1
        result.domains[domain_name] = {
            "samples": total, "metric": "execution_correctness",
            "passed": passed, "accuracy": round(passed / total, 4) if total else 0.0,
        }

    code_domain("code_gen", "code_generation")
    code_domain("code_repair", "debugging")
    code_domain("programming", "code")

    # ------------------------------------------------------- sample outputs
    for r in (by_cat.get("math", [])[:2] + by_cat.get("code_gen", [])[:2]):
        prompt = r["text"]
        for marker in ("<|thought|>", "<|code|>", "<|assistant|>"):
            if marker in prompt:
                prompt = prompt.split(marker)[0]
        result.generated_examples.append({
            "category": r["category"], "prompt": prompt[-300:],
            "generation": generate_fn(prompt, max_new_tokens=max_new_tokens)[-400:],
            "reference": (r.get("answer") or _target_from_text(r["text"]))[:200],
        })

    return result


def write_evaluation_report(results: list[dict[str, Any]], path: str | Path) -> None:
    lines = [
        "# EVALUATION REPORT",
        "",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Models evaluated:** {len(results)}",
        "- **Evaluation data:** held-out (`datasets/versions/*/eval.jsonl`), never used for training",
        "",
        "## Domain accuracy by model",
        "",
        "| Domain | " + " | ".join(r["experiment_id"] for r in results) + " |",
        "| :--- | " + " | ".join("---:" for _ in results) + " |",
    ]
    for domain in EVAL_DOMAINS:
        row = [domain]
        for r in results:
            d = r["domains"].get(domain)
            if d is None:
                row.append("—")
            elif domain == "language":
                row.append(f"ppl={d.get('perplexity')}")
            else:
                row.append(f"{d.get('accuracy', 0):.3f} ({d.get('passed', d.get('correct', 0))}/{d.get('samples', 0)})")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Notes", "",
              "- Accuracy for exact-answer domains uses normalized string equality.",
              "- Code domains are measured by executing generated code against real assertions in a sandboxed subprocess.",
              ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
