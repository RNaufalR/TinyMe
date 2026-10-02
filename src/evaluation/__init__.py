"""Independent evaluation engine (audit §19/§20)."""
from .evaluator import (  # noqa: F401
    DOMAINS, EvaluationResult, build_suite, evaluate_model, execute_candidate,
    extract_code, extract_final, render_prompt, render_target, write_evaluation_report,
)

__all__ = ["DOMAINS", "EvaluationResult", "build_suite", "evaluate_model", "execute_candidate",
           "extract_code", "extract_final", "render_prompt", "render_target",
           "write_evaluation_report"]
