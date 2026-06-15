"""Modular package for safety abstention bias research."""

from .config import ExperimentConfig, ModelConfig
from .datasets import (
    SafetyPromptRecord,
    build_benchmark_dataset,
    load_harmbench_records,
    load_jailbreakbench_records,
    load_xstest_records,
)
from .gate import AbstentionDecision, RiskAssessment, assess_prompt, decide_action
from .hybrid import (
    BaselineGeneration,
    HybridOrchestrator,
    LLMRiskAssessment,
    PairJudgeVerdict,
    ResponseVerification,
)
from .metrics import build_paper_tables, compute_abstention_metrics
from .pipelines import run_baseline_eval, run_judge_eval, run_paag_eval
from .solution import build_safe_answer, evaluate_safe_answer
from .training import NaiveBayesAbstentionModel, train_abstention_model

__all__ = [
    "AbstentionDecision",
    "BaselineGeneration",
    "ExperimentConfig",
    "HybridOrchestrator",
    "LLMRiskAssessment",
    "ModelConfig",
    "PairJudgeVerdict",
    "RiskAssessment",
    "ResponseVerification",
    "SafetyPromptRecord",
    "NaiveBayesAbstentionModel",
    "assess_prompt",
    "build_benchmark_dataset",
    "build_paper_tables",
    "compute_abstention_metrics",
    "decide_action",
    "evaluate_safe_answer",
    "build_safe_answer",
    "load_harmbench_records",
    "load_jailbreakbench_records",
    "load_xstest_records",
    "run_baseline_eval",
    "run_judge_eval",
    "run_paag_eval",
    "train_abstention_model",
]
