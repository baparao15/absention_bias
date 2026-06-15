from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(slots=True)
class ModelConfig:
    provider: str = field(default_factory=lambda: os.getenv("ABSTENTION_PROVIDER", "openai_compatible"))
    model_name: str = field(default_factory=lambda: os.getenv("ABSTENTION_MODEL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("ABSTENTION_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("ABSTENTION_BASE_URL", ""))
    timeout_sec: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_TIMEOUT_SEC", "90")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_LLM_MAX_RETRIES", "8")))
    retry_base_sec: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_LLM_RETRY_BASE_SEC", "2.5")))
    min_interval_sec: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_LLM_MIN_INTERVAL_SEC", "2.2")))


@dataclass(slots=True)
class HybridConfig:
    scorer_uncertain_low: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_SCORER_UNCERTAIN_LOW", "0.35")))
    scorer_uncertain_high: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_SCORER_UNCERTAIN_HIGH", "0.75")))
    llm_refuse_threshold: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_LLM_REFUSE_THRESHOLD", "0.70")))
    llm_answer_threshold: float = field(default_factory=lambda: float(os.getenv("ABSTENTION_LLM_ANSWER_THRESHOLD", "0.35")))
    max_baseline_calls: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_MAX_BASELINE_CALLS", "1000")))
    max_assessor_calls: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_MAX_ASSESSOR_CALLS", "350")))
    max_verifier_calls: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_MAX_VERIFIER_CALLS", "250")))
    max_judge_calls: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_MAX_JUDGE_CALLS", "1000")))
    enable_llm_baseline: bool = field(default_factory=lambda: os.getenv("ABSTENTION_ENABLE_LLM_BASELINE", "0") == "1")
    enable_llm_assessor: bool = field(default_factory=lambda: os.getenv("ABSTENTION_ENABLE_LLM_ASSESSOR", "0") == "1")
    enable_llm_responder: bool = field(default_factory=lambda: os.getenv("ABSTENTION_ENABLE_LLM_RESPONDER", "0") == "1")
    enable_llm_verifier: bool = field(default_factory=lambda: os.getenv("ABSTENTION_ENABLE_LLM_VERIFIER", "0") == "1")
    enable_llm_judge: bool = field(default_factory=lambda: os.getenv("ABSTENTION_ENABLE_LLM_JUDGE", "0") == "1")
    stop_on_llm_error: bool = field(default_factory=lambda: os.getenv("ABSTENTION_STOP_ON_LLM_ERROR", "1") == "1")
    resume_from_checkpoints: bool = field(default_factory=lambda: os.getenv("ABSTENTION_RESUME_FROM_CHECKPOINTS", "1") == "1")
    checkpoint_every_rows: int = field(default_factory=lambda: int(os.getenv("ABSTENTION_CHECKPOINT_EVERY_ROWS", "1")))
    trace_llm_prompts: bool = field(default_factory=lambda: os.getenv("ABSTENTION_TRACE_LLM_PROMPTS", "1") == "1")
    verify_paag_refusals: bool = field(default_factory=lambda: os.getenv("ABSTENTION_VERIFY_PAAG_REFUSALS", "0") == "1")
    verify_paag_answers: bool = field(default_factory=lambda: os.getenv("ABSTENTION_VERIFY_PAAG_ANSWERS", "1") == "1")


@dataclass(slots=True)
class ExperimentConfig:
    xstest_path: Path = Path("data/xstest_prompts.csv")
    harmbench_val_path: Path = Path("data/HarmBench/behavior_datasets/harmbench_behaviors_text_val.csv")
    harmbench_test_path: Path = Path("data/HarmBench/behavior_datasets/harmbench_behaviors_text_test.csv")
    jailbreakbench_llama2_path: Path = Path("data/JailBreakBench/llama2.json")
    jailbreakbench_vicuna_path: Path = Path("data/JailBreakBench/vicuna.json")
    benchmark_csv: Path = Path("results/abstention_bias/benchmark.csv")
    benchmark_jsonl: Path = Path("results/abstention_bias/benchmark.jsonl")
    baseline_results_csv: Path = Path("results/abstention_bias/baseline_results.csv")
    paag_results_csv: Path = Path("results/abstention_bias/paag_results.csv")
    judge_results_csv: Path = Path("results/abstention_bias/judge_results.csv")
    paper_table_csv: Path = Path("results/abstention_bias/paper_tables.csv")
    summary_metrics_csv: Path = Path("results/abstention_bias/summary_metrics.csv")
    dataset_comparison_csv: Path = Path("results/abstention_bias/dataset_comparison.csv")
    causal_comparison_csv: Path = Path("results/abstention_bias/causal_comparison.csv")
    error_analysis_csv: Path = Path("results/abstention_bias/error_analysis.csv")
    ml_model_comparison_csv: Path = Path("results/abstention_bias/ml_model_comparison.csv")
    ml_predictions_csv: Path = Path("results/abstention_bias/ml_predictions.csv")
    ml_risk_predictions_csv: Path = Path("results/abstention_bias/ml_risk_predictions.csv")
    model_artifact_json: Path = Path("results/abstention_bias/abstention_nb_model.json")
    baseline_cache_jsonl: Path = Path("results/abstention_bias/baseline_cache.jsonl")
    assessor_cache_jsonl: Path = Path("results/abstention_bias/assessor_cache.jsonl")
    verifier_cache_jsonl: Path = Path("results/abstention_bias/verifier_cache.jsonl")
    judge_cache_jsonl: Path = Path("results/abstention_bias/judge_cache.jsonl")
    llm_trace_jsonl: Path = Path("results/abstention_bias/llm_traces.jsonl")
    run_manifest_json: Path = Path("results/abstention_bias/run_manifest.json")
    figures_dir: Path = Path("results/abstention_bias/figures")
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider=os.getenv("ABSTENTION_BASELINE_PROVIDER", "openai_compatible"),
            model_name=os.getenv("ABSTENTION_BASELINE_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("ABSTENTION_BASELINE_API_KEY", os.getenv("ABSTENTION_RESPONDER_API_KEY", "")),
            base_url=os.getenv(
                "ABSTENTION_BASELINE_BASE_URL",
                os.getenv("ABSTENTION_RESPONDER_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
            ),
            timeout_sec=int(os.getenv("ABSTENTION_BASELINE_TIMEOUT_SEC", "90")),
        )
    )
    assessor_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider=os.getenv("ABSTENTION_ASSESSOR_PROVIDER", "openai_compatible"),
            model_name=os.getenv("ABSTENTION_ASSESSOR_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("ABSTENTION_ASSESSOR_API_KEY", ""),
            base_url=os.getenv("ABSTENTION_ASSESSOR_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
            timeout_sec=int(os.getenv("ABSTENTION_ASSESSOR_TIMEOUT_SEC", "90")),
        )
    )
    responder_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider=os.getenv("ABSTENTION_RESPONDER_PROVIDER", "openai_compatible"),
            model_name=os.getenv("ABSTENTION_RESPONDER_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("ABSTENTION_RESPONDER_API_KEY", ""),
            base_url=os.getenv("ABSTENTION_RESPONDER_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
            timeout_sec=int(os.getenv("ABSTENTION_RESPONDER_TIMEOUT_SEC", "90")),
        )
    )
    verifier_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider=os.getenv("ABSTENTION_VERIFIER_PROVIDER", "openai_compatible"),
            model_name=os.getenv("ABSTENTION_VERIFIER_MODEL", "gpt-4.1-mini"),
            api_key=os.getenv("ABSTENTION_VERIFIER_API_KEY", ""),
            base_url=os.getenv("ABSTENTION_VERIFIER_BASE_URL", "https://api.openai.com/v1/chat/completions"),
            timeout_sec=int(os.getenv("ABSTENTION_VERIFIER_TIMEOUT_SEC", "90")),
        )
    )
    judge_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            provider=os.getenv("ABSTENTION_JUDGE_PROVIDER", "openai_compatible"),
            model_name=os.getenv("ABSTENTION_JUDGE_MODEL", os.getenv("ABSTENTION_VERIFIER_MODEL", "gpt-4.1-mini")),
            api_key=os.getenv("ABSTENTION_JUDGE_API_KEY", os.getenv("ABSTENTION_VERIFIER_API_KEY", "")),
            base_url=os.getenv(
                "ABSTENTION_JUDGE_BASE_URL",
                os.getenv("ABSTENTION_VERIFIER_BASE_URL", "https://api.openai.com/v1/chat/completions"),
            ),
            timeout_sec=int(os.getenv("ABSTENTION_JUDGE_TIMEOUT_SEC", "90")),
        )
    )
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        for path in [
            self.benchmark_csv.parent,
            self.baseline_results_csv.parent,
            self.paag_results_csv.parent,
            self.judge_results_csv.parent,
            self.paper_table_csv.parent,
            self.summary_metrics_csv.parent,
            self.dataset_comparison_csv.parent,
            self.causal_comparison_csv.parent,
            self.error_analysis_csv.parent,
            self.ml_model_comparison_csv.parent,
            self.ml_predictions_csv.parent,
            self.ml_risk_predictions_csv.parent,
            self.model_artifact_json.parent,
            self.baseline_cache_jsonl.parent,
            self.assessor_cache_jsonl.parent,
            self.verifier_cache_jsonl.parent,
            self.judge_cache_jsonl.parent,
            self.llm_trace_jsonl.parent,
            self.run_manifest_json.parent,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
