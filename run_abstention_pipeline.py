from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from abstention_bias.config import ExperimentConfig
from abstention_bias.hybrid import PipelineHalt
from abstention_bias.metrics import build_paper_tables, compute_abstention_metrics
from abstention_bias.pipelines import (
    export_paper_artifacts,
    prepare_benchmark,
    run_baseline_eval,
    run_judge_eval,
    run_paag_eval,
)


def _write_run_manifest(config: ExperimentConfig) -> None:
    manifest = {
        "models": {
            "baseline": asdict(config.baseline_model),
            "assessor": asdict(config.assessor_model),
            "responder": asdict(config.responder_model),
            "verifier": asdict(config.verifier_model),
            "judge": asdict(config.judge_model),
        },
        "hybrid": asdict(config.hybrid),
        "outputs": {
            "benchmark_csv": str(config.benchmark_csv),
            "baseline_results_csv": str(config.baseline_results_csv),
            "paag_results_csv": str(config.paag_results_csv),
            "judge_results_csv": str(config.judge_results_csv),
            "llm_trace_jsonl": str(config.llm_trace_jsonl),
        },
    }
    for model in manifest["models"].values():
        if model.get("api_key"):
            model["api_key"] = "***redacted***"
    config.run_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    config.run_manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def _clean_outputs(config: ExperimentConfig) -> None:
    generated_files = [
        config.benchmark_csv,
        config.benchmark_jsonl,
        config.baseline_results_csv,
        config.paag_results_csv,
        config.judge_results_csv,
        config.paper_table_csv,
        config.summary_metrics_csv,
        config.dataset_comparison_csv,
        config.causal_comparison_csv,
        config.error_analysis_csv,
        config.model_artifact_json,
        config.baseline_cache_jsonl,
        config.assessor_cache_jsonl,
        config.verifier_cache_jsonl,
        config.judge_cache_jsonl,
        config.llm_trace_jsonl,
        config.run_manifest_json,
    ]
    for path in generated_files:
        path = Path(path)
        if path.exists():
            path.unlink()
    if config.figures_dir.exists():
        for figure in config.figures_dir.glob("*.png"):
            figure.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the abstention-bias PAAG experiment.")
    parser.add_argument("--clean", action="store_true", help="Remove prior generated results before starting.")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.ensure_dirs()
    if args.clean:
        _clean_outputs(config)
    _write_run_manifest(config)

    try:
        benchmark = prepare_benchmark(config)
        baseline = run_baseline_eval(config)
        paag = run_paag_eval(config)
        judge = run_judge_eval(config, baseline, paag)
        export_paper_artifacts(config, baseline, paag, judge)
    except PipelineHalt as halt:
        print("")
        print(f"Run halted safely during {halt.stage}.")
        print("Completed rows are checkpointed. Update the API key and rerun without --clean to continue.")
        raise SystemExit(2) from halt

    print(f"Benchmark rows: {len(benchmark)}")
    print("Baseline metrics:", compute_abstention_metrics(baseline))
    print("PAAG metrics:", compute_abstention_metrics(paag))
    print("Judge summary rows:", len(judge))
    print(build_paper_tables(paag).to_string(index=False))


if __name__ == "__main__":
    main()
