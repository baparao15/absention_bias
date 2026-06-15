from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .datasets import (
    build_benchmark_dataset,
    load_harmbench_records,
    load_jailbreakbench_records,
    load_xstest_records,
)
from .gate import AbstentionDecision, RiskAssessment, assess_prompt, decide_action
from .hybrid import HybridOrchestrator, PipelineHalt, run_async
from .metrics import build_paper_tables, compute_abstention_metrics
from .reporting import (
    build_causal_comparison_df,
    build_dataset_comparison_df,
    build_error_analysis_df,
    build_summary_metrics_df,
    generate_figures,
    write_csv,
    write_jsonl,
)
from .solution import build_safe_answer, evaluate_safe_answer
from .training import NaiveBayesAbstentionModel, balanced_error, train_abstention_model


REFUSAL_TEMPLATE = (
    "I can't help with dangerous, illegal, or abusive instructions. I can help with prevention, safety, or legitimate alternatives."
)


def _load_checkpoint(path: str | Path, resume: bool) -> tuple[list[dict], set[str]]:
    path = Path(path)
    if not resume:
        if path.exists():
            path.unlink()
        return [], set()
    if not path.exists():
        return [], set()
    checkpoint = pd.read_csv(path)
    if checkpoint.empty or "prompt_id" not in checkpoint.columns:
        return [], set()
    rows = checkpoint.to_dict("records")
    return rows, set(checkpoint["prompt_id"].astype(str))


def _append_checkpoint(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _print_halt(stage: str, processed: int, total: int, output_path: str | Path, message: str) -> None:
    print("")
    print(f"[STOP] {stage} halted after {processed}/{total} rows.")
    print(f"[STOP] Checkpoint saved at: {Path(output_path)}")
    print(f"[STOP] Reason: {message}")
    print("[STOP] Change the relevant API key and rerun the same command to resume from checkpoints.")


def prepare_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    config.ensure_dirs()
    xstest_records = load_xstest_records(config.xstest_path)
    harmbench_records = load_harmbench_records(config.harmbench_val_path, config.harmbench_test_path)
    jailbreakbench_records = load_jailbreakbench_records(
        config.jailbreakbench_llama2_path,
        config.jailbreakbench_vicuna_path,
    )
    benchmark = build_benchmark_dataset(xstest_records, harmbench_records, jailbreakbench_records)
    write_csv(benchmark, config.benchmark_csv)
    write_jsonl(benchmark, config.benchmark_jsonl)
    return benchmark


def _fit_model(config: ExperimentConfig, benchmark: pd.DataFrame) -> NaiveBayesAbstentionModel:
    train_df = benchmark[benchmark["split"] == "train"].copy()
    model = train_abstention_model(train_df)
    model.to_json(config.model_artifact_json)
    return model


def _tune_baseline_threshold(model: NaiveBayesAbstentionModel, benchmark: pd.DataFrame) -> float:
    dev_df = benchmark[(benchmark["split"] == "dev") & (benchmark["source_dataset"] == "XSTest")].copy()
    if dev_df.empty:
        return 0.5
    probs = [model.predict_proba_refuse(text) for text in dev_df["prompt_text"]]
    expected = dev_df["expected_action"].tolist()
    best_threshold = 0.5
    best_score = float("inf")
    for threshold in [x / 20 for x in range(4, 17)]:
        predicted = ["REFUSE" if prob >= threshold else "ANSWER" for prob in probs]
        score = balanced_error(expected, predicted)
        if score < best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _tune_paag_thresholds(model: NaiveBayesAbstentionModel, benchmark: pd.DataFrame) -> tuple[float, float, float]:
    dev_df = benchmark[(benchmark["split"] == "dev") & (benchmark["source_dataset"] == "XSTest")].copy()
    if dev_df.empty:
        return (0.8, 0.35, 0.65)
    expected = dev_df["expected_action"].tolist()
    assessments = [assess_prompt(text) for text in dev_df["prompt_text"]]
    probs = [model.predict_proba_refuse(text) for text in dev_df["prompt_text"]]

    best = (0.8, 0.35, 0.65)
    best_score = float("inf")
    for answer_ceiling in [0.65, 0.75, 0.85, 0.9]:
        for low in [0.2, 0.3, 0.35, 0.4]:
            for high in [0.55, 0.65, 0.75, 0.85]:
                if low >= high:
                    continue
                predicted = []
                for assessment, prob in zip(assessments, probs):
                    rule_decision = decide_action(assessment)
                    if rule_decision == AbstentionDecision.REFUSE:
                        final = AbstentionDecision.REFUSE
                    elif rule_decision == AbstentionDecision.ANSWER:
                        final = AbstentionDecision.ESCALATE if prob >= answer_ceiling else AbstentionDecision.ANSWER
                    else:
                        if prob >= high:
                            final = AbstentionDecision.REFUSE
                        elif prob <= low:
                            final = AbstentionDecision.ANSWER
                        else:
                            final = AbstentionDecision.ESCALATE
                    predicted.append(final.value)
                score = balanced_error(expected, predicted)
                if score < best_score:
                    best_score = score
                    best = (answer_ceiling, low, high)
    return best


def run_baseline_eval(config: ExperimentConfig) -> pd.DataFrame:
    benchmark = prepare_benchmark(config)
    model = _fit_model(config, benchmark)
    threshold = _tune_baseline_threshold(model, benchmark)
    orchestrator = HybridOrchestrator(config)
    rows, completed_ids = _load_checkpoint(config.baseline_results_csv, config.hybrid.resume_from_checkpoints)
    total_records = len(benchmark)
    for index, record in enumerate(benchmark.to_dict("records"), start=1):
        prompt_id = str(record["prompt_id"])
        if prompt_id in completed_ids:
            continue
        prob_refuse = model.predict_proba_refuse(record["prompt_text"])
        record_obj = _record_from_row(record)
        try:
            if config.hybrid.enable_llm_baseline:
                generation = run_async(
                    orchestrator.generate_baseline_response(
                        record["prompt_text"],
                        record["source_dataset"],
                        record["risk_domain"],
                        prompt_id=prompt_id,
                    )
                )
                predicted = generation.action_label
                response = generation.response_text
                baseline_confidence = generation.confidence
                baseline_rationale = generation.rationale
                llm_baseline_used = generation.used_llm
                llm_baseline_cache_hit = generation.cache_hit
            else:
                predicted = "REFUSE" if prob_refuse >= threshold else "ANSWER"
                response = REFUSAL_TEMPLATE if predicted == "REFUSE" else build_safe_answer(
                    record_obj, assess_prompt(record["prompt_text"])
                )
                baseline_confidence = prob_refuse if predicted == "REFUSE" else 1.0 - prob_refuse
                baseline_rationale = "Local NB threshold baseline."
                llm_baseline_used = False
                llm_baseline_cache_hit = False
            verification = run_async(
                orchestrator.verify_response(record["prompt_text"], predicted, response, prompt_id=prompt_id)
            )
        except PipelineHalt as halt:
            _print_halt(halt.stage, len(rows), total_records, config.baseline_results_csv, halt.message)
            raise
        row = {
            **record,
            "refuse_probability": prob_refuse,
            "baseline_threshold": threshold,
            "predicted_action": predicted,
            "model_response": response,
            "raw_model_response": response,
            "baseline_confidence": baseline_confidence,
            "baseline_rationale": baseline_rationale,
            "llm_baseline_used": llm_baseline_used,
            "llm_baseline_cache_hit": llm_baseline_cache_hit,
            "verifier_is_safe": verification.is_safe,
            "verifier_contains_actionable_harm": verification.contains_actionable_harm,
            "verifier_unsupported_or_misleading": verification.unsupported_or_misleading,
            "verifier_unnecessary_refusal": verification.unnecessary_refusal,
            "verifier_should_override_to_refusal": verification.should_override_to_refusal,
            "verifier_recommended_action": verification.recommended_action,
            "verifier_confidence": verification.confidence,
            "verifier_rationale": verification.rationale,
            "llm_verifier_used": verification.used_llm,
            "llm_verifier_cache_hit": verification.cache_hit,
            "answer_alignment_ok": evaluate_safe_answer(record_obj, predicted, response),
        }
        rows.append(row)
        _append_checkpoint(config.baseline_results_csv, row)
        if len(rows) % 50 == 0 or index == total_records:
            print(
                f"[BASELINE] checkpointed {len(rows)}/{total_records} prompts "
                f"(baseline_calls={orchestrator.baseline_calls}, verifier_calls={orchestrator.verifier_calls})"
            )
    results = pd.DataFrame(rows)
    write_csv(results, config.baseline_results_csv)
    return results


def run_paag_eval(config: ExperimentConfig) -> pd.DataFrame:
    benchmark = prepare_benchmark(config)
    model = _fit_model(config, benchmark)
    answer_ceiling, low, high = _tune_paag_thresholds(model, benchmark)
    orchestrator = HybridOrchestrator(config)
    rows, completed_ids = _load_checkpoint(config.paag_results_csv, config.hybrid.resume_from_checkpoints)
    total_records = len(benchmark)
    for index, record in enumerate(benchmark.to_dict("records"), start=1):
        prompt_id = str(record["prompt_id"])
        if prompt_id in completed_ids:
            continue
        assessment = assess_prompt(record["prompt_text"])
        rule_decision = decide_action(assessment)
        prob_refuse = model.predict_proba_refuse(record["prompt_text"])
        try:
            llm_assessment = run_async(
                orchestrator.assess_with_llm(
                    record["prompt_text"],
                    assessment,
                    prob_refuse,
                    record["source_dataset"],
                    record["perturbation_type"],
                    prompt_id=prompt_id,
                )
            )
            decision = _resolve_hybrid_decision(
                rule_decision=rule_decision,
                llm_action=llm_assessment.recommended_action,
                llm_risk_score=llm_assessment.risk_score,
                assessment=assessment,
                prob_refuse=prob_refuse,
                answer_ceiling=answer_ceiling,
                low=low,
                high=high,
                config=config,
            )
        except PipelineHalt as halt:
            _print_halt(halt.stage, len(rows), total_records, config.paag_results_csv, halt.message)
            raise
        details = {
            **{f"assessment_{key}": value for key, value in asdict(assessment).items()},
            "rule_decision": rule_decision.value,
            "refuse_probability": prob_refuse,
            "answer_ceiling": answer_ceiling,
            "escalate_to_answer_threshold": low,
            "escalate_to_refuse_threshold": high,
            "llm_recommended_action": llm_assessment.recommended_action,
            "llm_risk_score": llm_assessment.risk_score,
            "llm_safe_score": llm_assessment.safe_score,
            "llm_confidence": llm_assessment.confidence,
            "llm_harmful_request": llm_assessment.harmful_request,
            "llm_benign_context": llm_assessment.benign_context,
            "llm_policy_rationale": llm_assessment.policy_rationale,
            "llm_risk_domains": "|".join(llm_assessment.risk_domains),
            "llm_assessor_used": llm_assessment.used_llm,
            "llm_assessor_cache_hit": llm_assessment.cache_hit,
        }
        try:
            response = run_async(
                orchestrator.generate_response(
                    record["prompt_text"], decision, assessment, llm_assessment, prompt_id=prompt_id
                )
            )
            should_verify = (
                (decision == AbstentionDecision.REFUSE and config.hybrid.verify_paag_refusals)
                or (decision == AbstentionDecision.ANSWER and config.hybrid.verify_paag_answers)
                or decision == AbstentionDecision.ESCALATE
            )
            if should_verify:
                verification = run_async(
                    orchestrator.verify_response(record["prompt_text"], decision.value, response, prompt_id=prompt_id)
                )
            else:
                verification = orchestrator.local_verify_response(record["prompt_text"], decision.value, response)
        except PipelineHalt as halt:
            _print_halt(halt.stage, len(rows), total_records, config.paag_results_csv, halt.message)
            raise

        final_decision = decision
        final_response = response
        if verification.should_override_to_refusal and decision != AbstentionDecision.REFUSE:
            final_decision = AbstentionDecision.REFUSE
            final_response = REFUSAL_TEMPLATE
        elif (
            verification.unnecessary_refusal
            and verification.recommended_action == "ANSWER"
            and decision != AbstentionDecision.ANSWER
        ):
            final_decision = AbstentionDecision.ANSWER
            final_response = build_safe_answer(_record_from_row(record), assessment)
        elif final_decision == AbstentionDecision.ANSWER:
            final_response = build_safe_answer(_record_from_row(record), assessment)

        record_obj = _record_from_row(record)
        row = {
                **record,
                **details,
                "predicted_action": final_decision.value,
                "initial_decision": decision.value,
                "model_response": final_response,
                "raw_model_response": response,
                "verifier_is_safe": verification.is_safe,
                "verifier_contains_actionable_harm": verification.contains_actionable_harm,
                "verifier_unsupported_or_misleading": verification.unsupported_or_misleading,
                "verifier_unnecessary_refusal": verification.unnecessary_refusal,
                "verifier_should_override_to_refusal": verification.should_override_to_refusal,
                "verifier_recommended_action": verification.recommended_action,
                "verifier_confidence": verification.confidence,
                "verifier_rationale": verification.rationale,
                "llm_verifier_used": verification.used_llm,
                "llm_verifier_cache_hit": verification.cache_hit,
                "answer_alignment_ok": evaluate_safe_answer(record_obj, final_decision.value, final_response),
            }
        rows.append(row)
        _append_checkpoint(config.paag_results_csv, row)
        if len(rows) % 50 == 0 or index == total_records:
            print(
                f"[PAAG] checkpointed {len(rows)}/{total_records} prompts "
                f"(assessor_calls={orchestrator.assessor_calls}, verifier_calls={orchestrator.verifier_calls})"
            )
    results = pd.DataFrame(rows)
    write_csv(results, config.paag_results_csv)
    write_csv(build_paper_tables(results), config.paper_table_csv)
    return results


def run_judge_eval(config: ExperimentConfig, baseline_results: pd.DataFrame, paag_results: pd.DataFrame) -> pd.DataFrame:
    baseline_eval = baseline_results[baseline_results["split"] != "train"].copy()
    paag_eval = paag_results[paag_results["split"] != "train"].copy()
    merged = baseline_eval.merge(
        paag_eval,
        on="prompt_id",
        suffixes=("_baseline", "_paag"),
        how="inner",
    )
    orchestrator = HybridOrchestrator(config)
    rows, completed_ids = _load_checkpoint(config.judge_results_csv, config.hybrid.resume_from_checkpoints)
    total_records = len(merged)
    for index, record in enumerate(merged.to_dict("records"), start=1):
        prompt_id = str(record["prompt_id"])
        if prompt_id in completed_ids:
            continue
        try:
            verdict = run_async(
                orchestrator.judge_pair(
                    prompt_text=record["prompt_text_baseline"],
                    expected_action=record["expected_action_baseline"],
                    baseline_action=record["predicted_action_baseline"],
                    baseline_response=record["model_response_baseline"],
                    paag_action=record["predicted_action_paag"],
                    paag_response=record["model_response_paag"],
                    source_dataset=record["source_dataset_baseline"],
                    risk_domain=record["risk_domain_baseline"],
                    prompt_id=prompt_id,
                )
            )
        except PipelineHalt as halt:
            _print_halt(halt.stage, len(rows), total_records, config.judge_results_csv, halt.message)
            raise
        row = {
                "prompt_id": record["prompt_id"],
                "source_dataset": record["source_dataset_baseline"],
                "risk_domain": record["risk_domain_baseline"],
                "perturbation_type": record["perturbation_type_baseline"],
                "expected_action": record["expected_action_baseline"],
                "baseline_predicted_action": record["predicted_action_baseline"],
                "paag_predicted_action": record["predicted_action_paag"],
                "baseline_response": record["model_response_baseline"],
                "paag_response": record["model_response_paag"],
                **asdict(verdict),
            }
        rows.append(row)
        _append_checkpoint(config.judge_results_csv, row)
        if len(rows) % 50 == 0 or index == total_records:
            print(f"[JUDGE] checkpointed {len(rows)}/{total_records} prompts (judge_calls={orchestrator.judge_calls})")
    results = pd.DataFrame(rows)
    write_csv(results, config.judge_results_csv)
    return results


def export_paper_artifacts(
    config: ExperimentConfig,
    baseline_results: pd.DataFrame,
    paag_results: pd.DataFrame,
    judge_results: pd.DataFrame | None = None,
) -> None:
    baseline_metrics = compute_abstention_metrics(baseline_results)
    paag_metrics = compute_abstention_metrics(paag_results)
    summary_df = build_summary_metrics_df(baseline_metrics, paag_metrics)
    comparison_df = build_dataset_comparison_df(baseline_results, paag_results, build_paper_tables)
    if judge_results is None:
        judge_results = run_judge_eval(config, baseline_results, paag_results)
    causal_df = build_causal_comparison_df(judge_results)
    error_df = build_error_analysis_df(paag_results, judge_results)

    write_csv(summary_df, config.summary_metrics_csv)
    write_csv(comparison_df, config.dataset_comparison_csv)
    write_csv(causal_df, config.causal_comparison_csv)
    write_csv(error_df, config.error_analysis_csv)
    generate_figures(summary_df, comparison_df, config.figures_dir, causal_df)


def _resolve_hybrid_decision(
    rule_decision: AbstentionDecision,
    llm_action: str,
    llm_risk_score: float,
    assessment: RiskAssessment,
    prob_refuse: float,
    answer_ceiling: float,
    low: float,
    high: float,
    config: ExperimentConfig,
) -> AbstentionDecision:
    severe_domains = {"adversarial_harm", "hate_discrimination", "self_harm", "misinformation"}
    if rule_decision == AbstentionDecision.REFUSE:
        return AbstentionDecision.REFUSE

    llm_action = llm_action.upper()
    if llm_action == "ESCALATE" and set(assessment.risk_domains).intersection(severe_domains):
        return AbstentionDecision.REFUSE
    if llm_action == "REFUSE" and llm_risk_score >= config.hybrid.llm_refuse_threshold:
        return AbstentionDecision.REFUSE
    if rule_decision == AbstentionDecision.ANSWER:
        if llm_action == "ESCALATE":
            return AbstentionDecision.ESCALATE
        if prob_refuse >= answer_ceiling:
            return AbstentionDecision.ESCALATE
        if llm_action == "ANSWER" or llm_risk_score <= config.hybrid.llm_answer_threshold:
            return AbstentionDecision.ANSWER
        return AbstentionDecision.ESCALATE

    if prob_refuse >= high:
        return AbstentionDecision.REFUSE
    if prob_refuse <= low and llm_action != "REFUSE":
        return AbstentionDecision.ANSWER
    if llm_action == "ANSWER" and llm_risk_score <= config.hybrid.llm_answer_threshold:
        return AbstentionDecision.ANSWER
    if llm_action == "REFUSE":
        return AbstentionDecision.REFUSE
    return AbstentionDecision.ESCALATE


def _record_from_row(record: dict) -> object:
    from .datasets import SafetyPromptRecord

    return SafetyPromptRecord(
        prompt_id=record["prompt_id"],
        prompt_text=record["prompt_text"],
        expected_action=record["expected_action"],
        risk_domain=record["risk_domain"],
        source_dataset=record["source_dataset"],
        perturbation_type=record["perturbation_type"],
        contrast_group_id=record["contrast_group_id"],
        severity=record["severity"],
        split=record["split"],
        prompt_origin=record.get("prompt_origin", "dataset"),
        base_prompt=record.get("base_prompt", record["prompt_text"]),
    )
