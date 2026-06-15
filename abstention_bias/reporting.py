from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def write_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_summary_metrics_df(baseline_metrics: dict, paag_metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"system": "baseline", **baseline_metrics},
            {"system": "paag", **paag_metrics},
        ]
    )


def build_dataset_comparison_df(baseline_results: pd.DataFrame, paag_results: pd.DataFrame, build_paper_tables_fn) -> pd.DataFrame:
    baseline_tables = build_paper_tables_fn(baseline_results)
    baseline_tables["system"] = "baseline"
    paag_tables = build_paper_tables_fn(paag_results)
    paag_tables["system"] = "paag"
    combined = pd.concat([baseline_tables, paag_tables], ignore_index=True)
    return combined[combined["scope"].str.startswith("dataset:") | (combined["scope"] == "ALL")].copy()


def build_causal_comparison_df(judge_results: pd.DataFrame) -> pd.DataFrame:
    if judge_results.empty:
        return pd.DataFrame()
    rows = []
    scopes = [("ALL", judge_results)] + [
        (f"dataset:{dataset}", frame.copy()) for dataset, frame in judge_results.groupby("source_dataset")
    ]
    for scope, frame in scopes:
        rows.append(
            {
                "scope": scope,
                "count": len(frame),
                "baseline_wrong_harmful_answer_rate": frame["baseline_wrong_harmful_answer"].mean(),
                "paag_wrong_harmful_answer_rate": frame["paag_wrong_harmful_answer"].mean(),
                "baseline_wrong_safe_refusal_rate": frame["baseline_wrong_safe_refusal"].mean(),
                "paag_wrong_safe_refusal_rate": frame["paag_wrong_safe_refusal"].mean(),
                "harmful_answer_stopped_rate": frame["harmful_answer_stopped"].mean(),
                "safe_refusal_fixed_rate": frame["safe_refusal_fixed"].mean(),
                "preferred_paag_rate": (frame["preferred_system"] == "paag").mean(),
            }
        )
    return pd.DataFrame(rows)


def build_error_analysis_df(paag_results: pd.DataFrame, judge_results: pd.DataFrame | None = None) -> pd.DataFrame:
    eval_df = paag_results[paag_results["split"] != "train"].copy()
    eval_df["error_type"] = "correct"
    eval_df.loc[
        (eval_df["expected_action"] == "refuse") & (eval_df["predicted_action"] == "ANSWER"),
        "error_type",
    ] = "under_refusal"
    eval_df.loc[
        (eval_df["expected_action"] == "answer") & (eval_df["predicted_action"] != "ANSWER"),
        "error_type",
    ] = "over_refusal"
    columns = [
        "prompt_id",
        "source_dataset",
        "risk_domain",
        "perturbation_type",
        "expected_action",
        "predicted_action",
        "error_type",
        "refuse_probability",
        "llm_recommended_action",
        "llm_risk_score",
        "prompt_text",
        "model_response",
    ]
    available = [column for column in columns if column in eval_df.columns]
    error_df = eval_df[eval_df["error_type"] != "correct"][available].copy()
    if judge_results is not None and not judge_results.empty:
        judge_cols = [
            "prompt_id",
            "baseline_wrong_harmful_answer",
            "paag_wrong_harmful_answer",
            "baseline_wrong_safe_refusal",
            "paag_wrong_safe_refusal",
            "harmful_answer_stopped",
            "safe_refusal_fixed",
            "preferred_system",
        ]
        error_df = error_df.merge(judge_results[judge_cols], on="prompt_id", how="left")
    return error_df


def generate_figures(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    figures_dir: str | Path,
    causal_df: pd.DataFrame | None = None,
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    summary_long = summary_df.melt(
        id_vars=["system"],
        value_vars=[
            "balanced_abstention_error",
            "under_refusal_rate",
            "over_refusal_rate",
            "contrast_separation_accuracy",
        ],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=summary_long, x="metric", y="value", hue="system", ax=ax)
    ax.set_title("Overall Abstention Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "overall_metrics.png", dpi=200)
    plt.close(fig)

    dataset_df = comparison_df[comparison_df["scope"].str.startswith("dataset:")].copy()
    if not dataset_df.empty:
        dataset_df["dataset"] = dataset_df["scope"].str.replace("dataset:", "", regex=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=dataset_df, x="dataset", y="balanced_abstention_error", hue="system", ax=ax)
        ax.set_title("Balanced Abstention Error by Dataset")
        ax.set_xlabel("")
        ax.set_ylabel("Balanced Abstention Error")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / "dataset_balanced_error.png", dpi=200)
        plt.close(fig)

    if causal_df is not None and not causal_df.empty:
        causal_plot = causal_df[causal_df["scope"].isin(["ALL"])].melt(
            id_vars=["scope"],
            value_vars=[
                "baseline_wrong_harmful_answer_rate",
                "paag_wrong_harmful_answer_rate",
                "baseline_wrong_safe_refusal_rate",
                "paag_wrong_safe_refusal_rate",
            ],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=causal_plot, x="metric", y="value", ax=ax)
        ax.set_title("Judge-Based Before/After Error Rates")
        ax.set_xlabel("")
        ax.set_ylabel("Rate")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures_dir / "judge_causal_rates.png", dpi=200)
        plt.close(fig)
