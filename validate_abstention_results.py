from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PLACEHOLDER = "Here is a concise answer to the request."
EXPECTED_BASELINE_ROWS = 980
EXPECTED_PAAG_ROWS = 980
EXPECTED_JUDGE_ROWS = 605


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[MISSING] {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _count_false(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    normalized = df[column].astype(str).str.lower()
    return int((normalized != "true").sum())


def _count_placeholder(df: pd.DataFrame) -> int:
    if "model_response" not in df.columns:
        return 0
    return int((df["model_response"].astype(str) == PLACEHOLDER).sum())


def _trace_summary(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    errors = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            if record.get("status") == "error":
                errors += 1
    return total, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate abstention-bias experiment outputs.")
    parser.add_argument("--results-dir", default="results/abstention_bias")
    args = parser.parse_args()

    root = Path(args.results_dir)
    baseline = _read_csv(root / "baseline_results.csv")
    paag = _read_csv(root / "paag_results.csv")
    judge = _read_csv(root / "judge_results.csv")
    trace_total, trace_errors = _trace_summary(root / "llm_traces.jsonl")

    issues = {
        "baseline_rows": len(baseline),
        "paag_rows": len(paag),
        "judge_rows": len(judge),
        "baseline_missing_rows": max(0, EXPECTED_BASELINE_ROWS - len(baseline)),
        "paag_missing_rows": max(0, EXPECTED_PAAG_ROWS - len(paag)),
        "judge_missing_rows": max(0, EXPECTED_JUDGE_ROWS - len(judge)),
        "baseline_llm_not_used": _count_false(baseline, "llm_baseline_used"),
        "baseline_verifier_llm_not_used": _count_false(baseline, "llm_verifier_used"),
        "paag_assessor_llm_not_used": _count_false(paag, "llm_assessor_used"),
        "paag_verifier_llm_not_used": _count_false(paag, "llm_verifier_used"),
        "judge_llm_not_used": _count_false(judge, "used_llm"),
        "baseline_placeholder_answers": _count_placeholder(baseline),
        "paag_placeholder_answers": _count_placeholder(paag),
        "llm_trace_records": trace_total,
        "llm_trace_errors": trace_errors,
    }

    for key, value in issues.items():
        print(f"{key}: {value}")

    blocking = [
        "baseline_missing_rows",
        "paag_missing_rows",
        "judge_missing_rows",
        "baseline_llm_not_used",
        "baseline_verifier_llm_not_used",
        "judge_llm_not_used",
        "baseline_placeholder_answers",
    ]
    failed = any(issues[key] for key in blocking)
    if failed:
        print("[FAIL] Results still contain fallback contamination or LLM errors.")
        return 1
    print("[PASS] Results are clean for final reporting checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
