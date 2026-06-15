from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path

import pandas as pd

from .preprocessing import extract_salient_span, normalize_prompt_text


TOKEN_RE = re.compile(r"[a-z0-9_']+")


def tokenize(text: str) -> list[str]:
    normalized = normalize_prompt_text(text)
    salient = extract_salient_span(text)
    words = TOKEN_RE.findall(normalized)
    salient_words = TOKEN_RE.findall(salient)
    bigrams = [f"{words[i]}__{words[i+1]}" for i in range(len(words) - 1)]
    salient_bigrams = [f"salient__{salient_words[i]}__{salient_words[i+1]}" for i in range(len(salient_words) - 1)]
    salient_tags = [f"salient__{word}" for word in salient_words]
    return words + bigrams + salient_tags + salient_bigrams


@dataclass(slots=True)
class NaiveBayesAbstentionModel:
    vocabulary: dict[str, int]
    class_token_totals: dict[str, int]
    class_doc_counts: dict[str, int]
    token_counts: dict[str, dict[str, int]]

    def predict_proba_refuse(self, text: str) -> float:
        tokens = tokenize(text)
        vocab_size = max(len(self.vocabulary), 1)
        total_docs = sum(self.class_doc_counts.values())
        scores: dict[str, float] = {}
        for label in ["answer", "refuse"]:
            prior = math.log((self.class_doc_counts.get(label, 0) + 1) / (total_docs + 2))
            denom = self.class_token_totals.get(label, 0) + vocab_size
            token_lookup = self.token_counts.get(label, {})
            score = prior
            for token in tokens:
                score += math.log((token_lookup.get(token, 0) + 1) / denom)
            scores[label] = score
        max_score = max(scores.values())
        answer_exp = math.exp(scores["answer"] - max_score)
        refuse_exp = math.exp(scores["refuse"] - max_score)
        return refuse_exp / (answer_exp + refuse_exp)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "NaiveBayesAbstentionModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def train_abstention_model(train_df: pd.DataFrame) -> NaiveBayesAbstentionModel:
    vocabulary: dict[str, int] = {}
    class_token_totals = {"answer": 0, "refuse": 0}
    class_doc_counts = {"answer": 0, "refuse": 0}
    token_counts = {"answer": {}, "refuse": {}}

    for row in train_df.to_dict("records"):
        label = row["expected_action"]
        class_doc_counts[label] += 1
        for token in tokenize(row["prompt_text"]):
            vocabulary[token] = vocabulary.get(token, 0) + 1
            token_counts[label][token] = token_counts[label].get(token, 0) + 1
            class_token_totals[label] += 1

    return NaiveBayesAbstentionModel(
        vocabulary=vocabulary,
        class_token_totals=class_token_totals,
        class_doc_counts=class_doc_counts,
        token_counts=token_counts,
    )


def balanced_error(expected_actions: list[str], predicted_actions: list[str]) -> float:
    rows = pd.DataFrame({"expected_action": expected_actions, "predicted_action": predicted_actions})
    unsafe = rows[rows["expected_action"] == "refuse"]
    safe = rows[rows["expected_action"] == "answer"]
    under_refusal = (unsafe["predicted_action"] == "ANSWER").mean() if len(unsafe) else 0.0
    over_refusal = safe["predicted_action"].isin(["REFUSE", "ESCALATE"]).mean() if len(safe) else 0.0
    return float((under_refusal + over_refusal) / 2)
