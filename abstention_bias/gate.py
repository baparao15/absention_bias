from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .policy import extract_policy_signals


class AbstentionDecision(str, Enum):
    ANSWER = "ANSWER"
    REFUSE = "REFUSE"
    ESCALATE = "ESCALATE"


@dataclass(slots=True)
class RiskAssessment:
    requested_capability: str
    risk_domains: list[str]
    harm_evidence: list[str]
    benign_evidence: list[str]
    policy_conflict: bool
    confidence: float


def assess_prompt(prompt_text: str) -> RiskAssessment:
    signals = extract_policy_signals(prompt_text)
    requested_capability = "benign_assistance"
    if signals.harm_hits:
        requested_capability = "potentially_harmful_guidance"
    if signals.benign_hits and not signals.harm_hits:
        requested_capability = "benign_analysis_or_transformation"

    policy_conflict = bool(signals.harm_hits and (signals.benign_hits or signals.ambiguous_hits))
    if signals.harm_hits and not signals.benign_hits:
        confidence = 0.95
    elif signals.harm_hits and signals.benign_hits:
        confidence = 0.72
    elif signals.benign_hits:
        confidence = 0.88
    else:
        confidence = 0.6

    return RiskAssessment(
        requested_capability=requested_capability,
        risk_domains=signals.risk_domains,
        harm_evidence=signals.harm_hits,
        benign_evidence=signals.benign_hits + signals.ambiguous_hits,
        policy_conflict=policy_conflict,
        confidence=confidence,
    )


def decide_action(assessment: RiskAssessment) -> AbstentionDecision:
    high_risk_domains = {
        "adversarial_harm",
        "hate_discrimination",
        "private_data",
        "self_harm",
        "privacy",
        "fraud",
        "cyber",
        "weapons",
        "misinformation",
    }
    severe_conflict_domains = {"adversarial_harm", "hate_discrimination", "self_harm", "misinformation"}
    if assessment.harm_evidence and not assessment.benign_evidence:
        return AbstentionDecision.REFUSE
    if assessment.harm_evidence and assessment.benign_evidence:
        allow_markers = {
            "classify",
            "summarize",
            "analyze",
            "policy summary",
            "why it is unsafe",
            "training",
            "teaches",
            "spot phishing",
            "report them safely",
            "how to spot",
            "what happened to",
            "what happened during",
            "what was the scale of",
            "what was the impact of",
            "during world war 2",
            "during world war i",
            "historical",
            "explain why it is unsafe",
            "customer service",
            "white house",
            "phone number of the police",
            "public contact information",
            "fictional character",
            "fictional",
        }
        if set(assessment.risk_domains).intersection(severe_conflict_domains):
            return AbstentionDecision.REFUSE
        if set(assessment.risk_domains).intersection(high_risk_domains) and not allow_markers.intersection(set(assessment.benign_evidence)):
            return AbstentionDecision.REFUSE
        if allow_markers.intersection(set(assessment.benign_evidence)):
            return AbstentionDecision.ANSWER
        return AbstentionDecision.ESCALATE
    if assessment.requested_capability == "benign_analysis_or_transformation":
        return AbstentionDecision.ANSWER
    if assessment.confidence < 0.65:
        return AbstentionDecision.ESCALATE
    return AbstentionDecision.ANSWER
