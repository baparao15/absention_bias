from __future__ import annotations

from dataclasses import dataclass

from .preprocessing import extract_salient_span, normalize_prompt_text


HARMFUL_KEYWORDS = {
    "cyber": ["malware", "ransomware", "credential", "payload", "phishing", "hack", "exploit", "bypass"],
    "weapons": ["weapon", "bomb", "explosive", "detonator", "poison"],
    "fraud": ["steal", "fraud", "counterfeit", "scam", "forge"],
    "adversarial_harm": [
        "defamatory article",
        "mine cryptocurrencies while evading detection",
        "keylogger",
        "cheat in online exams",
        "manipulate someone into committing suicide",
        "justify ethnic genocide",
        "torture ring",
        "extort money",
        "gamble their life savings",
        "bribe a university official",
        "hide income",
        "falsify documents",
        "insider trading",
        "ddos attacks",
        "fake charity",
        "fake weight loss drug",
        "hijacks a user's computer",
        "laundry detergent",
    ],
    "privacy": ["doxx", "stalk", "track someone's", "spy on", "steal password"],
    "private_data": [
        "login",
        "password",
        "bank account details",
        "credit card details",
        "birth certificate",
        "criminal history",
        "date of birth",
        "driver's license number",
        "passport number",
        "social security number",
        "medical records",
        "tax records",
        "phone number",
        "email address",
        "home address",
    ],
    "hate_discrimination": [
        "denied the right to vote",
        "equal rights",
        "should be discriminated against",
        "not have access to healthcare",
        "inferior or dangerous",
        "xenophobic speech",
        "body shaming",
        "sexist",
        "racist",
        "voting rights",
        "overrepresented in government",
        "join political parties",
        "political beliefs",
        "restore jim crow",
        "under taliban rule",
    ],
    "self_harm": [
        "commit suicide",
        "go out like",
        "date rape drug",
        "anorexia",
        "bulimia",
        "end my life",
        "die like",
        "assassination",
    ],
    "misinformation": [
        "prove that",
        "hoax",
        "justify",
        "morally justified",
        "good thing",
        "directly involved",
        "complicit in",
        "never took place",
        "overstate what happened",
        "replicate the many benefits",
        "right to kill people",
        "pulse nightclub shooting justified",
        "dylann roof right",
        "discourages the hiring of individuals with disabilities",
    ],
}

BENIGN_CONTEXT_MARKERS = [
    "classify",
    "summarize",
    "analyze",
    "policy summary",
    "is this dangerous",
    "why it is unsafe",
    "defensive",
    "educational",
    "classroom",
    "quote",
    "quoted",
    "training",
    "teaches",
    "spot phishing",
    "report them safely",
    "store household chemicals safely",
    "reduce fire and explosion risk",
    "improve privacy settings",
    "legitimate documents",
    "instead of a bank statement",
    "reduce unwanted tracking",
    "how to spot",
    "why the following request should not be fulfilled",
    "what happened to",
    "what happened during",
    "what was the scale of",
    "what was the impact of",
    "during world war 1",
    "during world war 2",
    "during world war i",
    "historical",
    "explain why it is unsafe",
    "safety alternatives",
    "customer service",
    "white house",
    "phone number of the police",
    "public contact information",
    "fictional character",
    "fictional",
]

AMBIGUOUS_MARKERS = [
    "for research only",
    "hypothetically",
    "curious about",
    "without using it",
]


@dataclass(slots=True)
class PolicySignals:
    risk_domains: list[str]
    harm_hits: list[str]
    benign_hits: list[str]
    ambiguous_hits: list[str]


def extract_policy_signals(prompt_text: str) -> PolicySignals:
    text = normalize_prompt_text(prompt_text)
    salient = extract_salient_span(prompt_text)
    combined_text = f"{text} {salient}".strip()
    risk_domains: list[str] = []
    harm_hits: list[str] = []
    for domain, keywords in HARMFUL_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in combined_text]
        if hits:
            risk_domains.append(domain)
            harm_hits.extend(hits)
    benign_hits = [marker for marker in BENIGN_CONTEXT_MARKERS if marker in combined_text]
    ambiguous_hits = [marker for marker in AMBIGUOUS_MARKERS if marker in combined_text]
    return PolicySignals(
        risk_domains=sorted(set(risk_domains)),
        harm_hits=sorted(set(harm_hits)),
        benign_hits=sorted(set(benign_hits)),
        ambiguous_hits=sorted(set(ambiguous_hits)),
    )
