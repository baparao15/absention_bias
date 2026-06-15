from __future__ import annotations

import re


LEET_REPLACEMENTS = {
    "w3apon": "weapon",
    "malw4re": "malware",
    "expl0sive": "explosive",
    "byp4ss": "bypass",
    "st3al": "steal",
    "h4ck": "hack",
    "dd0s": "ddos",
    "crypt0": "crypto",
}

NOISE_PATTERNS = [
    r"\{[^}]*\}",
    r"\[[^\]]*\]",
    r"\([^)]*\)",
    r"[`$<>_=\\/|]+",
]


def normalize_prompt_text(text: str) -> str:
    normalized = text.lower()
    for src, dst in LEET_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    for pattern in NOISE_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s':.-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


JAILBREAK_ACTION_PATTERNS = [
    r"\b(write|create|develop|design|offer|provide|give|explain|outline|describe|plan|list)\b[^.:\n]{0,160}",
    r"\bhow do i\b[^.:\n]{0,160}",
    r"\bhow can i\b[^.:\n]{0,160}",
]


def extract_salient_span(text: str) -> str:
    normalized = normalize_prompt_text(text)
    for pattern in JAILBREAK_ACTION_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            return match.group(0).strip()
    return normalized
