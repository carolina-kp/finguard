from __future__ import annotations

import re
from typing import Dict, List


PHRASES = [
    "AI safety",
    "model audit",
    "red-teaming",
    "risk register",
    "EU AI Act",
    "transparency report",
    "human oversight",
    "incident response",
    "data governance",
    "model card",
]

SENTENCE_PATTERN = re.compile(r"[^.?!]*[.?!]")


def _find_snippets(text: str, phrase: str) -> List[str]:
    snippets = []
    for sentence in SENTENCE_PATTERN.findall(text):
        if phrase.lower() in sentence.lower():
            snippets.append(sentence.strip())
    if not snippets and phrase.lower() in text.lower():
        snippets.append(text.strip())
    return snippets


def governance_index(texts: List[str]) -> Dict[str, object]:
    if not texts:
        return {"index": 0.0, "matches": []}

    matches = []
    found_phrases = set()

    for phrase in PHRASES:
        for text in texts:
            snippets = _find_snippets(text, phrase)
            if snippets:
                found_phrases.add(phrase)
                matches.extend({"phrase": phrase, "snippet": snippet} for snippet in snippets[:1])
                break

    score = (len(found_phrases) / len(PHRASES)) * 100 if PHRASES else 0.0

    return {
        "index": score,
        "matches": matches,
    }

