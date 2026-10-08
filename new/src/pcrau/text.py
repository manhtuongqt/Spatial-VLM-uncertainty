from __future__ import annotations

import hashlib
import re
from typing import Sequence


RELATION_CLASSES = [
    "direct",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "nearer_than",
    "farther_than",
    "between_in_depth",
    "nearer_than_both",
]

_PATTERNS: Sequence[tuple[str, str]] = (
    ("between_in_depth", r"\bbetween\b.*?\bdepth\b"),
    ("nearer_than_both", r"\b(?:nearer|closer)\b.*?\bboth\b"),
    ("right_of", r"\bright\s+of\b"),
    ("left_of", r"\bleft\s+of\b"),
    ("front_of", r"\bfront\s+of\b"),
    ("behind", r"\bbehind\b"),
    ("farther_than", r"\b(?:farther|further)\b"),
    ("nearer_than", r"\b(?:nearer|closer)\b"),
)


def words(prompt: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", prompt.lower())


def tokenize(prompt: str, max_tokens: int, vocab_size: int) -> tuple[list[int], list[bool]]:
    tokens = words(prompt)[:max_tokens]
    ids = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        ids.append(int.from_bytes(digest[:8], "big") % (vocab_size - 1) + 1)
    ids = ids or [1]
    mask = [True] * len(ids)
    padding = max_tokens - len(ids)
    return ids + [0] * padding, mask + [False] * padding


def parse_relations(prompt: str, maximum: int = 3) -> list[str]:
    token_list = words(prompt)
    text = " ".join(token_list)
    found: list[tuple[int, str]] = []
    for name, pattern in _PATTERNS:
        match = re.search(pattern, text)
        if match:
            found.append((match.start(), name))
    ordered: list[str] = []
    for _, name in sorted(found):
        if name not in ordered:
            ordered.append(name)
    if "nearer_than_both" in ordered and "nearer_than" in ordered:
        ordered.remove("nearer_than")
    return (ordered or ["direct"])[:maximum]


def relation_ids(prompt: str, maximum: int = 3) -> tuple[list[int], list[bool]]:
    relations = parse_relations(prompt, maximum)
    ids = [RELATION_CLASSES.index(name) for name in relations]
    mask = [True] * len(ids)
    return ids + [0] * (maximum - len(ids)), mask + [False] * (maximum - len(ids))


def prompt_anchor_mask(prompt: str, maximum: int = 3) -> list[bool]:
    """Infer slot count from prompt-visible relation syntax only."""
    relations = parse_relations(prompt, maximum)
    if relations == ["direct"]:
        count = 0
    elif any(name in {"between_in_depth", "nearer_than_both"} for name in relations):
        count = 2
    else:
        count = min(maximum, len(relations))
    return [index < count for index in range(maximum)]
