"""Opt-in, prompt-only typed queries; never imported by the baseline forward.

English grammar v1: one `locate the ...` clause, direct or left/right of.
Spans use the existing text.words tokenizer, not an LLM tokenizer or oracle IDs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .text import words


PARSER_VERSION = "pcrau_query_parser_s1_v1"
_TOKENS = re.compile(r"[a-z0-9]+")
_COMMAND = re.compile(r"\blocate\s+the\s+(?P<core>[^.!?\n]+)(?:[.!?]|$)")
_HORIZONTAL = re.compile(r"(?P<target>.+?)\s+that\s+is\s+(?P<side>left|right)\s+of\s+the\s+(?P<anchor>.+)")
_PHRASE = re.compile(r"[a-z0-9]+(?:[\s-]+[a-z0-9]+)*")
_RESERVED = re.compile(r"\b(and|or|not|no|that|is|between|near|nearer|closer|farther|further|behind|front|left|right|than|from|of|above|below|under|over|beside|inside|outside|next|to|on|in|depth|frame)\b")
_OTHER_FRAME = re.compile(r"\b(world|object|robot|base_link)\s+(coordinate\s+)?frame\b|\brelative\s+to\s+the\s+robot\b|\bfrom\s+(?:the\s+)?(?:person|object|robot)['’]s\s+(?:view|perspective)")


@dataclass(frozen=True)
class PhraseSpan:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class TypedQuery:
    status: str
    reason: str | None
    predicate: str | None
    reference_frame: str | None
    target: PhraseSpan | None
    anchors: tuple[PhraseSpan, ...]
    command_char_span: tuple[int, int] | None
    total_tokens: int
    max_tokens: int
    max_anchors: int
    parser_version: str = PARSER_VERSION

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def token_masks(self) -> dict[str, list]:
        """All false on failure; unsupported queries cannot condition a branch."""
        target = [False] * self.max_tokens
        anchors = [[False] * self.max_tokens for _ in range(self.max_anchors)]
        active = [False] * self.max_anchors
        if self.supported:
            assert self.target is not None
            target[self.target.token_start:self.target.token_end] = [True] * (self.target.token_end-self.target.token_start)
            for slot, phrase in enumerate(self.anchors):
                anchors[slot][phrase.token_start:phrase.token_end] = [True] * (phrase.token_end-phrase.token_start)
                active[slot] = True
        return {"target_token_mask": target, "anchor_token_masks": anchors, "anchor_slot_mask": active}

    def to_dict(self) -> dict:
        return {**asdict(self), **self.token_masks()}


def parse_query(prompt: str, max_tokens: int = 72, max_anchors: int = 3) -> TypedQuery:
    """Fail closed outside v1 grammar; frame for horizontal queries is image.

    Character/token spans are half-open, measured in the original full prompt.
    Truncation of a required phrase is unsupported. Truncating instructions after
    complete phrases does not invalidate their spans; total_tokens records this.
    No ontology, annotation, masks, record/sample IDs or image features are read.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if max_tokens <= 0 or max_anchors < 0:
        raise ValueError("max_tokens must be positive; max_anchors nonnegative")
    text = prompt.lower()
    count = len(words(prompt))

    def fail(reason: str, status: str = "unsupported") -> TypedQuery:
        return TypedQuery(status, reason, None, None, None, (), None, count, max_tokens, max_anchors)

    if len(text) != len(prompt):
        return fail("unicode_case_expansion")
    if len(re.findall(r"\blocate\b", text)) != 1:
        return fail("missing_locate_command" if not re.search(r"\blocate\b", text) else "multiple_locate_commands",
                    "unsupported" if not re.search(r"\blocate\b", text) else "ambiguous")
    command = _COMMAND.search(text)
    if command is None:
        return fail("outside_locate_grammar")
    if re.search(r"(?:\bdo\s+not|\bnever|\bdon't)\s*$", text[:command.start()]):
        return fail("negated_command")
    if _OTHER_FRAME.search(text):
        return fail("unsupported_reference_frame")
    core_raw = command.group("core")
    offset = command.start("core")
    horizontal = _HORIZONTAL.fullmatch(core_raw.strip())
    offset += len(core_raw) - len(core_raw.lstrip())
    if horizontal:
        if max_anchors < 1:
            return fail("anchor_capacity")
        predicate = horizontal.group("side") + "_of"
        segments = [(horizontal.group(name), offset + horizontal.start(name), offset + horizontal.end(name))
                    for name in ("target", "anchor")]
    else:
        predicate = "direct"
        segments = [(core_raw.strip(), offset, offset + len(core_raw.strip()))]
    for value, _, _ in segments:
        if not _PHRASE.fullmatch(value.strip()) or _RESERVED.search(value):
            return fail("unsupported_phrase_or_scope")
    # Additional relation clauses outside the parsed command are not ignored.
    outside = text[:command.start()] + text[command.end():]
    if re.search(r"\b(left\s+of|right\s+of|behind|between|nearer\s+than|farther\s+than|front\s+of)\b", outside):
        return fail("additional_relation_clause")
    tokens = list(_TOKENS.finditer(text))
    assert [m.group() for m in tokens] == words(prompt)
    spans = []
    for value, start, end in segments:
        start += len(value)-len(value.lstrip())
        end -= len(value)-len(value.rstrip())
        indices = [i for i, token in enumerate(tokens) if token.start() >= start and token.end() <= end]
        if not indices:
            return fail("empty_phrase_tokens")
        if indices[-1] >= max_tokens:
            return fail("required_phrase_truncated")
        spans.append(PhraseSpan(" ".join(tokens[i].group() for i in indices), start, end, indices[0], indices[-1]+1))
    return TypedQuery("supported", None, predicate, "image" if horizontal else None,
                      spans[0], tuple(spans[1:]), (command.start(),command.end()), count, max_tokens, max_anchors)
