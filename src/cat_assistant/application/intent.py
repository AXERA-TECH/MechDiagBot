"""Small deterministic intent helpers for actuation requests.

The rule planner and the simulated control adapter both need the same answer to
"is this text actually asking for an action?", and neither may be fooled by
substrings such as "确认" inside "不确认".  Keeping the checks here makes the
two paths agree and keeps keyword-driven safety out of the orchestration file.
"""

from __future__ import annotations

import re
from enum import Enum


class ControlIntent(str, Enum):
    STOP = "stop"
    START = "start"
    CONFLICT = "conflict"
    QUESTION = "question"
    NONE = "none"


STOP_TERMS: tuple[str, ...] = (
    "停止",
    "停机",
    "停下",
    "停一下",
    "刹车",
    "熄火",
    "关机",
    "关闭",
    "shutdown",
    "power off",
    "turn off",
    "stop",
    "halt",
)
START_TERMS: tuple[str, ...] = (
    "启动",
    "发动",
    "开机",
    "开始运行",
    "点火",
    "power on",
    "turn on",
    "start",
    "ignite",
)

# Negation / question cues are checked in a short prefix before an action verb,
# so "是否停机" and "千万不要停机" are both read as *not* a stop command.
_NEGATION_PHRASES: tuple[str, ...] = (
    "不要",
    "不用",
    "不需",
    "不必",
    "不得",
    "不能",
    "不可",
    "无法",
    "没法",
    "不想",
    "不会",
    "不准",
    "禁止",
    "严禁",
    "请勿",
    "勿",
    "别",
    "无需",
    "不执行",
    "不确认",
    "是否",
    "能否",
    "可否",
    "要不要",
    "需不需要",
    "是不是",
    "不是",
    "有没有",
    "为什么",
    "怎么",
    "如何",
    "取消",
    "撤销",
    "暂停",
)
_NEGATION_WORDS: tuple[str, ...] = (
    "no",
    "not",
    "never",
    "don't",
    "dont",
    "do not",
    "without",
    "cannot",
    "can't",
    "won't",
    "shouldn't",
    "mustn't",
    "cancel",
    "abort",
)

_QUESTION_PHRASES: tuple[str, ...] = (
    "?",
    "？",
    "吗",
    "呢",
    "怎么",
    "怎样",
    "如何",
    "什么",
    "为什么",
    "是否",
    "能否",
    "可否",
    "会不会",
    "可不可以",
    "要不要",
    "需不需要",
    "有没有",
)
_QUESTION_WORDS: tuple[str, ...] = (
    "what",
    "why",
    "how",
    "whether",
    "which",
    "can i",
    "could i",
    "should i",
    "do i",
)

_PREFIX_WINDOW = 16
_LATIN_TERM_RE = re.compile(r"[a-z ]+")
# "发动" is a start verb, but "发动机" (engine) is a noun: "停止发动机" must not
# be read as stop+start. A term-specific lookahead keeps the verb usable.
_TERM_FOLLOWING_EXCLUSIONS = {"发动": "机"}


def _term_occurrences(text: str, term: str) -> list[int]:
    if not term:
        return []
    if term in _TERM_FOLLOWING_EXCLUSIONS:
        excluded = _TERM_FOLLOWING_EXCLUSIONS[term]
        pattern = rf"{re.escape(term)}(?!{re.escape(excluded)})"
        return [match.start() for match in re.finditer(pattern, text)]
    if _LATIN_TERM_RE.fullmatch(term):
        pattern = rf"(?<![a-z]){re.escape(term)}(?![a-z])"
        return [match.start() for match in re.finditer(pattern, text)]
    return [index for index in range(len(text)) if text.startswith(term, index)]


def _contains_latin_term(text: str, term: str) -> bool:
    return bool(_term_occurrences(text, term))


def _is_negated_at(text: str, index: int) -> bool:
    prefix = text[max(0, index - _PREFIX_WINDOW):index]
    if any(phrase in prefix for phrase in _NEGATION_PHRASES):
        return True
    return any(_contains_latin_term(prefix, word) for word in _NEGATION_WORDS)


def has_positive_action(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    """True when at least one term occurs without a negation in front of it."""
    lowered = text.casefold()
    for term in terms:
        for index in _term_occurrences(lowered, term.casefold()):
            if not _is_negated_at(lowered, index):
                return True
    return False


def looks_like_question(text: str) -> bool:
    lowered = text.casefold()
    if any(phrase in lowered for phrase in _QUESTION_PHRASES):
        return True
    # Sentence-final 么 ("发动了吗"/"启动了么") is interrogative, but a bare 么
    # also occurs in "那么"/"这么" discourse markers.
    if re.search(r"么[\s!！?？。.]*$", lowered):
        return True
    return any(_contains_latin_term(lowered, word) for word in _QUESTION_WORDS)


def has_negation_cue(text: str) -> bool:
    lowered = text.casefold()
    if any(phrase in lowered for phrase in _NEGATION_PHRASES):
        return True
    return any(_contains_latin_term(lowered, word) for word in _NEGATION_WORDS)


def control_command(text: str) -> ControlIntent:
    """Classify an actuator request, refusing questions and mixed commands."""
    lowered = text.casefold()
    if looks_like_question(lowered):
        return ControlIntent.QUESTION
    stop = has_positive_action(lowered, STOP_TERMS)
    start = has_positive_action(lowered, START_TERMS)
    if stop and start:
        return ControlIntent.CONFLICT
    if stop:
        return ControlIntent.STOP
    if start:
        return ControlIntent.START
    return ControlIntent.NONE
