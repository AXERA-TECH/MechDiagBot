"""Approval bookkeeping for side-effecting plans.

A pending plan is only as trustworthy as the identity and the machine state it
was approved against.  This module keeps all of that in one immutable record
(:class:`PendingApproval`) and parses the operator's reply with an explicit
allow-list, so "我不确认，请不要执行" can never be mistaken for consent the way a
substring check on "确认" would.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from cat_assistant.domain.models import MachineSnapshot
from cat_assistant.domain.plans import TaskPlan


class ConfirmationKind(str, Enum):
    CONFIRM = "confirm"
    CANCEL = "cancel"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """A plan waiting for one specific operator on one specific machine state."""

    plan: TaskPlan
    session_id: str
    operator_id: str | None
    machine_id: str
    plan_digest: str
    machine_signature: tuple[str, bool, str, tuple[str, ...]]
    requested_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


def plan_digest(plan: TaskPlan) -> str:
    """Stable fingerprint of the plan that is about to be approved."""
    payload = [
        {
            "step_id": step.step_id,
            "capability": step.capability,
            "arguments": step.arguments,
            "depends_on": list(step.depends_on),
        }
        for step in plan.steps
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def machine_signature(
    machine: MachineSnapshot,
) -> tuple[str, bool, str, tuple[str, ...]]:
    """The parts of a snapshot that make a stored plan still applicable.

    Comparing a fresh snapshot against this tuple catches "the engine was
    already stopped / the fault code cleared while the plan sat in the approval
    queue" without freezing the whole telemetry payload into the approval.
    """
    return (
        machine.machine_id,
        machine.engine_running,
        machine.operating_state,
        tuple(sorted(machine.fault_codes)),
    )


# Replies that mean "yes, run the stored plan".  Matching is on a whole
# normalized phrase, never a substring: "yesterday" must not read as "yes",
# and "确认一下油量" is a new question, not an approval.
_CONFIRM_PHRASES = frozenset(
    {
        "确认",
        "确认执行",
        "确认继续",
        "确认无误",
        "确定",
        "确定执行",
        "同意",
        "同意执行",
        "执行吧",
        "yes",
        "y",
        "approve",
        "approved",
        "proceed",
        "confirm",
        "confirmed",
        "goahead",
        "yesplease",
        "pleaseproceed",
        "pleasegoahead",
    }
)

# Cancellation is checked first and may appear anywhere in the reply, because
# "确认取消" / "no, do not proceed" must never execute.
_CANCEL_PHRASES = (
    "取消",
    "撤销",
    "撤回",
    "作废",
    "终止",
    "拒绝",
    "算了",
    "不要执行",
    "不要确认",
    "不予执行",
    "不执行",
    "不确认",
    "不同意",
    "不批准",
    "不用了",
    "别执行",
    "先别",
)
_CANCEL_WORDS = (
    "cancel",
    "cancelled",
    "canceled",
    "abort",
    "aborted",
    "reject",
    "rejected",
    "decline",
    "declined",
    "dont",
    "don't",
    "do not",
    "not approved",
    "not confirm",
    "no",
    "nope",
)

_SEPARATOR_RE = re.compile(r"[\s!！?？。.,，、;；:：~～\"'“”‘’()（）\[\]【】]+")
# A cancellation embedded in a longer request ("检查故障，不要执行任何操作")
# must block a pending approval, but with nothing pending the same text is a
# normal request. These are the phrasings that stand alone as a cancel reply.
_STANDALONE_CANCEL = frozenset(
    {
        "取消",
        "取消操作",
        "取消执行",
        "取消吧",
        "撤销",
        "撤回",
        "作废",
        "终止",
        "拒绝",
        "拒绝执行",
        "算了",
        "不要执行",
        "不执行",
        "不确认",
        "不同意",
        "别执行",
        "先别执行",
        "不用了",
        "cancel",
        "abort",
        "reject",
        "decline",
        "no",
        "nope",
        "dont",
        "don't",
        "do not proceed",
        "not approved",
    }
)
# Narrower than _CONFIRM_PHRASES: with no pending plan, a bare "好"/"ok" is an
# ordinary conversational acknowledgement, while "确认执行" is clearly an
# approval reply that has nothing to approve.
_STANDALONE_CONFIRM = frozenset(
    {
        "确认",
        "确认执行",
        "确认继续",
        "确定执行",
        "同意",
        "同意执行",
        "执行吧",
        "yes",
        "y",
        "approve",
        "approved",
        "proceed",
        "confirm",
        "confirmed",
        "goahead",
        "pleaseproceed",
    }
)


def _normalized(text: str) -> str:
    return _SEPARATOR_RE.sub("", text.casefold())


def _contains_word(text: str, word: str) -> bool:
    pattern = rf"(?<![a-z]){re.escape(word)}(?![a-z])"
    return re.search(pattern, text) is not None


def parse_confirmation(text: str) -> ConfirmationKind:
    """Classify an operator reply as confirmation, cancellation or neither."""
    raw = str(text or "").strip().casefold()
    if not raw:
        return ConfirmationKind.NONE
    if any(phrase in raw for phrase in _CANCEL_PHRASES):
        return ConfirmationKind.CANCEL
    if any(_contains_word(raw, word) for word in _CANCEL_WORDS):
        return ConfirmationKind.CANCEL
    normalized = _normalized(raw)
    if normalized in _CONFIRM_PHRASES or raw in _CONFIRM_PHRASES:
        return ConfirmationKind.CONFIRM
    return ConfirmationKind.NONE


def is_standalone_cancel(text: str) -> bool:
    """True when the whole message is a cancel reply, not a request that merely
    mentions not executing something."""
    raw = str(text or "").strip().casefold()
    if not raw:
        return False
    normalized = _normalized(raw)
    return normalized in _STANDALONE_CANCEL or raw in _STANDALONE_CANCEL


def is_standalone_confirmation(text: str) -> bool:
    """True when the whole message is a clear approval reply."""
    raw = str(text or "").strip().casefold()
    if not raw:
        return False
    normalized = _normalized(raw)
    return normalized in _STANDALONE_CONFIRM or raw in _STANDALONE_CONFIRM
