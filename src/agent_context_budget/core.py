"""Track estimated token usage across a conversation, warn at thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ContextBudgetError(Exception):
    """Raised when the context budget is exhausted and stop_on_limit=True."""


@dataclass
class BudgetWarning:
    """Emitted when a warning threshold is crossed."""

    used_tokens: int
    max_tokens: int
    remaining_tokens: int
    pct_used: float          # 0.0–1.0
    message: str


# ---------------------------------------------------------------------------
# Token estimation (no deps)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars/4 + overhead.

    Deliberately conservative — overshoots slightly so we never cut too close.
    The same heuristic used by major LLM provider planning docs.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _message_tokens(msg: Any) -> int:
    """Estimate tokens for one message dict."""
    overhead = 4  # role + framing
    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
    if isinstance(content, list):
        # Anthropic content blocks
        text = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = str(content) if content is not None else ""
    return estimate_tokens(text) + overhead


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------

@dataclass
class ContextBudget:
    """Track estimated token usage and enforce a per-conversation budget.

    Args:
        max_tokens: total token budget for the conversation.
        warn_at: thresholds (float 0.0–1.0 fraction, or int absolute) at which
            the warning callback fires. Default: [0.7, 0.9].
        on_warn: called with a BudgetWarning each time a threshold is crossed.
        stop_on_limit: if True (default), raise ContextBudgetError when
            max_tokens is exceeded.

    Usage::

        budget = ContextBudget(max_tokens=8000)
        for turn in agent_loop():
            budget.add_messages(messages)   # raises at 8001 tokens
    """

    max_tokens: int
    warn_at: list[float | int] = field(default_factory=list)
    on_warn: Callable[[BudgetWarning], None] | None = None
    stop_on_limit: bool = True
    _used: int = field(default=0, init=False, repr=False)
    _warned: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not self.warn_at:
            self.warn_at = [0.7, 0.9]
        # Normalise thresholds to absolute token counts
        self._thresholds: list[int] = sorted(
            {
                max(1, int(w * self.max_tokens)) if isinstance(w, float) else int(w)
                for w in self.warn_at
            }
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    @property
    def used_tokens(self) -> int:
        return self._used

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self._used)

    @property
    def pct_used(self) -> float:
        return self._used / self.max_tokens

    @property
    def is_exhausted(self) -> bool:
        return self._used >= self.max_tokens

    def add(self, tokens: int) -> bool:
        """Add `tokens` to the used count.

        Returns True if still within budget, False (or raises) if exceeded.

        Raises:
            ValueError: if `tokens` is negative. Negative additions would
                silently corrupt the running total (used could go below zero,
                making ``remaining_tokens`` exceed ``max_tokens``), so they are
                rejected outright.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        self._used += tokens
        self._check_warnings()
        if self._used > self.max_tokens:
            if self.stop_on_limit:
                raise ContextBudgetError(
                    f"Context budget exhausted: {self._used}/{self.max_tokens} tokens used"
                )
            return False
        return True

    def add_text(self, text: str) -> bool:
        """Estimate tokens for text and add to used count."""
        return self.add(estimate_tokens(text))

    def add_message(self, msg: dict[str, Any]) -> bool:
        """Estimate tokens for a single message dict and add to used count."""
        return self.add(_message_tokens(msg))

    def add_messages(self, messages: list[dict[str, Any]]) -> bool:
        """Estimate tokens for a list of message dicts and add to used count."""
        total = sum(_message_tokens(m) for m in messages)
        return self.add(total)

    def reset(self) -> None:
        """Reset used token count to 0."""
        self._used = 0
        self._warned.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a dict snapshot of current state."""
        return {
            "used_tokens": self._used,
            "max_tokens": self.max_tokens,
            "remaining_tokens": self.remaining_tokens,
            "pct_used": round(self.pct_used, 4),
            "is_exhausted": self.is_exhausted,
        }

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _check_warnings(self) -> None:
        for threshold in self._thresholds:
            if self._used >= threshold and threshold not in self._warned:
                self._warned.add(threshold)
                pct = threshold / self.max_tokens
                w = BudgetWarning(
                    used_tokens=self._used,
                    max_tokens=self.max_tokens,
                    remaining_tokens=self.remaining_tokens,
                    pct_used=self.pct_used,
                    message=(
                        f"Context budget {pct:.0%} used: "
                        f"{self._used}/{self.max_tokens} tokens "
                        f"({self.remaining_tokens} remaining)"
                    ),
                )
                if self.on_warn:
                    self.on_warn(w)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_context_budget(
    max_tokens: int,
    *,
    warn_at: list[float | int] | None = None,
    on_warn: Callable[[BudgetWarning], None] | None = None,
    stop_on_limit: bool = True,
) -> ContextBudget:
    """Factory for ContextBudget with optional defaults."""
    return ContextBudget(
        max_tokens=max_tokens,
        warn_at=warn_at if warn_at is not None else [0.7, 0.9],
        on_warn=on_warn,
        stop_on_limit=stop_on_limit,
    )
