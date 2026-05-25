"""agent-context-budget: track estimated token usage and warn before the limit.

Public API:
    ContextBudget(max_tokens, warn_at, on_warn, stop_on_limit)
    BudgetWarning   — emitted at warning thresholds
    ContextBudgetError — raised when budget exhausted
    estimate_tokens(text) -> int
    make_context_budget(...) -> ContextBudget
"""

from .core import (
    BudgetWarning,
    ContextBudget,
    ContextBudgetError,
    estimate_tokens,
    make_context_budget,
)

__all__ = [
    "ContextBudget",
    "BudgetWarning",
    "ContextBudgetError",
    "estimate_tokens",
    "make_context_budget",
]
__version__ = "0.1.0"
