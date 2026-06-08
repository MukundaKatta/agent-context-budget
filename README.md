# agent-context-budget

Track estimated token usage across a conversation and warn before the context limit hits.

Zero dependencies. Python 3.10+. MIT.

Long-running agent loops drift toward the model's context window one message at
a time. `agent-context-budget` keeps a running estimate of how many tokens
you've spent, fires callbacks at the thresholds you choose (e.g. 70% and 90%),
and either raises or returns `False` once the budget is exhausted — so you can
trim or summarize history *before* the provider rejects the request.

## Install

```bash
pip install agent-context-budget
```

## Usage

```python
from agent_context_budget import make_context_budget, ContextBudgetError
import logging

budget = make_context_budget(
    max_tokens=8000,
    warn_at=[0.7, 0.9],
    on_warn=lambda w: logging.warning(w.message),
)

for turn in agent_loop():
    budget.add_messages(messages)   # raises ContextBudgetError at 8001 tokens
    response = call_llm(messages)
    messages.append({"role": "assistant", "content": response.text})
```

## Add tokens different ways

```python
budget.add(500)                          # raw token count
budget.add_text("some string")           # estimates tokens for text
budget.add_message({"role": "user", "content": "hello"})  # one message
budget.add_messages(messages)            # whole message list
```

## Token estimation

`chars/4 + 4 per message` — the rough heuristic documented by major providers for
planning purposes. Slightly over-estimates so trimming never cuts too close. No
tokenizer dependency.

## Snapshot for logging

```python
snap = budget.snapshot()
# {"used_tokens": 5600, "max_tokens": 8000, "remaining_tokens": 2400,
#  "pct_used": 0.7, "is_exhausted": False}
```

## stop_on_limit=False

```python
budget = make_context_budget(8000, stop_on_limit=False)
while True:
    if not budget.add_messages(messages):
        # trim and retry
        messages = trim(messages)
        budget.reset()
```

## API

### `make_context_budget(max_tokens, *, warn_at=None, on_warn=None, stop_on_limit=True)`

Factory with default warn_at=[0.7, 0.9].

`max_tokens` must be `>= 1` (otherwise `ValueError`). Each `add*` call must
contribute a non-negative number of tokens; passing a negative count to `add()`
raises `ValueError` rather than silently corrupting the running total.

### `ContextBudget`

```python
@dataclass
class ContextBudget:
    max_tokens: int
    warn_at: list[float | int]
    on_warn: Callable[[BudgetWarning], None] | None
    stop_on_limit: bool

    @property
    def used_tokens(self) -> int: ...
    @property
    def remaining_tokens(self) -> int: ...
    @property
    def pct_used(self) -> float: ...
    @property
    def is_exhausted(self) -> bool: ...

    def add(self, tokens: int) -> bool: ...
    def add_text(self, text: str) -> bool: ...
    def add_message(self, msg: dict) -> bool: ...
    def add_messages(self, messages: list[dict]) -> bool: ...
    def reset(self) -> None: ...
    def snapshot(self) -> dict: ...
```

### `BudgetWarning`

```python
@dataclass
class BudgetWarning:
    used_tokens: int
    max_tokens: int
    remaining_tokens: int
    pct_used: float
    message: str
```

### `estimate_tokens(text) -> int`

Rough token estimate for a string.

## Development

The library has no runtime dependencies, and the test suite uses only the
Python standard library (`unittest`) — no test runner to install:

```bash
python -m unittest discover -s tests
```

CI runs the same command across Python 3.10–3.13 (see
`.github/workflows/ci.yml`).

## License

MIT
