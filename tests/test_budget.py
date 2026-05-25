"""Tests for agent-context-budget."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from agent_context_budget import (
    BudgetWarning,
    ContextBudget,
    ContextBudgetError,
    estimate_tokens,
    make_context_budget,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_short_string():
    # "hi" = 2 chars → max(1, (2+3)//4) = max(1,1) = 1
    assert estimate_tokens("hi") == 1


def test_estimate_four_chars():
    # "abcd" → max(1, (4+3)//4) = max(1,1) = 1
    assert estimate_tokens("abcd") == 1


def test_estimate_eight_chars():
    # 8 chars → max(1, (8+3)//4) = max(1,2) = 2
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_scales_with_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert long > short


# ---------------------------------------------------------------------------
# ContextBudget — basic
# ---------------------------------------------------------------------------

def test_initial_used_zero():
    b = ContextBudget(max_tokens=1000)
    assert b.used_tokens == 0


def test_initial_remaining_full():
    b = ContextBudget(max_tokens=1000)
    assert b.remaining_tokens == 1000


def test_add_increases_used():
    b = ContextBudget(max_tokens=1000)
    b.add(100)
    assert b.used_tokens == 100


def test_add_decreases_remaining():
    b = ContextBudget(max_tokens=1000)
    b.add(300)
    assert b.remaining_tokens == 700


def test_add_returns_true_below_limit():
    b = ContextBudget(max_tokens=1000)
    assert b.add(500) is True


def test_add_raises_at_limit():
    b = ContextBudget(max_tokens=100)
    with pytest.raises(ContextBudgetError):
        b.add(101)


def test_add_stop_on_limit_false_returns_false():
    b = ContextBudget(max_tokens=100, stop_on_limit=False)
    b.add(100)
    result = b.add(1)
    assert result is False


def test_pct_used():
    b = ContextBudget(max_tokens=100)
    b.add(50)
    assert b.pct_used == pytest.approx(0.5)


def test_is_exhausted_false_below():
    b = ContextBudget(max_tokens=100)
    b.add(99)
    assert b.is_exhausted is False


def test_is_exhausted_true_at_limit():
    b = ContextBudget(max_tokens=100, stop_on_limit=False)
    b.add(100)
    assert b.is_exhausted is True


# ---------------------------------------------------------------------------
# add_text / add_message / add_messages
# ---------------------------------------------------------------------------

def test_add_text():
    b = ContextBudget(max_tokens=10000)
    b.add_text("Hello, world!")
    assert b.used_tokens > 0


def test_add_message_dict():
    b = ContextBudget(max_tokens=10000)
    msg = {"role": "user", "content": "Hello, world!"}
    b.add_message(msg)
    assert b.used_tokens > 0


def test_add_messages_list():
    b = ContextBudget(max_tokens=10000)
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
    ]
    b.add_messages(msgs)
    assert b.used_tokens > 0


def test_add_messages_content_blocks():
    b = ContextBudget(max_tokens=10000)
    msgs = [{"role": "system", "content": [{"type": "text", "text": "Be helpful."}]}]
    b.add_messages(msgs)
    assert b.used_tokens > 0


def test_add_multiple_messages_more_tokens():
    b1 = ContextBudget(max_tokens=10000)
    b2 = ContextBudget(max_tokens=10000)
    b1.add_message({"role": "user", "content": "Hi"})
    b2.add_messages([
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there!"},
    ])
    assert b2.used_tokens > b1.used_tokens


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_warning_triggered_at_threshold():
    warnings = []
    b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
    b.add(70)
    assert len(warnings) == 1


def test_warning_not_triggered_below():
    warnings = []
    b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
    b.add(69)
    assert warnings == []


def test_warning_fires_once():
    count = [0]
    b = ContextBudget(max_tokens=100, warn_at=[0.5], on_warn=lambda _: count.__setitem__(0, count[0]+1))
    b.add(50)
    b.add(1)
    assert count[0] == 1


def test_multiple_warnings():
    warnings = []
    b = ContextBudget(max_tokens=100, warn_at=[0.5, 0.9], on_warn=warnings.append)
    b.add(50)   # fires 50% warning → warnings[0]
    b.add(40)   # fires 90% warning → warnings[1]
    assert len(warnings) == 2
    assert warnings[0].pct_used == pytest.approx(0.5, abs=0.05)
    assert warnings[1].pct_used == pytest.approx(0.9, abs=0.05)


def test_warning_message_content():
    warnings = []
    b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
    b.add(70)
    assert "70%" in warnings[0].message or "70" in warnings[0].message


def test_warning_remaining_correct():
    warnings = []
    b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
    b.add(70)
    assert warnings[0].remaining_tokens == 30


def test_default_warnings_at_70_90():
    warns = []
    b = ContextBudget(max_tokens=100, on_warn=warns.append)
    b.add(70)
    assert len(warns) == 1
    b.add(20)
    assert len(warns) == 2


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_used():
    b = ContextBudget(max_tokens=1000)
    b.add(500)
    b.reset()
    assert b.used_tokens == 0


def test_reset_allows_warnings_to_fire_again():
    warns = []
    b = ContextBudget(max_tokens=100, warn_at=[0.5], on_warn=warns.append)
    b.add(50)
    b.reset()
    warns.clear()
    b.add(50)
    assert len(warns) == 1


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def test_snapshot_keys():
    b = ContextBudget(max_tokens=1000)
    b.add(200)
    snap = b.snapshot()
    assert snap["used_tokens"] == 200
    assert snap["max_tokens"] == 1000
    assert snap["remaining_tokens"] == 800
    assert snap["pct_used"] == pytest.approx(0.2)
    assert snap["is_exhausted"] is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_max_tokens_zero_raises():
    with pytest.raises(ValueError):
        ContextBudget(max_tokens=0)


def test_max_tokens_negative_raises():
    with pytest.raises(ValueError):
        ContextBudget(max_tokens=-1)


# ---------------------------------------------------------------------------
# make_context_budget factory
# ---------------------------------------------------------------------------

def test_factory_creates_budget():
    b = make_context_budget(4096)
    assert b.max_tokens == 4096


def test_factory_custom_warn():
    b = make_context_budget(1000, warn_at=[0.5])
    # 0.5 * 1000 = 500
    warns = []
    b.on_warn = warns.append
    b.add(500)
    assert len(warns) == 1
