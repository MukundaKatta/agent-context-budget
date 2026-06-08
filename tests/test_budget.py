"""Tests for agent-context-budget.

Uses the Python standard-library ``unittest`` framework only (no third-party
test dependencies). Run with::

    python3 -m unittest discover -s tests
"""

import os
import sys
import unittest

# Make the package importable when running straight from a checkout, without
# requiring an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_context_budget import (  # noqa: E402
    BudgetWarning,
    ContextBudget,
    ContextBudgetError,
    estimate_tokens,
    make_context_budget,
)


class EstimateTokensTests(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_short_string(self):
        # "hi" = 2 chars -> max(1, (2+3)//4) = max(1, 1) = 1
        self.assertEqual(estimate_tokens("hi"), 1)

    def test_four_chars(self):
        # "abcd" -> max(1, (4+3)//4) = max(1, 1) = 1
        self.assertEqual(estimate_tokens("abcd"), 1)

    def test_eight_chars(self):
        # 8 chars -> max(1, (8+3)//4) = max(1, 2) = 2
        self.assertEqual(estimate_tokens("abcdefgh"), 2)

    def test_scales_with_length(self):
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 100)
        self.assertGreater(long, short)

    def test_never_negative(self):
        self.assertGreaterEqual(estimate_tokens("x"), 0)


class ContextBudgetBasicTests(unittest.TestCase):
    def test_initial_used_zero(self):
        b = ContextBudget(max_tokens=1000)
        self.assertEqual(b.used_tokens, 0)

    def test_initial_remaining_full(self):
        b = ContextBudget(max_tokens=1000)
        self.assertEqual(b.remaining_tokens, 1000)

    def test_add_increases_used(self):
        b = ContextBudget(max_tokens=1000)
        b.add(100)
        self.assertEqual(b.used_tokens, 100)

    def test_add_decreases_remaining(self):
        b = ContextBudget(max_tokens=1000)
        b.add(300)
        self.assertEqual(b.remaining_tokens, 700)

    def test_add_returns_true_below_limit(self):
        b = ContextBudget(max_tokens=1000)
        self.assertIs(b.add(500), True)

    def test_add_zero_is_allowed(self):
        b = ContextBudget(max_tokens=1000)
        self.assertIs(b.add(0), True)
        self.assertEqual(b.used_tokens, 0)

    def test_add_raises_at_limit(self):
        b = ContextBudget(max_tokens=100)
        with self.assertRaises(ContextBudgetError):
            b.add(101)

    def test_add_exactly_at_limit_does_not_raise(self):
        b = ContextBudget(max_tokens=100)
        self.assertIs(b.add(100), True)
        self.assertTrue(b.is_exhausted)

    def test_add_stop_on_limit_false_returns_false(self):
        b = ContextBudget(max_tokens=100, stop_on_limit=False)
        b.add(100)
        self.assertIs(b.add(1), False)

    def test_pct_used(self):
        b = ContextBudget(max_tokens=100)
        b.add(50)
        self.assertAlmostEqual(b.pct_used, 0.5)

    def test_is_exhausted_false_below(self):
        b = ContextBudget(max_tokens=100)
        b.add(99)
        self.assertFalse(b.is_exhausted)

    def test_is_exhausted_true_at_limit(self):
        b = ContextBudget(max_tokens=100, stop_on_limit=False)
        b.add(100)
        self.assertTrue(b.is_exhausted)

    def test_remaining_never_negative(self):
        b = ContextBudget(max_tokens=100, stop_on_limit=False)
        b.add(150)
        self.assertEqual(b.remaining_tokens, 0)


class NegativeAddTests(unittest.TestCase):
    """Regression tests: negative additions would silently corrupt the total."""

    def test_add_negative_raises(self):
        b = ContextBudget(max_tokens=100)
        with self.assertRaises(ValueError):
            b.add(-1)

    def test_add_negative_does_not_mutate_used(self):
        b = ContextBudget(max_tokens=100)
        b.add(40)
        with self.assertRaises(ValueError):
            b.add(-10)
        # State must be unchanged after the rejected call.
        self.assertEqual(b.used_tokens, 40)


class AddTextMessageTests(unittest.TestCase):
    def test_add_text(self):
        b = ContextBudget(max_tokens=10000)
        b.add_text("Hello, world!")
        self.assertGreater(b.used_tokens, 0)

    def test_add_message_dict(self):
        b = ContextBudget(max_tokens=10000)
        b.add_message({"role": "user", "content": "Hello, world!"})
        self.assertGreater(b.used_tokens, 0)

    def test_add_messages_list(self):
        b = ContextBudget(max_tokens=10000)
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        b.add_messages(msgs)
        self.assertGreater(b.used_tokens, 0)

    def test_add_messages_empty_list(self):
        b = ContextBudget(max_tokens=10000)
        self.assertIs(b.add_messages([]), True)
        self.assertEqual(b.used_tokens, 0)

    def test_add_messages_content_blocks(self):
        b = ContextBudget(max_tokens=10000)
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "Be helpful."}]}
        ]
        b.add_messages(msgs)
        self.assertGreater(b.used_tokens, 0)

    def test_message_with_none_content(self):
        # A tool-call message may have content=None; it should not crash.
        b = ContextBudget(max_tokens=10000)
        b.add_message({"role": "assistant", "content": None})
        # Only the per-message overhead is counted.
        self.assertEqual(b.used_tokens, 4)

    def test_add_multiple_messages_more_tokens(self):
        b1 = ContextBudget(max_tokens=10000)
        b2 = ContextBudget(max_tokens=10000)
        b1.add_message({"role": "user", "content": "Hi"})
        b2.add_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
            ]
        )
        self.assertGreater(b2.used_tokens, b1.used_tokens)


class WarningTests(unittest.TestCase):
    def test_warning_triggered_at_threshold(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
        b.add(70)
        self.assertEqual(len(warnings), 1)

    def test_warning_payload_is_budget_warning(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
        b.add(70)
        self.assertIsInstance(warnings[0], BudgetWarning)

    def test_warning_not_triggered_below(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
        b.add(69)
        self.assertEqual(warnings, [])

    def test_warning_fires_once(self):
        count = [0]
        b = ContextBudget(
            max_tokens=100,
            warn_at=[0.5],
            on_warn=lambda _: count.__setitem__(0, count[0] + 1),
        )
        b.add(50)
        b.add(1)
        self.assertEqual(count[0], 1)

    def test_multiple_warnings(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.5, 0.9], on_warn=warnings.append)
        b.add(50)  # 50% warning
        b.add(40)  # 90% warning
        self.assertEqual(len(warnings), 2)
        self.assertAlmostEqual(warnings[0].pct_used, 0.5, delta=0.05)
        self.assertAlmostEqual(warnings[1].pct_used, 0.9, delta=0.05)

    def test_absolute_int_threshold(self):
        warnings = []
        b = ContextBudget(max_tokens=1000, warn_at=[500], on_warn=warnings.append)
        b.add(499)
        self.assertEqual(warnings, [])
        b.add(1)
        self.assertEqual(len(warnings), 1)

    def test_warning_message_content(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
        b.add(70)
        self.assertIn("70", warnings[0].message)

    def test_warning_remaining_correct(self):
        warnings = []
        b = ContextBudget(max_tokens=100, warn_at=[0.7], on_warn=warnings.append)
        b.add(70)
        self.assertEqual(warnings[0].remaining_tokens, 30)

    def test_default_warnings_at_70_90(self):
        warns = []
        b = ContextBudget(max_tokens=100, on_warn=warns.append)
        b.add(70)
        self.assertEqual(len(warns), 1)
        b.add(20)
        self.assertEqual(len(warns), 2)

    def test_no_on_warn_callback_is_safe(self):
        # Crossing a threshold with no callback registered must not raise.
        b = ContextBudget(max_tokens=100, warn_at=[0.5])
        self.assertIs(b.add(60), True)


class ResetTests(unittest.TestCase):
    def test_reset_clears_used(self):
        b = ContextBudget(max_tokens=1000)
        b.add(500)
        b.reset()
        self.assertEqual(b.used_tokens, 0)

    def test_reset_allows_warnings_to_fire_again(self):
        warns = []
        b = ContextBudget(max_tokens=100, warn_at=[0.5], on_warn=warns.append)
        b.add(50)
        b.reset()
        warns.clear()
        b.add(50)
        self.assertEqual(len(warns), 1)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_keys_and_values(self):
        b = ContextBudget(max_tokens=1000)
        b.add(200)
        snap = b.snapshot()
        self.assertEqual(snap["used_tokens"], 200)
        self.assertEqual(snap["max_tokens"], 1000)
        self.assertEqual(snap["remaining_tokens"], 800)
        self.assertAlmostEqual(snap["pct_used"], 0.2)
        self.assertIs(snap["is_exhausted"], False)


class ValidationTests(unittest.TestCase):
    def test_max_tokens_zero_raises(self):
        with self.assertRaises(ValueError):
            ContextBudget(max_tokens=0)

    def test_max_tokens_negative_raises(self):
        with self.assertRaises(ValueError):
            ContextBudget(max_tokens=-1)


class FactoryTests(unittest.TestCase):
    def test_factory_creates_budget(self):
        b = make_context_budget(4096)
        self.assertEqual(b.max_tokens, 4096)

    def test_factory_returns_context_budget(self):
        self.assertIsInstance(make_context_budget(4096), ContextBudget)

    def test_factory_default_warns_at_70_90(self):
        warns = []
        b = make_context_budget(100, on_warn=warns.append)
        b.add(70)
        b.add(20)
        self.assertEqual(len(warns), 2)

    def test_factory_custom_warn(self):
        warns = []
        b = make_context_budget(1000, warn_at=[0.5], on_warn=warns.append)
        b.add(500)  # 0.5 * 1000
        self.assertEqual(len(warns), 1)

    def test_factory_stop_on_limit_false(self):
        b = make_context_budget(100, stop_on_limit=False)
        b.add(100)
        self.assertIs(b.add(1), False)


if __name__ == "__main__":
    unittest.main()
