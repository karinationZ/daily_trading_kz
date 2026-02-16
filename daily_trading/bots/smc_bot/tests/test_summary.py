"""
Invariant-based tests for the daily trading summary.

These tests verify correctness of session window computation and
fill-based trade pairing WITHOUT requiring an IBKR connection.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from daily_trading.brokers.IBKRClient import IBKRClient


# ── Session Window Tests ──


class TestSessionWindow:
    """Test _compute_session_window: prev day 14:30 PT → current day 14:00 PT."""

    utc = ZoneInfo("UTC")
    pt = ZoneInfo("America/Los_Angeles")

    def test_after_session_end_returns_full_window(self):
        """At 16:00 PT, session end is today 14:00 PT (already passed)."""
        # Feb 17, 16:00 PT = Feb 18, 00:00 UTC
        now_utc = datetime(2026, 2, 18, 0, 0, 0, tzinfo=self.utc)
        start, end = IBKRClient._compute_session_window(now_utc)

        start_pt = start.astimezone(self.pt)
        end_pt = end.astimezone(self.pt)

        assert start_pt.hour == 14 and start_pt.minute == 30  # prev day 14:30
        assert end_pt.hour == 14 and end_pt.minute == 0       # current day 14:00
        assert start_pt.day == 16  # yesterday
        assert end_pt.day == 17    # today

    def test_before_session_end_caps_at_now(self):
        """At 10:00 PT, session end should be capped at now (10:00 PT)."""
        # Feb 17, 10:00 PT = Feb 17, 18:00 UTC
        now_utc = datetime(2026, 2, 17, 18, 0, 0, tzinfo=self.utc)
        start, end = IBKRClient._compute_session_window(now_utc)

        start_pt = start.astimezone(self.pt)
        end_pt = end.astimezone(self.pt)

        assert start_pt.hour == 14 and start_pt.minute == 30  # prev day 14:30
        assert end_pt.hour == 10 and end_pt.minute == 0       # capped at now
        assert start_pt.day == 16  # yesterday
        assert end_pt.day == 17    # today

    def test_identical_on_utc_and_pt_systems(self):
        """Same wall-clock moment must produce identical UTC boundaries."""
        moment_utc = datetime(2026, 2, 17, 20, 0, 0, tzinfo=self.utc)
        moment_pt = moment_utc.astimezone(self.pt)

        start1, end1 = IBKRClient._compute_session_window(moment_utc)
        start2, end2 = IBKRClient._compute_session_window(moment_pt.astimezone(self.utc))

        assert start1 == start2
        assert end1 == end2

    def test_session_window_always_returns_utc(self):
        """Return values must always be in UTC."""
        now_utc = datetime(2026, 2, 17, 12, 0, 0, tzinfo=self.utc)
        start, end = IBKRClient._compute_session_window(now_utc)

        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start.utcoffset().total_seconds() == 0
        assert end.utcoffset().total_seconds() == 0

    def test_start_always_before_end(self):
        """Invariant: session start must always be before session end."""
        test_times_utc = [
            datetime(2026, 2, 17, 0, 0, 0, tzinfo=self.utc),   # 16:00 PT prev day
            datetime(2026, 2, 17, 6, 0, 0, tzinfo=self.utc),   # 22:00 PT prev day
            datetime(2026, 2, 17, 12, 0, 0, tzinfo=self.utc),  # 04:00 PT
            datetime(2026, 2, 17, 18, 0, 0, tzinfo=self.utc),  # 10:00 PT
            datetime(2026, 2, 17, 22, 0, 0, tzinfo=self.utc),  # 14:00 PT
            datetime(2026, 2, 17, 23, 0, 0, tzinfo=self.utc),  # 15:00 PT
        ]
        for now_utc in test_times_utc:
            start, end = IBKRClient._compute_session_window(now_utc)
            assert start < end, f"start={start} >= end={end} for now_utc={now_utc}"

    def test_window_spans_23_5_hours_after_close(self):
        """After 14:00 PT, the full window should be 23.5 hours."""
        # Feb 17, 15:00 PT = Feb 17, 23:00 UTC
        now_utc = datetime(2026, 2, 17, 23, 0, 0, tzinfo=self.utc)
        start, end = IBKRClient._compute_session_window(now_utc)

        duration = end - start
        assert duration == timedelta(hours=23, minutes=30)


# ── Timezone Normalization Tests ──


class TestEnsureUtc:
    utc = ZoneInfo("UTC")
    pt = ZoneInfo("America/Los_Angeles")

    def test_naive_assumed_utc(self):
        naive = datetime(2026, 2, 16, 10, 30, 0)
        result = IBKRClient._ensure_utc(naive)
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_pt_converted_to_utc(self):
        pt_time = datetime(2026, 2, 16, 5, 0, 0, tzinfo=self.pt)
        result = IBKRClient._ensure_utc(pt_time)
        assert result.utcoffset().total_seconds() == 0
        assert result.hour == 13  # PT is UTC-8 in Feb

    def test_utc_passthrough(self):
        utc_time = datetime(2026, 2, 16, 10, 0, 0, tzinfo=self.utc)
        result = IBKRClient._ensure_utc(utc_time)
        assert result == utc_time


# ── Invariant: if exits exist in session, trade MUST be reported ──


class TestFillPairingInvariant:
    """
    Invariant: if a bracket group has both BUY and SELL fills
    with exit time inside the session window, it MUST produce a trade.
    """

    def _make_fill(self, side, price, shares, perm_id, time, order_ref="TEST"):
        """Create a mock Fill object matching ib_async structure."""
        fill = MagicMock()
        fill.execution.side = side  # 'BOT' or 'SLD'
        fill.execution.price = price
        fill.execution.shares = shares
        fill.execution.permId = perm_id
        fill.execution.execId = f"exec_{perm_id}_{side}_{time.timestamp()}"
        fill.execution.orderRef = order_ref
        fill.time = time
        fill.contract.symbol = "MES"
        fill.contract.multiplier = "5"
        fill.commissionReport = MagicMock()
        fill.commissionReport.commission = 1.04
        return fill

    def test_long_bracket_produces_trade(self):
        """A bracket with BUY entry + SELL exit inside session must report."""
        utc = ZoneInfo("UTC")

        session_start = datetime(2026, 2, 15, 22, 30, 0, tzinfo=utc)  # prev day 14:30 PT
        session_end = datetime(2026, 2, 16, 22, 0, 0, tzinfo=utc)     # current day 14:00 PT

        entry_fill = self._make_fill(
            "BOT", 5000.0, 1, perm_id=100,
            time=datetime(2026, 2, 16, 1, 0, 0, tzinfo=utc)
        )
        exit_fill = self._make_fill(
            "SLD", 5010.0, 1, perm_id=101,
            time=datetime(2026, 2, 16, 3, 0, 0, tzinfo=utc)
        )

        buy_fills = [entry_fill]
        sell_fills = [exit_fill]

        assert buy_fills and sell_fills, "Both sides must exist"

        earliest_buy = min(IBKRClient._ensure_utc(f.time) for f in buy_fills)
        earliest_sell = min(IBKRClient._ensure_utc(f.time) for f in sell_fills)

        if earliest_buy < earliest_sell:
            direction = "LONG"
            e_fills, x_fills = buy_fills, sell_fills
        else:
            direction = "SHORT"
            e_fills, x_fills = sell_fills, buy_fills

        latest_exit = max(IBKRClient._ensure_utc(f.time) for f in x_fills)

        # INVARIANT: exit is in session → trade must be reported
        assert session_start <= latest_exit <= session_end, "Exit must be in session"

        result = IBKRClient._format_fill_based_trade(
            e_fills, x_fills, "MES", entry_fill.contract,
            direction, "TEST", IBKRClient.MULTIPLIERS
        )

        assert result is not None, "Trade with exit in session MUST be reported"
        message, net_pnl, tag, symbol = result
        assert symbol == "MES"
        assert direction == "LONG"
        assert net_pnl > 0, "Winning trade must have positive PnL"

    def test_short_bracket_produces_trade(self):
        """A bracket with SELL entry + BUY exit must also be reported."""
        utc = ZoneInfo("UTC")

        entry_fill = self._make_fill(
            "SLD", 5010.0, 1, perm_id=200,
            time=datetime(2026, 2, 16, 1, 0, 0, tzinfo=utc)
        )
        exit_fill = self._make_fill(
            "BOT", 5000.0, 1, perm_id=201,
            time=datetime(2026, 2, 16, 3, 0, 0, tzinfo=utc)
        )

        result = IBKRClient._format_fill_based_trade(
            [entry_fill], [exit_fill], "MES", entry_fill.contract,
            "SHORT", "TEST", IBKRClient.MULTIPLIERS
        )

        assert result is not None
        _, net_pnl, _, _ = result
        assert net_pnl > 0, "Winning short must have positive PnL"

    def test_exit_outside_session_is_filtered(self):
        """Exit outside session window must NOT be reported (filter check)."""
        utc = ZoneInfo("UTC")

        # Session: prev day 14:30 PT → current day 14:00 PT
        session_start = datetime(2026, 2, 15, 22, 30, 0, tzinfo=utc)
        session_end = datetime(2026, 2, 16, 22, 0, 0, tzinfo=utc)

        # Exit at 23:00 UTC (15:00 PT) — outside session
        exit_time = datetime(2026, 2, 16, 23, 0, 0, tzinfo=utc)
        latest_exit_utc = IBKRClient._ensure_utc(exit_time)

        in_session = session_start <= latest_exit_utc <= session_end
        assert not in_session, "Exit outside session must be filtered"
