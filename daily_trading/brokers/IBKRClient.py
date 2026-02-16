import asyncio
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, ClassVar, List
from zoneinfo import ZoneInfo

import pandas as pd
from ib_async import (
    Contract,
    Future,
    IB,
    LimitOrder,
    MarketOrder,
    util,
)

from daily_trading.settings import logger


class IBKRClient:
    """
    Refactored IBKR Client using ib_async (TWS/Gateway API).
    Standardizes futures trading, tick rounding, and bracket execution.
    """

    # TICK SIZES: Conform to exchange variations
    TICK_SIZES = {
        "MNQ": 0.25,  # Micro E-mini Nasdaq-100
        "MES": 0.25,  # Micro E-mini S&P 500
        "MGC": 0.10,  # Micro Gold
        "MBT": 5.00,  # Micro Bitcoin
        "MCL": 0.01,  # Micro WTI Crude Oil
        "SIL": 0.005,  # Micro Silver (CME)
    }

    MULTIPLIERS = {
        "MNQ": 2,
        "MES": 5,
        "MGC": 10,
        "MBT": 0.1,
        "MCL": 100,
        "SIL": 1000,
        "NQ": 20,
        "ES": 50,
        "GC": 100,
        "CL": 1000,
        "RTY": 50,
        "M2K": 5,
    }

    # Map intervals to IB durations and bar sizes
    # Format: (BarSize, Duration)
    TIMEFRAME_CONFIG = {
        "1m": ("1 min", "2 D"),
        "5m": ("5 mins", "5 D"),
        "15m": ("15 mins", "2 W"),
        "1h": ("1 hour", "3 M"),
        "1d": ("1 day", "1 Y"),
    }
    _order_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self._conid_cache = {}

    async def connect(self):
        """Persistent connection to TWS/Gateway."""
        if not self.ib.isConnected():
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            # CRITICAL: Tell IBKR you want LIVE data (Type 1)
            self.ib.reqMarketDataType(1)
            logger.info("Connected to IBKR and set Market Data Type to LIVE")

    def disconnect(self):
        self.ib.disconnect()

    def round_to_tick(self, price: float, symbol: str) -> float:
        """Adjusts price to conform to the instrument's tick size."""
        tick = self.TICK_SIZES.get(symbol, 0.01)
        return round(math.floor(price / tick + 0.5) * tick, 4)

    def get_active_positions(self) -> List[Any]:
        """Returns currently open positions."""
        return self.ib.positions()

    def get_open_orders(self) -> List[Any]:
        """
        Returns a list of active 'Trade' objects (wrapping Order + Contract).
        We return Trades (instead of raw Orders) so the MasterBot
        can access 'trade.contract.symbol'.
        """
        return self.ib.openTrades()

    async def get_futures_contract(self, symbol: str) -> Contract | None:
        """Resolves the front-month futures contract with AGGRESSIVE Physical Delivery Safety."""
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]

        contract_map = {
            # --- CASH SETTLED (Indices) ---
            "MNQ": {"symbol": "MNQ", "exchange": "CME", "currency": "USD", "active_months": [3, 6, 9, 12], "min_days": 8, "physical_delivery": False},
            "MES": {"symbol": "MES", "exchange": "CME", "currency": "USD", "active_months": [3, 6, 9, 12], "min_days": 8, "physical_delivery": False},
            "MBT": {"symbol": "MBT", "exchange": "CME", "currency": "USD", "active_months": range(1, 13), "min_days": 2, "physical_delivery": False},
            # --- PHYSICAL DELIVERY (Commodities) ---
            # FIX: First Notice Day is often the last day of the PREVIOUS month.
            # Using Feb 1st - 3 days = Jan 29th is TOO LATE.
            # We use Feb 1st - 8 days = Jan 24th. This ensures we are out well before FND.
            "MGC": {
                "symbol": "MGC",
                "exchange": "COMEX",
                "currency": "USD",
                "active_months": [2, 4, 6, 8, 12],
                "min_days": 8,
                "physical_delivery": True,
            },
            "MCL": {"symbol": "MCL", "exchange": "NYMEX", "currency": "USD", "active_months": range(1, 13), "min_days": 8, "physical_delivery": True},
        }

        target = contract_map.get(symbol)
        if not target:
            raise ValueError(f"Unknown symbol: {symbol}")

        contract = Future(
            symbol=target["symbol"],
            exchange=target["exchange"],
            currency=target["currency"],
            multiplier=target.get("multiplier", ""),
        )

        details = await self.ib.reqContractDetailsAsync(contract)
        if not details:
            raise RuntimeError(f"Could not resolve contract for {symbol}")

        today = datetime.now()
        valid_contracts = []

        target_mult = target.get("multiplier")
        active_months = target.get("active_months")
        min_days = target.get("min_days", 7)
        is_physical = target.get("physical_delivery", False)

        for d in details:
            if target_mult and d.contract.multiplier != target_mult:
                continue

            # Parse the Expiration String (YYYYMMDD)
            exp_str = d.contract.lastTradeDateOrContractMonth
            exp_date = datetime.strptime(exp_str, "%Y%m%d")

            if active_months and exp_date.month not in active_months:
                continue

            # --- SAFETY LOGIC ---
            if is_physical:
                # PHYSICAL: Calculate deadline relative to the Start of Delivery Month
                # Example: Feb '26 contract -> Starts Feb 1st.
                # Buffer 8 days: Feb 1 - 8 days = Jan 24th.
                # Since Today (Jan 28) > Jan 24, this contract is INVALID.
                delivery_month_start = exp_date.replace(day=1)
                deadline = delivery_month_start - timedelta(days=min_days)
            else:
                # CASH SETTLED: Relative to actual expiration day
                deadline = exp_date - timedelta(days=min_days)

            # Check if we are safely before the deadline
            if today < deadline:
                valid_contracts.append(d.contract)

        if not valid_contracts:
            # If no contract is found, it usually means our buffer is too aggressive
            # or the next month isn't loaded.
            # In this case, we fallback to finding the contract with the furthest expiration
            # that is still "Frontish" (e.g. within next 4 months) to ensure we get the April contract.
            logger.warning(f"⚠️ No contracts passed strict buffer {min_days}d. Searching for next available month...")

            future_contracts = []
            for d in details:
                exp_date = datetime.strptime(d.contract.lastTradeDateOrContractMonth, "%Y%m%d")
                if exp_date > today + timedelta(days=25):  # Look for contracts at least ~1 month out
                    if active_months and exp_date.month not in active_months:
                        continue
                    future_contracts.append(d.contract)

            if future_contracts:
                valid_contracts = future_contracts

        if not valid_contracts:
            raise RuntimeError(f"No valid liquid front-month contracts found for {symbol} (Physical={is_physical})")

        # Pick the nearest valid contract (which should now be April, since Feb is excluded)
        front_month = min(valid_contracts, key=lambda c: c.lastTradeDateOrContractMonth)

        logger.info(f"✅ Resolved {symbol} -> {front_month.localSymbol} (Exp: {front_month.lastTradeDateOrContractMonth})")

        # Clear cache if we just switched contracts to force update
        self._conid_cache[symbol] = front_month
        return front_month

    async def get_klines(self, symbol: str, interval: str, outside_rth: bool = True) -> pd.DataFrame:
        contract = await self.get_futures_contract(symbol)
        bar_size, duration = self.TIMEFRAME_CONFIG.get(interval, ("5 mins", "2 D"))

        # FIX: Use an empty string for endDateTime to get 'until now'
        # but ensure keepUpToDate=False for historical calls
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",  # Empty string means 'Now'
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=not outside_rth,
            formatDate=1,
            keepUpToDate=False,  # Set this to False for simple DataFrame fetching
        )

        df = util.df(bars)
        if df is not None and not df.empty:
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        return pd.DataFrame()

    def get_active_sl_tp(self, symbol: str, order_ref: str, direction: str) -> tuple[float, float]:
        """
        Retreives the active Stop Loss (auxPrice) and Take Profit (lmtPrice)
        for a specific strategy tag and direction.

        Returns: (sl_price, tp_price). Returns 0.0 if not found.
        """
        # 1. Force a refresh to ensure we see orders placed seconds ago
        open_trades = self.ib.openTrades()

        sl_price = 0.0
        tp_price = 0.0

        # We only look for closing orders (Opposite to position direction)
        required_action = "SELL" if direction == "LONG" else "BUY"

        for t in open_trades:
            # 1. Strict Tag & Symbol Match
            if t.contract.symbol != symbol or t.order.orderRef != order_ref:
                continue

            # 2. Action Match (Must be a closing order)
            if t.order.action != required_action:
                continue

            # 3. Classify based on Order Type
            # STOP LOSS (STP, TRAIL, STP LMT uses auxPrice)
            if t.order.orderType in ['STP', 'TRAIL', 'STP LMT']:
                sl_price = t.order.auxPrice

            # TAKE PROFIT (LMT uses lmtPrice)
            elif t.order.orderType == 'LMT':
                tp_price = t.order.lmtPrice

        return sl_price, tp_price

    async def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        order_ref: str = "TradingBot",
    ):
        contract = await self.get_futures_contract(symbol)
        price = self.round_to_tick(price, symbol)
        order = LimitOrder(side.upper(), quantity, price, orderRef=order_ref, outsideRth=True)
        trade = self.ib.placeOrder(contract, order)
        return trade

    async def place_market_order(self, symbol: str, quantity: int, side: str, order_ref: str = "TradingBot"):
        contract = await self.get_futures_contract(symbol)
        order = MarketOrder(side.upper(), quantity, orderRef=order_ref, outsideRth=True)
        trade = self.ib.placeOrder(contract, order)
        return trade

    async def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        entry: float,
        sl: float,
        tp: float,
        direction: str,
        order_ref: str = "TradingBot",
    ):
        """Place Entry + SL + TP with thread-safe order ID management."""

        # CRITICAL: Lock prevents concurrent order ID conflicts across strategies
        async with self._order_lock:
            contract = await self.get_futures_contract(symbol)

            side = "BUY" if direction.upper() == "LONG" else "SELL"

            entry = self.round_to_tick(entry, symbol)
            sl = self.round_to_tick(sl, symbol)
            tp = self.round_to_tick(tp, symbol)

            # Create bracket - gets 3 sequential order IDs
            bracket = self.ib.bracketOrder(
                action=side,
                quantity=quantity,
                limitPrice=entry,
                takeProfitPrice=tp,
                stopLossPrice=sl,
            )

            # Apply custom settings to all orders
            for order in bracket:
                order.orderRef = order_ref
                order.tif = "GTC"
                order.outsideRth = True

            # Place orders sequentially with delays
            parent_trade = self.ib.placeOrder(contract, bracket.parent)
            await asyncio.sleep(0.2)

            tp_trade = self.ib.placeOrder(contract, bracket.takeProfit)
            await asyncio.sleep(0.2)

            sl_trade = self.ib.placeOrder(contract, bracket.stopLoss)
            await asyncio.sleep(0.3)  # Final delay before releasing lock

            logger.info(f"✅ Bracket: {symbol} {direction} @ {entry} " f"(SL={sl}, TP={tp}) Ref={order_ref}")

            return [parent_trade, tp_trade, sl_trade]

    async def cancel_all_orders(self, order_ref: str):
        """Cancels orders only if they are still active."""
        # Get all trades tracked by the client
        open_trades = self.ib.openTrades()

        for trade in open_trades:
            if order_ref is None or trade.order.orderRef == order_ref:
                # ONLY CANCEL IF NOT ALREADY DONE/CANCELLED
                # trade.isDone() returns True if Filled, Cancelled, or Inactive
                if not trade.isDone():
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"🗑️ Cancelling Order {trade.order.orderId} (Ref: {trade.order.orderRef})")

    async def diagnostic_market_data(self, symbol: str):
        contract = await self.get_futures_contract(symbol)

        # Request a ticker to see current market data type
        ticker = self.ib.reqTickers(contract)[0]
        await asyncio.sleep(1)  # Wait for data to stream in

        logger.info(f"--- DIAGNOSTIC FOR {symbol} ---")
        logger.info(f"Market Data Type (Expected 1): {ticker.marketDataType}")
        logger.info(f"Last Price: {ticker.last}")
        logger.info(f"Bid: {ticker.bid} | Ask: {ticker.ask}")

        if math.isnan(ticker.last):
            logger.error("❌ ERROR: No live price received. IBKR is blocking real-time for this session.")
        elif ticker.marketDataType != 1:
            logger.error(f"❌ ERROR: IBKR forced data type to {ticker.marketDataType} (Delayed).")
        else:
            logger.info("✅ SUCCESS: API has Real-Time access. The delay is in the reqHistoricalData cache.")

    async def close_positions_smartly(self, symbol):
        positions = self.get_active_positions()
        found = False

        for p in positions:
            if p.contract.symbol == symbol and p.position != 0:
                found = True
                size = p.position
                action = "SELL" if size > 0 else "BUY"
                qty = abs(int(size))

                logger.info(f"FOUND: {symbol} | Size: {size}")
                logger.info(f"ACTION: {action} {qty} to Close.")

                await self.place_market_order(
                    symbol=symbol,
                    quantity=qty,
                    side=action,
                    order_ref="FORCE_CLOSE",
                )
                logger.info("Order Placed - FORCE_CLOSE")

        if not found:
            logger.info(f"No open position found for {symbol}. You are flat.")

    async def get_active_bracket_levels(self, symbol: str, order_ref: str, direction: str) -> dict:
        """
        Fetches ACTUAL live StopLoss and TakeProfit prices from IBKR.
        Uses direction to distinguish between Entry Limit and TP Limit orders.
        """
        open_trades = self.ib.openTrades()

        sl_price = 0.0
        tp_price = 0.0
        found = False

        # Determine expected TP Side (Opposite to Strategy Direction)
        # If Strategy is LONG, TP is a SELL order.
        tp_side = "SELL" if direction == "LONG" else "BUY"

        for t in open_trades:
            if t.contract.symbol == symbol and t.order.orderRef == order_ref:

                # 1. STOP LOSS (Order Type 'STP' or 'TRAIL')
                # SL is always on the exit side (same as TP side)
                if t.order.orderType in ["STP", "TRAIL", "STP LMT"]:
                    sl_price = t.order.auxPrice
                    found = True

                # 2. TAKE PROFIT (Order Type 'LMT' on the Exit Side)
                # We check t.order.action to ensure we don't grab the Entry Limit order
                elif t.order.orderType == "LMT" and t.order.action == tp_side:
                    tp_price = t.order.lmtPrice
                    found = True

        return {"sl": sl_price, "tp": tp_price, "found": found}

    async def get_last_fill_price(self, symbol: str) -> float:
        """
        Fetches the price of the most recent execution for a symbol.
        Used to report TRUE exit price (accounting for slippage).
        """
        # Request recent executions
        fills = self.ib.fills()

        # Filter for our symbol
        symbol_execs = [fill for fill in fills if fill.contract.symbol == symbol]

        if not symbol_execs:
            return 0.0

        # Sort by time descending (newest first)
        last_exec = sorted(symbol_execs, key=lambda x: x.time, reverse=True)[0]

        # Return average price of that fill
        return last_exec.execution.avgPrice

    def check_flatness(self, symbol: str) -> (bool, float):
        """
        NON-BLOCKING: Uses the internal ib_async cache.
        Returns (True, 0.0) if flat.
        """
        # Use .positions() (the cache) instead of .reqPositionsAsync() (the network)
        positions = self.ib.positions()

        for p in positions:
            if p.contract.symbol == symbol and p.position != 0:
                return False, p.position

        return True, 0.0

    # ************** REPORTING LOGIC **************

    @staticmethod
    def _compute_session_window(now_utc: datetime) -> tuple[datetime, datetime]:
        """
        Compute the trading session window: previous day 14:30 PT → current day 14:00 PT.

        All math is done in PT then returned as UTC.
        This is timezone-safe and deterministic regardless of system timezone.
        """
        pt = ZoneInfo("America/Los_Angeles")
        now_pt = now_utc.astimezone(pt)

        # Session end: today at 14:00 PT
        session_end = now_pt.replace(hour=14, minute=0, second=0, microsecond=0)

        # Session start: yesterday at 14:30 PT
        yesterday_pt = now_pt - timedelta(days=1)
        session_start = yesterday_pt.replace(hour=14, minute=30, second=0, microsecond=0)

        # If we haven't reached today's 14:00 yet, use up to now
        if now_pt < session_end:
            session_end = now_pt

        # Convert to UTC for all comparisons
        utc = ZoneInfo("UTC")
        return session_start.astimezone(utc), session_end.astimezone(utc)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Normalize any datetime to UTC. Naive datetimes are assumed UTC."""
        utc = ZoneInfo("UTC")
        if dt.tzinfo is None:
            return dt.replace(tzinfo=utc)
        return dt.astimezone(utc)

    async def get_daily_summary(self, account_id='DUA614247') -> str:
        """
        Deterministic, timezone-safe daily trading summary.
        Covers prev day 14:30 PT → current day 14:00 PT.
        Uses reqExecutionsAsync for fill data that survives reconnections.
        """
        if not self.ib.isConnected():
            return "Error: Not connected to IBKR."

        utc = ZoneInfo("UTC")
        pt_tz = ZoneInfo("America/Los_Angeles")
        now_utc = datetime.now(utc)

        # Compute session window: prev day 14:30 PT → current day 14:00 PT
        session_start_utc, session_end_utc = self._compute_session_window(now_utc)

        # Log session boundaries
        logger.info(f"📅 Session window (UTC): {session_start_utc.strftime('%Y-%m-%d %H:%M')} → {session_end_utc.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"📅 Session window (PT):  {session_start_utc.astimezone(pt_tz).strftime('%Y-%m-%d %H:%M')} → {session_end_utc.astimezone(pt_tz).strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"📅 System time: UTC={now_utc.strftime('%H:%M')} PT={now_utc.astimezone(pt_tz).strftime('%H:%M')}")

        # Get account summary
        summary = await self.ib.accountSummaryAsync()

        net_liq = unrealized = realized = 0.0
        for v in summary:
            if v.account == account_id and v.currency == "USD":
                if v.tag == "NetLiquidation":
                    net_liq = float(v.value)
                elif v.tag == "UnrealizedPnL":
                    unrealized = float(v.value)
                elif v.tag == "RealizedPnL":
                    realized = float(v.value)

        # Get detailed trades using fill-based approach
        detailed_trades, trade_pnls = await self.get_detailed_trades(session_start_utc, session_end_utc)

        # Calculate session PnL
        session_pnl = sum(pnl for _, pnl, _, _ in trade_pnls)

        # Calculate PnL breakdowns
        tag_pnls = defaultdict(float)
        symbol_pnls = defaultdict(float)

        for _, pnl, tag, symbol in trade_pnls:
            tag_pnls[tag] += pnl
            symbol_pnls[symbol] += pnl

        # Display times in PT for readability
        start_pt = session_start_utc.astimezone(pt_tz)
        end_pt = session_end_utc.astimezone(pt_tz)
        now_pt = now_utc.astimezone(pt_tz)

        lines = [
            "\n📊 **DAILY TRADING SUMMARY**\n",
            f"Session: {start_pt.strftime('%Y-%m-%d %H:%M')} PT – {end_pt.strftime('%Y-%m-%d %H:%M')} PT",
            "",
            "📈 **ACCOUNT SUMMARY:**",
            f"💰 Net Liq: ${net_liq:,.2f}",
            f"{'🟢' if session_pnl >= 0 else '🔴'} Session PnL: ${session_pnl:+,.2f}",
            f"{'🟢' if unrealized >= 0 else '🔴'} Open PnL: ${unrealized:+,.2f}",
            "────────────────",
            "",
            "📋 **EXECUTED TRADES:**",
            "",
        ]

        if detailed_trades:
            lines.extend(detailed_trades)
            lines.append("")
        else:
            lines.append("  (no completed trades found)\n")

        if symbol_pnls:
            lines.append("📊 **TOTALS BY SYMBOL:**")
            for symbol, total in sorted(symbol_pnls.items()):
                icon = "🟢" if total >= 0 else "🔴"
                lines.append(f"{icon} {symbol}: ${total:+,.2f}")
            lines.append("")

        lines.extend([f"\nReport generated: {now_pt.strftime('%Y-%m-%d %H:%M:%S')} PT\n"])

        return "\n".join(lines)

    @staticmethod
    def _format_fill_based_trade(entry_fills, exit_fills, symbol, contract, direction, tag, multipliers):
        """
        Format a completed trade from consolidated fill lists.

        entry_fills / exit_fills: list of Fill objects (from ib_async)
        Each Fill has: .time, .execution.price, .execution.shares, .commissionReport
        """
        pt_tz = ZoneInfo("America/Los_Angeles")

        entry_time = min(f.time for f in entry_fills)
        entry_time_pt = IBKRClient._ensure_utc(entry_time).astimezone(pt_tz)
        exit_time = max(f.time for f in exit_fills)
        exit_time_pt = IBKRClient._ensure_utc(exit_time).astimezone(pt_tz)

        # Weighted average prices
        entry_shares_total = sum(f.execution.shares for f in entry_fills)
        entry_price = sum(f.execution.price * f.execution.shares for f in entry_fills) / entry_shares_total

        exit_shares_total = sum(f.execution.shares for f in exit_fills)
        exit_price = sum(f.execution.price * f.execution.shares for f in exit_fills) / exit_shares_total

        # Use the smaller of entry/exit qty (handles partial exits)
        qty = min(entry_shares_total, exit_shares_total)

        # Multiplier: prefer contract.multiplier, fall back to lookup
        mult = multipliers.get(symbol, 1)
        if hasattr(contract, 'multiplier') and contract.multiplier:
            try:
                mult = float(contract.multiplier)
            except (ValueError, TypeError):
                pass

        # PnL
        if direction == "LONG":
            pnl = (exit_price - entry_price) * mult * qty
        else:
            pnl = (entry_price - exit_price) * mult * qty

        # Commissions
        total_commission = 0.0
        for f in entry_fills + exit_fills:
            if hasattr(f, 'commissionReport') and f.commissionReport:
                comm = f.commissionReport.commission
                if comm and comm > 0:
                    total_commission += comm

        net_pnl = pnl - total_commission

        # Format output
        entry_date_str = entry_time_pt.strftime("%Y-%m-%d")
        entry_time_str = entry_time_pt.strftime("%H:%M")
        exit_date_str = exit_time_pt.strftime("%Y-%m-%d")
        exit_time_str = exit_time_pt.strftime("%H:%M")
        entry_price_fmt = f"{entry_price:,.2f}"
        exit_price_fmt = f"{exit_price:,.2f}"

        emoji = "🟢" if net_pnl >= 0 else "🔴"

        if entry_date_str == exit_date_str:
            message = (
                f"{emoji} [{tag}] {symbol} {direction} ({int(qty)}) ${net_pnl:+,.2f}\n"
                f"   ⏳ {entry_date_str} {entry_time_str} @ {entry_price_fmt} ➜ {exit_time_str} @ {exit_price_fmt}"
            )
        else:
            message = (
                f"{emoji} [{tag}] {symbol} {direction} ({int(qty)}) ${net_pnl:+,.2f}\n"
                f"   ⏳ {entry_date_str} {entry_time_str} @ {entry_price_fmt} ➜ {exit_date_str} {exit_time_str} @ {exit_price_fmt}"
            )

        return message, net_pnl, tag, symbol

    async def get_detailed_trades(self, start_time_utc, end_time_utc):
        """
        Fill-based, bracket-aware trade analysis.

        Uses reqExecutionsAsync() to get fills that survive API reconnections
        (unlike ib.trades() which only has fills from the current session).

        Groups fills by bracket order (permId linkage), then pairs
        entry fills with exit fills to produce completed round-trip trades.

        All timestamps normalized to UTC before any comparison.

        Returns: (list of formatted messages, list of (message, pnl, tag, symbol))
        """
        utc = ZoneInfo("UTC")

        logger.info(f"📊 Session window (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')} → {end_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 System timezone: {datetime.now().astimezone().tzinfo}")

        # ── Step 1: Get ALL fills from TWS (survives reconnections) ──
        all_fills = await self.ib.reqExecutionsAsync()
        logger.info(f"📊 reqExecutionsAsync returned {len(all_fills)} fills")

        # Also check ib.fills() for any fills only in memory
        memory_fills = self.ib.fills()
        logger.info(f"📊 ib.fills() returned {len(memory_fills)} fills")

        # Merge: use execId as dedup key
        fill_by_exec_id = {}
        for f in all_fills:
            fill_by_exec_id[f.execution.execId] = f
        for f in memory_fills:
            fill_by_exec_id[f.execution.execId] = f

        all_fills_deduped = list(fill_by_exec_id.values())
        logger.info(f"📊 Total unique fills after merge: {len(all_fills_deduped)}")

        # Log sample fills for debugging
        for i, f in enumerate(all_fills_deduped[:5]):
            raw_time = f.time
            norm_time = self._ensure_utc(raw_time)
            logger.info(
                f"📊 Fill[{i}]: {f.contract.symbol} {f.execution.side} "
                f"qty={f.execution.shares} @ {f.execution.price} "
                f"permId={f.execution.permId} orderRef={f.execution.orderRef} "
                f"raw_time={raw_time} (tzinfo={raw_time.tzinfo}) norm_utc={norm_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # ── Step 2: Get bracket relationships from ib.trades() ──
        # This gives us parentPermId linkage (fills don't have it directly)
        trades = self.ib.trades()
        logger.info(f"📊 ib.trades() returned {len(trades)} trade objects")

        # Build permId → parentPermId map and permId → orderRef map
        perm_to_parent = {}
        perm_to_order_ref = {}
        for trade in trades:
            perm_id = trade.order.permId
            parent_perm_id = getattr(trade.order, 'parentPermId', 0)
            if parent_perm_id and parent_perm_id != 0:
                perm_to_parent[perm_id] = parent_perm_id
            perm_to_order_ref[perm_id] = trade.order.orderRef or ""

        # ── Step 3: Group fills into bracket groups ──
        # Each fill has execution.permId → use perm_to_parent to find bracket root
        bracket_groups = defaultdict(list)  # root_permId → [Fill, ...]

        for fill in all_fills_deduped:
            perm_id = fill.execution.permId
            # Walk up to the bracket root
            root_id = perm_id
            if perm_id in perm_to_parent:
                root_id = perm_to_parent[perm_id]

            bracket_groups[root_id].append(fill)

        logger.info(f"📊 Bracket groups formed: {len(bracket_groups)}")

        # ── Step 4: For each bracket group, identify entry/exit fills ──
        detailed_messages = []
        trade_data = []
        multipliers = self.MULTIPLIERS

        for root_id, fills in bracket_groups.items():
            symbol = fills[0].contract.symbol
            contract = fills[0].contract

            # Separate BUY and SELL fills
            buy_fills = [f for f in fills if f.execution.side == 'BOT']
            sell_fills = [f for f in fills if f.execution.side == 'SLD']

            buy_qty = sum(f.execution.shares for f in buy_fills)
            sell_qty = sum(f.execution.shares for f in sell_fills)

            logger.info(
                f"📊 Group {root_id} ({symbol}): "
                f"BUY fills={len(buy_fills)} (qty={buy_qty}), "
                f"SELL fills={len(sell_fills)} (qty={sell_qty})"
            )

            if not buy_fills or not sell_fills:
                # Only one side → open position, not a completed trade
                if buy_fills or sell_fills:
                    side_fills = buy_fills or sell_fills
                    logger.info(f"📊   → Open position (only {'BUY' if buy_fills else 'SELL'} side)")
                continue

            # Determine direction: the side with the EARLIEST fill is the entry
            earliest_buy = min(self._ensure_utc(f.time) for f in buy_fills)
            earliest_sell = min(self._ensure_utc(f.time) for f in sell_fills)

            if earliest_buy < earliest_sell:
                direction = "LONG"
                entry_fills = buy_fills
                exit_fills = sell_fills
            else:
                direction = "SHORT"
                entry_fills = sell_fills
                exit_fills = buy_fills

            # ── Session filter: exit must be within session window ──
            latest_exit_utc = max(self._ensure_utc(f.time) for f in exit_fills)

            if latest_exit_utc < start_time_utc or latest_exit_utc > end_time_utc:
                pt_tz = ZoneInfo("America/Los_Angeles")
                logger.info(
                    f"📊   → Skipping (exit {latest_exit_utc.strftime('%H:%M:%S')} UTC "
                    f"/ {latest_exit_utc.astimezone(pt_tz).strftime('%H:%M:%S')} PT "
                    f"outside session window)"
                )
                continue

            # Get order ref tag from the entry's permId
            entry_perm_id = entry_fills[0].execution.permId
            tag = perm_to_order_ref.get(entry_perm_id, "") or perm_to_order_ref.get(root_id, "") or symbol

            logger.info(
                f"📊   → {direction} trade, exit at {latest_exit_utc.strftime('%H:%M:%S')} UTC, tag={tag}"
            )

            result = self._format_fill_based_trade(
                entry_fills, exit_fills, symbol, contract, direction, tag, multipliers
            )
            if result:
                message, net_pnl, tag, symbol = result
                detailed_messages.append(message)
                trade_data.append((message, net_pnl, tag, symbol))
                logger.info(f"📊   → ✅ {symbol} {direction} PnL=${net_pnl:+.2f}")
            else:
                logger.warning(f"📊   → ❌ format returned None for group {root_id}")

        # Sort by entry time
        def extract_datetime(msg):
            match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})', msg)
            if match:
                date_str, time_str = match.groups()
                return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            return datetime.min

        if detailed_messages:
            paired = sorted(zip(detailed_messages, trade_data), key=lambda x: extract_datetime(x[0]))
            detailed_messages = [p[0] for p in paired]
            trade_data = [p[1] for p in paired]

        logger.info(f"📊 RESULT: {len(detailed_messages)} completed trades in session")

        return detailed_messages, trade_data
