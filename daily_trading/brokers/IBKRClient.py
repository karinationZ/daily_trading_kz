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
    # This is the main report function that compiles all the data into a comprehensive text report.

    async def get_daily_summary(self, account_id='DUA614247') -> str:
        """
        Clean daily trading summary with net PnL after commissions.
        """
        if not self.ib.isConnected():
            return "Error: Not connected to IBKR."

        # Session start - Trading day starts at midnight PT (00:00)
        pt_tz = ZoneInfo("America/Los_Angeles")
        now_pt = datetime.now(pt_tz)

        # Session starts at midnight (00:00) of current day in PT
        session_date = now_pt.date()
        start_pt = datetime(session_date.year, session_date.month, session_date.day, 0, 0, 0, tzinfo=pt_tz)
        end_pt = now_pt

        logger.info(f"📅 Session window: {start_pt.strftime('%Y-%m-%d %H:%M %Z')} → {end_pt.strftime('%H:%M %Z')}")

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

        # Get detailed trades
        detailed_trades, trade_pnls = await self.get_detailed_trades(start_pt, end_pt)

        # Calculate session PnL
        session_pnl = sum(pnl for _, pnl, _, _ in trade_pnls)

        # Calculate PnL breakdowns
        tag_pnls = defaultdict(float)
        symbol_pnls = defaultdict(float)

        for _, pnl, tag, symbol in trade_pnls:
            tag_pnls[tag] += pnl
            symbol_pnls[symbol] += pnl

        # Build report - clean and simple like your original
        lines = [
            "\n📊 **DAILY TRADING SUMMARY**\n",
            f"Start: {start_pt.strftime('%Y-%m-%d %H:%M')} PT – {end_pt.strftime('%H:%M')} PT",
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

        # Add detailed trades
        if detailed_trades:
            lines.extend(detailed_trades)
            lines.append("")  # Add spacing
        else:
            lines.append("  (no completed trades found)\n")

        # Add totals by symbol
        if symbol_pnls:
            lines.append("📊 **TOTALS BY SYMBOL:**")
            for symbol, total in sorted(symbol_pnls.items()):
                icon = "🟢" if total >= 0 else "🔴"
                lines.append(f"{icon} {symbol}: ${total:+,.2f}")
            lines.append("")

        # Footer
        lines.extend([f"\nReport generated: {end_pt.strftime('%Y-%m-%d %H:%M:%S')} PT\n"])

        return "\n".join(lines)

    @staticmethod
    def format_detailed_trade(entry_trade, exit_trade, multipliers):
        """Format a single trade pair - clean and simple like your original."""

        # Get contract details
        contract = entry_trade.contract
        symbol = contract.symbol

        # Get entry details
        entry_fills = entry_trade.fills
        entry_time = min(fill.time for fill in entry_fills)
        pt_tz = ZoneInfo("America/Los_Angeles")
        entry_time_pt = entry_time.astimezone(pt_tz)

        entry_price = sum(fill.execution.price * fill.execution.shares for fill in entry_fills) / sum(fill.execution.shares for fill in entry_fills)
        entry_qty = sum(fill.execution.shares for fill in entry_fills)

        # Get exit details
        exit_fills = exit_trade.fills
        exit_time = max(fill.time for fill in exit_fills)
        exit_time_pt = exit_time.astimezone(pt_tz)

        exit_price = sum(fill.execution.price * fill.execution.shares for fill in exit_fills) / sum(fill.execution.shares for fill in exit_fills)
        exit_qty = sum(fill.execution.shares for fill in exit_fills)

        # Verify quantities match
        if entry_qty != exit_qty:
            return None

        # Get multiplier
        mult = multipliers.get(symbol, 1)
        if hasattr(contract, 'multiplier') and contract.multiplier:
            mult = float(contract.multiplier)

        # Calculate PnL based on direction
        if entry_trade.order.action == 'BUY' and exit_trade.order.action == 'SELL':
            direction = "LONG"
            pnl = (exit_price - entry_price) * mult * entry_qty
        elif entry_trade.order.action == 'SELL' and exit_trade.order.action == 'BUY':
            direction = "SHORT"
            pnl = (entry_price - exit_price) * mult * entry_qty
        else:
            return None

        # Calculate commissions
        total_commission = 0
        for fill in entry_fills:
            if hasattr(fill, 'commissionReport') and fill.commissionReport:
                total_commission += fill.commissionReport.commission
        for fill in exit_fills:
            if hasattr(fill, 'commissionReport') and fill.commissionReport:
                total_commission += fill.commissionReport.commission

        # Net PnL after commissions
        net_pnl = pnl - total_commission

        # Get order reference tag
        tag = entry_trade.order.orderRef or symbol

        # Format dates and times
        entry_date_str = entry_time_pt.strftime("%Y-%m-%d")
        entry_time_str = entry_time_pt.strftime("%H:%M")
        exit_date_str = exit_time_pt.strftime("%Y-%m-%d")
        exit_time_str = exit_time_pt.strftime("%H:%M")

        # Format numbers
        entry_price_fmt = f"{entry_price:,.2f}"
        exit_price_fmt = f"{exit_price:,.2f}"

        # Simple clean format like your original
        emoji = "🟢" if net_pnl >= 0 else "🔴"

        if entry_date_str == exit_date_str:
            # Same day trade
            message = (
                f"{emoji} [{tag}] {symbol} {direction} ({int(entry_qty)}) ${net_pnl:+,.2f}\n"
                f"   ⏳ {entry_date_str} {entry_time_str} @ {entry_price_fmt} ➜ {exit_time_str} @ {exit_price_fmt}"
            )
        else:
            # Multi-day trade
            message = (
                f"{emoji} [{tag}] {symbol} {direction} ({int(entry_qty)}) ${net_pnl:+,.2f}\n"
                f"   ⏳ {entry_date_str} {entry_time_str} @ {entry_price_fmt} ➜ {exit_date_str} {exit_time_str} @ {exit_price_fmt}"
            )

        return message, net_pnl, tag, symbol

    async def get_detailed_trades(self, start_time, end_time):
        """
        Get detailed trade-by-trade analysis from self.ib.trades()
        Returns: (list of formatted messages, list of (message, pnl, tag, symbol))
        """
        trades = self.ib.trades()

        # DEBUGGING: Log critical info
        logger.info(f"🔍 DEBUG - Session window: {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} → {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"🔍 DEBUG - Total trades in ib.trades(): {len(trades)}")
        logger.info(f"🔍 DEBUG - System timezone: {datetime.now().astimezone().tzinfo}")

        # Log first few trades for inspection
        for i, trade in enumerate(trades[:3]):
            if trade.fills:
                first_fill = trade.fills[0]
                logger.info(f"🔍 DEBUG - Trade {i}: Symbol={trade.contract.symbol}, "
                           f"Fill time={first_fill.time}, "
                           f"Fill time tzinfo={first_fill.time.tzinfo}, "
                           f"Status={trade.orderStatus.status}")

        # Group trades by parent-child relationships
        trade_groups = {}
        child_to_parent = {}

        # First pass: identify parent-child relationships
        for trade in trades:
            if hasattr(trade.order, 'parentPermId') and trade.order.parentPermId:
                child_to_parent[trade.order.permId] = trade.order.parentPermId

        # Second pass: group ALL trades first (don't filter by time yet)
        for trade in trades:
            # Skip cancelled orders
            if trade.orderStatus.status == 'Cancelled':
                logger.debug(f"🔍 Skipped cancelled: {trade.contract.symbol}")
                continue

            # If this is a child order, add to parent's group
            if trade.order.permId in child_to_parent:
                parent_id = child_to_parent[trade.order.permId]
                if parent_id not in trade_groups:
                    trade_groups[parent_id] = []
                trade_groups[parent_id].append(trade)
            else:
                # This might be a parent or standalone trade
                if trade.order.permId not in trade_groups:
                    trade_groups[trade.order.permId] = [trade]
                else:
                    trade_groups[trade.order.permId].append(trade)

        # DEBUG: Summary
        logger.info(f"🔍 DEBUG SUMMARY - Trade groups created: {len(trade_groups)}")

        # Process each trade group
        detailed_messages = []
        trade_data = []  # (message, pnl, tag, symbol)
        multipliers = self.MULTIPLIERS

        logger.info(f"🔍 DEBUG - Processing {len(trade_groups)} trade groups...")

        for group_id, group_trades in trade_groups.items():
            logger.debug(f"🔍 Processing group {group_id}: {len(group_trades)} trades")

            # Find entry and exit trades
            entry_trade = None
            exit_trade = None

            # Collect all filled trades
            buy_trades = []
            sell_trades = []

            for trade in group_trades:
                logger.debug(f"🔍   Trade: {trade.contract.symbol} {trade.order.action} "
                            f"parentPermId={getattr(trade.order, 'parentPermId', None)} "
                            f"fills={len(trade.fills) if trade.fills else 0} "
                            f"status={trade.orderStatus.status}")

                # Only consider filled trades
                if trade.order.action in ['BUY', 'SELL'] and trade.fills and trade.orderStatus.status == 'Filled':
                    if trade.order.action == 'BUY':
                        buy_trades.append(trade)
                    else:
                        sell_trades.append(trade)

            # Determine entry/exit based on action pairing
            # LONG trade: BUY (entry) → SELL (exit)
            # SHORT trade: SELL (entry) → BUY (exit)
            if buy_trades and sell_trades:
                # Both BUY and SELL exist → round trip completed
                # Use earliest of each as entry/exit
                entry_trade = min(buy_trades, key=lambda t: min(f.time for f in t.fills))
                exit_trade = min(sell_trades, key=lambda t: min(f.time for f in t.fills))

                # Determine direction by comparing times
                entry_time = min(f.time for f in entry_trade.fills)
                exit_time = min(f.time for f in exit_trade.fills)

                if entry_time > exit_time:
                    # SELL came before BUY → SHORT position
                    entry_trade, exit_trade = exit_trade, entry_trade

                logger.debug(f"🔍     → LONG: Entry=BUY, Exit=SELL")
            elif len(buy_trades) >= 2:
                # Multiple BUYs → SHORT trade (first SELL entry, second BUY exit)
                # Sort by time
                sorted_buys = sorted(buy_trades, key=lambda t: min(f.time for f in t.fills))
                entry_trade = sorted_buys[1]  # Later BUY might be covering a short
                exit_trade = sorted_buys[0]
                # Actually this is ambiguous, skip it
                entry_trade = None
                exit_trade = None
            elif len(sell_trades) >= 2:
                # Multiple SELLs → LONG trade (first BUY entry, second SELL exit)
                sorted_sells = sorted(sell_trades, key=lambda t: min(f.time for f in t.fills))
                entry_trade = sorted_sells[0]
                exit_trade = sorted_sells[1]
                # Actually this is also ambiguous, skip it
                entry_trade = None
                exit_trade = None
            else:
                # Only one side filled → incomplete trade (still open)
                entry_trade = None
                exit_trade = None

            # Debug: Report what was found
            logger.info(f"🔍 Group {group_id}: BUYs={len(buy_trades)}, SELLs={len(sell_trades)}, "
                       f"Entry={'✓' if entry_trade else '✗'}, Exit={'✓' if exit_trade else '✗'}")

            # If we found both entry and exit, check time filter and format
            if entry_trade and exit_trade and entry_trade.fills and exit_trade.fills:
                # Filter by EXIT time (include if exit happened within session window)
                exit_fills = exit_trade.fills
                exit_time = max(fill.time for fill in exit_fills)

                # Ensure exit_time is timezone-aware
                if exit_time.tzinfo is None:
                    pt_tz = ZoneInfo("America/Los_Angeles")
                    exit_time = exit_time.replace(tzinfo=pt_tz)

                # Check if exit is within session window
                if exit_time < start_time or exit_time > end_time:
                    logger.info(f"🔍   → ⏭️ Skipping - Exit outside session window: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    continue

                logger.info(f"🔍   → Formatting trade pair (exit at {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')})...")
                result = self.format_detailed_trade(entry_trade, exit_trade, multipliers)
                if result:
                    message, net_pnl, tag, symbol = result
                    detailed_messages.append(message)
                    trade_data.append((message, net_pnl, tag, symbol))
                    logger.info(f"🔍   → ✅ Trade formatted: {symbol} PnL=${net_pnl:+.2f}")
                else:
                    logger.warning(f"🔍   → ❌ format_detailed_trade returned None")
            else:
                logger.warning(f"🔍   → ❌ Incomplete pair - skipping (Entry={bool(entry_trade)}, Exit={bool(exit_trade)})")

        # Sort by entry time (extract from message)
        def extract_datetime(msg):
            # Extract from format like: 2026-02-13 20:59 @ 24,810.25 ➜ 2026-02-13 21:06 @ 24,780.25
            match = re.search(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})', msg)
            if match:
                date_str, time_str = match.groups()
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                return dt
            return datetime.min

        # Sort messages by entry time
        if detailed_messages:
            paired = sorted(zip(detailed_messages, trade_data), key=lambda x: extract_datetime(x[0]))
            detailed_messages = [p[0] for p in paired]
            trade_data = [p[1] for p in paired]

        logger.info(f"🔍 DEBUG FINAL - Successfully formatted {len(detailed_messages)} completed trades")

        return detailed_messages, trade_data
