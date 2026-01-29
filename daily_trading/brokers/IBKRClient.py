import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import Any, List

import pandas as pd
from ib_async import (
    Contract,
    ExecutionFilter,
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
        "15m": ("15 mins", "10 D"),
        "1h": ("1 hour", "1 M"),
        "1d": ("1 day", "3 M"),
    }

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
        order = LimitOrder(side.upper(), quantity, price, orderRef=order_ref, outsideRth=False)
        trade = self.ib.placeOrder(contract, order)
        return trade

    async def place_market_order(self, symbol: str, quantity: int, side: str, order_ref: str = "TradingBot"):
        contract = await self.get_futures_contract(symbol)
        order = MarketOrder(side.upper(), quantity, orderRef=order_ref, outsideRth=False)
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
        """Natively handles Entry + SL + TP Bracket."""
        contract = await self.get_futures_contract(symbol)

        side = "BUY" if direction.upper() == "LONG" else "SELL"
        entry = self.round_to_tick(entry, symbol)
        sl = self.round_to_tick(sl, symbol)
        tp = self.round_to_tick(tp, symbol)

        # Create bracket orders WITHOUT outsideRth (it's ignored anyway)
        brackets = self.ib.bracketOrder(
            side,
            quantity,
            limitPrice=entry,
            stopLossPrice=sl,
            takeProfitPrice=tp,
            outsideRth=False,
            # Set to False for all futures to avoid warning
        )

        trades = []
        for ord in brackets:
            ord.orderRef = order_ref
            ord.outsideRth = False  # Doesn't matter for futures, but avoids warning
            ord.tif = "GTC"  # Good-Til-Cancelled (works across sessions)
            # Alternative: ord.tif = "GTD" with ord.goodTillDate for specific expiry
            trade = self.ib.placeOrder(contract, ord)
            trades.append(trade)

        return trades

    async def cancel_all_orders(self, order_ref: str):
        """Cancels orders. If order_ref is provided, filters for that bot tag."""
        open_trades = self.ib.openTrades()
        for trade in open_trades:
            if order_ref is None or trade.order.orderRef == order_ref:
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled Order ID: {trade.order.orderId} (Ref: {trade.order.orderRef})")

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
                    order_ref="EMERGENCY_CLOSE",
                )
                logger.info("Order Placed - EMERGENCY_CLOSE")

        if not found:
            logger.info(f"No open position found for {symbol}. You are flat.")

    async def get_daily_summary(self) -> str:
        try:
            values = self.ib.accountValues()
            account_id = self.ib.managedAccounts()[0]
            net_liq = realized = unrealized = 0.0

            for v in values:
                # Filter for the specific account and currency
                if v.account == account_id and v.currency == "USD":
                    if v.tag == "NetLiquidation":
                        net_liq = float(v.value)
                    elif v.tag == "RealizedPnL":
                        realized = float(v.value)
                    elif v.tag == "UnrealizedPnL":
                        unrealized = float(v.value)

            # 2. Get Today's Executions (UTC)
            executions = await asyncio.wait_for(self.ib.reqExecutionsAsync(ExecutionFilter()), timeout=15)
            utc_now = datetime.now(timezone.utc)
            start_of_day = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)

            # CRITICAL FIX: Sort by Time AND execId to handle simultaneous fills correctly
            tz = timezone.utc
            todays_fills = sorted(
                [f for f in executions if f.time.astimezone(tz) >= start_of_day],
                key=lambda x: (x.time, x.execution.execId),
            )

            # 3. FIFO MATCHING LOGIC
            raw_trades = []
            open_inventory = {}
            calculated_total_pnl = 0.0
            inventory_mismatch = False
            orphan_note = ""

            for fill in todays_fills:
                sym = fill.contract.symbol
                side = fill.execution.side
                price = fill.execution.avgPrice
                qty = fill.execution.shares
                order_ref = fill.execution.orderRef
                fill_time = fill.time.astimezone(tz)

                if sym not in open_inventory:
                    open_inventory[sym] = []

                needed_side = "SLD" if side == "BOT" else "BOT"

                # Match against inventory
                while qty > 0 and open_inventory[sym] and open_inventory[sym][0]["side"] == needed_side:
                    match = open_inventory[sym][0]
                    match_qty = min(qty, match["qty"])

                    is_long = match["side"] == "BOT"
                    mult = self.MULTIPLIERS.get(sym, 1.0)

                    diff = price - match["price"] if is_long else match["price"] - price
                    pnl = diff * match_qty * mult
                    calculated_total_pnl += pnl

                    raw_trades.append(
                        {
                            "symbol": sym,
                            "direction": "LONG" if is_long else "SHORT",
                            "qty": match_qty,
                            "pnl": pnl,
                            "entry_dt": match["time"],
                            "exit_dt": fill_time,
                            "entry_price": match["price"],
                            "exit_price": price,
                            "strategy_tag": order_ref,
                        }
                    )

                    qty -= match_qty
                    match["qty"] -= match_qty
                    if match["qty"] <= 0:
                        open_inventory[sym].pop(0)

                if qty > 0:
                    open_inventory[sym].append({"price": price, "qty": qty, "side": side, "time": fill_time})

            # 4. AGGREGATION STEP
            consolidated_trades = []
            if raw_trades:
                # Sort by Exit Time
                raw_trades.sort(key=lambda x: x["exit_dt"])

                current = raw_trades[0]

                for i in range(1, len(raw_trades)):
                    next_t = raw_trades[i]

                    time_diff = (next_t["exit_dt"] - current["exit_dt"]).total_seconds() / 60

                    is_same_group = current["symbol"] == next_t["symbol"] and current["direction"] == next_t["direction"] and time_diff < 5.0

                    if is_same_group:
                        total_qty = current["qty"] + next_t["qty"]

                        current_entry_val = current["entry_price"] * current["qty"]
                        next_entry_val = next_t["entry_price"] * next_t["qty"]
                        avg_entry = (current_entry_val + next_entry_val) / total_qty

                        current_exit_val = current["exit_price"] * current["qty"]
                        next_exit_val = next_t["exit_price"] * next_t["qty"]
                        avg_exit = (current_exit_val + next_exit_val) / total_qty

                        current["qty"] = total_qty
                        current["pnl"] += next_t["pnl"]
                        current["entry_price"] = avg_entry
                        current["exit_price"] = avg_exit

                        if next_t["entry_dt"] < current["entry_dt"]:
                            current["entry_dt"] = next_t["entry_dt"]
                        if next_t["exit_dt"] > current["exit_dt"]:
                            current["exit_dt"] = next_t["exit_dt"]

                    else:
                        consolidated_trades.append(current)
                        current = next_t

                consolidated_trades.append(current)

            # 5. Check Overnight
            positions = self.ib.positions()
            actual_pos_map = {p.contract.symbol: p.position for p in positions if p.account == account_id and p.position != 0}

            for sym, items in open_inventory.items():
                calc_qty = sum([i["qty"] * (1 if i["side"] == "BOT" else -1) for i in items])
                actual_qty = actual_pos_map.get(sym, 0.0)
                if calc_qty != 0 and actual_qty == 0:
                    inventory_mismatch = True
                    orphan_note += f"\n⚠️ <b>Overnight Close:</b> {sym} {abs(calc_qty)} contracts."

            # 6. Build Report
            pnl_icon = "🟢" if realized >= 0 else "🔴"
            lines = [
                f"📅 **REPORT: {utc_now.strftime('%Y-%m-%d')} (UTC)**",
                f"👤 Account: `{account_id}`",
                "────────────────",
                f"💰 Net Liq: ${net_liq:,.2f}",
                f"{pnl_icon} Daily PnL: ${realized:,.2f}",
                f"🎲 Open PnL: ${unrealized:,.2f}",
                "────────────────",
            ]

            if actual_pos_map:
                lines.append("🏗️ **Open Positions:**")
                for sym, size in actual_pos_map.items():
                    side = "LONG" if size > 0 else "SHORT"
                    lines.append(f"• {sym}: {side} {abs(size)}")
            else:
                lines.append("🧘 **Current Status: FLAT**")

            lines.append("────────────────")

            if consolidated_trades:
                lines.append(f"📊 **Today's Trades (Calc: ${calculated_total_pnl:+.2f}):**")
                for t in consolidated_trades:
                    icon = "🟢" if t["pnl"] >= 0 else "🔴"
                    t_entry = t["entry_dt"].strftime("%H:%M")
                    t_exit = t["exit_dt"].strftime("%H:%M")

                    strategy_info = f" {t['strategy_tag']}" if t['strategy_tag'] else ""
                    header = f"{icon} {t['symbol']} {t['direction']} ({int(t['qty'])}) <b>${t['pnl']:+.2f}</b>"
                    details = f"   ⏳ {t_entry} @ {t['entry_price']:,.2f} ➜ {t_exit} {tz} @ {t['exit_price']:,.2f}"
                    lines.append(f"\n{strategy_info} - {header}\n{details}\n")
            else:
                lines.append("💤 No round-trip trades completed today.")

            if inventory_mismatch:
                lines.append("────────────────")
                lines.append(f"📉 <i>Note: Realized PnL includes overnight positions.</i>{orphan_note}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Report Error: {e}")
            return f"⚠️ Report Error: {e}"

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
