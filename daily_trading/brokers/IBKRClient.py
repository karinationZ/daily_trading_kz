import asyncio
import math
from datetime import datetime, timedelta
from typing import Any, List
from zoneinfo import ZoneInfo

import pandas as pd
from ib_async import Contract, ExecutionFilter, Future, IB, LimitOrder, MarketOrder, util

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
        "SIL": 0.005  # Micro Silver (CME)
    }

    # Map intervals to IB durations and bar sizes
    # Format: (BarSize, Duration)
    TIMEFRAME_CONFIG = {
        "1m": ("1 min", "2 D"),
        "5m": ("5 mins", "5 D"),
        "15m": ("15 mins", "10 D"),
        "1h": ("1 hour", "1 M"),
        "1d": ("1 day", "3 M")
    }

    def __init__(self, host: str = '127.0.0.1', port: int = 4002, client_id: int = 1):
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
            logger.info(f"Connected to IBKR and set Market Data Type to LIVE")

    def disconnect(self):
        self.ib.disconnect()

    def round_to_tick(self, price: float, symbol: str) -> float:
        """Adjusts price to conform to the instrument's tick size."""
        tick = self.TICK_SIZES.get(symbol, 0.01)
        return round(math.floor(price / tick + 0.5) * tick, 4)

    async def get_active_positions(self) -> List[Any]:
        """Returns currently open positions."""
        return await self.ib.reqPositionsAsync()

    async def get_open_orders(self) -> List[Any]:
        """
        Returns a list of active 'Trade' objects (wrapping Order + Contract).
        We return Trades (instead of raw Orders) so the MasterBot
        can access 'trade.contract.symbol'.
        """
        # 1. Force a refresh from TWS to ensure we capture orders
        # that might have been placed before this bot started.
        await self.ib.reqOpenOrdersAsync()

        # 2. Return the list of Trade objects tracked by the client
        return self.ib.openTrades()

    async def get_futures_contract(self, symbol: str) -> Contract | None:
        """Resolves the front-month futures contract."""
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]

        contract_map = {
            "MNQ": {"symbol": "MNQ", "exchange": "CME", "currency": "USD"},
            "MES": {"symbol": "MES", "exchange": "CME", "currency": "USD"},
            "MBT": {"symbol": "MBT", "exchange": "CME", "currency": "USD"},
            "MGC": {"symbol": "MGC", "exchange": "COMEX", "currency": "USD"},
            "SIL": {"symbol": "SI", "exchange": "COMEX", "currency": "USD", "multiplier": "1000"},
            "MCL": {"symbol": "MCL", "exchange": "NYMEX", "currency": "USD"},
        }
        target = contract_map[symbol]
        contract = Future(symbol=target['symbol'],
                          exchange=target['exchange'],
                          currency=target['currency'],
                          multiplier=target.get('multiplier', ''))

        # Qualify (fetch details from IB)
        details = await self.ib.reqContractDetailsAsync(contract)
        if not details:
            raise RuntimeError(f"Could not resolve contract for {symbol}")

        # Filter for front-month (the closest expiration at least 5 days away)
        today = datetime.now()
        valid_contracts = []
        for d in details:
            exp_date = datetime.strptime(d.contract.lastTradeDateOrContractMonth, "%Y%m%d")
            if exp_date > today + timedelta(days=10):
                valid_contracts.append(d.contract)

        front_month = min(valid_contracts, key=lambda c: c.lastTradeDateOrContractMonth)
        self._conid_cache[symbol] = front_month
        return front_month

    async def get_klines(self, symbol: str, interval: str, outside_rth: bool = True) -> pd.DataFrame:
        contract = await self.get_futures_contract(symbol)
        bar_size, duration = self.TIMEFRAME_CONFIG.get(interval, ("5 mins", "2 D"))

        # FIX: Use an empty string for endDateTime to get 'until now'
        # but ensure keepUpToDate=False for historical calls
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',  # Empty string means 'Now'
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=not outside_rth,
            formatDate=1,
            keepUpToDate=False  # Set this to False for simple DataFrame fetching
        )

        df = util.df(bars)
        if df is not None and not df.empty:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        return pd.DataFrame()

    async def place_limit_order(self, symbol: str, quantity: int, price: float, side: str,
                                order_ref: str = "TradingBot"):
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

    async def place_bracket_order(self, symbol: str, quantity: int, entry: float, sl: float, tp: float, direction: str,
                                  order_ref: str = "TradingBot"):
        """Natively handles Entry + SL + TP Bracket."""
        contract = await self.get_futures_contract(symbol)

        side = 'BUY' if direction.upper() == 'LONG' else 'SELL'
        entry = self.round_to_tick(entry, symbol)
        sl = self.round_to_tick(sl, symbol)
        tp = self.round_to_tick(tp, symbol)

        # ib_async utility to create the three order objects
        brackets = self.ib.bracketOrder(
            side,
            quantity,
            limitPrice=entry,
            stopLossPrice=sl,
            takeProfitPrice=tp
        )

        trades = []
        for ord in brackets:
            ord.orderRef = order_ref
            ord.outsideRth = True
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

        print(f"--- DIAGNOSTIC FOR {symbol} ---")
        print(f"Market Data Type (Expected 1): {ticker.marketDataType}")
        print(f"Last Price: {ticker.last}")
        print(f"Bid: {ticker.bid} | Ask: {ticker.ask}")

        if math.isnan(ticker.last):
            print("❌ ERROR: No live price received. IBKR is blocking real-time for this session.")
        elif ticker.marketDataType != 1:
            print(f"❌ ERROR: IBKR forced data type to {ticker.marketDataType} (Delayed).")
        else:
            print("✅ SUCCESS: API has Real-Time access. The delay is in the reqHistoricalData cache.")

    async def close_positions_smartly(self, symbol):
        positions = await self.get_active_positions()
        found = False

        for p in positions:
            if p.contract.symbol == symbol and p.position != 0:
                found = True
                size = p.position
                action = "SELL" if size > 0 else "BUY"
                qty = abs(int(size))

                print(f"FOUND: {symbol} | Size: {size}")
                print(f"ACTION: {action} {qty} to Close.")

                await self.place_market_order(
                    symbol=symbol,
                    quantity=qty,
                    side=action,
                    order_ref="EMERGENCY_CLOSE"
                )
                print("Order Placed.")

        if not found:
            print(f"No open position found for {symbol}. You are flat.")

    async def get_daily_summary(self) -> str:
        try:
            accounts = self.ib.managedAccounts()
            if not accounts:
                return "⚠️ No accounts."
            account_id = accounts[0]

            # 1. Get Account Summary (Net Liq, PnL)
            summary = await self.ib.accountSummaryAsync()
            net_liq = realized = unrealized = 0.0

            for row in summary:
                if (row.account == account_id or row.account == 'All') and row.currency == 'USD':
                    if row.tag == 'NetLiquidation':
                        net_liq = float(row.value)
                    elif row.tag == 'RealizedPnL':
                        realized = float(row.value)
                    elif row.tag == 'UnrealizedPnL':
                        unrealized = float(row.value)

            # 2. Get ACTUAL Current Positions (The Truth)
            positions = await self.ib.reqPositionsAsync()
            current_holdings = {}
            for p in positions:
                if p.account == account_id and p.position != 0:
                    current_holdings[p.contract.symbol] = p.position

            # 3. Get Daily Activity (Volume)
            executions = await self.ib.reqExecutionsAsync(ExecutionFilter())
            ny_now = datetime.now(ZoneInfo("America/New_York"))
            start_of_day = ny_now.replace(hour=0, minute=0, second=0, microsecond=0)

            todays_fills = []
            activity = {}

            for fill in executions:
                if fill.time.astimezone(ZoneInfo("America/New_York")) >= start_of_day:
                    todays_fills.append(fill)
                    sym = fill.contract.symbol
                    if sym not in activity: activity[sym] = {'b': 0, 's': 0}

                    if fill.execution.side == 'BOT':
                        activity[sym]['b'] += fill.execution.shares
                    elif fill.execution.side == 'SLD':
                        activity[sym]['s'] += fill.execution.shares

            # 4. Construct Message
            pnl_icon = "🟢" if realized >= 0 else "🔴"

            lines = [
                f"📅 **REPORT: {ny_now.strftime('%Y-%m-%d')}**",
                f"👤 Account: `{account_id}`",
                "────────────────",
                f"💰 Net Liq: ${net_liq:,.2f}",
                f"{pnl_icon} Realized: ${realized:,.2f}",
                f"🎲 Open PnL: ${unrealized:,.2f}",
                "────────────────"
            ]

            # --- NEW: DISPLAY ACTUAL POSITIONS ---
            if current_holdings:
                lines.append("🏗️ **Current Positions:**")
                for sym, size in current_holdings.items():
                    side = "LONG" if size > 0 else "SHORT"
                    lines.append(f"• {sym}: {side} {abs(size)}")
            else:
                lines.append("🧘 **Current Status: No Open/Pending Orders**")

            lines.append("────────────────")

            # --- DISPLAY VOLUME (Removed "Direction" logic to avoid confusion) ---
            if not todays_fills:
                lines.append("💤 No trades executed today.")
            else:
                lines.append(f"📊 **Today's Activity ({len(todays_fills)} Fills):**")
                for sym, counts in activity.items():
                    # Just report what we did, don't try to guess position
                    lines.append(f"• **{sym}**: Bought {int(counts['b'])} | Sold {int(counts['s'])}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Report Error: {e}")
            return f"⚠️ Report Error: {e}"

    async def get_active_bracket_levels(self, symbol: str, order_ref: str, direction: str) -> dict:
        """
        Fetches ACTUAL live StopLoss and TakeProfit prices from IBKR.
        Uses direction to distinguish between Entry Limit and TP Limit orders.
        """
        await self.ib.reqOpenOrdersAsync()
        open_trades = self.ib.openTrades()

        sl_price = 0.0
        tp_price = 0.0
        found = False

        # Determine expected TP Side (Opposite to Strategy Direction)
        # If Strategy is LONG, TP is a SELL order.
        tp_side = 'SELL' if direction == 'LONG' else 'BUY'

        for t in open_trades:
            if t.contract.symbol == symbol and t.order.orderRef == order_ref:

                # 1. STOP LOSS (Order Type 'STP' or 'TRAIL')
                # SL is always on the exit side (same as TP side)
                if t.order.orderType in ['STP', 'TRAIL', 'STP LMT']:
                    sl_price = t.order.auxPrice
                    found = True

                # 2. TAKE PROFIT (Order Type 'LMT' on the Exit Side)
                # We check t.order.action to ensure we don't grab the Entry Limit order
                elif t.order.orderType == 'LMT' and t.order.action == tp_side:
                    tp_price = t.order.lmtPrice
                    found = True

        return {"sl": sl_price, "tp": tp_price, "found": found}

    async def get_last_fill_price(self, symbol: str) -> float:
        """
        Fetches the price of the most recent execution for a symbol.
        Used to report TRUE exit price (accounting for slippage).
        """
        # Request recent executions
        execs = await self.ib.reqExecutionsAsync(ExecutionFilter())

        # Filter for our symbol
        symbol_execs = [e for e in execs if e.contract.symbol == symbol]

        if not symbol_execs:
            return 0.0

        # Sort by time descending (newest first)
        last_exec = sorted(symbol_execs, key=lambda x: x.time, reverse=True)[0]

        # Return average price of that fill
        return last_exec.execution.avgPrice
    async def check_flatness(self, symbol: str) -> (bool, float):
        """
        Returns (True, 0.0) if flat, or (False, size) if position exists.
        """
        # Force refresh
        positions = await self.ib.reqPositionsAsync()

        for p in positions:
            if p.contract.symbol == symbol and p.position != 0:
                return False, p.position

        return True, 0.0
