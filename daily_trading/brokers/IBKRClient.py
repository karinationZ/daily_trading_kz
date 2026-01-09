import asyncio
import math
from datetime import datetime, timedelta
from typing import Any, List, Optional

import pandas as pd
from ib_async import Contract, Future, IB, LimitOrder, MarketOrder, util

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

    # ======================
    # CONTRACT RESOLUTION
    # ======================

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

    # ======================
    # MARKET DATA
    # ======================

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

    # ======================
    # ORDERS
    # ======================

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

    async def place_bracket_order(self, symbol: str,
                                  quantity: int,
                                  entry: float,
                                  sl: float,
                                  tp: float,
                                  direction: str,
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

    # ======================
    # ORDER MANAGEMENT
    # ======================

    async def cancel_all_orders(self, order_ref: Optional[str] = None):
        """Cancels orders. If order_ref is provided, filters for that bot tag."""
        open_trades = self.ib.openTrades()
        for trade in open_trades:
            if order_ref is None or trade.order.orderRef == order_ref:
                self.ib.cancelOrder(trade.order)
                logger.info(f"Cancelled Order ID: {trade.order.orderId} (Ref: {trade.order.orderRef})")

    async def get_active_positions(self) -> List[Any]:
        """Returns currently open positions."""
        return await self.ib.reqPositionsAsync()

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

# async def run_gateway_qa_test():
#     # 1. Init Client (Gateway Paper Port 4002)
#     cl = IBKRClient(port=4002, client_id=88)
#
#     for symbol in ["SIL", "MGC", "MNQ", "MES", "MCL"]:
#         await cl.connect()
#
#         # STEP 2: Contract Resolution
#         print(f"\n[2/7] Resolving {symbol} Contract...")
#         contract = await cl.get_futures_contract(symbol)
#         print(f"✅ Contract Qualified: {contract.localSymbol} (Expiry: {contract.lastTradeDateOrContractMonth})")
#
#         # STEP 3: Market Data
#         print(f"\n[3/7] Fetching 5m Historical Data...")
#         df = await cl.get_klines(symbol, "5m")
#         if df.empty:
#             raise ValueError("No data returned. Check Market Data Subscriptions in IBKR.")
#         last_price = df.iloc[-1]['close']
#         print(f"✅ Received {len(df)} candles. Last Close: {last_price}")
#
#
# if __name__ == "__main__":
#     asyncio.run(run_gateway_qa_test())
