import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

# Import the Sub-Bots
from daily_trading.bots.smc_bot.bots.ibkr.orb_bot import ORBBot
from daily_trading.bots.smc_bot.bots.ibkr.supply_demand_tap_bot import SupplyDemandTapBot
from daily_trading.brokers.IBKRClient import IBKRClient
from daily_trading.settings import IBKR_PORT, logger


class MasterBot:
    def __init__(self):
        # 1. Initialize ONE Shared Connection
        self.ib_client = IBKRClient(port=IBKR_PORT, client_id=1)

        # 2. Init Bots with Shared Client
        self.orb_bot = ORBBot(ib_client=self.ib_client)
        self.sd_bot = SupplyDemandTapBot(ib_client=self.ib_client)

        # 3. Categorize Symbols for Fetching
        # ORB needs 5m
        self.symbols_5m = set(self.orb_bot.active_symbols + self.sd_bot.active_symbols)
        # SD needs 1h
        self.symbols_1h = set(self.sd_bot.active_symbols)

    async def get_account_state(self):
        """
        Fetches Positions and Orders ONCE.
        Returns:
            - positions: List of IBKR Position objects
            - orders: List of IBKR Trade objects
            - blocked: Set of symbols that are busy
        """
        blocked = set()
        positions = []
        orders = []

        try:
            # 1. Fetch Raw Data
            positions = await self.ib_client.get_active_positions()
            orders = await self.ib_client.get_open_orders()

            # 2. Build Blocked Set
            for pos in positions:
                if pos.position != 0:
                    blocked.add(pos.contract.symbol)

            for o in orders:
                blocked.add(o.contract.symbol)

        except Exception as e:
            logger.error(f"State Fetch Failed: {e}")

        return positions, orders, blocked

    async def get_blocked_symbols(self):
        """
        Returns a set of symbols that are currently active (Open Orders or Positions).
        Used to prevent bots from double-trading.
        """
        blocked = set()
        try:
            # Check Live Positions
            positions = await self.ib_client.get_active_positions()
            for pos in positions:
                if pos.position != 0:
                    blocked.add(pos.contract.symbol)

            # Check Pending Orders
            orders = await self.ib_client.get_open_orders()
            for o in orders:
                blocked.add(o.contract.symbol)
        except Exception as e:
            logger.error(f"Conflict Check Failed: {e}")
        return blocked

    async def run(self):
        logger.info("🚀 IBKR MASTER BOT STARTING...")

        try:
            await self.ib_client.connect()
            logger.info("✅ IBKR Connection Verified.")
        except Exception as e:
            msg = f"FATAL: IBKR Connect Failed: {e}"
            logger.critical(msg)
            self.orb_bot.telegram_bot_debug.send_message(msg)
            return
        last_logged_hour = -1
        while True:
            try:
                # 1. Time Check (New York Time)
                ny_now = datetime.now(ZoneInfo("America/New_York"))
                is_weekday = ny_now.weekday() < 5
                current_hour = ny_now.hour + (ny_now.minute / 60)

                # Active Hours: 01:45 AM to 05:15 PM NY
                if is_weekday and (1.75 <= current_hour <= 17.25):
                    last_logged_hour = -1

                    # --- A. Data Fetching Phase (Centralized) ---
                    # Structure: { "MNQ": { "5m": df, "1h": df } }
                    market_data = {}

                    # Fetch 5m Data
                    for sym in self.symbols_5m:
                        df = await self.ib_client.get_klines(sym, "5m")
                        if not df.empty:
                            if sym not in market_data: market_data[sym] = {}
                            market_data[sym]["5m"] = df
                        # Tiny sleep prevents IBKR Pacing Violation
                        await asyncio.sleep(0.1)

                        # Fetch 1h Data (Every 5 mins is fine)
                    for sym in self.symbols_1h:
                        df = await self.ib_client.get_klines(sym, "1h")
                        if not df.empty:
                            if sym not in market_data: market_data[sym] = {}
                            market_data[sym]["1h"] = df
                        await asyncio.sleep(0.1)

                        # --- B. Execution Phase (UPDATED) ---
                        # 1. Fetch State ONCE
                    positions, orders, blocked = await self.get_account_state()

                    if blocked:
                        msg = f"🔒 There are open positions or orders for Symbols: {blocked}"
                        logger.info(msg)
                        await self.orb_bot.telegram_bot_debug(msg)
                    if positions:
                        msg = f"💼 Found Opened positions: {positions}"
                        logger.info(msg)
                        await self.orb_bot.telegram_bot_debug(msg)
                    if orders:
                        msg = f"💼 Found Pending Orders: {orders}"
                        logger.info(msg)
                        await self.orb_bot.telegram_bot_debug(msg)

                    # 2. Pass State to Bots
                    await asyncio.gather(
                        self.orb_bot.execute_cycle(market_data, blocked, positions, orders),
                        self.sd_bot.execute_cycle(market_data, blocked, positions, orders)
                    )

                    # --- C. Wait for Next Candle ---
                    await self.sleep_until_next_candle()

                else:
                    # 💤 SLEEP MODE
                    if ny_now.hour != last_logged_hour:
                        msg = "💤 Market Closed. Sleeping..."
                        last_logged_hour = ny_now.hour
                        await self.orb_bot.telegram_bot_debug.send_message(msg)
                        logger.info(msg)
                        await asyncio.sleep(60)

                    # Maintenance: Keep connection alive
                    if not self.ib_client.ib.isConnected():
                        try:
                            logger.info("Reconnecting IBKR...")
                            await self.ib_client.connect()
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e}")

                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"MASTER LOOP CRITICAL ERROR: {e}")
                await asyncio.sleep(60)

    @staticmethod
    async def sleep_until_next_candle():
        """Syncs the loop to the 5-minute candle close."""
        now = datetime.now()
        # Calculate seconds remaining until next 5m mark
        minutes = 5 - (now.minute % 5)
        seconds = (minutes * 60) - now.second
        if seconds <= 0: seconds = 300

        logger.debug(f"⏳ Waiting {seconds}s for next candle...")
        await asyncio.sleep(seconds)
