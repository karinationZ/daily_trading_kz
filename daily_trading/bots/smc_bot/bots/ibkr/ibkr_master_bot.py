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

    @staticmethod
    def is_market_open() -> bool:
        """
        Determines if the Futures Market is generally open (23 hours / 5 days).
        Base Timezone: America/New_York
        """
        ny_now = datetime.now(ZoneInfo("America/New_York"))
        weekday = ny_now.weekday()  # 0=Mon, ... 4=Fri, 5=Sat, 6=Sun
        hour = ny_now.hour

        # 1. Daily Maintenance (17:00 - 18:00 ET) - Market is always closed
        if 17 <= hour < 18:
            return False

        # 2. Weekend Logic
        # Friday: Closes at 17:00 ET
        if weekday == 4 and hour >= 17:
            return False

        # Saturday: Closed all day
        if weekday == 5:
            return False

        # Sunday: Opens at 18:00 ET
        if weekday == 6 and hour < 18:
            return False

        return True

    async def ibkr_check_connection(self):
        """
        Checks connection. Returns True if connected.
        If disconnected, attempts to reconnect multiple times before returning False.
        """
        # 1. Fast Path: If already connected, return immediately.
        if self.ib_client.ib.isConnected():
            return True

        logger.warning("⚠️ IBKR Disconnected. Attempting Reconnect sequence...")

        # 2. Retry Loop (Try 3 times rapidly before failing back to main loop)
        for i in range(1, 6):
            try:
                await self.ib_client.connect()
                # Give it a moment to stabilize
                await asyncio.sleep(1)

                if self.ib_client.ib.isConnected():
                    msg = "✅ IBKR Connection Verified / Restored."
                    logger.info(msg)
                    await self.orb_bot.telegram_bot_debug.send_message(msg)
                    return True
            except Exception as e:
                logger.warning(f"Reconnect attempt {i}/3 failed: {e}")
                await asyncio.sleep(2)  # Short pause between attempts

        # 3. If we get here, all quick retries failed
        logger.critical("FATAL: IBKR Connect Failed after retries.")
        return False

    async def run(self):
        logger.info("🚀 IBKR MASTER BOT STARTING...")
        await self.orb_bot.telegram_bot_debug.send_message("🚀 IBKR MASTER BOT STARTING...")

        # Initial Check
        is_connected = await self.ibkr_check_connection()

        last_logged_hour = -1
        last_logged_hour_summary = -1

        # Only fetch summary if connected
        if is_connected:
            try:
                report = await self.ib_client.get_daily_summary()
                await self.orb_bot.telegram_bot.send_message(report)
            except Exception:
                logger.error("Failed to fetch startup summary.")

        while True:
            try:
                # 1. Time Check
                ny_now = datetime.now(ZoneInfo("America/New_York"))
                current_hour_int = ny_now.hour

                if self.is_market_open():
                    last_logged_hour = -1

                    while not await self.ibkr_check_connection():
                        logger.error("🚫 Connection Lost. Pausing 5s before next retry...")
                        await asyncio.sleep(5)

                        # Optional: Safety check - if market closed while we were retrying, break out
                        if not self.is_market_open():
                            break

                        # 4H Summary Logic
                    if current_hour_int % 4 == 0 and current_hour_int != last_logged_hour_summary:
                        report = await self.ib_client.get_daily_summary()
                        logger.info(f"4H Summary:\n{report}")
                        await self.orb_bot.telegram_bot.send_message(report)
                        last_logged_hour_summary = current_hour_int

                    # --- A. Data Fetching Phase ---
                    market_data = {}

                    # Fetch 5m Data
                    for sym in self.symbols_5m:
                        df = await self.ib_client.get_klines(sym, "5m")
                        if not df.empty:
                            if sym not in market_data: market_data[sym] = {}
                            market_data[sym]["5m"] = df
                        await asyncio.sleep(0.1)

                    # Fetch 1h Data
                    for sym in self.symbols_1h:
                        df = await self.ib_client.get_klines(sym, "1h")
                        if not df.empty:
                            if sym not in market_data: market_data[sym] = {}
                            market_data[sym]["1h"] = df
                        await asyncio.sleep(0.1)

                    # --- B. Execution Phase ---
                    positions, orders, blocked = await self.get_account_state()
                    active_holdings = [p for p in positions if p.position != 0]

                    if blocked:
                        msg = f"🔒 Blocked Symbols: {blocked}"
                        logger.info(msg)
                        # await self.orb_bot.telegram_bot_debug.send_message(msg)

                    if active_holdings:
                        pretty_pos = [f"{p.contract.symbol}: {p.position}" for p in active_holdings]
                        msg = f"💼 Active Holdings: {pretty_pos}"
                        logger.info(msg)
                        await self.orb_bot.telegram_bot_debug.send_message(msg)

                    if orders:
                        pretty_orders = [f"{o.order.action} {o.order.totalQuantity} {o.contract.symbol}" for o in
                                         orders]
                        msg = f"⏳ Pending Orders: {pretty_orders}"
                        logger.info(msg)
                        await self.orb_bot.telegram_bot_debug.send_message(msg)

                    # Pass State to Bots
                    await asyncio.gather(
                        self.orb_bot.execute_cycle(market_data, blocked, positions, orders),
                        self.sd_bot.execute_cycle(market_data, blocked, positions, orders)
                    )

                    # --- C. Wait for Next Candle ---
                    await self.sleep_until_next_candle()

                else:
                    # 💤 SLEEP MODE
                    last_logged_hour_summary = -1
                    if ny_now.hour != last_logged_hour:
                        msg = "💤 Market Closed. Sleeping..."
                        last_logged_hour = ny_now.hour
                        await self.orb_bot.telegram_bot_debug.send_message(msg)
                        logger.info(msg)
                        await asyncio.sleep(60)

                    # Maintenance: Check connection quietly
                    await self.ibkr_check_connection()
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
        if seconds <= 0:
            seconds = 300

        logger.debug(f"⏳ Waiting {seconds}s for next candle...")
        await asyncio.sleep(seconds)
