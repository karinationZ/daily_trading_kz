import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.ibkr.bos_bot import BosOrderBlockBot
from daily_trading.bots.smc_bot.bots.ibkr.bos_fvg_bot import BosFvgBot

# Import the Sub-Bots
from daily_trading.bots.smc_bot.bots.ibkr.orb_bot import ORBBot
from daily_trading.brokers.IBKRClient import IBKRClient
from daily_trading.settings import IBKR_PORT, logger


class MasterBot:
    def __init__(self):
        # 1. Initialize ONE Shared Connection
        self.ib_client = IBKRClient(port=IBKR_PORT, client_id=1)

        # 2. Init Bots with Shared Client
        self.orb_bot = ORBBot(ib_client=self.ib_client)
        self.bos_ob_bot = BosOrderBlockBot(ib_client=self.ib_client)
        self.bos_fvg_bot = BosFvgBot(ib_client=self.ib_client)

        # 3. Categorize Symbols for Fetching
        # ORB & BOS needs 5m
        self.symbols_5m = set(self.orb_bot.active_symbols + self.bos_ob_bot.active_symbols)
        # BOS need 1h
        self.symbols_1h = set(self.bos_ob_bot.active_symbols)

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
            positions = self.ib_client.get_active_positions()
            orders = self.ib_client.get_open_orders()

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
            positions = self.ib_client.get_active_positions()
            for pos in positions:
                if pos.position != 0:
                    blocked.add(pos.contract.symbol)

            # Check Pending Orders
            orders = self.ib_client.get_open_orders()
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
                    await self.orb_bot.telegram_debug_channel.send_message(msg)
                    return True
            except Exception as e:
                logger.warning(f"Reconnect attempt {i}/3 failed: {e}")
                await asyncio.sleep(2)  # Short pause between attempts

        # 3. If we get here, all quick retries failed
        logger.critical("FATAL: IBKR Connect Failed after retries.")
        return False

    async def run(self):
        logger.info("🚀 IBKR MASTER BOT STARTING...")
        await self.orb_bot.telegram_debug_channel.send_message("🚀 IBKR MASTER BOT STARTING...")

        # Initial Check
        is_connected = await self.ibkr_check_connection()

        last_logged_hour = -1
        last_logged_hour_summary = -1

        # Only fetch summary if connected
        if is_connected:
            try:
                report = await self.ib_client.get_daily_summary()
                await self.orb_bot.telegram_main_channel.send_message(report)
            except Exception as e:
                logger.error(f"Failed to fetch startup summary.\n{e}")

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

                        # 1H Summary Logic
                    if current_hour_int != last_logged_hour_summary:
                        report = await self.ib_client.get_daily_summary()
                        logger.info(f"1H Summary:\n{report}")
                        await self.orb_bot.telegram_main_channel.send_message(report)
                        last_logged_hour_summary = current_hour_int

                    # --- A. Data Fetching Phase ---
                    market_data = {}

                    # --- B. Execution Phase ---
                    positions, orders, blocked = await self.get_account_state()
                    active_holdings = [p for p in positions if p.position != 0]

                    if blocked:
                        msg = f"🔒 Blocked Symbols: {blocked}"
                        logger.info(msg)
                        await self.orb_bot.telegram_debug_channel.send_message(msg)

                    if active_holdings:
                        pretty_pos = [f"{p.contract.symbol}: {p.position}" for p in active_holdings]
                        msg = f"💼 Active Holdings: {pretty_pos}"
                        logger.info(msg)
                        await self.orb_bot.telegram_debug_channel.send_message(msg)

                    if orders:
                        pretty_orders = [f"{o.order.action} {o.order.totalQuantity} {o.contract.symbol}" for o in orders]
                        msg = f"⏳ Pending Orders: {pretty_orders}"
                        logger.info(msg)
                        await self.orb_bot.telegram_debug_channel.send_message(msg)

                    logger.info("🤖 Executing bots strategies ...")
                    try:
                        # Fetch 5m Data
                        for sym in self.symbols_5m:
                            df_5m = await self.ib_client.get_klines(sym, "5m")
                            market_data[sym] = {"5m": df_5m}
                            if df_5m.empty:
                                logger.warning(f"⚠️ ALARM {sym}: '5m' DataFrame is empty.")

                        # Fetch all 1h data
                        for sym in self.symbols_1h:
                            if sym not in market_data:
                                market_data[sym] = {}
                            market_data[sym]["1h"] = await self.ib_client.get_klines(sym, "1h")

                        logger.info("📊 Starting ORB Bot Cycle...")
                        await self.orb_bot.execute_cycle(market_data, self.orb_bot.strategies)
                        await asyncio.sleep(0.5)

                        # # BOS 5m FVG Bot
                        # logger.info("📊 Starting BOS 5m FVG Cycle...")
                        # await self.bos_fvg_bot.execute_cycle(market_data, self.bos_fvg_bot.strategies)
                        # await asyncio.sleep(0.5)

                        logger.info("📊 Starting BOS Bot Cycle...")
                        # BOS 1H Order Block Bot
                        await self.bos_ob_bot.execute_cycle(market_data, self.bos_ob_bot.strategies)

                        logger.info("✅ All bots completed sequentially")

                    except Exception as e:
                        logger.error(f"❌ Bot execution error: {e}", exc_info=True)

                    await self.sleep_until_next_candle()

                else:
                    # 💤 SLEEP MODE
                    last_logged_hour_summary = -1
                    if ny_now.hour != last_logged_hour:
                        msg = "💤 Market Closed. Sleeping..."
                        last_logged_hour = ny_now.hour
                        await self.orb_bot.telegram_debug_channel.send_message(msg)
                        logger.info(msg)
                        await asyncio.sleep(60)

                    # Maintenance: Check connection quietly
                    await self.ibkr_check_connection()
                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"MASTER LOOP CRITICAL ERROR: {e}")
                await asyncio.sleep(60)

    async def sleep_until_next_candle(self):
        """
        Waits until the next 5-minute boundary (e.g., 11:10:00, 11:15:00, 11:20:00).
        Aligns to clock time regardless of when bot started.
        Checks connection every 10 seconds while waiting.
        """
        try:
            now = datetime.now(ZoneInfo("America/New_York"))

            # Calculate next 5-minute boundary
            current_minute = now.minute
            current_second = now.second

            # How many minutes until next 5-minute mark?
            minutes_to_next = 5 - (current_minute % 5)
            if minutes_to_next == 5 and current_second == 0:
                # If we're exactly on a 5-minute mark, wait for next one
                minutes_to_next = 5

            # Calculate exact target time
            target = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_next)

            # How many seconds to wait?
            wait_seconds = (target - now).total_seconds()

            logger.info(f"⏳ Current time: {now.strftime('%H:%M:%S')}")
            logger.info(f"⏰ Next candle: {target.strftime('%H:%M:%S')} (waiting {wait_seconds:.1f}s)")

            # Wait with periodic connection checks
            check_interval = 10  # Check connection every 10 seconds
            elapsed = 0

            while elapsed < wait_seconds:
                # Calculate how long to sleep this iteration
                remaining = wait_seconds - elapsed
                sleep_time = min(check_interval, remaining)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    elapsed += sleep_time

                # Periodic connection check (but not on the last iteration)
                if elapsed < wait_seconds:
                    is_connected = await self.ibkr_check_connection()
                    if not is_connected:
                        logger.warning("⚠️ Connection lost during wait period")

            # Final timestamp log
            final_time = datetime.now(ZoneInfo("America/New_York"))
            logger.info(f"🎯 Candle time reached! Current time: {final_time.strftime('%H:%M:%S.%f')[:-3]}")

            return

        except Exception as e:
            logger.error(f"❌ Error in sleep_until_next_candle: {e}", exc_info=True)
            # Fallback: sleep 5 minutes
            await asyncio.sleep(300)
