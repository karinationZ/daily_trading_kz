import asyncio
import traceback
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SMCBase import BIAS
from daily_trading.bots.smc_bot.strategy.ORBStrategy import ORBStrategy
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeAction
from daily_trading.brokers.IBKRClient import IBKRClient
from daily_trading.settings import IBKR_PORT, logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

DEPLOYMENT_CONFIG = {
    # ==========================================================================
    # GROUP 1: MNQ (The Profit Engine)
    # ==========================================================================

    # MNQ NY: The Aggressive Runner
    # Champ: SW15 | ATR1.8 | THR0.4 | RR2.5
    # Stats: Score 84.8 | PnL $2,810 | WR 61.9%
    ("MNQ", "NY"): {
        # Strategy Logic
        "swing_length": 15,
        "internal_length": 4,
        "fvg_offset": 1,  # Aggressive Momentum Entry
        "fvg_threshold": 0.4,
        "zone_precision_factor": 1.8,
        "tp_multiple": 2.5,  # Targeting Home Runs
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # MNQ LONDON: The Consistent Scalper
    # Champ: SW12 | ATR1.8 | THR0.2 | RR1.2
    # Stats: Score 73.6 | PnL $2,556 | WR 60.0%
    ("MNQ", "LONDON"): {
        # Strategy Logic
        "swing_length": 12,
        "internal_length": 5,
        "fvg_offset": 4,  # Deep Reversal Entry
        "fvg_threshold": 0.2,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,  # Scalping Target
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # ==========================================================================
    # GROUP 2: MES (The Stability Anchor)
    # ==========================================================================

    # MES LONDON: Structural Grinder
    # Champ: SW26 | ATR1.8 | THR0.25
    # Stats: Score 71.4 | PnL $1,185 | WR 54.55%
    ("MES", "LONDON"): {
        # Strategy Logic
        "swing_length": 26,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,  # Conservative
        # Dynamic Buffers (Increased for London Low Volatility)
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,  # Increased per your setup
        "tick_buffer_mult": 0.05  # Increased per your setup
    },

    # MES NY: Deep Value
    # Champ: SW60 | ATR1.8 | THR0.25
    # Stats: Score 71.5 | PnL $713 | WR 54.84%
    ("MES", "NY"): {
        # Strategy Logic
        "swing_length": 50,  # Very Deep Structure to filter NY Chop
        "internal_length": 5,
        "fvg_offset": 1,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # ==========================================================================
    # GROUP 3: COMMODITIES (MGC/SIL)
    # ==========================================================================

    # MGC LONDON: Gold Volatility
    # Champ: SW50 | ATR2.0 | THR0.2
    # Stats: Score 73.3 | PnL $2,385 | WR 53.23%
    ("MGC", "LONDON"): {
        # Strategy Logic
        "swing_length": 50,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.2,
        "zone_precision_factor": 2.0,
        "tp_multiple": 1.2,
        # Dynamic Buffers (Doubled for London Gold)
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.2,  # <--- INCREASED
        "tick_buffer_mult": 0.1  # <--- INCREASED
    },

    # SIL LONDON: Silver Patience
    # Champ: SW75 | ATR2.5 | THR0.25
    # Stats: Score 70.2 | PnL $1,096 | WR 51.43%
    ("SIL", "LONDON"): {
        # Strategy Logic
        "swing_length": 75,  # Massive Structure
        "internal_length": 10,
        "fvg_offset": 5,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 2.5,  # Wide Zones
        "tp_multiple": 1.2,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.2,  # <--- INCREASED
        "tick_buffer_mult": 0.1  # <--- INCREASED
    },

    # SIL NY: Momentum
    # Champ: SW60 | ATR2.5 | THR0.3
    # Stats: PnL $1,043 | WR 50.0%
    ("SIL", "NY"): {
        "swing_length": 60,
        "internal_length": 15,
        "fvg_offset": 1,
        "fvg_threshold": 0.3,
        "zone_precision_factor": 2.5,
        "tp_multiple": 1.5,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    }
}


class ORBBot(SMCBotBase):
    """
    Institutional ORB Portfolio Manager.

    Architecture:
    - 1 Render Worker / Process.
    - 1 IBKR Connection.
    - N Independent Strategy Instances (from DEPLOYMENT_CONFIG).

    Logic:
    1. Wake up every 5 minutes.
    2. Fetch data ONCE per symbol (Efficiency).
    3. Pass data to relevant Strategy instances (Isolation).
    4. Execute signals via IBKR.
    """

    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250):
        # Pass empty symbols to base because we manage them dynamically
        super().__init__(symbols=[], main_tf="5m", telegram_chat_id=telegram_chat_id)

        self.ib_client = IBKRClient(port=IBKR_PORT, client_id=1)
        self.strategies: List[ORBStrategy] = []
        self.active_symbols: List[str] = []
        self.risk = risk

        # 1. Initialize Portfolio from Golden Config
        self._initialize_strategies()

        logger.info(
            f"🤖 ORBBot Initialized with {len(self.strategies)} strategies across {len(self.active_symbols)} symbols.")

    def _initialize_strategies(self):
        """Builds strategy instances based on the Deployment Config."""
        unique_symbols = set()

        for (symbol, session_name), config in DEPLOYMENT_CONFIG.items():
            try:
                tz = ZoneInfo("America/New_York") if session_name == "NY" else ZoneInfo("Europe/London")

                strategy = ORBStrategy(
                    symbol=symbol,
                    timezone=tz,
                    main_tf="5m",
                    risk=self.risk,
                    # Injection from Config
                    swing_length=config.get("swing_length", 50),
                    internal_length=config.get("internal_length", 5),
                    fvg_offset=config.get("fvg_offset", 3),
                    zone_precision_factor=config.get("zone_precision_factor", 2.0),
                    tp_multiple=config.get("tp_multiple", 1.2),
                    # Dynamic Buffers
                    entry_buffer_mult=config["entry_buffer_mult"],
                    sl_buffer_mult=config["sl_buffer_mult"],
                    tick_buffer_mult=config["tick_buffer_mult"],
                    atr_length_buffer=14
                )
                self.strategies.append(strategy)
                unique_symbols.add(symbol)

            except Exception as e:
                logger.error(f"Failed to init strategy for {symbol} {session_name}: {e}")

        self.active_symbols = list(unique_symbols)

    async def handle_execution(self, strategy: ORBStrategy, analysis: SignalSummary):
        """
        Translates TradeActions into IBKR Orders.
        Uses a unique 'order_ref' tag per strategy instance to prevent collisions.
        """
        if not analysis or not analysis.trade_signal:
            return

        sig = analysis.trade_signal
        symbol = strategy.symbol
        # Unique Tag: MNQ_NY_ORB
        session_short = "NY" if "New_York" in str(strategy.timezone) else "LON"
        ext_tag = f"{symbol}_{session_short}_ORB"

        try:
            # --- ENTER ---
            if sig.action == TradeAction.ENTER:
                strategy.log(f"SIGNAL: ENTER {sig.direction.name} @ {sig.price}. Placing Bracket.")
                direction = "LONG" if sig.direction == BIAS.LONG else "SHORT"
                await self.ib_client.place_bracket_order(
                    symbol=symbol,
                    quantity=sig.size,
                    entry=sig.price,
                    sl=sig.sl, tp=sig.tp,
                    direction=direction,
                    order_ref=ext_tag
                )


            # --- CANCEL ---
            elif sig.action == TradeAction.CANCEL:
                strategy.log(f"SIGNAL: CANCEL. Cleaning book.")
                await self.ib_client.cancel_all_orders(order_ref=ext_tag)

            # --- EXIT ---
            elif sig.action == TradeAction.EXIT:
                # Check for EOD hard exit
                ts_now = datetime.now(strategy.timezone)
                s_info = SESSION_MAP[strategy.timezone.key]
                end_hour = s_info.get("end_hour", 16)
                end_minute = s_info.get("end_minute", 0)

                # Check if we are in the last 6 minutes of the session
                session_end = ts_now.replace(hour=end_hour, minute=end_minute, second=0)
                is_eod = ts_now >= (session_end - timedelta(minutes=6))

                if is_eod:
                    strategy.log(f"EOD LIQUIDATION: Market Exit.")
                    await self.ib_client.cancel_all_orders(order_ref=ext_tag)
                    side = "SELL" if sig.direction == BIAS.LONG else "BUY"
                    await self.ib_client.place_market_order(symbol, quantity=sig.size, side=side, order_ref=ext_tag)
                else:
                    # Normal TP/SL Hit logic
                    strategy.log(f"STRATEGY EXIT: Cleaning up.")
                    await self.ib_client.cancel_all_orders(order_ref=ext_tag)

        except Exception as e:
            strategy.log(f"EXECUTION FAILURE: {e}", level="error")

    async def sleep_until_next_interval(self, interval_minutes: int = 5):
        """Synced Sleep."""
        now = datetime.now()
        minutes_to_sleep = interval_minutes - (now.minute % interval_minutes)
        seconds_to_sleep = (minutes_to_sleep * 60) - now.second
        if seconds_to_sleep <= 0: seconds_to_sleep = interval_minutes * 60
        logger.debug(f"SYNC: Waiting {seconds_to_sleep}s for next candle.")
        await asyncio.sleep(seconds_to_sleep)

    @staticmethod
    def is_strategy_in_window(strategy: ORBStrategy) -> bool:
        """
        Checks if the strategy is within its operating window (Session +/- Buffers).
        Prevents running NY logic during Asian session, etc.
        """
        # 1. Get Setup Info
        tz_key = str(strategy.timezone)
        s_info = SESSION_MAP.get(tz_key)
        if not s_info:
            raise ValueError(f"Unknown timezone/session: {tz_key}")

        now = datetime.now(strategy.timezone)

        # 2. Define Buffers (in minutes)
        PRE_SESSION_BUFFER = 60  # Start 1 hour before session open
        POST_SESSION_BUFFER = 30  # End 30 mins after session close

        # 3. Construct Session Start/End times for Today
        start_time = now.replace(
            hour=s_info.get("start_hour", 9),
            minute=s_info.get("start_minute", 0),
            second=0, microsecond=0
        )
        end_time = now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0, microsecond=0
        )

        # 4. Apply Buffers
        active_start = start_time - timedelta(minutes=PRE_SESSION_BUFFER)
        active_end = end_time + timedelta(minutes=POST_SESSION_BUFFER)

        # 5. Check Window (simple comparison, no midnight wrapping needed)
        return active_start <= now <= active_end

    async def run_strategy_cycle(self, strategy: ORBStrategy, df):
        """
        Isolated execution context for a single strategy.
        """
        try:
            # 1. Run Analysis
            analysis = strategy.analyze(df)

            # 2. Report to Telegram (Only Significant Actions)
            # We filter out 'PENDING' and 'IN_TRADE' statuses to avoid spamming the channel every 5 minutes.
            if analysis.trade_signal:

                is_significant = analysis.trade_signal.action in [
                    TradeAction.ENTER,
                    TradeAction.EXIT,
                    TradeAction.CANCEL
                ]

                # Double Check: If it's CANCEL, only send if it was an active pending order getting killed (e.g. Session End)
                # If it's just "No Trade Found", we don't send.
                if is_significant:
                    action_emoji = {
                        TradeAction.ENTER: "🚀 ENTER",
                        TradeAction.EXIT: "💰 EXIT",
                        TradeAction.CANCEL: "🚫 CANCEL"
                    }.get(analysis.trade_signal.action, "⚠️ UPDATE")

                    msg = (
                        f"{action_emoji} {strategy.log_tag}\n"
                        f"Action: {analysis.trade_signal.action.name}\n"
                        f"Direction: {analysis.trade_signal.direction.name}\n"
                        f"Price: {analysis.trade_signal.price:.2f}\n"
                        f"SL: {analysis.trade_signal.sl:.2f} | TP: {analysis.trade_signal.tp:.2f}\n"
                        f"Time: {analysis.trade_signal.timestamp.strftime('%H:%M')}"
                    )

                    # Send Message with Chart (analysis.fig)
                    # Note: getattr handles cases where fig might be None
                    await self.telegram_bot.send_message(msg, getattr(analysis.trade_signal, 'fig', None))

            # 3. Execute Orders
            await self.handle_execution(strategy, analysis)

        except Exception as e:
            strategy.log(f"CYCLE ERROR: {e}", level="error")
            logger.error(traceback.format_exc())

    async def run(self):
        """
        Main Loop: Active from London Prep (01:45 NY) to NY Close (17:15 NY).
        """
        logger.info("🚀 ORB Portfolio Manager ACTIVATED.")

        try:
            await self.ib_client.connect()
            logger.info("✅ IBKR Connection Verified.")
        except Exception as e:
            logger.critical(f"FATAL: IBKR Connect Failed: {e}")
            return

        await self.telegram_bot.send_message("🚀 ORB Bot ACTIVATED.")

        while True:
            try:
                # Master Clock: NY Time
                ny_now = datetime.now(ZoneInfo("America/New_York"))
                current_hour = ny_now.hour + (ny_now.minute / 60)

                # 🟢 ACTIVE WINDOW: 01:45 AM to 17:15 PM (NY Time)
                if 1.75 <= current_hour <= 17.25:

                    # 1. Fetch Data
                    market_data = {}
                    for symbol in self.active_symbols:
                        try:
                            df = await self.ib_client.get_klines(symbol, "5m")
                            if not df.empty:
                                market_data[symbol] = df
                        except Exception as e:
                            logger.error(f"Data Fetch Error {symbol}: {e}")

                    # 2. Run Strategies
                    tasks = []
                    for strategy in self.strategies:
                        # --- THE FIX IS HERE ---
                        # We must check if THIS specific strategy is open (e.g. don't run NY logic at 4am)
                        if self.is_strategy_in_window(strategy) and strategy.symbol in market_data:
                            tasks.append(self.run_strategy_cycle(strategy, market_data[strategy.symbol]))

                    if tasks:
                        await asyncio.gather(*tasks)

                    await self.sleep_until_next_interval(5)

                else:
                    # 💤 SLEEP MODE
                    if ny_now.minute == 0:
                        logger.info(f"💤 MARKET CLOSED. Sleeping... (NY: {ny_now.strftime('%H:%M')})")

                    if not self.ib_client.ib.isConnected():
                        try:
                            await self.ib_client.connect()
                        except:
                            pass

                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"LOOP ERROR: {e}")
                await asyncio.sleep(60)
