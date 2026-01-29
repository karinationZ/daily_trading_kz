import asyncio
import traceback
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.ORBStrategy import ORBStrategy
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

CONFIG = [
    # MNQ NY: Stats: PnL $2,181 | WR 40% | RR 5.0
    {
        'name': 'MNQ_NY_sw15_int4_off1_atr1.8_rr5.0_thr0.25_eb0.0_sl0.15',
        'symbol': 'MNQ',
        'tz_name': 'NY',
        'sw': 15,
        'int': 4,
        'off': 1,
        'atr': 1.8,
        'thr': 0.25,
        'rr': 5.0,
        'entry_buffer_mult': 0.0,
        'sl_buffer_mult': 0.15,
        'tick_buffer_mult': 0.05,
    },
    # MGC NY: Stats: PnL $3,056 | WR 69% | RR 2.0
    # SW:28 | INT:3 | OFF:1 | ATR:2.3 | THR:0.25 | SL:0.12 | EB:0.0
    {
        'name': 'MGC_NY_sw28_int3_off1_atr2.3_rr2.0_thr0.25_eb0.02_sl0.12',
        'symbol': 'MGC',
        'tz_name': 'NY',
        'sw': 28,
        'int': 3,
        'rr': 2.0,
        'entry_buffer_mult': 0.0,
        'sl_buffer_mult': 0.12,
        'off': 1,
        'atr': 2.3,
        'thr': 0.25,
        'tick_buffer_mult': 0.05,
    },
]


class ORBBot(SMCBotBase):
    """
    Institutional ORB Portfolio Manager.
    Refactored for MasterBot Architecture.
    """

    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        # Pass the shared ib_client to the base
        super().__init__(
            symbols=[],
            main_tf="5m",
            telegram_chat_id=telegram_chat_id,
            ib_client=ib_client,
        )

        self.strategies: List[ORBStrategy] = []
        self.active_symbols: List[str] = []
        self.risk = risk

        self._initialize_strategies()  # (Keep your existing init logic)
        logger.info(f"🤖 ORB Bot Manager Ready: {len(self.strategies)} strategies.")

    def _initialize_strategies(self):
        """Builds strategy instances based on the Deployment Config."""
        unique_symbols = set()

        for config in CONFIG:
            symbol = config["symbol"]
            session_name = config["tz_name"]
            try:
                if session_name == "NY":
                    tz = ZoneInfo("America/New_York")
                elif session_name == "LONDON":
                    tz = ZoneInfo("Europe/London")
                else:
                    tz = ZoneInfo("Asia/Tokyo")

                strategy = ORBStrategy(
                    symbol=symbol,
                    timezone=tz,
                    main_tf="5m",
                    risk=self.risk,
                    # Injection from Config
                    swing_length=config["sw"],
                    internal_length=config["int"],
                    fvg_offset=config["off"],
                    zone_precision_factor=config["atr"],
                    tp_multiple=config["rr"],
                    # Dynamic Buffers
                    entry_buffer_mult=config["entry_buffer_mult"],
                    sl_buffer_mult=config["sl_buffer_mult"],
                    tick_buffer_mult=config["tick_buffer_mult"],
                    priority=2,
                )
                self.strategies.append(strategy)
                unique_symbols.add(symbol)

            except Exception as e:
                logger.error(f"Failed to init strategy for {symbol} {session_name}: {e}")

        self.active_symbols = list(unique_symbols)

    async def execute_cycle(self, market_data: dict):
        """
        The Entry Point for MasterBot.

         It takes the centralized data, filters it, and triggers
         'run_strategy_cycle' for valid strategies.

         market_data: Dict structure { "MNQ": {"5m": df, "1h": df} }
         blocked_symbols: Set of symbols active in other bots (conflict prevention).
        """
        logger.info(f"🔵 {self.__class__.__name__}.execute_cycle() STARTED")

        tasks = []
        for strategy in self.strategies:
            # 2. Extract Data: ORBBot only needs "5m" data.
            sym_data = market_data.get(strategy.symbol, {})
            df_5m = sym_data.get("5m")

            # 3. Validation: Check if data exists and strategy is in time window.
            if df_5m is not None and not df_5m.empty:
                curr = df_5m.iloc[-1]
                ts_now = curr.name.tz_convert(self.timezone)
                is_session_started = strategy.is_pre_session_started(ts_now, strategy.timezone)
                if self.is_strategy_in_window(strategy) and is_session_started:
                    # Call the EXISTING logic function
                    coro = self.run_strategy_cycle(strategy, df_5m)
                    tasks.append(asyncio.wait_for(coro, timeout=30))  # Each strategy gets 30s max
                else:
                    strategy.log.info(f"💤 {strategy.tz_short} Session has not started yet!")
        if tasks:
            logger.info(f"🔧 Running {len(tasks)} strategy tasks...")
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=180)

                # Check each result
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # 1. Identify WHICH strategy failed
                        strategy_name = self.strategies[i].symbol if i < len(self.strategies) else "Unknown"

                        # 2. Extract the full stack trace as a string
                        tb_lines = traceback.format_exception(type(result), result, result.__traceback__)
                        tb_text = "".join(tb_lines)

                        # 3. Log the full text clearly
                        logger.error(f"❌ STRATEGY FAILURE: {strategy_name} (Task {i})\n{tb_text}")
                    else:
                        logger.info(f"✅ Task {i} completed successfully")

            except asyncio.TimeoutError:
                logger.error(f"⏰ TIMEOUT: {self.__class__.__name__} tasks took longer than 3 min!")
                raise
            except Exception as e:
                logger.error(f"❌ Unexpected error in gather: {e}", exc_info=True)
                raise
        else:
            logger.info(f"⚪ No active strategies in {self.__class__.__name__}")

        logger.info(f"🟢 {self.__class__.__name__}.execute_cycle() COMPLETED")
