import asyncio
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosOrderBlockStrategy import (
    BosOrderBlockStrategy,
)
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

# # 10. MES_ASIA - Score: 85.82 | WR: 68.42% | Edge: 35.09% | Trades: 19
# {
#     'name': 'MES_ASIA_sw50_int3_rr2.0_mit0.75_sl0.1',
#     'symbol': 'MES',
#     'tz_name': 'ASIA',
#     'sw': 50,
#     'int': 3,
#     'rr': 2.0,
#     'mit': 0.75,
#     'sl_buffer_mult': 0.1
# },

CONFIG = [
    # MES_NY - Score: 101.98 | WR: 54.29% | Edge: 20.96% | Trades: 105
    # SW:8 | INT:5 | MIT:0.5 | SL:0.5
    {'name': 'MES_NY_sw8_int5_rr2.0_mit0.5_sl0.5', 'symbol': 'MES', 'tz_name': 'NY', 'sw': 8, 'int': 5, 'rr': 2.0, 'mit': 0.5, 'sl_buffer_mult': 0.5},
    # MES_LONDON - Score: 112.96 | WR: 61.54% | Edge: 32.97% | Trades: 91
    # SW:8 | INT:3 | MIT:0.1 | SL:0.1
    {
        'name': 'MES_LONDON_sw8_int3_rr2.5_mit0.1_sl0.1',
        'symbol': 'MES',
        'tz_name': 'LONDON',
        'sw': 8,
        'int': 3,
        'rr': 2.5,
        'mit': 0.1,
        'sl_buffer_mult': 0.1,
    },
    # MNQ_NY - Score: 101 | WR: 54.74% | Edge: 21.4% | Trades: 95
    # SW:50 | INT:5 | MIT:0.1 | SL:0.5
    {
        'name': 'MNQ_NY_sw50_int5_rr2.0_mit0.1_sl0.5',
        'symbol': 'MNQ',
        'tz_name': 'NY',
        'sw': 50,
        'int': 5,
        'rr': 2.0,
        'mit': 0.1,
        'sl_buffer_mult': 0.5,
    },
    # MNQ_LONDON - Score: 64.14 | WR: 58.33% | Edge: 29.76% | Trades: 24
    # SW:50 | INT:5 | MIT:0.75 | SL:1.0
    {
        'name': 'MNQ_LONDON_sw50_int5_rr2.5_mit0.75_sl1.0',
        'symbol': 'MNQ',
        'tz_name': 'LONDON',
        'sw': 50,
        'int': 5,
        'rr': 2.5,
        'mit': 0.75,
        'sl_buffer_mult': 1.0,
    },
    # MGC_NY - Score: 93.8 | WR: 71.43% | Edge: 31.4% | Trades: 70
    # SW:12 | INT:3 | MIT:0.75 | SL:1.0
    {
        'name': 'MGC_NY_sw12_int3_rr1.5_mit0.75_sl1.0',
        'symbol': 'MGC',
        'tz_name': 'NY',
        'sw': 12,
        'int': 3,
        'rr': 1.5,
        'mit': 0.75,
        'sl_buffer_mult': 1.0,
    },
    # MGC_ASIA - Score: 103.69 | WR: 65.12% | Edge: 19.67% | Trades: 43
    # SW:8 | INT:3 | MIT:0.75 | SL:0.1
    {
        'name': 'MGC_ASIA_sw8_int3_rr1.2_mit0.75_sl0.1',
        'symbol': 'MGC',
        'tz_name': 'ASIA',
        'sw': 8,
        'int': 3,
        'rr': 1.2,
        'mit': 0.75,
        'sl_buffer_mult': 0.1,
    },
    # MGC_LONDON - Score: 80 | WR: 58| Edge: 18% | Trades: 43
    # SW:24 | INT:3 | MIT:0.5 | SL:1.5
    {
        'name': 'MGC_LONDON_sw24_int3_rr1.5_mit0.5_sl1.5',
        'symbol': 'MGC',
        'tz_name': 'LONDON',
        'sw': 24,
        'int': 3,
        'rr': 1.5,
        'mit': 0.5,
        'sl_buffer_mult': 1.5,
    },
    # MBT_LONDON - Score: 105.34 | WR: 57.14% | Edge: 17.14% | Trades: 49
    # SW:45 | INT:5 | MIT:0.1 | SL:1.2
    {
        'name': 'MBT_LONDON_sw45_int5_rr1.5_mit0.1_sl1.2',
        'symbol': 'MBT',
        'tz_name': 'LONDON',
        'sw': 45,
        'int': 5,
        'rr': 1.5,
        'mit': 0.1,
        'sl_buffer_mult': 1.2,
    },
]


class BosOrderBlockBot(SMCBotBase):
    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        super().__init__(
            symbols=[],
            main_tf="5m",
            telegram_chat_id=telegram_chat_id,
            ib_client=ib_client,
        )
        self.strategies: List[BosOrderBlockStrategy] = []
        self.active_symbols = []
        self.risk = risk
        self._initialize_strategies()
        logger.info(f"🤖 BOS 1H OB Bot Manager Ready: {len(self.strategies)} strategies.")

    def _initialize_strategies(self):
        unique_symbols = set()
        for cfg in CONFIG:
            symbol = cfg["symbol"]
            session_name = cfg["tz_name"]
            if session_name == "NY":
                tz = ZoneInfo("America/New_York")
            elif session_name == "LONDON":
                tz = ZoneInfo("Europe/London")
            else:
                tz = ZoneInfo("Asia/Tokyo")
            try:
                strategy = BosOrderBlockStrategy(
                    symbol=symbol,
                    timezone=tz,
                    main_tf="1h",
                    swing_length=cfg["sw"],
                    internal_length=cfg["int"],
                    risk=self.risk,
                    tp_multiple=cfg["rr"],
                    entry_depth=cfg["mit"],
                    sl_buffer_mult=cfg["sl_buffer_mult"],
                    priority=1,
                )
                self.strategies.append(strategy)
                unique_symbols.add(symbol)
            except Exception as e:
                logger.error(f"Failed to init SD Strategy {symbol}: {e}")
        self.active_symbols = list(unique_symbols)

    async def execute_cycle(self, market_data: dict, blocked_symbols: set, current_positions, open_orders):
        """
        Called by MasterBot.
        Needs both 5m and 1h data.
        """
        logger.info(f"🔵 {self.__class__.__name__}.execute_cycle() STARTED [VERSION 2.0 WITH 120s TIMEOUT]")

        tasks = []
        for strategy in self.strategies:
            sym_data = market_data.get(strategy.symbol, {})
            df_5m = sym_data.get("5m")
            df_1h = sym_data.get("1h")

            # 3. Validate Data
            if df_5m is None or df_5m.empty or df_1h is None or df_1h.empty:
                continue

            curr = df_5m.iloc[-1]
            ts_now = curr.name.tz_convert(self.timezone)
            is_session_started = strategy.is_pre_session_started(ts_now, strategy.timezone)

            if self.is_strategy_in_window(strategy) and is_session_started:
                await self.sync_state(strategy, current_positions, open_orders)
                # Run Logic
                tasks.append(self.run_strategy_cycle(strategy, df_5m, df_1h))
            else:
                strategy.log.info(f"💤 {strategy.tz_short} Session has not started yet!")

        if tasks:
            logger.info(f"🔧 Running {len(tasks)} strategy tasks...")
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=180)

                # Check each result
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Task {i} failed with exception: {result}", exc_info=result)
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
