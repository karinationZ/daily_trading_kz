from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosFvgStrategy import BosFvgStrategy
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

CONFIG = [
    # NY
    {'name': 'MNQ_NY_rr1.2_mit0.95_sl0.8', 'symbol': 'MNQ', 'tz_name': 'NY', 'rr': 1.2, 'mit': 0.95, 'sl_buffer_mult': 0.8},
    {'name': 'MES_NY_rr1.2_mit0.95_sl0.5', 'symbol': 'MES', 'tz_name': 'NY', 'rr': 1.2, 'mit': 0.95, 'sl_buffer_mult': 0.5},
    {'name': 'MGC_NY_rr1.2_mit0.5_sl0.8', 'symbol': 'MGC', 'tz_name': 'NY', 'rr': 1.2, 'mit': 0.5, 'sl_buffer_mult': 0.8},
    {'name': 'MBT_NY_rr1.4_mit0.75_sl0.8', 'symbol': 'MBT', 'tz_name': 'NY', 'rr': 1.4, 'mit': 0.75, 'sl_buffer_mult': 0.8},
    # ASIA
    {'name': 'MES_ASIA_rr1.8_mit0.75_sl0.1', 'symbol': 'MES', 'tz_name': 'ASIA', 'rr': 1.8, 'mit': 0.75, 'sl_buffer_mult': 0.1},
    {'name': 'MGC_ASIA_rr1.2_mit0.5_sl0.8', 'symbol': 'MGC', 'tz_name': 'ASIA', 'rr': 1.2, 'mit': 0.5, 'sl_buffer_mult': 0.8},
    {'name': 'MNQ_ASIA_rr1.2_mit0.75_sl0.2', 'symbol': 'MNQ', 'tz_name': 'ASIA', 'rr': 1.2, 'mit': 0.75, 'sl_buffer_mult': 0.2},
    {'name': 'MBT_ASIA_rr1.2_mit0.95_sl0.5', 'symbol': 'MBT', 'tz_name': 'ASIA', 'rr': 1.2, 'mit': 0.95, 'sl_buffer_mult': 0.5},
    # LONDON
    {'name': 'MES_LONDON_rr1.2_mit0.95_sl0.8', 'symbol': 'MES', 'tz_name': 'LONDON', 'rr': 1.2, 'mit': 0.95, 'sl_buffer_mult': 0.8},
    {'name': 'MNQ_LONDON_rr1.2_mit0.75_sl0.5', 'symbol': 'MNQ', 'tz_name': 'LONDON', 'rr': 1.2, 'mit': 0.75, 'sl_buffer_mult': 0.5},
    {'name': 'MGC_LONDON_rr1.4_mit0.5_sl0.5', 'symbol': 'MGC', 'tz_name': 'LONDON', 'rr': 1.4, 'mit': 0.5, 'sl_buffer_mult': 0.5},
    {'name': 'MBT_LONDON_rr1.2_mit0.95_sl0.5', 'symbol': 'MBT', 'tz_name': 'LONDON', 'rr': 1.2, 'mit': 0.95, 'sl_buffer_mult': 0.5},
]


class BosFvgBot(SMCBotBase):
    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        super().__init__(
            symbols=[],
            main_tf="5m",
            telegram_chat_id=telegram_chat_id,
            ib_client=ib_client,
        )
        self.strategies: List[BosFvgStrategy] = []
        self.active_symbols = []
        self.risk = risk
        self._initialize_strategies()
        logger.info(f"🤖 BOS FVG OB Bot Manager Ready: {len(self.strategies)} strategies.")

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
                strategy = BosFvgStrategy(
                    symbol=cfg["symbol"],
                    timezone=tz,
                    main_tf="5m",
                    risk=self.risk,
                    tp_multiple=cfg["rr"],
                    fvg_mitigation_ratio=cfg["mit"],
                    sl_buffer_mult=cfg["sl_buffer_mult"],
                    priority=2,
                )
                self.strategies.append(strategy)
                unique_symbols.add(symbol)
            except Exception as e:
                logger.error(f"Failed to init SD Strategy {symbol}: {e}")
        self.active_symbols = list(unique_symbols)
