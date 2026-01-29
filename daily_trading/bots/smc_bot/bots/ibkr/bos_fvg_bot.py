from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosFvgStrategy import BosFvgStrategy
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

CONFIG = []


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
