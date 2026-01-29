from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosFvgStrategy import BosFvgStrategy
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

CONFIG = []


class BosOrderBlockBot(SMCBotBase):
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
