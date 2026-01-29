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
                    priority=3,
                )
                self.strategies.append(strategy)
                unique_symbols.add(symbol)

            except Exception as e:
                logger.error(f"Failed to init strategy for {symbol} {session_name}: {e}")

        self.active_symbols = list(unique_symbols)
