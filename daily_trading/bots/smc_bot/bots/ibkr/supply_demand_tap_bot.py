import asyncio
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeStatus
from daily_trading.bots.smc_bot.strategy.SupplyDemandZoneTapStrategy import SupplyDemandZoneTapStrategy
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

# Define your configuration here
SD_CONFIG = {
    # --------------------------------------------------------------------------
    # GROUP 1: MES (Micro S&P 500)
    # --------------------------------------------------------------------------

    # MES LONDON
    # Stats: WR 65.22% | PnL 2050 | Score 75.22
    ("MES", "LONDON"): {
        "sw": 35,  # swing_length
        "int": 3,  # internal_length
        "off": 3,  # fvg_offset
        "atr": 1.8,  # zone_precision_factor
        "thr": 0.15,  # fvg_threshold
        "eb_mult": 2.0,  # entry_buffer_mult
        "sl_atr": 1.5,  # sl_atr_multiplier
        "rr": 1.5  # tp_multiple
    },

    # MES NY
    # Stats: WR 51.22% | PnL 2715 | Score 76.48
    ("MES", "NY"): {
        "sw": 50,
        "int": 5,
        "off": 3,
        "atr": 1.5,
        "thr": 0.15,
        "eb_mult": 1.5,
        "sl_atr": 1.5,
        "rr": 2.0
    },

    # --------------------------------------------------------------------------
    # GROUP 2: MNQ (Micro Nasdaq)
    # --------------------------------------------------------------------------

    # MNQ LONDON
    # Stats: WR 50.0% | PnL 2483 | Score 76.0
    ("MNQ", "LONDON"): {
        "sw": 35,
        "int": 3,
        "off": 3,
        "atr": 2.0,
        "thr": 0.15,
        "eb_mult": 2.0,
        "sl_atr": 2.0,
        "rr": 2.0
    },
    ("MNQ", "ASIA"): {
        "sw": 35,
        "int": 3,
        "off": 3,
        "atr": 2.0,
        "thr": 0.15,
        "eb_mult": 2.0,
        "sl_atr": 2.0,
        "rr": 2.0
    },

    # MNQ NY
    # Stats: WR 69.23% | PnL 1406 | Score 75.11
    ("MNQ", "NY"): {
        "sw": 8,  # Very fast swing for NY volatility
        "int": 5,
        "off": 2,
        "atr": 1.5,
        "thr": 0.25,
        "eb_mult": 2.0,
        "sl_atr": 2.0,
        "rr": 1.2  # Scalping Target
    },

    # --------------------------------------------------------------------------
    # GROUP 3: MGC (Micro Gold)
    # --------------------------------------------------------------------------

    # MGC LONDON
    # Stats: WR 55.77% | PnL 2831 | Score 74.30
    ("MGC", "LONDON"): {
        "sw": 8,
        "int": 3,
        "off": 3,
        "atr": 1.8,
        "thr": 0.15,
        "eb_mult": 1.5,
        "sl_atr": 1.5,
        "rr": 1.5
    },
    ("MGC", "ASIA"): {
        "sw": 8,
        "int": 3,
        "off": 3,
        "atr": 1.8,
        "thr": 0.15,
        "eb_mult": 1.5,
        "sl_atr": 1.5,
        "rr": 1.5
    },

    # MGC NY
    # Stats: WR 73.68% | PnL 1168 | Score 76.03
    ("MGC", "NY"): {
        "sw": 50,
        "int": 5,
        "off": 3,
        "atr": 1.5,
        "thr": 0.25,
        "eb_mult": 1.5,
        "sl_atr": 2.0,
        "rr": 1.2
    }
}


class SupplyDemandTapBot(SMCBotBase):
    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        super().__init__(symbols=[], main_tf="5m", telegram_chat_id=telegram_chat_id, ib_client=ib_client)
        self.strategies: List[SupplyDemandZoneTapStrategy] = []
        self.active_symbols = []
        self.risk = risk
        self._initialize_strategies()
        logger.info(f"🤖 SupplyDemandBot Manager Ready: {len(self.strategies)} strategies.")

    def _initialize_strategies(self):
        unique_symbols = set()
        for (symbol, session_name), cfg in SD_CONFIG.items():
            if session_name == "NY":
                tz = ZoneInfo("America/New_York")
            elif session_name == "LONDON":
                tz = ZoneInfo("Europe/London")
            else:
                tz = ZoneInfo("Asia/Tokyo")
            try:
                strategy = SupplyDemandZoneTapStrategy(
                    symbol=symbol,
                    timezone=tz,
                    risk=self.risk,
                    tp_multiple=cfg["rr"],
                    fvg_based_only=True,
                    swing_length=cfg["sw"],
                    internal_length=cfg["int"],
                    fvg_offset=cfg["off"],
                    zone_precision_factor=cfg["atr"],
                    fvg_threshold=cfg["thr"],
                    sl_atr_multiplier=cfg["sl_atr"],
                    eb_mult=cfg["eb_mult"]
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
        tasks = []
        for strategy in self.strategies:
            # --- 1. STATE RESTORATION ---
            # Attempt to restore state from IBKR (in case of crash/restart)
            self.sync_state(strategy, current_positions, open_orders)

            # --- 2. CONFLICT CHECK ---
            # If the strategy is ACTIVE (Restored), we allow it to run even if 'blocked'
            # (because it is blocked by itself).
            # If it is INACTIVE and blocked, we skip it.
            if strategy.symbol in blocked_symbols and strategy.trade_status == TradeStatus.NONE:
                # Only log occasionally to avoid spam
                # logger.debug(f"Skipping {strategy.symbol} (Busy in another strategy)")
                continue

            # 2. Extract Data
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
                # 4. Run Logic
                tasks.append(self.run_strategy_cycle(strategy, df_5m, df_1h))
            else:
                strategy.log.info(f"💤 {strategy.tz_short} Session has not started yet!")

        if tasks:
            await asyncio.gather(*tasks)
