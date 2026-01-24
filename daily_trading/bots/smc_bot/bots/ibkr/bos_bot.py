import asyncio
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosOrderBlockStrategy import (
    BosOrderBlockStrategy,
)
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeStatus
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

# Define your configuration here
# --- MES ASIA | WinRate 70.0% | Score 88.6 | SUGGEST EXCLUDE (Weak Stats) | Low Sample (20 trades)
# TRADER NOTE: Sample size < 25 is too risky for live money.
# High win rate likely due to luck/low volatility in Asia.
# ("MES", "ASIA"): {
#     "swing_length": 50,
#     "internal_length": 3,
#     "entry_depth": 0.75,
#     "tp_multiple": 2.0,
#     "sl_buffer": 0.1
# },
# --- MNQ LONDON | WinRate 43.4% | Score 86.4 | CAUTION | Low WR (43.4%)
# TRADER NOTE: 43% WR is psychologically hard to trade manually,
# but math works (2.5R). Ensure bot can handle drawdown streaks.
# ("MNQ", "LONDON"): {
#     "swing_length": 12,
#     "internal_length": 3,
#     "entry_depth": 0.25,
#     "tp_multiple": 2.5,
#     "sl_buffer": 0.1
# },
CONFIG = {
    # --- MBT LONDON | WinRate 54.72% | Score 107.4 | APPROVED | Consistent Performer
    ("MBT", "LONDON"): {"swing_length": 45, "internal_length": 5, "entry_depth": 0.1, "tp_multiple": 1.5, "sl_buffer": 1.2},
    # --- MES LONDON | WinRate 51.35% | Score 107.4 | APPROVED | Strong Edge
    ("MES", "LONDON"): {"swing_length": 24, "internal_length": 3, "entry_depth": 0.1, "tp_multiple": 2.5, "sl_buffer": 0.1},
    # --- MES NY | WinRate 62.22% | Score 100.4 | APPROVED | Solid Bread & Butter
    ("MES", "NY"): {"swing_length": 8, "internal_length": 5, "entry_depth": 0.75, "tp_multiple": 1.2, "sl_buffer": 1.0},
    # --- MGC ASIA | WinRate 58.0% | Score 93.5 | APPROVED | Decent Edge
    ("MGC", "ASIA"): {"swing_length": 8, "internal_length": 3, "entry_depth": 0.75, "tp_multiple": 1.2, "sl_buffer": 0.1},
    # --- MGC NY | WinRate 55.74% | Score 105.0 | APPROVED | Strong Edge
    ("MGC", "NY"): {"swing_length": 24, "internal_length": 5, "entry_depth": 0.75, "tp_multiple": 2.0, "sl_buffer": 0.1},
    # --- MNQ NY | WinRate 63.46% | Score 107.9 | APPROVED | High Confidence Sample
    # TRADER NOTE: This is a Portfolio Anchor. High volume, proven edge.
    ("MNQ", "NY"): {"swing_length": 12, "internal_length": 5, "entry_depth": 0.25, "tp_multiple": 1.2, "sl_buffer": 0.1},
    # --- SIL ASIA | WinRate 80.95% | Score 109.1 | APPROVED | Strong Edge
    # TRADER NOTE: The Stat Winner. Exceptional edge (35%+).
    # Must trade this.
    ("SIL", "ASIA"): {"swing_length": 8, "internal_length": 5, "entry_depth": 0.75, "tp_multiple": 1.2, "sl_buffer": 0.1},
    # --- SIL LONDON | WinRate 59.52% | Score 101.3 | APPROVED | Strong Edge
    ("SIL", "LONDON"): {"swing_length": 50, "internal_length": 5, "entry_depth": 0.5, "tp_multiple": 2.0, "sl_buffer": 0.1},
    # --- SIL NY | WinRate 63.64% | Score 106.9 | APPROVED | Strong Edge
    ("SIL", "NY"): {"swing_length": 50, "internal_length": 5, "entry_depth": 0.75, "tp_multiple": 1.5, "sl_buffer": 0.1},
}


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
        for (symbol, session_name), cfg in CONFIG.items():
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
                    swing_length=cfg["swing_length"],
                    internal_length=cfg["internal_length"],
                    risk=self.risk,
                    tp_multiple=cfg["tp_multiple"],
                    entry_depth=cfg["entry_depth"],
                    sl_buffer_mult=cfg["sl_buffer"],
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
                strategy.log.warning(f"Skipping {strategy.symbol} (Busy in another strategy)")
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
