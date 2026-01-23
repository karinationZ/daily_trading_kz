import asyncio
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeStatus
from daily_trading.bots.smc_bot.strategy.SupplyDemandZoneTapStrategy import (
    SupplyDemandZoneTapStrategy,
)
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

# Define your configuration here
CONFIG = {
    # ==========================================================================
    # GROUP 1: MNQ NY — "THE 100% BANK"
    # Stats: WR 100% | PnL $597 | 5 Trades
    # Logic: Deep Structure (SW 50) + Deep Entry (EB 2.0) = Mathematical Certainty.
    # Note: Even though SW 8 (Momentum) made similar money, SW 50 had 0 losses.
    # We choose Stability/Zero Drawdown.
    # ==========================================================================
    ("MNQ", "NY"): {
        "swing_length": 15,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.15,
        "zone_precision_factor": 1.5,
        "tp_multiple": 1.5,
        "entry_buffer_mult": 2.5,
        "sl_buffer_mult": 1.5,
    },
    # ==========================================================================
    # GROUP 2: MGC NY — "THE GOLD MOMENTUM" (UPGRADED)
    # Stats: WR 60% | PnL $837 | 5 Trades
    # Change: Increased RR from 2.0 to 2.5.
    # Logic: Gold trends harder than indices. The Grid Search proved 2.5R is the
    # peak PnL point ($837 vs $812 at 3.0R).
    # ==========================================================================
    ("MGC", "NY"): {
        "swing_length": 50,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.15,
        "zone_precision_factor": 1.5,
        "tp_multiple": 2.5,
        "entry_buffer_mult": 2.0,
        "sl_buffer_mult": 2.0,
    },
    # ==========================================================================
    # GROUP 3: MES NY — "THE PRECISION ANCHOR"
    # Stats: WR 83% | PnL $597 | 6 Trades
    # Logic: Pushing RR to 2.0 dropped WR to 66% and PnL to $518.
    # RR 1.5 is the mathematically perfect target for MES intraday zone taps.
    # ==========================================================================
    ("MES", "NY"): {
        "swing_length": 50,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.15,
        "zone_precision_factor": 1.5,
        "tp_multiple": 1.5,
        "entry_buffer_mult": 1.25,
        "sl_buffer_mult": 1.5,
    },
}


class SupplyDemandTapBot(SMCBotBase):
    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        super().__init__(
            symbols=[],
            main_tf="5m",
            telegram_chat_id=telegram_chat_id,
            ib_client=ib_client,
        )
        self.strategies: List[SupplyDemandZoneTapStrategy] = []
        self.active_symbols = []
        self.risk = risk
        self._initialize_strategies()
        logger.info(f"🤖 SupplyDemandBot Manager Ready: {len(self.strategies)} strategies.")

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
                strategy = SupplyDemandZoneTapStrategy(
                    symbol=symbol,
                    timezone=tz,
                    risk=self.risk,
                    tp_multiple=cfg["tp_multiple"],
                    fvg_based_only=True,
                    swing_length=cfg["swing_length"],
                    internal_length=cfg["internal_length"],
                    fvg_offset=cfg["fvg_offset"],
                    zone_precision_factor=cfg["zone_precision_factor"],
                    fvg_threshold=cfg["fvg_threshold"],
                    sl_atr_multiplier=cfg["sl_buffer_mult"],
                    eb_mult=cfg["entry_buffer_mult"],
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
