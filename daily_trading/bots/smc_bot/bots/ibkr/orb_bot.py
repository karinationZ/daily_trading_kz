import asyncio
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.ORBStrategy import ORBStrategy
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeStatus
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

ORB_CONFIG = {
    # ==========================================================================
    # GROUP 1: MNQ (THE MOMENTUM ENGINE)
    # ==========================================================================

    # MNQ NY: The "5R" Sniper
    # Stats: PnL $2,181 | WR 40% | RR 5.0
    # Change: RR increased to 5.0. SL Buffer increased to 0.15 to accommodate volatility.
    ("MNQ", "NY"): {
        "swing_length": 15,  # Tied with 12/18, kept 15 for consistency
        "internal_length": 4,
        "fvg_offset": 1,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 1.8,
        "tp_multiple": 5.0,  # <--- UPGRADED from 4.25
        "entry_buffer_mult": 0.0,
        "sl_buffer_mult": 0.15,  # <--- INCREASED from 0.10 for stability
        "tick_buffer_mult": 0.05
    },

    # MNQ LONDON: The 62% Scalper
    # Stats: PnL $1,821 | WR 62% | RR 1.0
    # Status: Unchanged. Leaderboard confirms these exact settings.
    ("MNQ", "LONDON"): {
        "swing_length": 12,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.2,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.0,
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.15,
        "tick_buffer_mult": 0.05
    },

    # MNQ ASIA: The Hyper Scalper
    # Stats: PnL $1,423 | WR 55.8% | RR 1.0
    # Status: Settings confirmed. PnL reflects current data snapshot.
    ("MNQ", "ASIA"): {
        "swing_length": 5,
        "internal_length": 5,
        "fvg_offset": 5,
        "fvg_threshold": 0.15,
        "zone_precision_factor": 2.0,
        "tp_multiple": 1.0,
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.15,
        "tick_buffer_mult": 0.05
    },

    # ==========================================================================
    # GROUP 2: MGC (GOLD) (THE CASH MACHINE)
    # ==========================================================================

    # MGC LONDON: The Golden King
    # Stats: PnL $3,528 | WR 51.6% | RR 1.6
    # Status: Performance updated ($3528 vs old $3417). No param changes needed.
    ("MGC", "LONDON"): {
        "swing_length": 24,
        "internal_length": 4,
        "fvg_offset": 3,
        "fvg_threshold": 0.2,
        "zone_precision_factor": 2.3,
        "tp_multiple": 1.6,
        "entry_buffer_mult": 0.02,
        "sl_buffer_mult": 0.15,
        "tick_buffer_mult": 0.05
    },

    # MGC NY: The Elite 2.0
    # Stats: PnL $3,056 | WR 69% | RR 2.0
    # Change: Major overhaul. RR up to 2.0. Offset tightened to 1. Threshold 0.25.
    ("MGC", "NY"): {
        "swing_length": 28,  # Selected 28 over 24 for stability at higher RR
        "internal_length": 3,
        "fvg_offset": 1,  # <--- TIGHTENED from 2
        "fvg_threshold": 0.25,  # <--- INCREASED from 0.2
        "zone_precision_factor": 2.3,  # <--- ADJUSTED from 2.5
        "tp_multiple": 2.0,  # <--- UPGRADED from 1.7
        "entry_buffer_mult": 0.02,
        "sl_buffer_mult": 0.12,
        "tick_buffer_mult": 0.05
    },

    # MGC ASIA: The Steady Earner
    # Stats: PnL $2,366 | WR 51.7% | RR 1.4
    # Status: Unchanged.
    ("MGC", "ASIA"): {
        "swing_length": 24,
        "internal_length": 4,
        "fvg_offset": 3,
        "fvg_threshold": 0.1,
        "zone_precision_factor": 2.0,
        "tp_multiple": 1.4,
        "entry_buffer_mult": 0.02,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    }
}


class ORBBot(SMCBotBase):
    """
    Institutional ORB Portfolio Manager.
    Refactored for MasterBot Architecture.
    """

    def __init__(self, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID, risk=250, ib_client=None):
        # Pass the shared ib_client to the base
        super().__init__(symbols=[], main_tf="5m", telegram_chat_id=telegram_chat_id, ib_client=ib_client)

        self.strategies: List[ORBStrategy] = []
        self.active_symbols: List[str] = []
        self.risk = risk

        self._initialize_strategies()  # (Keep your existing init logic)
        logger.info(f"🤖 ORBBot Manager Ready: {len(self.strategies)} strategies.")

    def _initialize_strategies(self):
        """Builds strategy instances based on the Deployment Config."""
        unique_symbols = set()

        for (symbol, session_name), config in ORB_CONFIG.items():
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
                    swing_length=config.get("swing_length", 50),
                    internal_length=config.get("internal_length", 5),
                    fvg_offset=config.get("fvg_offset", 3),
                    zone_precision_factor=config.get("zone_precision_factor", 2.0),
                    tp_multiple=config.get("tp_multiple", 1.2),
                    # Dynamic Buffers
                    entry_buffer_mult=config["entry_buffer_mult"],
                    sl_buffer_mult=config["sl_buffer_mult"],
                    tick_buffer_mult=config["tick_buffer_mult"])
                self.strategies.append(strategy)
                unique_symbols.add(symbol)

            except Exception as e:
                logger.error(f"Failed to init strategy for {symbol} {session_name}: {e}")

        self.active_symbols = list(unique_symbols)

    async def execute_cycle(self, market_data: dict, blocked_symbols: set, current_positions, open_orders):
        """
       The Entry Point for MasterBot.

        It takes the centralized data, filters it, and triggers
        'run_strategy_cycle' for valid strategies.

        market_data: Dict structure { "MNQ": {"5m": df, "1h": df} }
        blocked_symbols: Set of symbols active in other bots (conflict prevention).
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
                strategy.log.warning(f"Skipping {strategy.symbol} (Busy in another strategy)")
                continue

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
                    tasks.append(self.run_strategy_cycle(strategy, df_5m))
                else:
                    strategy.log.info(f"💤 {strategy.tz_short} Session has not started yet!")
        if tasks:
            await asyncio.gather(*tasks)
