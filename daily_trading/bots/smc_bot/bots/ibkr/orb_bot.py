import asyncio
import traceback
from typing import List
from zoneinfo import ZoneInfo

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.ORBStrategy import ORBStrategy
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction, TradeStatus
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID

ORB_CONFIG = {
    # ==========================================================================
    # GROUP 1: MNQ (The Profit Engine)
    # ==========================================================================

    # MNQ NY: The Aggressive Runner
    # Champ: SW15 | ATR1.8 | THR0.4 | RR2.5
    # Stats: Score 86.7 | PnL $3162 | WR 66.7%
    ("MNQ", "NY"): {
        # Strategy Logic
        "swing_length": 15,
        "internal_length": 4,
        "fvg_offset": 1,  # Aggressive Momentum Entry
        "fvg_threshold": 0.4,
        "zone_precision_factor": 1.8,
        "tp_multiple": 2.5,  # Targeting Home Runs
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # MNQ LONDON: The Consistent Scalper
    # Champ: SW12 | ATR1.8 | THR0.2 | RR1.2
    # Stats: Score 73.6 | PnL $2,556 | WR 60.0%
    ("MNQ", "LONDON"): {
        # Strategy Logic
        "swing_length": 12,
        "internal_length": 5,
        "fvg_offset": 4,  # Deep Reversal Entry
        "fvg_threshold": 0.2,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,  # Scalping Target
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # ==========================================================================
    # GROUP 2: MES (The Stability Anchor)
    # ==========================================================================

    # MES LONDON: Structural Grinder
    # Champ: SW26 | ATR1.8 | THR0.25
    # Stats: Score 71.4 | PnL $1,185 | WR 54.55%
    ("MES", "LONDON"): {
        # Strategy Logic
        "swing_length": 26,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,  # Conservative
        # Dynamic Buffers (Increased for London Low Volatility)
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,  # Increased per your setup
        "tick_buffer_mult": 0.05  # Increased per your setup
    },

    # MES NY: Deep Value
    # Champ: SW50 | ATR1.8 | THR0.25
    # Stats: Score 74.3 | PnL $1522 | WR 61.8%
    ("MES", "NY"): {
        # Strategy Logic
        "swing_length": 50,
        "internal_length": 5,
        "fvg_offset": 1,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 1.8,
        "tp_multiple": 1.2,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # ==========================================================================
    # GROUP 3: COMMODITIES (MGC/SIL)
    # ==========================================================================

    # MGC LONDON: Gold Volatility
    # Champ: SW12 | ATR2.5 | THR0.2
    # Stats: Score 73.3 | PnL $2,693 | WR 53.23%
    ("MGC", "LONDON"): {
        # Strategy Logic
        "swing_length": 12,
        "internal_length": 5,
        "fvg_offset": 3,
        "fvg_threshold": 0.2,
        "zone_precision_factor": 2.5,
        "tp_multiple": 1.2,
        # Dynamic Buffers (Doubled for London Gold)
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.2,  # <--- INCREASED
        "tick_buffer_mult": 0.1  # <--- INCREASED
    },

    # MGC LONDON: Gold Volatility
    # Champ: SW12 | ATR2.5 | THR0.2
    # Stats: Score 76.2 | PnL $2,522 | WR 60.63%
    ("MGC", "NY"): {
        # Strategy Logic
        "swing_length": 12,
        "internal_length": 3,
        "fvg_offset": 1,
        "fvg_threshold": 0.2,
        "zone_precision_factor": 2.5,
        "tp_multiple": 1.5,
        # Dynamic Buffers (standard)
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.1,
        "tick_buffer_mult": 0.05
    },

    # SIL LONDON: Silver Patience
    # Champ: SW75 | ATR2.5 | THR0.25
    # Stats: Score 70.2 | PnL $1,096 | WR 51.43%
    ("SIL", "LONDON"): {
        # Strategy Logic
        "swing_length": 75,  # Massive Structure
        "internal_length": 10,
        "fvg_offset": 5,
        "fvg_threshold": 0.25,
        "zone_precision_factor": 2.5,  # Wide Zones
        "tp_multiple": 1.2,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
        "sl_buffer_mult": 0.2,  # <--- INCREASED
        "tick_buffer_mult": 0.1  # <--- INCREASED
    },

    # SIL NY: Momentum
    # Champ: SW60 | ATR2.5 | THR0.3
    # Stats: PnL $1,043 | WR 50.0%
    ("SIL", "NY"): {
        "swing_length": 60,
        "internal_length": 15,
        "fvg_offset": 1,
        "fvg_threshold": 0.3,
        "zone_precision_factor": 2.5,
        "tp_multiple": 1.5,
        # Dynamic Buffers
        "entry_buffer_mult": 0.05,
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
                tz = ZoneInfo("America/New_York") if session_name == "NY" else ZoneInfo("Europe/London")

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
                    tick_buffer_mult=config["tick_buffer_mult"],
                    atr_length_buffer=14
                )
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
                # logger.debug(f"Skipping {strategy.symbol} (Busy in another strategy)")
                continue

            # 2. Extract Data: ORBBot only needs "5m" data.
            sym_data = market_data.get(strategy.symbol, {})
            df_5m = sym_data.get("5m")

            # 3. Validation: Check if data exists and strategy is in time window.
            if df_5m is not None and not df_5m.empty:
                if self.is_strategy_in_window(strategy):
                    # Call the EXISTING logic function
                    tasks.append(self.run_strategy_cycle(strategy, df_5m))

        if tasks:
            await asyncio.gather(*tasks)

    async def run_strategy_cycle(self, strategy: ORBStrategy, df):
        """
        Isolated execution context for a single strategy.
        """
        try:
            # 1. Run Analysis
            analysis = strategy.analyze(df)

            # 2. Report to Telegram (Only Significant Actions)
            # We filter out 'PENDING' and 'IN_TRADE' statuses to avoid spamming the channel every 5 minutes.
            if analysis.trade_signal:

                is_significant = analysis.trade_signal.action in [
                    TradeAction.ENTER,
                    TradeAction.EXIT,
                    TradeAction.CANCEL
                ]

                # Double Check: If it's CANCEL, only send if it was an active pending order getting killed
                # If it's just "No Trade Found", we don't send.
                if is_significant:
                    action_emoji = {
                        TradeAction.ENTER: "🚀 ENTER",
                        TradeAction.EXIT: "💰 EXIT",
                        TradeAction.CANCEL: "🚫 CANCEL"
                    }.get(analysis.trade_signal.action, "⚠️ UPDATE")

                    msg = (
                        f"{action_emoji} {strategy.log_tag}\n"
                        f"Action: {analysis.trade_signal.action.name}\n"
                        f"Direction: {analysis.trade_signal.direction.name}\n"
                        f"Size: {analysis.trade_signal.size}\n"
                        f"Price: {analysis.trade_signal.price:.2f}\n"
                        f"SL: {analysis.trade_signal.sl:.2f} | TP: {analysis.trade_signal.tp:.2f}\n"
                        f"Time: {analysis.trade_signal.timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
                    )

                    # Send Message with Chart (analysis.fig)
                    # Note: getattr handles cases where fig might be None
                    await self.telegram_bot.send_message(msg, getattr(analysis.trade_signal, 'fig', None))

            # 3. Execute Orders
            await self.handle_execution(strategy, analysis)

        except Exception as e:
            strategy.log(f"CYCLE ERROR: {e}", level="error")
            logger.error(traceback.format_exc())
