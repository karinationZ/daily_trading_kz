from datetime import timedelta
from typing import Dict, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector

# Assuming FvgDetector exists in your codebase based on the context
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    ExitType,
)
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    SESSION_MAP,
    SignalSummary,
    StrategyBase,
    TradeAction,
    TradeSignal,
    TradeStatus,
)
from daily_trading.settings import logger

CONFIG = {
    # ==========================================================================
    # GROUP 1: BITCOIN (MBT) — "THE TREND BEAST"
    # Logic: Deep Entry (0.95) + Tight Stop (1.2) = Maximum PnL.
    # Bitcoin respects the deep FVG fill perfectly before continuing.
    # ==========================================================================
    # MBT NY (Row 3: $17,258 PnL | 58% WR)
    ("MBT", "NY"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,  # Validated Winner
    },
    # MBT LONDON/ASIA (Row 5 & 6: ~$11k PnL)
    ("MBT", "LONDON"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    ("MBT", "ASIA"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    # ==========================================================================
    # GROUP 2: SILVER (SIL) — "THE AGGRESSIVE RUNNER"
    # Logic: NY prefers shallower entry (0.75) than London/Asia (0.95).
    # All sessions prefer tight stops (1.2).
    # ==========================================================================
    # SIL NY (Row 4: $15,518 PnL)
    ("SIL", "NY"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.75,  # NY moves faster, 0.75 fills more often
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    # SIL LONDON/ASIA (Row 1/2: ~$11k PnL)
    # Note: These had RR 2.0 in the screenshot, but 43% WR.
    # Row 16 (RR 1.2) had 55% WR and similar PnL. We prefer RR 1.2 for consistency.
    ("SIL", "LONDON"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    ("SIL", "ASIA"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    # ==========================================================================
    # GROUP 3: MES (S&P 500) — "THE PRECISION SCALP"
    # Logic: Deep Entry (0.95) + Tight Stop (1.2) is the clear winner across board.
    # ==========================================================================
    # MES NY (Row 8: $14,094 PnL)
    ("MES", "NY"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    # MES ASIA/LONDON (Row 10/11: $6,905 PnL)
    ("MES", "ASIA"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.5,
    },
    ("MES", "LONDON"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.5,
    },
    # ==========================================================================
    # GROUP 4: MNQ (NASDAQ) — "THE SPLIT PERSONALITY"
    # Logic: London/Asia need 0.75 Entry. NY allows 0.95 Deep Entry.
    # ==========================================================================
    # MNQ LONDON (Row 7: $6,528 PnL)
    ("MNQ", "LONDON"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.75,  # Speed required
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.5,
    },
    # MNQ NY (Row 14: $9,159 PnL)
    ("MNQ", "NY"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,  # Deep value works in NY
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,  # Tight stop
    },
    # ==========================================================================
    # GROUP 5: MGC (GOLD) — "THE MOMENTUM REVERSAL"
    # Logic: Data shows Gold prefers SHALLOW entry (0.25) in London/Asia.
    # This contradicts the "Deep" logic of indices. Gold trends harder.
    # ==========================================================================
    # MGC LONDON/ASIA (Row 12/13: $9,214 PnL)
    ("MGC", "LONDON"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.25,  # Aggressive Entry
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    ("MGC", "ASIA"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.25,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
    # MGC NY (Row 17: $6,320 PnL)
    ("MGC", "NY"): {
        "htf": "1h",
        "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,  # NY Gold chops more, needs deep entry
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2,
    },
}


class BosFvgStrategy(StrategyBase):
    """
    STRATEGY: Multi-Timeframe Trend Following (SMC) - BOS + LTF FVG

    OVERVIEW:
    A "Smart Money Concept" strategy that aligns Higher Timeframe (HTF) market structure
    with Lower Timeframe (LTF) entry signals.

    LOGIC:
    1. HTF (e.g., 1H): Identify Trend Bias via Break of Structure (BOS).
    2. LTF (e.g., 15m): Identify "Fresh" Fair Value Gaps (FVG) matching the bias.
    3. ENTRY: Limit Order inside the FVG (at `fvg_mitigation_ratio`).
    4. EXIT: SL outside FVG, Fixed RR TP.

    ROBUSTNESS:
    - Stateless recovery (checks FVG invalidation on restart).
    - Zombie protection (bad math/state detection).
    - Session cutoffs.
    """

    def __init__(
        self,
        symbol: str,
        timezone: ZoneInfo,
        main_tf="15m",  # LTF for Entry execution
        risk: float = 150.0,
        tp_multiple: float = 2.0,
        fvg_mitigation_ratio: float = 0.5,  # 0.5 = Entry at 50% of FVG
        sl_buffer_mult: float = 0.1,
        priority: int = 1,
    ):

        super().__init__(main_tf, priority=priority)

        self.symbol = symbol
        self.timezone = timezone

        # Logging
        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS-FVG | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)

        # Config
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = fvg_mitigation_ratio
        self.sl_buffer_mult = sl_buffer_mult
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

        # Session Config
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

        # --- DETECTORS ---
        # HTF: For Trend Bias
        self.htf_bos_detector = BosChochDetector(main_tf="1h")

        # LTF: For Entry Signals
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.5)

        # --- STATE ---
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[str] = set()

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        self.log.info("Resetting Session State.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    # -------------------------------------------------------------------------
    # HELPERS: Safety & Math
    # -------------------------------------------------------------------------

    @staticmethod
    def is_fvg_invalidated(df_ltf: pd.DataFrame, fvg: Dict) -> bool:
        """
        Restart Safety: Checks if the FVG has been VIOLATED (Candle Closed beyond it)
        in the history between FVG creation and Now.
        """
        fvg_time = fvg["time"]

        # Normalize Timezone
        try:
            if df_ltf.index.tz is None:
                fvg_time = fvg_time.replace(tzinfo=None)
            elif fvg_time.tzinfo is None:
                fvg_time = fvg_time.tz_localize(df_ltf.index.tz)
            else:
                fvg_time = fvg_time.tz_convert(df_ltf.index.tz)
        except Exception:
            fvg_time = fvg_time.replace(tzinfo=None)

        # Get history strictly AFTER FVG creation (excluding current forming candle)
        history = df_ltf[df_ltf.index > fvg_time].iloc[:-1]

        if history.empty:
            return False

        # Check Invalidation (Close beyond the gap)
        if fvg["bias"] == 1:  # Bullish FVG
            # Invalidated if a candle CLOSES below the bottom
            if history["close"].min() < fvg["bottom"]:
                return True
        else:  # Bearish FVG
            # Invalidated if a candle CLOSES above the top
            if history["close"].max() > fvg["top"]:
                return True

        return False

    # -------------------------------------------------------------------------
    # LOGIC: Find Setup
    # -------------------------------------------------------------------------

    def find_setup(self, df_ltf: pd.DataFrame, htf_bias: BIAS, bos_time, ts_now) -> Optional[TradeSignal]:
        """
        Scans LTF for fresh FVGs that align with HTF Bias.
        """
        # 1. Get LTF FVGs
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get("signals", [])
        atr = self.ltf_fvg_detector.get_current_atr(df_ltf)

        if not fvg_signals:
            return None

        current_price = df_ltf.iloc[-1]["close"]
        target_bias_val = BIAS.LONG.num_value if htf_bias == BIAS.LONG else BIAS.SHORT.num_value

        # Iterate FVGs (Newest First)
        for fvg in fvg_signals:

            # A. Filter by Bias
            if fvg["bias"] != target_bias_val:
                continue

            # B. Filter by Mitigation (Must be Fresh)
            if fvg.get("mitigated", True):
                continue

            # C. Filter by Cache (Already processed)
            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                continue

            # D. Filter by Historical Invalidation (Restart Safety)
            if self.is_fvg_invalidated(df_ltf, fvg):
                self.processed_zone_ids.add(fvg_id)
                continue

            # E. Calculate Entry Params
            top = fvg["top"]
            bottom = fvg["bottom"]
            height = top - bottom

            if htf_bias == BIAS.LONG:
                # Buy Limit at X% depth
                entry = top - (height * self.entry_depth)
                sl = bottom - (atr * self.sl_buffer_mult)

                # Logic: Entry must be below current price for a Limit Buy
                if entry > current_price:
                    continue  # Price is below entry, we missed it or need market order (skipping for limit strategy)
            else:
                # Sell Limit at X% depth
                entry = bottom + (height * self.entry_depth)
                sl = top + (atr * self.sl_buffer_mult)

                # Logic: Entry must be above current price for a Limit Sell
                if entry < current_price:
                    continue

            # F. Calculate Position Size
            pos_math = self.calculate_trade_position(
                symbol=self.symbol, tp_multiple=self.tp_multiple, risk=self.risk, entry_price=entry, sl=sl, direction=htf_bias
            )

            # Gatekeeper
            if pos_math["size"] <= 0 or pos_math["sl"] <= 0:
                continue

            # G. Create Signal
            return TradeSignal(
                action=TradeAction.ENTER,
                symbol=self.symbol,
                direction=htf_bias,
                price=pos_math["entry"] if "entry" in pos_math else entry,
                sl=pos_math["sl"],
                tp=pos_math["tp"],
                size=pos_math["size"],
                timestamp=ts_now,
                metadata={
                    "fvg_id": fvg_id,
                    "bos_ts": bos_time,
                },
            )

        return None

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None, reason=None) -> TradeSignal:
        """Helper to create signals and manage state transitions."""
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(
            symbol=self.symbol,
            action=action,
            direction=self.active_trade.direction,
            price=report_price,
            sl=self.active_trade.sl,
            tp=self.active_trade.tp,
            size=self.active_trade.size,
            reason=reason,
            timestamp=timestamp,
        )

        # Pass metadata
        if self.active_trade and self.active_trade.metadata:
            sig.metadata.update(self.active_trade.metadata)

        # State Reset on Terminal Actions
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        curr = df_ltf.iloc[-1]
        ts_now = curr.name.tz_convert(self.timezone)
        ts_utc = curr.name.tz_convert("UTC")

        # Session Cutoff Calculation
        s_info = SESSION_MAP[self.timezone.key]
        session_end = ts_now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0,
        )
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)
        entry_cutoff = session_end - timedelta(minutes=30)  # No new entries last 30m

        just_filled = False

        # ==================================================================
        # 1. FILL CHECK (Limit Orders)
        # ==================================================================
        if self.trade_status == TradeStatus.PENDING:
            filled = False
            if self.active_trade.direction == BIAS.LONG:
                if curr["low"] <= self.active_trade.price:
                    filled = True
            else:
                if curr["high"] >= self.active_trade.price:
                    filled = True

            if filled:
                self.trade_status = TradeStatus.FILLED
                self.active_trade.entry_timestamp = ts_utc
                just_filled = True
                self.log.info(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price:.2f}")

        # ==================================================================
        # 2. EXIT CHECK (Stop Loss, Take Profit, Zombies, Cutoff)
        # ==================================================================
        if self.trade_status == TradeStatus.FILLED:
            # --- ZOMBIE KILLER (Safety) ---
            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.log.error(f"🧟 ZOMBIE TRADE DETECTED ({self.symbol}). Cleaning up.")
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG

            exit_type = None
            exit_price = None
            exit_reason = "NORMAL"

            # A. GAP CHECK (If not just filled)
            if not just_filled:
                if is_long:
                    if curr["open"] <= sl:
                        exit_type, exit_price = "LOSS", curr["open"]
                    elif curr["open"] >= tp:
                        exit_type, exit_price = "WIN", curr["open"]
                else:
                    if curr["open"] >= sl:
                        exit_type, exit_price = "LOSS", curr["open"]
                    elif curr["open"] <= tp:
                        exit_type, exit_price = "WIN", curr["open"]

            # B. STANDARD CHECK (Trust Wicks)
            if not exit_type:
                if is_long:
                    hit_sl = curr["low"] <= sl
                    hit_tp = curr["high"] >= tp
                else:
                    hit_sl = curr["high"] >= sl
                    hit_tp = curr["low"] <= tp

                # Pessimistic conflict resolution
                if hit_sl and hit_tp:
                    hit_tp = False

                if hit_sl:
                    exit_type = "LOSS"
                    exit_price = sl
                elif hit_tp:
                    exit_type = "WIN"
                    exit_price = tp

            # C. TIME CUTOFF
            is_cutoff = ts_now >= cutoff_ts
            if not exit_type and is_cutoff:
                exit_price = curr["close"]
                exit_reason = "CUTOFF"

                # Determine Result Manually
                pnl = (exit_price - self.active_trade.price) if is_long else (self.active_trade.price - exit_price)
                exit_type = "WIN" if pnl > 0 else "LOSS"

            # --- EXECUTION ---
            if exit_type:
                sig = self.create_signal(TradeAction.EXIT, ts_utc, custom_price=exit_price)
                sig.trade_result = exit_type
                sig.metadata["exit_reason"] = exit_reason

                if exit_reason == "CUTOFF":
                    sig.metadata["exit_type"] = ExitType.MARKET_FORCE
                else:
                    sig.metadata["exit_type"] = ExitType.SL_TP_HIT

                self.log.info(f"🎯 TRADE EXIT ({exit_type}) {self.symbol} | Reason: {exit_reason}")
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="IN_TRADE")

        # ==================================================================
        # 3. HTF TREND ANALYSIS
        # ==================================================================
        trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
        bos_signals = self.htf_bos_detector.signals

        # If no structure, no trade
        if trend_bias == BIAS.NEUTRAL or not bos_signals:
            return SignalSummary(direction="NEUTRAL")

        last_bos = bos_signals[-1]
        bos_time = last_bos.get("time")

        # ==================================================================
        # 4. NEW ENTRY SEARCH
        # ==================================================================
        if self.trade_status == TradeStatus.NONE:
            if ts_now < entry_cutoff:
                new_trade = self.find_setup(df_ltf, trend_bias, bos_time, ts_utc)

                if new_trade:
                    self.active_trade = new_trade
                    self.trade_status = TradeStatus.PENDING
                    # Mark FVG as processed immediately to avoid recalc
                    self.processed_zone_ids.add(new_trade.metadata["fvg_id"])

                    self.log.info(f"🚀 LIMIT ORDER PLACED: {new_trade.direction.name} @ {new_trade.price:.2f}")
                    return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        # ==================================================================
        # 5. PENDING MANAGEMENT (Cutoff & Structure Update)
        # ==================================================================
        if self.trade_status == TradeStatus.PENDING:
            should_cancel = False
            cancel_reason = ""

            # A. Time Cutoff
            if ts_now >= cutoff_ts:
                should_cancel = True
                cancel_reason = "CUTOFF"

            # B. Check for BOS Change (Trailing Entry Logic)
            # If HTF forms a NEW BOS, the old FVG might be outdated or trend accelerated.
            # Cancel old limit to look for a new one in the next tick.
            elif self.active_trade and self.active_trade.metadata:
                current_trade_bos_ts = self.active_trade.metadata.get("bos_ts")

                if current_trade_bos_ts and bos_time != current_trade_bos_ts:
                    should_cancel = True
                    cancel_reason = "NEW_STRUCTURE"

            if should_cancel:
                self.log.info(f"🚫 CANCELLING PENDING ENTRY | Reason: {cancel_reason}")
                return SignalSummary(
                    direction="CANCELLED",
                    trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc),
                )

        return SignalSummary(direction="NEUTRAL")
