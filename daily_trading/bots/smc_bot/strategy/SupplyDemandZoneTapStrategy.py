from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import (
    SupplyDemandZonesDetector,
)
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    ExitType,
    SESSION_MAP,
    SignalSummary,
    StrategyBase,
    TradeAction,
    TradeSignal,
    TradeStatus,
)
from daily_trading.settings import logger


class SupplyDemandZoneTapStrategy(StrategyBase):
    """
    Institutional HTF Zone Tap Strategy.

    Logic Hierarchy:
        - Trigger: Monitors 5m TF for a 'Tap' (Touch) of the 1H Zone.
        - Confirmation: 5m candle must show 'Absorption' (wick) and not be a 'Falling Knife'.
        - Execution: Limit order placed at the 1H zone edge.
         - Safety: Hard cutoff at session end and automatic mitigation of zones after one trade.
    """

    def __init__(
        self,
        symbol: str,
        timezone: ZoneInfo,
        main_tf="5m",
        risk: float = 150.0,
        tp_multiple: float = 1.5,
        swing_length: int = 45,
        internal_length: int = 5,
        sl_atr_multiplier=2.0,
        eb_mult=2.0,
    ):
        super().__init__(main_tf)
        self.symbol = symbol
        self.timezone = timezone
        self.risk = risk
        self.tp_multiple = tp_multiple
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

        self.sl_atr_multiplier = sl_atr_multiplier
        self.eb_mult = eb_mult
        self.enter_buffer = self.ENTRY_BUFFER[symbol]

        # Detectors
        self.sd_detector_1h = SupplyDemandZonesDetector(
            fvg_based_only=False,
            swing_length=swing_length,
            internal_length=internal_length,
        )

        # State Management
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()
        self.zone_usage_counts: Dict[pd.Timestamp, int] = defaultdict(int)

        # Caching
        self.cached_zones_1h: List[Dict] = []
        self.last_processed_hour: int = -1
        self.point_value = self.POINT_VALUE[symbol]

        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"1H TAP | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)

        # Unique IBKR Tag
        self.ext_tag = f"{symbol}_{self.tz_short}_1H_TAP"

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        logger.info(f"{self.symbol} | New Session: Resetting Strategy State and Cache.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.zone_usage_counts.clear()
        self.cached_zones_1h = []
        self.last_processed_hour = -1

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """
        Refactored Trigger Logic:
        1. Fixes buffer math (allows entry at edge + buffer).
        2. Checks for falling knife.
        """
        atr = self.sd_detector_1h.get_current_atr(ohlc)
        # Adjusted buffer (Striking distance)
        entry_buffer = self.enter_buffer * self.eb_mult

        curr = ohlc.iloc[-1]

        # CHANGE 2: Fix Buffer Math
        # Previous code forced price DEEP into zone (Top - Buffer).
        # Correct logic is (Top + Buffer) to catch it as it arrives.
        if direction == BIAS.LONG:
            # Low must drop BELOW the (Zone Top + Buffer)
            trigger_price = zone["top"] + entry_buffer
            touched = curr["low"] <= trigger_price
        else:
            # High must rise ABOVE the (Zone Bottom - Buffer)
            trigger_price = zone["bottom"] - entry_buffer
            touched = curr["high"] >= trigger_price

        if not touched:
            return False

        # Check for Falling Knife (Velocity/Momentum check)
        is_falling_knf = self.sd_detector_1h.is_falling_knife(curr, direction, atr)

        if is_falling_knf:
            self.log.info(f"🔪 Falling Knife Detected | {curr.name} | Skipping Trigger.")
            return False

        return True

    @staticmethod
    def is_zone_already_mitigated(ohlc_5m: pd.DataFrame, zone: Dict, threshold: float = 0.5) -> bool:
        """
        Checks if the zone was ALREADY mitigated (penetrated > threshold%)
        by any candle AFTER zone creation but BEFORE the current candle.

        Args:
            threshold: 0.5 means 50% of the zone depth.
        """
        zone_start = zone["start_time"]

        # 1. Timezone Alignment (Robust)
        try:
            if zone_start.tzinfo is None:
                # Zone is naive, localize to match OHLC
                zone_start = zone_start.tz_localize(ohlc_5m.index.tz)
            else:
                # Zone is aware, convert to match OHLC
                zone_start = zone_start.tz_convert(ohlc_5m.index.tz)
        except TypeError:
            # Fallback: If OHLC is naive and Zone is aware (or vice versa mismatch)
            # We strip TZ from both to compare raw times
            zone_start = zone_start.replace(tzinfo=None)
            # Note: This relies on the loop filtering below, assumes df index is processed

        # 2. Get Historical Data (Between Zone Creation and Current Candle)
        # We look strictly > zone_start to avoid checking the candle that created the zone.
        # We use [:-1] to exclude the CURRENT candle (because we want to trade the current one).
        history = ohlc_5m[ohlc_5m.index > zone_start].iloc[:-1]

        if history.empty:
            return False  # No history exists, so it can't be mitigated

        # 3. Calculate Logic
        top = zone["top"]
        bottom = zone["bottom"]
        height = top - bottom

        # 4. Check for Penetration
        if zone["bias"] == BIAS.LONG:
            # Demand Zone: Entry is at TOP.
            # We check if Price went LOWER than (Top - 50% Height)
            limit_price = top - (height * threshold)

            # Use 'low' for Demand (deepest point of candle)
            deepest_dip = history["low"].min()

            if deepest_dip <= limit_price:
                # self.log.info(f"🚫 Zone {zone_start} mitigated! Low {deepest_dip:.2f} < Limit {limit_price:.2f}")
                return True

        else:  # BIAS.SHORT
            # Supply Zone: Entry is at BOTTOM.
            # We check if Price went HIGHER than (Bottom + 50% Height)
            limit_price = bottom + (height * threshold)

            # Use 'high' for Supply (highest point of candle)
            highest_peak = history["high"].max()

            if highest_peak >= limit_price:
                # self.log.info(f"🚫 Zone {zone_start} mitigated! High {highest_peak:.2f} > Limit {limit_price:.2f}")
                return True

        return False

    def calculate_trade_params(self, ohlc, zone: Dict, direction: BIAS) -> Dict:
        """Calculates entry at zone edge and SL beyond HTF zone boundary."""
        curr_candle = ohlc.iloc[-1]
        adjusted_buffer = self.enter_buffer * self.eb_mult
        if direction == BIAS.LONG:
            entry_price = curr_candle["low"] - adjusted_buffer
            sl = self.sd_detector_1h.calculate_atr_stop_loss(
                ohlc,
                entry_price,
                direction=direction,
                atr_multiplier=self.sl_atr_multiplier,
            )
            r = entry_price - sl
            tp = entry_price + (r * self.tp_multiple)
        else:
            entry_price = curr_candle["high"] + adjusted_buffer
            sl = self.sd_detector_1h.calculate_atr_stop_loss(
                ohlc,
                entry_price,
                direction=direction,
                atr_multiplier=self.sl_atr_multiplier,
            )

            r = sl - entry_price
            tp = entry_price - (r * self.tp_multiple)
        # 1. Calculate Stop Distance
        r = abs(entry_price - sl)
        # Safety: Prevent division by zero if SL is exactly on Entry
        if r == 0:
            r = 0.25
        # 2. Calculate Risk per 1 Contract
        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0:
            risk_per_contract = 5.0  # Arbitrary small fallback to prevent crash

        # 3. Calculate Raw Size (Pure Math)
        raw_size = int(self.risk / risk_per_contract)
        # 4. --- SAFETY CAP ---
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        if raw_size > max_contracts:
            self.log.warning(f"⚠️ RISK CAP: Math wanted {raw_size} contracts, capped at {size}. " f"(Risk per contract: ${risk_per_contract:.2f})")

        return {"entry": entry_price, "sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        """
        Creates a signal.
        """
        # 1. Determine which price to report
        # If it's an EXIT, we want the execution price, not the entry price.
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(
            symbol=self.symbol,
            action=action,
            direction=self.active_trade.direction,
            price=report_price,
            sl=self.active_trade.sl,
            tp=self.active_trade.tp,
            size=self.active_trade.size,
            timestamp=timestamp,
        )
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade = None
            self.trade_status = TradeStatus.NONE
        return sig

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        """Main strategy execution loop: High-Priority Fill & Exit Logic."""
        curr = df_ltf.iloc[-1]
        ts_now = curr.name.tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]

        # 1. HTF Caching (Zones refresh once per hour)
        if ts_now.hour != self.last_processed_hour:
            swing_zones, internal_zones = self.sd_detector_1h.calculate(df_htf)
            self.cached_zones_1h = swing_zones + internal_zones
            self.last_processed_hour = ts_now.hour
            self.log.info(f"1H Zones: {self.cached_zones_1h}")

        # 2. Define Cutoffs
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16), minute=s_info.get("end_minute", 0))
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)

        # 3. FILL LOGIC (High Priority)
        # We check this FIRST. If it fills, we update status immediately.
        filled = False
        if self.trade_status == TradeStatus.PENDING:
            is_long = self.active_trade.direction == BIAS.LONG
            if is_long:
                filled = curr["low"] <= self.active_trade.price
            else:
                filled = curr["high"] >= self.active_trade.price

            if filled:
                self.trade_status = TradeStatus.FILLED
                self.log.info(
                    f"⚡ ORDER FILLED | {self.symbol}: {self.active_trade.direction.name} " f"@ {self.active_trade.price:.2f} | Time: {curr.name}"
                )

        # 4. SESSION EXIT LOGIC (Evaluate both Active and Just-Filled trades)
        if self.trade_status == TradeStatus.FILLED:
            # --- SAFEGUARD: ZOMBIE KILLER ---
            # If data is corrupted (0.00), kill the trade to stop infinite loops.
            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.log.error(f"🧟 ZOMBIE TRADE DETECTED (Bad Data) | {self.symbol}. Clearing state.")
                self.active_trade = None
                self.trade_status = TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            # Check for SL/TP first
            is_long = self.active_trade.direction == BIAS.LONG

            # Detailed check to see WHICH one was hit
            hit_sl = (is_long and curr["low"] <= self.active_trade.sl) or (not is_long and curr["high"] >= self.active_trade.sl)

            hit_tp = (is_long and curr["high"] >= self.active_trade.tp) or (not is_long and curr["low"] <= self.active_trade.tp)

            hit_sl_tp = hit_sl or hit_tp

            # OR Check if we hit the Session Hard Exit
            is_cutoff_time = ts_now >= cutoff_ts

            if hit_sl_tp:
                # Determine actual exit price
                exit_price = self.active_trade.sl if hit_sl else self.active_trade.tp
                self.log.info(f"🎯 TRADE EXIT (HIT {'SL' if hit_sl else 'TP'}) | Close: {curr['close']:.2f}")

                # PASS THE EXIT PRICE HERE
                signal = self.create_signal(TradeAction.EXIT, df_ltf.index[-1], custom_price=exit_price)
                signal.metadata["exit_type"] = ExitType.SL_TP_HIT
                if hit_sl:
                    signal.trade_result = "LOSS"
                elif hit_tp:
                    signal.trade_result = "WIN"
                return SignalSummary(direction="EXIT", trade_signal=signal)

            elif is_cutoff_time:
                # Exit at current market close
                exit_price = curr["close"]
                self.log.info(f"⏰ SESSION CUTOFF EXIT | Closing at {exit_price:.2f}")
                signal = self.create_signal(TradeAction.EXIT, df_ltf.index[-1], custom_price=exit_price)
                signal.metadata["exit_type"] = ExitType.MARKET_FORCE

                # PASS THE MARKET CLOSE PRICE HERE
                return SignalSummary(direction="EXIT", trade_signal=signal)

            return SignalSummary(direction="FILLED" if filled else "IN_TRADE")

        # 5. PENDING ORDER CANCELLATION (Only if NOT filled)
        # We removed the redundant Fill check here. If it was filled, it was caught in Step 3.
        if self.trade_status == TradeStatus.PENDING:

            # A. Time Cutoff Cancel
            if ts_now >= cutoff_ts:
                return SignalSummary(
                    direction="CANCELLED",
                    trade_signal=self.create_signal(TradeAction.CANCEL, curr.name),
                )

            # B. Target Reached Early Cancel (Invalidation)
            trade = self.active_trade
            is_long = trade.direction == BIAS.LONG
            hit_target_early = (is_long and curr["high"] >= trade.tp) or (not is_long and curr["low"] <= trade.tp)

            if hit_target_early:
                return SignalSummary(
                    direction="CANCELLED",
                    trade_signal=self.create_signal(TradeAction.CANCEL, curr.name),
                )

            return SignalSummary(direction="PENDING")

        # 6. Scan for NEW Taps (Only if session is active)
        if self.trade_status == TradeStatus.NONE and ts_now < cutoff_ts:
            for zone in self.cached_zones_1h:
                z_id = zone["start_time"]

                # Check usage count < 2
                current_usage = self.zone_usage_counts[z_id]
                if current_usage >= 2:
                    continue

                # Check Historical Mitigation (Robust / Restart Safe)
                # If price went > 50% into the zone in the past, kill it.
                if self.is_zone_already_mitigated(df_ltf, zone, threshold=0.8):
                    # We add to processed so we don't re-calculate this heavy check every tick
                    self.zone_usage_counts[z_id] = 99
                    continue

                if self.validate_zone_reaction(df_ltf, zone["bias"], zone):
                    params = self.calculate_trade_params(df_ltf, zone, zone["bias"])

                    # Reject invalid math immediately
                    if params["entry"] <= 0 or params["sl"] <= 0 or params["size"] <= 0:
                        self.log.warning(f"⚠️ SKIPPING INVALID SETUP: {params}")
                        continue

                    # Basic sanity check (Don't enter if price is already past TP)
                    if (zone["bias"] == BIAS.LONG and curr["close"] >= params["tp"]) or (
                        zone["bias"] == BIAS.SHORT and curr["close"] <= params["tp"]
                    ):
                        continue

                    trade = TradeSignal(
                        action=TradeAction.ENTER,
                        symbol=self.symbol,
                        direction=zone["bias"],
                        price=params["entry"],
                        sl=params["sl"],
                        tp=params["tp"],
                        size=params["size"],
                        timestamp=curr.name,
                        metadata={"zone_id": z_id, "usage_count": current_usage + 1},
                    )
                    self.active_trade, self.trade_status = trade, TradeStatus.PENDING

                    # We increment immediately on Signal Generation.
                    self.zone_usage_counts[z_id] += 1
                    self.log.info(f"🚀 Signal Generated from Zone {z_id} (Usage: {self.zone_usage_counts[z_id]}/2)")

                    return SignalSummary(direction=trade.direction.name, trade_signal=trade)

        return SignalSummary(direction="NEUTRAL")
