from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
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


class BosOrderBlockStrategy(StrategyBase):
    """
    STRATEGY NAME: Institutional Structure Pro (BOS) - Limit Entry

    PHASE 1: STRUCTURAL BIAS (HTF)
    - Detects the last Break of Structure (BOS) on 1H TF to determine Trend.

    PHASE 2: ZONE IDENTIFICATION & ORDER PLACEMENT
    - Identifies Supply/Demand zones on the 1H TF (or Main TF, configurable).
    - Places a LIMIT ORDER inside the zone at `entry_depth` %.
    - Stop Loss placed just outside the zone + buffer.
    """

    def __init__(
        self,
        symbol: str,
        timezone: ZoneInfo,
        main_tf="1h",
        risk: float = 150.0,
        tp_multiple: float = 1.5,
        entry_depth: float = 0.5,  # 0.5 = 50% into zone
        sl_buffer_mult: float = 0.1,
        swing_length: int = 45,
        internal_length: int = 5,
        bos_left_bars: int = 5,
        bos_right_bars: int = 5,
        bos_volume_thresh: float = 0,
        fvg_based_only: bool = False,
        # fvg_offset: int = 10,
        # fvg_threshold=0.15,
        # zone_precision_factor: float = 1,
    ):

        super().__init__(main_tf)
        self.symbol = symbol
        self.timezone = timezone

        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"1H-BOS-OB | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{self.tz_short}_BOS"

        # Settings
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = entry_depth

        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

        # --- DETECTORS ---
        self.htf_bos_detector = BosChochDetector(
            main_tf=main_tf,
            left_bars=bos_left_bars,
            right_bars=bos_right_bars,
            volume_thresh=bos_volume_thresh,
        )

        self.sd_detector = SupplyDemandZonesDetector(
            fvg_based_only=fvg_based_only,
            swing_length=swing_length,
            internal_length=internal_length,
            # zone_precision_factor=zone_precision_factor,
            # fvg_offset=fvg_offset,
            # fvg_threshold=fvg_threshold,
        )

        # State Variables
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()

        # Buffers
        self.point_value = self.POINT_VALUE.get(symbol, 1.0)
        self.sl_buffer_mult = sl_buffer_mult
        self.zones = None
        self.last_processed_hour: int = -1

    def reset_session_state(self):
        """Resets daily state."""
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.last_processed_hour = -1

    # -------------------------------------------------------------------------
    # CALCULATION LOGIC
    # -------------------------------------------------------------------------

    def get_limit_levels(self, ohlc: pd.DataFrame, zone: Dict, direction: BIAS):
        """
        Calculates Limit Entry and Hard SL based on Zone boundaries + Depth.
        """
        atr = self.sd_detector.get_current_atr(ohlc, atr_length_buffer=14)
        if self.sl_buffer_mult > 0:
            sl_buffer_val = atr * self.sl_buffer_mult
        else:
            sl_buffer_val = self.TICK_SIZES[self.symbol]

        top = zone["top"]
        bottom = zone["bottom"]
        height = top - bottom

        if direction == BIAS.LONG:
            # BUY LIMIT @ Top - (Height * Depth) -> Lower is better
            # If depth is 0.0, entry is at 'top' (Aggressive limit)
            # If depth is 0.5, entry is at 50% of zone
            entry = top - (height * self.entry_depth)
            sl = bottom - sl_buffer_val
        else:
            # SELL LIMIT @ Bottom + (Height * Depth) -> Higher is better
            entry = bottom + (height * self.entry_depth)
            sl = top + sl_buffer_val

        return entry, sl

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """Calculates Size and TP."""
        r = abs(entry_price - sl)
        if r == 0:
            r = 0.25

        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0:
            risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        return {"sl": sl, "tp": tp, "size": size}

    # -------------------------------------------------------------------------
    # ENTRY SEARCH
    # -------------------------------------------------------------------------

    def find_setup(
        self,
        df_ltf: pd.DataFrame,
        df_htf: pd.DataFrame,
        swing_zones: List[Dict],
        internal_zones: List[Dict],
        ts_now,
    ):

        # 1. HTF BIAS
        trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
        bos_signals = self.htf_bos_detector.signals

        if trend_bias == BIAS.NEUTRAL or not bos_signals:
            return None

        last_bos = bos_signals[-1]
        bos_time = last_bos.get("time")

        # 2. SELECT ZONES MATCHING BIAS
        # Combine lists, prefer SWING if overlapping logic needed (but here we treat equal)
        all_candidates = [(z, "SWING") for z in swing_zones] + [(z, "INTERNAL") for z in internal_zones]

        best_signal = None
        current_close = df_ltf.iloc[-1]["close"]
        closest_dist = float("inf")

        for zone, zone_type in all_candidates:
            zone_id = zone["start_time"]
            if zone_id.tzinfo is None:
                zone_id = zone_id.tz_localize("UTC")
            else:
                zone_id = zone_id.tz_convert("UTC")

            # FILTER: Direction & Freshness
            if zone["bias"] != trend_bias:
                continue
            if zone_id in self.processed_zone_ids:
                continue

            # 3. CALCULATE LIMIT ORDER
            entry, sl = self.get_limit_levels(df_ltf, zone, trend_bias)

            # 4. DISTANCE CHECK
            # We want the zone closest to current price but NOT already passed
            dist = abs(current_close - entry)

            # Ensure we aren't placing a limit order that would immediately fill at a bad price
            # (e.g. Price is 100, we want to Buy Limit at 90. OK.)
            # (e.g. Price is 100, we want to Buy Limit at 110. BAD -> Market Buy.)
            valid_limit = False
            if trend_bias == BIAS.LONG:
                if current_close > entry:
                    valid_limit = True
            else:
                if current_close < entry:
                    valid_limit = True

            if not valid_limit:
                continue

            if dist < closest_dist:
                pos = self.calculate_trade_position(entry, sl, trend_bias)

                # Gatekeeper
                if pos["size"] <= 0:
                    continue

                best_signal = TradeSignal(
                    action=TradeAction.ENTER,
                    symbol=self.symbol,
                    direction=trend_bias,
                    price=entry,
                    sl=pos["sl"],
                    tp=pos["tp"],
                    size=pos["size"],
                    timestamp=ts_now,
                    metadata={
                        "zone_id": zone_id,
                        "type": zone_type,
                        "bos_ts": bos_time,
                    },
                )
                closest_dist = dist

        return best_signal

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
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

        # Propagate the entry_timestamp from the active trade to the exit signal
        if self.active_trade and hasattr(self.active_trade, "entry_timestamp"):
            sig.entry_timestamp = self.active_trade.entry_timestamp

        # Pass metadata for Exit logic
        if self.active_trade and self.active_trade.metadata:
            sig.metadata.update(self.active_trade.metadata)

        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        curr = df_ltf.iloc[-1]
        ts_now = curr.name.tz_convert(self.timezone)
        ts_utc = curr.name.tz_convert("UTC")

        s_info = SESSION_MAP[self.timezone.key]
        session_end = ts_now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0,
        )
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)
        entry_cutoff = session_end - timedelta(minutes=30)

        swing_zones, internal_zones = self.sd_detector.calculate(df_htf)
        self.zones = swing_zones + internal_zones

        if ts_now.hour != self.last_processed_hour:
            self.last_processed_hour = ts_now.hour
            self.log.info(f"1H Zones: {self.zones}")

        just_filled = False

        # 1. FILL CHECK (Limit Orders)
        if self.trade_status == TradeStatus.PENDING:
            is_long = self.active_trade.direction == BIAS.LONG
            filled = False
            if is_long:
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

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:
            # Zombie Killer
            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG

            exit_type = None
            exit_price = None
            exit_reason = "NORMAL"  # Metadata tag

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

            # B. STANDARD CHECK (Trusting Wicks)
            if not exit_type:
                if is_long:
                    hit_sl = curr["low"] <= sl
                    hit_tp = curr["high"] >= tp
                else:
                    hit_sl = curr["high"] >= sl
                    hit_tp = curr["low"] <= tp

                if hit_sl and hit_tp:
                    hit_tp = False  # Pessimistic

                if hit_sl:
                    exit_type = "LOSS"
                    exit_price = sl
                elif hit_tp:
                    exit_type = "WIN"
                    exit_price = tp

            # C. TIME CUTOFF (Calculated as WIN/LOSS)
            is_cutoff = ts_now >= cutoff_ts
            if not exit_type and is_cutoff:
                exit_price = curr["close"]
                exit_reason = "CUTOFF"  # Mark reason, but result is binary

                # Calculate PnL manually to determine result
                if is_long:
                    pnl = exit_price - self.active_trade.price
                else:
                    pnl = self.active_trade.price - exit_price

                # Assign Result
                exit_type = "WIN" if pnl > 0 else "LOSS"

            # --- EXECUTION ---
            if exit_type:
                # TODO: Mark zone as processed if we don't want to re-enter
                # zone_id = self.active_trade.metadata.get("zone_id")
                # if zone_id:
                #     self.processed_zone_ids.add(zone_id)

                sig = self.create_signal(TradeAction.EXIT, ts_utc, custom_price=exit_price)
                sig.trade_result = exit_type  # 'WIN' or 'LOSS'
                sig.metadata["exit_reason"] = exit_reason

                # CRITICAL: Tell Execution Handler how to behave
                if exit_reason == "CUTOFF":
                    # Time exit: Bot MUST cancel brackets and dump position at market
                    sig.metadata["exit_type"] = ExitType.MARKET_FORCE
                else:
                    # SL/TP hit: Bot assumes IBKR Bracket triggered; just monitors
                    sig.metadata["exit_type"] = ExitType.SL_TP_HIT

                self.log.info(f"🎯 TRADE EXIT ({exit_type}) {self.symbol} | Reason: {exit_reason}")
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="IN_TRADE")

        # 3. NEW ENTRY SEARCH (Limit Order Placement)
        if self.trade_status == TradeStatus.NONE:
            if ts_now < entry_cutoff:
                new_trade = self.find_setup(df_ltf, df_htf, swing_zones, internal_zones, ts_utc)
                if new_trade:
                    self.active_trade = new_trade
                    self.trade_status = TradeStatus.PENDING
                    self.log.info(f"🚀 LIMIT ORDER PLACED: {new_trade.direction.name} @ {new_trade.price:.2f}")
                    return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        # 4. PENDING MANAGEMENT (Cutoff & BOS Structure Update)
        if self.trade_status == TradeStatus.PENDING:
            should_cancel = False
            cancel_reason = ""

            # A. Time Cutoff
            if ts_now >= cutoff_ts:
                should_cancel = True
                cancel_reason = "CUTOFF"

            # B. Check for BOS Change (New BOS formed)
            else:
                self.htf_bos_detector.get_current_trend(df_htf)
                latest_signals = self.htf_bos_detector.signals

                if latest_signals and self.active_trade and self.active_trade.metadata:
                    latest_bos_ts = latest_signals[-1].get("time")
                    current_trade_bos_ts = self.active_trade.metadata.get("bos_ts")

                    # If the latest BOS timestamp is different from the one that created this trade,
                    # it means structure has evolved (New BOS).
                    # We cancel this order so the next cycle can find the new, better zone.
                    if current_trade_bos_ts and latest_bos_ts != current_trade_bos_ts:
                        should_cancel = True
                        cancel_reason = "NEW_STRUCTURE"

            if should_cancel:
                self.log.info(f"🚫 CANCELLING PENDING ENTRY | Reason: {cancel_reason}")
                return SignalSummary(
                    direction="CANCELLED",
                    trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc),
                )

        return SignalSummary(direction="NEUTRAL")
