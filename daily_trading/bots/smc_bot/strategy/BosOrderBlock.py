from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SESSION_MAP, SignalSummary, StrategyBase, TradeAction, \
    TradeSignal, \
    TradeStatus
from daily_trading.settings import logger


# TEST RESULT ARE EXACTLY THE SAME AS ORBStrategy - IDENTICAL
class BosOrderBlockStrategy(StrategyBase):
    """
    STRATEGY NAME: Institutional Structure Pro (BOS)
    VERSION: 1.0
    TYPE: Trend Following (SMC)

    OVERVIEW:
    This strategy identifies the dominant Market Structure (Trend) on a Higher Timeframe (HTF)
    via Break of Structure (BOS) logic. It then enters trades on the Trading Timeframe
    by waiting for price to retrace into premium Supply/Demand zones that align with that trend.

    PHASES OF EXECUTION:

    PHASE 1: STRUCTURAL BIAS (HTF)
    - Resamples data to a user-defined HTF (e.g., 1H).
    - Detects the last Break of Structure (BOS) to determine Trend (Bullish/Bearish).

    PHASE 2: ZONE IDENTIFICATION
    - Identifies Supply/Demand zones on the Main Timeframe (e.g., 5m).
    - Filters: Only zones matching the HTF Trend are considered.

    PHASE 3: THE TRIGGER (RETRACE)
    - Waits for price to retrace to the zone.
    - Validates "Falling Knife" scenarios (skips if momentum is too strong against us).

    PHASE 4: EXECUTION & MANAGEMENT
    - Same robust execution logic as ORB Strategy (Pending orders, hard stops, ATR buffers).
    """

    def __init__(self, symbol: str,
                 timezone: ZoneInfo,
                 main_tf="5m",
                 htf_tf="1h",  # Higher Timeframe for Trend Detection
                 risk: float = 150.0,
                 tp_multiple: float = 1.5,
                 wick_ratio: float = 0.10,
                 atr_length=200,
                 swing_length: int = 50,
                 internal_length: int = 5,
                 zone_precision_factor: float = 2.0,
                 fvg_based_only: bool = True,
                 fvg_offset: int = 3,
                 fvg_threshold=0.15,
                 entry_buffer_mult: float = 0.05,
                 sl_buffer_mult: float = 0.1,
                 tick_buffer_mult: float = 0.05,
                 bos_left_bars: int = 8,
                 bos_right_bars: int = 3,
                 bos_volume_thresh: float = 0):

        super().__init__(main_tf)
        self.symbol = symbol
        self.timezone = timezone
        self.htf_tf = htf_tf

        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{self.tz_short}_BOS"

        # Risk & Timings
        self.risk = risk
        self.tp_multiple = tp_multiple
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = 5
        else:
            self.end_buffer_minutes = 0

        # --- DETECTORS ---

        # 1. Structure Detector (Runs on HTF)
        self.htf_bos_detector = BosChochDetector(
            main_tf=htf_tf,
            left_bars=bos_left_bars,
            right_bars=bos_right_bars,
            wick_ratio=wick_ratio,
            volume_thresh=bos_volume_thresh
        )

        # 2. Zone Detector (Runs on Main TF)
        self.sd_detector = SupplyDemandZonesDetector(
            atr_length=atr_length,
            fvg_based_only=fvg_based_only,
            swing_length=swing_length,
            internal_length=internal_length,
            zone_precision_factor=zone_precision_factor,
            fvg_offset=fvg_offset,
            fvg_threshold=fvg_threshold,
            wick_ratio=wick_ratio
        )

        # State Variables
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()
        self.active_bos_timestamp = None  # Tracks the specific BOS leg we are trading

        # Buffers (Static or loaded from constants)
        self.point_value = self.POINT_VALUE.get(symbol, 1.0)
        # Fallbacks if symbol not in map
        self.entry_buffer = self.ENTRY_BUFFER.get(symbol, 0.0001)
        self.sl_buffer = self.SL_BUFFER.get(symbol, 0.0001)
        self.tick_buffer = self.ENTRY_BUFFER.get(symbol, 0.0001)

        # Adaptive Param Multipliers
        self.entry_buffer_mult = entry_buffer_mult
        self.sl_buffer_mult = sl_buffer_mult
        self.tick_buffer_mult = tick_buffer_mult
        self.wick_ratio = wick_ratio
        self.zones = None

    def reset_session_state(self):
        """Resets daily state, though BOS trend might persist across days."""
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        # Note: We do NOT reset active_bos_timestamp here usually,
        # as HTF trend persists across sessions.

    # -------------------------------------------------------------------------
    # VALIDATION & CALCULATIONS (Shared Logic)
    # -------------------------------------------------------------------------

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Checks if price is in 'striking distance' and validates momentum."""
        atr = self.sd_detector.get_current_atr(ohlc)
        tick_buffer = atr * self.tick_buffer_mult

        curr = ohlc.iloc[-1]

        # 1. Striking Distance Check
        if direction == BIAS.LONG:
            touched = curr['low'] <= (zone['top'] + tick_buffer)
        else:
            touched = curr['high'] >= (zone['bottom'] - tick_buffer)

        if not touched:
            return False

        # 2. Falling Knife Check (Don't catch a runaway train)
        is_falling_knf = self.sd_detector.is_falling_knife(curr, direction, atr)
        if is_falling_knf:
            self.log.info(f"🔪 Falling knife detected. Skipping entry. | {curr.name}")
            return False

        return True

    def get_execution_levels(self, ohlc, zone: Dict, zone_type: str, direction: BIAS, current_close: float):
        """Calculates exact Entry and SL prices based on Volatility (ATR)."""
        atr = self.sd_detector.get_current_atr(ohlc)

        entry_buffer = atr * self.entry_buffer_mult
        sl_buffer = atr * self.sl_buffer_mult
        tick_buffer = atr * self.tick_buffer_mult
        min_atr_dist = atr * 1.0  # Minimum width for SL to breathe

        # Determine structural levels
        is_london_logic = self.timezone in [ZoneInfo("Europe/London"), ZoneInfo("Asia/Tokyo")]

        if is_london_logic:
            if zone_type == "SWING":
                if direction == BIAS.LONG:
                    entry = current_close + entry_buffer
                    structural_sl = zone["bottom"] - tick_buffer
                else:
                    entry = current_close - entry_buffer
                    structural_sl = zone["top"] + tick_buffer
            else:  # Internal
                if direction == BIAS.LONG:
                    entry = zone["top"] + entry_buffer
                    structural_sl = zone["bottom"] - sl_buffer
                else:
                    entry = zone["bottom"] - entry_buffer
                    structural_sl = zone["top"] + sl_buffer
        else:
            # NY Logic
            if direction == BIAS.LONG:
                entry = current_close + entry_buffer
                structural_sl = zone["bottom"] - sl_buffer
            else:
                entry = current_close - entry_buffer
                structural_sl = zone["top"] + sl_buffer

        # Enforce Minimum Volatility Distance
        current_dist = abs(entry - structural_sl)

        if direction == BIAS.LONG:
            volatility_sl = entry - min_atr_dist
            final_sl = min(structural_sl, volatility_sl)  # Take the lower (safer) one
        else:
            volatility_sl = entry + min_atr_dist
            final_sl = max(structural_sl, volatility_sl)  # Take the higher (safer) one

        return entry, final_sl

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """Calculates Size based on Risk and returns TP."""
        r = abs(entry_price - sl)
        if r == 0: r = 0.25  # Safety

        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0: risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        return {"sl": sl, "tp": tp, "size": size}

    # -------------------------------------------------------------------------
    # ENTRY LOGIC
    # -------------------------------------------------------------------------

    def check_for_entry_signal(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                               swing_zones: List[Dict], internal_zones: List[Dict]):
        """
        Combines HTF Trend + LTF Zones to find an entry.
        """
        # 1. GET HTF TREND BIAS
        # self.htf_bos_detector.calculate(df_htf) must have been called in analyze()
        trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
        bos_signals = self.htf_bos_detector.signals

        if trend_bias == BIAS.NEUTRAL or not bos_signals:
            return None

        # 2. CHECK BOS FRESHNESS
        last_bos = bos_signals[-1]
        bos_time = last_bos.get("time")

        # If structure shifted while we were pending (and not filled), we might need logic to reset
        # For now, we update the active timestamp
        self.active_bos_timestamp = bos_time

        # 3. FILTER ZONES BY BIAS
        # We only look for zones that match the HTF trend
        active_direction = trend_bias

        # Filter Zones
        all_candidates = [(z, "SWING") for z in swing_zones] + [(z, "INTERNAL") for z in internal_zones]

        # Define Time Lookback (Don't trade ancient zones)
        # Using 4 hours for 5m chart is standard, or session start
        last_local = df_ltf.index[-1].tz_convert(self.timezone)
        # s_info = SESSION_MAP[self.timezone.key]
        # session_start = last_local.replace(hour=s_info["start_hour"], minute=s_info["start_minute"]).tz_convert("UTC")
        lookback_limit = last_local - timedelta(hours=4)  # Configurable lookback?

        current_close = df_ltf.iloc[-1]["close"]
        best_signal, min_distance = None, float('inf')

        for zone, zone_type in all_candidates:
            zone_id = zone['start_time']
            if zone_id.tzinfo is None:
                zone_id = zone_id.tz_localize("UTC")
            else:
                zone_id = zone_id.tz_convert("UTC")

            # FILTER: Direction match, Not processed, Not too old
            if zone["bias"] != active_direction: continue
            if zone_id in self.processed_zone_ids: continue
            if zone_id < lookback_limit: continue

            # FILTER: Price Action Trigger
            if not self.validate_zone_reaction(df_ltf, active_direction, zone):
                continue

            # CALC: Levels
            entry_p, sl_p = self.get_execution_levels(df_ltf, zone, zone_type, active_direction, current_close)

            # Setup Trade
            signal, updated_dist = self.trade_setup(entry_p, sl_p, active_direction,
                                                    current_close, min_distance,
                                                    zone_id, zone_type, df_ltf.index[-1])
            if signal:
                best_signal, min_distance = signal, updated_dist

        if best_signal:
            self.processed_zone_ids.add(best_signal.metadata['zone_id'])

        return best_signal

    def trade_setup(self, entry, sl, direction, current_close, min_dist, zone_id, zone_type, ts):
        if entry <= 0 or sl <= 0: return None, min_dist

        try:
            pos = self.calculate_trade_position(entry, sl, direction)

            # Invalid if TP already hit
            if (direction == BIAS.LONG and current_close >= pos["tp"]) or \
                    (direction == BIAS.SHORT and current_close <= pos["tp"]):
                return None, min_dist

            dist = abs(entry - current_close)
            if dist < min_dist:
                signal = TradeSignal(
                    action=TradeAction.ENTER, symbol=self.symbol, direction=direction,
                    price=entry, sl=pos["sl"], tp=pos["tp"], size=pos["size"],
                    timestamp=ts, metadata={"zone_id": zone_id, "type": zone_type, "bos_ts": self.active_bos_timestamp}
                )
                return signal, dist
        except ValueError:
            pass
        return None, min_dist

    # -------------------------------------------------------------------------
    # MAIN ANALYZE LOOP
    # -------------------------------------------------------------------------

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        ts_now = df_ltf.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        curr = df_ltf.iloc[-1]

        # 1. TIME/SESSION CUTOFFS
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16),
                                     minute=s_info.get("end_minute", 0),
                                     second=0, microsecond=0)

        # Hard cutoff for pending orders (e.g., don't fill 5 mins before close)
        pending_cutoff = session_end - timedelta(minutes=self.end_buffer_minutes)
        # Cutoff for looking for NEW trades (e.g., 30 mins before close)
        entry_search_cutoff = session_end - timedelta(minutes=30)

        # 2. RUN DETECTORS
        # A. Resample & Run HTF Structure

        # B. Run LTF Zones
        swing_zones, internal_zones = self.sd_detector.calculate(df_ltf)
        self.zones = swing_zones + internal_zones

        # 3. FILL LOGIC (Check if pending became filled)
        filled = False
        if self.trade_status == TradeStatus.PENDING:
            is_long = self.active_trade.direction == BIAS.LONG
            if is_long:
                filled = curr["low"] <= self.active_trade.price
            else:
                filled = curr["high"] >= self.active_trade.price

            if filled:
                self.trade_status = TradeStatus.FILLED
                self.log.info(f"⚡ FILLED | {self.symbol} | {curr.name}")

        # 4. EXIT LOGIC
        if self.trade_status == TradeStatus.FILLED:
            is_long = self.active_trade.direction == BIAS.LONG

            # SL/TP Check
            hit_sl = (is_long and curr["low"] <= self.active_trade.sl) or \
                     (not is_long and curr["high"] >= self.active_trade.sl)
            hit_tp = (is_long and curr["high"] >= self.active_trade.tp) or \
                     (not is_long and curr["low"] <= self.active_trade.tp)

            # EOD Time Exit Check
            is_cutoff_time = ts_now >= pending_cutoff

            if hit_sl or hit_tp:
                exit_price = self.active_trade.sl if hit_sl else self.active_trade.tp
                signal = self.create_signal(TradeAction.EXIT, df_ltf.index[-1], custom_price=exit_price)
                signal.trade_result = "LOSS" if hit_sl else "WIN"
                self.log.info(f"🎯 EXIT ({signal.trade_result})")
                return SignalSummary(direction="EXIT", trade_signal=signal)

            elif is_cutoff_time:
                self.log.info(f"⏰ EOD EXIT")
                exit_price = curr['close']
                pnl = (exit_price - self.active_trade.price) * (1 if is_long else -1)
                sig = self.create_signal(TradeAction.EXIT, curr.name, custom_price=exit_price)
                sig.trade_result = "WIN" if pnl > 0 else "LOSS"
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="FILLED" if filled else "IN_TRADE")

        # 5. PENDING MANAGEMENT
        if self.trade_status == TradeStatus.PENDING:
            # Cancel if session ends
            if ts_now >= pending_cutoff:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, df_ltf.index[-1]))

            # Cancel if TP hit before entry (Invalidation)
            curr = df_ltf.iloc[-1]
            if (self.active_trade.direction == BIAS.LONG and curr['high'] >= self.active_trade.tp) or \
                    (self.active_trade.direction == BIAS.SHORT and curr['low'] <= self.active_trade.tp):
                self.log.info("🚫 CANCEL: Price hit TP before Entry.")
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, df_ltf.index[-1]))

            # Cancel if Structure Changed (Optional: Only if you want strict strict trend following)
            # current_trend = self.htf_bos_detector.get_current_trend(df_htf)
            # if current_trend != self.active_trade.direction:
            #     return SignalSummary(direction="CANCELLED", ...)

            return SignalSummary(direction="PENDING")

        # 6. NEW ENTRY SEARCH
        if self.trade_status == TradeStatus.NONE:
            if ts_now < entry_search_cutoff:
                new_trade = self.check_for_entry_signal(df_ltf, df_htf, swing_zones, internal_zones)
                if new_trade:
                    self.active_trade, self.trade_status = new_trade, TradeStatus.PENDING
                    self.log.info(f"🚀 NEW SIGNAL (BOS) | {new_trade.direction.name} @ {new_trade.price}")
                    return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        return SignalSummary(direction="NEUTRAL")

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        report_price = custom_price if custom_price is not None else self.active_trade.price
        sig = TradeSignal(symbol=self.symbol,
                          action=action,
                          direction=self.active_trade.direction,
                          price=report_price,
                          sl=self.active_trade.sl,
                          tp=self.active_trade.tp,
                          size=self.active_trade.size,
                          timestamp=timestamp)

        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            # Clean up logic
            if self.active_trade:
                self.processed_zone_ids.add(self.active_trade.metadata['zone_id'])
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig
