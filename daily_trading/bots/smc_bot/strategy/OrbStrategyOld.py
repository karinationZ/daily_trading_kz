from datetime import timedelta
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.ORBDetector import ORBDetector, SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SMCBase import BIAS, SMCBase
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeAction, TradeSignal, TradeStatus


class ORBStrategyOld(SMCBase):
    """
    Institutional ORB Strategy (5-Minute Timeframe).

    EXACT REPLICA OF 'OLD CODE' (55.8% Win Rate):
    - Restored 0.5 SL Buffer for MNQ.
    - Restored '+' buffer Striking Distance Trigger.
    - Restored 10% Wick Rejection threshold.
    - String-safe Timezone handling to ensure NY logic triggers.
    """

    POINT_VALUE = {"MNQ": 2.0, "MES": 5.0, "MGC": 10.0, "MBT": 0.10}
    ENTRY_BUFFER = {"MNQ": 0.5, "MES": 0.5, "MGC": 0.2, "MBT": 5.0}

    # FIX 1: Restored to Old Code values
    SL_BUFFER = {"MNQ": 0.5, "MES": 0.5, "MGC": 0.5, "MBT": 5.0}

    def __init__(self, symbol: str, timezone: ZoneInfo, main_tf="5m", risk: float = 150.0, tp_multiple: float = 1.5,
                 end_buffer_minutes: int = 15, fvg_offset: int = 3):
        super().__init__(main_tf)
        self.symbol = symbol
        self.timezone = timezone
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.end_buffer_minutes = end_buffer_minutes

        self.orb_detector = ORBDetector(orb_minutes=15, timezone=timezone)
        self.sd_detector = SupplyDemandZonesDetector(fvg_based_only=True, fvg_offset=fvg_offset)

        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()

        self.point_value = self.POINT_VALUE[symbol]
        self.entry_buffer = self.ENTRY_BUFFER[symbol]
        self.sl_buffer = self.SL_BUFFER[symbol]
        self.tick_buffer = self.ENTRY_BUFFER[symbol]

    def reset_session_state(self):
        self.orb_results = None
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    def _is_falling_knife(self, candle: pd.Series, direction: BIAS) -> bool:
        """Exact logic and 10% threshold from the high-performing OLD code."""
        full_range = candle['high'] - candle['low']
        if full_range <= 0: return False
        is_bearish = candle['close'] < candle['open']
        is_bullish = candle['close'] > candle['open']

        if direction == BIAS.LONG:
            rejection_wick = min(candle['open'], candle['close']) - candle['low']
        else:
            rejection_wick = candle['high'] - max(candle['open'], candle['close'])

        wick_ratio = rejection_wick / full_range
        return (direction == BIAS.LONG and is_bearish and wick_ratio < 0.10) or \
               (direction == BIAS.SHORT and is_bullish and wick_ratio < 0.10)

    def _is_absorption(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Trigger Logic: Restored to 'Striking Distance' (+ buffer for Longs)."""
        curr = ohlc.iloc[-1]
        if direction == BIAS.LONG:
            # FIX 2: Restored to + buffer to catch front-runs
            touched = curr['low'] <= (zone['top'] + self.tick_buffer)
        else:
            # FIX 2: Restored to - buffer
            touched = curr['high'] >= (zone['bottom'] - self.tick_buffer)

        if not touched: return False
        return not self._is_falling_knife(curr, direction)

    def _get_execution_levels(self, zone, zone_type, direction, current_close) -> Tuple[float, float]:
        """Restored exact 'Old Code' logic using string-safe Timezone keys."""
        tz_key = str(self.timezone)

        if "London" in tz_key:
            if zone_type == "SWING":
                entry = (current_close + self.entry_buffer) if direction == BIAS.LONG else (
                        current_close - self.entry_buffer)
                sl = (zone["bottom"] - self.tick_buffer) if direction == BIAS.LONG else (zone["top"] + self.tick_buffer)
            else:
                entry = (zone["top"] + self.entry_buffer) if direction == BIAS.LONG else (
                        zone["bottom"] - self.entry_buffer)
                sl = (zone["bottom"] - self.sl_buffer) if direction == BIAS.LONG else (zone["top"] + self.sl_buffer)
        else:
            # NY LOGIC: Matches high-performer
            sl = (zone["bottom"] - self.sl_buffer) if direction == BIAS.LONG else (zone["top"] + self.sl_buffer)
            entry = (current_close + self.entry_buffer) if direction == BIAS.LONG else (
                    current_close - self.entry_buffer)

        return entry, sl

    def _trade_setup(self, entry, sl, direction, current_close, min_dist, zone_id, zone_type, ts) -> Tuple[
        Optional[TradeSignal], float]:
        try:
            pos = self.calculate_trade_position(entry, sl, direction)
            if (direction == BIAS.LONG and current_close >= pos["tp"]) or \
                    (direction == BIAS.SHORT and current_close <= pos["tp"]):
                return None, min_dist

            dist = abs(entry - current_close)
            if dist < min_dist:
                signal = TradeSignal(
                    action=TradeAction.ENTER, symbol=self.symbol, direction=direction,
                    price=entry, sl=pos["sl"], tp=pos["tp"], size=pos["size"],
                    timestamp=ts, metadata={"zone_id": zone_id, "type": zone_type}
                )
                return signal, dist
        except ValueError:
            pass
        return None, min_dist

    def check_zone_retrace(self, ohlc: pd.DataFrame, swing_zones: List[Dict], internal_zones: List[Dict]) -> Optional[
        TradeSignal]:
        if not self.orb_results or not self.orb_results["signals"]: return None
        latest_orb_sig = max(self.orb_results["signals"], key=lambda d: d["time"])
        direction = latest_orb_sig["direction"]

        last_local = ohlc.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        session_start = last_local.replace(hour=s_info["start_hour"], minute=s_info["start_minute"], second=0,
                                           microsecond=0).tz_convert("UTC")
        lookback_start = session_start - timedelta(hours=1)

        current_close = ohlc.iloc[-1]["close"]
        best_signal, min_distance = None, float('inf')
        proximity_limit = self.tick_buffer * 100

        all_candidates = [(z, "SWING") for z in swing_zones] + [(z, "INTERNAL") for z in internal_zones]

        for zone, zone_type in all_candidates:
            zone_id = zone['start_time'].tz_convert("UTC") if zone['start_time'].tzinfo else zone[
                'start_time'].tz_localize("UTC")
            if zone["bias"] != direction or zone_id < lookback_start or zone_id in self.processed_zone_ids:
                continue

            if not self._is_absorption(ohlc, direction, zone): continue

            # Support logic
            is_supported = any(
                abs(zone["bottom"] - sz["top"]) < proximity_limit for sz in swing_zones if sz["bias"] == direction)
            if not is_supported:
                orb_level = latest_orb_sig["orb_high"] if direction == BIAS.LONG else latest_orb_sig["orb_low"]
                is_supported = abs(zone["bottom"] - orb_level) < proximity_limit
            if not is_supported and lookback_start <= zone_id: is_supported = True
            if not is_supported: continue

            # Execution
            entry_p, sl_p = self._get_execution_levels(zone, zone_type, direction, current_close)
            signal, updated_dist = self._trade_setup(entry_p, sl_p, direction, current_close, min_distance, zone_id,
                                                     zone_type, ohlc.index[-1])

            if signal:
                best_signal, min_distance = signal, updated_dist

        if best_signal: self.processed_zone_ids.add(best_signal.metadata['zone_id'])
        return best_signal

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        r = abs(entry_price - sl)
        tp = entry_price + r * self.tp_multiple if direction == BIAS.LONG else entry_price - r * self.tp_multiple
        size = max(1, int(self.risk / (r * self.point_value)))
        return {"sl": sl, "tp": tp, "size": size}

    def _create_signal(self, action: TradeAction, timestamp) -> TradeSignal:
        sig = TradeSignal(symbol=self.symbol, action=action, direction=self.active_trade.direction,
                          price=self.active_trade.price, sl=self.active_trade.sl, tp=self.active_trade.tp,
                          size=self.active_trade.size, timestamp=timestamp)
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            if self.active_trade: self.processed_zone_ids.add(self.active_trade.metadata['zone_id'])
        self.active_trade, self.trade_status = None, TradeStatus.NONE
        return sig

    def check_pending_validity(self, ohlc: pd.DataFrame, internal_zones: List[Dict], swing_zones: List[Dict]) -> \
            Optional[TradeSignal]:
        if not self.active_trade: return None
        curr = ohlc.iloc[-1]
        zone_id = self.active_trade.metadata.get('zone_id')
        if not any(z['start_time'] == zone_id for z in (internal_zones + swing_zones)):
            return self._create_signal(TradeAction.CANCEL, ohlc.index[-1])
        if (self.active_trade.direction == BIAS.LONG and curr['high'] >= self.active_trade.tp) or \
                (self.active_trade.direction == BIAS.SHORT and curr['low'] <= self.active_trade.tp):
            return self._create_signal(TradeAction.CANCEL, ohlc.index[-1])
        return None

    def analyze(self, ohlc: pd.DataFrame) -> SignalSummary:
        ts_now = ohlc.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        curr = ohlc.iloc[-1]

        if ts_now.hour == s_info["start_hour"] and ts_now.minute == s_info["start_minute"] and ts_now.second == 0:
            self.reset_session_state()

        session_end = ts_now.replace(hour=s_info.get("end_hour", 16), minute=s_info.get("end_minute", 0), second=0,
                                     microsecond=0)
        pending_cutoff = session_end - timedelta(minutes=self.end_buffer_minutes)
        final_bar_cutoff = session_end - timedelta(minutes=5)

        # 1. HARD EXIT
        if ts_now >= final_bar_cutoff and self.trade_status == TradeStatus.FILLED:
            return SignalSummary(direction="EXIT", trade_signal=self._create_signal(TradeAction.EXIT, ohlc.index[-1]))

        # 2. PENDING SAFETY
        if ts_now >= pending_cutoff and self.trade_status != TradeStatus.FILLED:
            if self.trade_status == TradeStatus.PENDING:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self._create_signal(TradeAction.CANCEL, ohlc.index[-1]))
            return SignalSummary(direction="NEUTRAL")

        # 3. Indicators
        self.orb_results = self.orb_detector.calculate(ohlc)
        swing_zones, internal_zones = self.sd_detector.calculate(ohlc)

        if self.trade_status == TradeStatus.FILLED:
            is_long = self.active_trade.direction == BIAS.LONG
            hit = (is_long and (curr["low"] <= self.active_trade.sl or curr["high"] >= self.active_trade.tp)) or \
                  (not is_long and (curr["high"] >= self.active_trade.sl or curr["low"] <= self.active_trade.tp))
            if hit: return SignalSummary(direction="EXIT",
                                         trade_signal=self._create_signal(TradeAction.EXIT, ohlc.index[-1]))
            return SignalSummary(direction="IN_TRADE")

        if self.trade_status == TradeStatus.PENDING:
            is_long = self.active_trade.direction == BIAS.LONG
            # Intra-bar Fill Check
            filled = (is_long and curr["low"] <= self.active_trade.price) or (
                    not is_long and curr["high"] >= self.active_trade.price)
            if filled:
                self.trade_status = TradeStatus.FILLED
                # Check for instant SL/TP hit in same bar
                hit = (is_long and (curr["low"] <= self.active_trade.sl or curr["high"] >= self.active_trade.tp)) or \
                      (not is_long and (curr["high"] >= self.active_trade.sl or curr["low"] <= self.active_trade.tp))
                if hit: return SignalSummary(direction="EXIT",
                                             trade_signal=self._create_signal(TradeAction.EXIT, ohlc.index[-1]))
                return SignalSummary(direction="FILLED")

            cancel_sig = self.check_pending_validity(ohlc, internal_zones, swing_zones)
            if cancel_sig: return SignalSummary(direction="CANCELLED", trade_signal=cancel_sig)
            return SignalSummary(direction="PENDING")

        if self.trade_status == TradeStatus.NONE and self.orb_results["signals"]:
            new_trade = self.check_zone_retrace(ohlc, swing_zones, internal_zones)
            if new_trade:
                self.active_trade, self.trade_status = new_trade, TradeStatus.PENDING
                return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        return SignalSummary(direction="NEUTRAL")
