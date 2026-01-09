from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SMCBase import BIAS, SMCBase
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeAction, TradeSignal, TradeStatus
from daily_trading.settings import logger


class SupplyDemandZoneTapStrategy(SMCBase):
    """
    Institutional HTF Zone Tap Strategy.

    Logic Hierarchy:
    1. Context: Calculates 1H Trend and S/D Zones once per hour.
    2. Filtering: Only allows trades in the direction of the 1H Trend.
    3. Trigger: Monitors 5m TF for a 'Tap' (Touch) of the 1H Zone.
    4. Confirmation: 5m candle must show 'Absorption' (wick) and not be a 'Falling Knife'.
    5. Execution: Limit order placed at the 1H zone edge.
    6. Safety: Hard cutoff at session end and automatic mitigation of zones after one trade.
    """

    # Updated Configuration with MCL
    POINT_VALUE = {
        "MNQ": 2.0,
        "MES": 5.0,
        "MGC": 10.0,
        "MBT": 0.10,
        "SIL": 1000.0,
        "MCL": 100.0  # $100 per 1.00 point ($1 per 0.01 tick)
    }

    TICK_BUFFER = {
        "MNQ": 0.5,
        "MES": 0.5,
        "MGC": 0.2,
        "MBT": 5.0,
        "SIL": 0.010,
        "MCL": 0.02  # 2 ticks (Tick size is 0.01)
    }

    ENTRY_BUFFER = {
        "MNQ": 0.5,
        "MES": 0.5,
        "MGC": 0.2,
        "MBT": 5.0,
        "SIL": 0.010,
        "MCL": 0.02  # 2 ticks (Tick size is 0.01)
    }

    SL_BUFFER = {
        "MNQ": 0.5,
        "MES": 0.5,
        "MGC": 0.5,
        "MBT": 5.0,
        "SIL": 0.025,
        "MCL": 0.05  # 5 ticks (Matches the conservative MGC/SIL profile)
    }

    def __init__(self, symbol: str, timezone: ZoneInfo, main_tf="5m", risk: float = 150.0,
                 tp_multiple: float = 1.5,
                 wick_ratio: float = 0.10,
                 swing_length: int = 8, internal_length: int = 2, zone_precision_factor: float = 3.0,
                 fvg_based_only: bool = True, fvg_offset: int = 2, fvg_threshold=0.12, sl_atr_multiplier=3.0, eb_mult=2.0):
        super().__init__(main_tf)
        self.symbol = symbol
        self.timezone = timezone
        self.risk = risk
        self.tp_multiple = tp_multiple
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = 5
        else:
            self.end_buffer_minutes = 0

        self.sl_atr_multiplier = sl_atr_multiplier
        self.eb_mult = eb_mult

        self.wick_ratio = wick_ratio
        # Detectors
        self.sd_detector_1h = SupplyDemandZonesDetector(fvg_based_only=fvg_based_only,
                                                        swing_length=swing_length,
                                                        internal_length=internal_length,
                                                        zone_precision_factor=zone_precision_factor,
                                                        fvg_offset=fvg_offset,
                                                        fvg_threshold=fvg_threshold)

        # State Management
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()

        # Caching
        self.cached_zones_1h: List[Dict] = []
        self.last_processed_hour: int = -1

        self.point_value = self.POINT_VALUE[symbol]
        self.tick_buffer = self.TICK_BUFFER[symbol]
        self.enter_buffer = self.ENTRY_BUFFER[symbol]

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        logger.info(f"{self.symbol} | New Session: Resetting Strategy State and Cache.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.cached_zones_1h = []
        self.last_processed_hour = -1

    def is_falling_knife(self, candle: pd.Series, direction: BIAS) -> bool:
        """Exact logic and {wick_ratio}% threshold"""
        full_range = candle['high'] - candle['low']
        if full_range <= 0: return False
        is_bearish = candle['close'] < candle['open']
        is_bullish = candle['close'] > candle['open']

        if direction == BIAS.LONG:
            rejection_wick = min(candle['open'], candle['close']) - candle['low']
        else:
            rejection_wick = candle['high'] - max(candle['open'], candle['close'])

        wick_ratio = rejection_wick / full_range
        _is_falling_knife = (direction == BIAS.LONG and is_bearish and wick_ratio < self.wick_ratio) or \
                            (direction == BIAS.SHORT and is_bullish and wick_ratio < self.wick_ratio)
        if _is_falling_knife:
            logger.info(f"Detected Falling Knife pattern at {candle.time}")
        return _is_falling_knife

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Trigger Logic: Restored to 'Striking Distance' (+ buffer for Longs)."""
        curr = ohlc.iloc[-1]
        if direction == BIAS.LONG:
            # Calculate Striking Distance for Longs
            touched = curr['low'] <= zone['top']

        else:
            # Calculate Striking Distance for Shorts
            touched = curr['high'] >= zone['bottom']

        if not touched:
            return False
        return not self.is_falling_knife(curr, direction)

    def calculate_trade_params(self, ohlc, zone: Dict, direction: BIAS) -> Dict:
        """Calculates entry at zone edge and SL beyond HTF zone boundary."""
        # curr_price = ohlc.iloc[-1]["close"]
        curr_candle = ohlc.iloc[-1]
        adjusted_buffer = self.enter_buffer * self.eb_mult

        if direction == BIAS.LONG:
            entry_price = curr_candle["low"] - adjusted_buffer
            # entry_price = zone["top"] + self.enter_buffer
            sl = self.calculate_atr_stop_loss(ohlc,
                                              entry_price,
                                              direction=direction,
                                              atr_length=14,
                                              atr_multiplier=self.sl_atr_multiplier)
            r = entry_price - sl
            tp = entry_price + (r * self.tp_multiple)
        else:
            entry_price = curr_candle["high"] + adjusted_buffer
            # entry_price = zone["bottom"] - self.enter_buffer
            sl = self.calculate_atr_stop_loss(ohlc,
                                              entry_price,
                                              direction=direction,
                                              atr_length=14,
                                              atr_multiplier=self.sl_atr_multiplier)
            r = sl - entry_price
            tp = entry_price - (r * self.tp_multiple)

        risk_per_contract = r * self.point_value
        size = max(1, int(self.risk / risk_per_contract))

        return {"entry": entry_price, "sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp: pd.Timestamp) -> TradeSignal:
        """Utility for state cleanup and signal generation."""
        sig = TradeSignal(
            symbol=self.symbol,
            action=action,
            direction=self.active_trade.direction,
            price=self.active_trade.price,
            sl=self.active_trade.sl,
            tp=self.active_trade.tp,
            size=self.active_trade.size,
            timestamp=timestamp
        )
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade = None
            self.trade_status = TradeStatus.NONE
        return sig

    def analyze(self, ohlc_5m: pd.DataFrame, ohlc_1h: pd.DataFrame) -> SignalSummary:
        """Main strategy execution loop: High-Priority Fill & Exit Logic."""
        curr_5m = ohlc_5m.iloc[-1]
        ts_now = curr_5m.name.tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]

        # 1. HTF Caching (Zones refresh once per hour)
        if ts_now.hour != self.last_processed_hour:
            swing_zones, internal_zones = self.sd_detector_1h.calculate(ohlc_1h)
            self.cached_zones_1h = swing_zones + internal_zones
            self.last_processed_hour = ts_now.hour

        # 2. Define Cutoffs
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16), minute=s_info.get("end_minute", 0))
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)

        # 3. Handle FILLED Trades (Check SL/TP EVERY 5m bar)
        if self.trade_status == TradeStatus.FILLED:
            # Check Hard Session Exit
            if ts_now >= cutoff_ts:
                return SignalSummary(direction="EXIT", trade_signal=self.create_signal(TradeAction.EXIT, curr_5m.name))

            trade = self.active_trade
            is_long = trade.direction == BIAS.LONG

            # SL/TP logic
            hit = (is_long and (curr_5m['low'] <= trade.sl or curr_5m['high'] >= trade.tp)) or \
                  (not is_long and (curr_5m['high'] >= trade.sl or curr_5m['low'] <= trade.tp))

            if hit:
                return SignalSummary(direction="EXIT", trade_signal=self.create_signal(TradeAction.EXIT, curr_5m.name))

            # If we hit the cutoff while in trade
            if ts_now >= cutoff_ts:
                return SignalSummary(direction="EXIT", trade_signal=self.create_signal(TradeAction.EXIT, curr_5m.name))

            return SignalSummary(direction="IN_TRADE")

        # 4. Handle PENDING Orders (Fill Detection FIRST)
        if self.trade_status == TradeStatus.PENDING:
            trade = self.active_trade
            is_long = trade.direction == BIAS.LONG

            # Check Fill
            filled = (is_long and curr_5m['low'] <= trade.price) or \
                     (not is_long and curr_5m['high'] >= trade.price)

            if filled:
                self.trade_status = TradeStatus.FILLED
                # Immediate SL/TP check on the same bar as fill
                hit = (is_long and (curr_5m['low'] <= trade.sl or curr_5m['high'] >= trade.tp)) or \
                      (not is_long and (curr_5m['high'] >= trade.sl or curr_5m['low'] <= trade.tp))

                if hit or ts_now >= cutoff_ts:
                    return SignalSummary(direction="EXIT",
                                         trade_signal=self.create_signal(TradeAction.EXIT, curr_5m.name))
                return SignalSummary(direction="FILLED")

            # If NOT filled, check for cancellation
            if ts_now >= cutoff_ts:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, curr_5m.name))

            # Target reached before entry = Cancel
            hit_target_early = (is_long and curr_5m['high'] >= trade.tp) or \
                               (not is_long and curr_5m['low'] <= trade.tp)
            if hit_target_early:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, curr_5m.name))

            return SignalSummary(direction="PENDING")

        # 5. Scan for NEW Taps (Only if session is active)
        if self.trade_status == TradeStatus.NONE and ts_now < cutoff_ts:
            for zone in self.cached_zones_1h:
                z_id = zone['start_time']
                if z_id in self.processed_zone_ids: continue

                if self.validate_zone_reaction(ohlc_5m, zone['bias'], zone):
                    params = self.calculate_trade_params(ohlc_5m, zone, zone['bias'])

                    # Basic sanity check
                    if (zone['bias'] == BIAS.LONG and curr_5m['close'] >= params['tp']) or \
                            (zone['bias'] == BIAS.SHORT and curr_5m['close'] <= params['tp']):
                        self.processed_zone_ids.add(z_id)
                        continue

                    trade = TradeSignal(
                        action=TradeAction.ENTER, symbol=self.symbol, direction=zone['bias'],
                        price=params['entry'], sl=params['sl'], tp=params['tp'],
                        size=params['size'], timestamp=curr_5m.name, metadata={'zone_id': z_id}
                    )
                    self.active_trade, self.trade_status = trade, TradeStatus.PENDING
                    self.processed_zone_ids.add(z_id)
                    return SignalSummary(direction=trade.direction.name, trade_signal=trade)

        return SignalSummary(direction="NEUTRAL")
