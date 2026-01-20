from datetime import timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SESSION_MAP, SignalSummary, StrategyBase, TradeAction, \
    TradeSignal, TradeStatus
from daily_trading.settings import logger

# Keep Config as is, mapped fvg_mitigation_ratio to zone_depth inside strategy
CONFIG = {
    # ... (Your existing config remains compatible) ...
}


class BosOrderBlockStrategy(StrategyBase):
    """
    STRATEGY: Multi-Timeframe Trend Following (SMC)

    REFACTORED:
    - Trend: HTF BOS/CHOCH (1H/4H)
    - Entry: Nearest Valid Swing or Internal Supply/Demand Zone (LTF)
    """

    def __init__(self,
                 symbol: str,
                 timezone: ZoneInfo,
                 main_tf="5m",
                 htf="1h",
                 risk: float = 150.0,
                 tp_multiple: float = 2.0,
                 fvg_mitigation_ratio: float = 0.5,  # Treated as Depth into Zone
                 sl_buffer_mult: float = 0.1,
                 atr_length: int = 14):

        super().__init__(main_tf)

        self.symbol = symbol
        self.timezone = timezone
        self.htf = htf
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = fvg_mitigation_ratio
        self.sl_buffer_mult = sl_buffer_mult

        # State
        self.active_bos_timestamp = None
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids = set()

        # Logging
        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS-SD | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

        # 1. Trend Detector
        self.htf_bos_detector = BosChochDetector(main_tf=htf)

        # 2. Zone Detector (Refactored)
        self.sd_detector = SupplyDemandZonesDetector(
            atr_length=atr_length,
            fvg_based_only=True,
            swing_length=24,
            internal_length=3,
            zone_precision_factor=2.3,
            fvg_offset=3,
            fvg_threshold=0.25,
            wick_ratio=0.10
        )

        # Session Cutoff
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = 5
        else:
            self.end_buffer_minutes = 0

    def reset_session_state(self):
        self.log.info("Resetting Session State.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.active_bos_timestamp = None

    # ==================================================================
    # 1. HELPERS: Zone Finding & Validation
    # ==================================================================

    @staticmethod
    def is_zone_invalidated(df_ltf: pd.DataFrame, zone: Dict) -> bool:
        """
        Checks if a zone was broken by a candle CLOSE between its creation and now.
        """
        zone_time = zone['start_time']

        # Timezone Normalization
        try:
            if df_ltf.index.tz is None:
                zone_time = zone_time.replace(tzinfo=None)
            elif zone_time.tzinfo is None:
                zone_time = zone_time.tz_localize(df_ltf.index.tz)
            else:
                zone_time = zone_time.tz_convert(df_ltf.index.tz)
        except Exception:
            zone_time = zone_time.replace(tzinfo=None)

        # Get history strictly AFTER zone creation (exclude current candle)
        history = df_ltf[df_ltf.index > zone_time].iloc[:-1]

        if history.empty:
            return False

        # Invalidation Logic
        if zone['bias'] == BIAS.LONG:  # Demand Zone
            # Broken if a candle CLOSES below the bottom
            if history['close'].min() < zone['bottom']:
                return True
        else:  # Supply Zone
            # Broken if a candle CLOSES above the top
            if history['close'].max() > zone['top']:
                return True

        return False

    def find_nearest_zone(self, df_ltf: pd.DataFrame, direction: BIAS, current_price: float):
        """
        Finds the nearest ACTIVE Supply/Demand zone matching the direction.
        Combines Swing and Internal zones.
        """
        # Calculate Zones
        swing_zones, internal_zones = self.sd_detector.calculate(df_ltf)

        # Combine lists
        all_zones = swing_zones + internal_zones
        if not all_zones:
            return None

        candidates = []

        for zone in all_zones:
            # 1. Basic Filters
            if not zone.get('active', False):
                continue
            if zone['bias'] != direction:
                continue

            # 2. Check if already processed
            # Use Start Time + Top Level as ID to differentiate overlapping zones
            z_id = f"{zone['start_time']}_{zone['top']}_{self.symbol}"
            if z_id in self.processed_zone_ids:
                continue

            # 3. Price Logic (Must be valid for Limit Order)
            # Longs: Zone must be BELOW price
            # Shorts: Zone must be ABOVE price
            if direction == BIAS.LONG:
                if zone['top'] >= current_price:
                    continue
            elif direction == BIAS.SHORT:
                if zone['bottom'] <= current_price:
                    continue

            # 4. History Validation
            if self.is_zone_invalidated(df_ltf, zone):
                self.processed_zone_ids.add(z_id)
                continue

            candidates.append(zone)

        if not candidates:
            return None

        # SORTING: Find the NEAREST zone to price
        if direction == BIAS.LONG:
            # For Demand (below price), we want the HIGHEST zone (closest to price)
            candidates.sort(key=lambda x: x['top'], reverse=True)
        else:
            # For Supply (above price), we want the LOWEST zone (closest to price)
            candidates.sort(key=lambda x: x['bottom'], reverse=False)

        return candidates[0]

    # ==================================================================
    # 2. TRADE CALCULATION
    # ==================================================================

    def calculate_setup_params(self, zone, direction: BIAS, df_ltf: pd.DataFrame):
        """
        Calculates Entry, SL, TP based on Zone geometry.
        """
        # We need ATR for buffer
        # Since SD detector calculates ATR internally, we can recalculate or fetch if exposed.
        # Recalculating here for safety:
        atr = self.sd_detector.get_current_atr(df_ltf)

        top = zone['top']
        bottom = zone['bottom']
        height = top - bottom

        # Entry Logic:
        # entry_depth 0.0 = Proximal Line (Top for Demand, Bottom for Supply)
        # entry_depth 0.5 = 50% into the zone

        if direction == BIAS.LONG:  # Demand
            # Entry: Top minus depth
            entry = top - (height * self.entry_depth)
            # SL: Bottom minus buffer
            sl = bottom - (atr * self.sl_buffer_mult)
        else:  # Supply
            # Entry: Bottom plus depth
            entry = bottom + (height * self.entry_depth)
            # SL: Top plus buffer
            sl = top + (atr * self.sl_buffer_mult)

        pos_math = self.calculate_trade_position(entry, sl, direction)

        return {
            "entry": entry, "sl": pos_math['sl'], "tp": pos_math['tp'],
            "size": pos_math['size'], "direction": direction
        }

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """ Calculates size with risk floors. """
        chart_dist = abs(entry_price - sl)
        if chart_dist == 0: chart_dist = 0.25

        hard_floor = self.HARD_MIN_DIST.get(self.symbol, 0.0)
        effective_dist = max(chart_dist, hard_floor)

        risk_per_contract = effective_dist * self.point_value
        if risk_per_contract <= 0: risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        if raw_size > max_contracts:
            self.log.warning(f"⚠️ RISK CAP: {raw_size} -> {size}")

        if direction == BIAS.LONG:
            tp = entry_price + (chart_dist * self.tp_multiple)
        else:
            tp = entry_price - (chart_dist * self.tp_multiple)

        return {"sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        report_price = custom_price if custom_price is not None else self.active_trade.price
        sig = TradeSignal(
            symbol=self.symbol, action=action, direction=self.active_trade.direction,
            price=report_price, sl=self.active_trade.sl, tp=self.active_trade.tp,
            size=self.active_trade.size, timestamp=timestamp
        )
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade = None
            self.trade_status = TradeStatus.NONE
        return sig

    def check_order_fill(self, candle) -> bool:
        if not self.active_trade: return False
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= self.active_trade.price
        else:
            return candle['high'] >= self.active_trade.price

    # ==================================================================
    # 3. MAIN LOOP
    # ==================================================================

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame, df_4h: pd.DataFrame = None) -> SignalSummary:
        curr = df_ltf.iloc[-1]
        ts_now = curr.name.tz_convert(self.timezone)
        ts_utc = curr.name.tz_convert("UTC")

        s_info = SESSION_MAP[self.timezone.key]
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16), minute=s_info.get("end_minute", 0), second=0)
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)

        just_filled = False

        # 1. FILL CHECK
        if self.trade_status == TradeStatus.PENDING:
            if self.check_order_fill(curr):
                self.trade_status = TradeStatus.FILLED
                just_filled = True
                self.log.info(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price:.2f}")

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:
            # Zombie Check
            if self.active_trade.sl <= 0:
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG
            exit_type, exit_price = None, None

            # Gap Check
            if not just_filled:
                if is_long and curr['open'] <= sl:
                    exit_type, exit_price = 'LOSS', curr['open']
                elif is_long and curr['open'] >= tp:
                    exit_type, exit_price = 'WIN', curr['open']
                elif not is_long and curr['open'] >= sl:
                    exit_type, exit_price = 'LOSS', curr['open']
                elif not is_long and curr['open'] <= tp:
                    exit_type, exit_price = 'WIN', curr['open']

            # Standard Check
            if not exit_type:
                hit_sl, hit_tp = False, False
                if just_filled:
                    # Strict Mode on Entry Candle
                    if is_long:
                        hit_sl = curr['low'] <= sl
                        hit_tp = curr['close'] >= tp
                    else:
                        hit_sl = curr['high'] >= sl
                        hit_tp = curr['close'] <= tp
                else:
                    if is_long:
                        hit_sl = curr['low'] <= sl
                        hit_tp = curr['high'] >= tp
                    else:
                        hit_sl = curr['high'] >= sl
                        hit_tp = curr['low'] <= tp

                if hit_sl and hit_tp: hit_tp = False  # Assume worst

                if hit_sl:
                    exit_type, exit_price = 'LOSS', sl
                elif hit_tp:
                    exit_type, exit_price = 'WIN', tp

            # Time Cutoff
            if not exit_type and ts_now >= cutoff_ts:
                exit_type, exit_price = 'CUTOFF', curr['close']

            if exit_type:
                sig = self.create_signal(TradeAction.EXIT, ts_utc, custom_price=exit_price)
                if exit_type in ['WIN', 'LOSS']: sig.trade_result = exit_type
                self.log.info(f"🎯 TRADE EXIT ({exit_type}) {self.symbol} @ {exit_price:.2f}")
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="IN_TRADE")

        # 3. ENTRY SEARCH
        if ts_now < cutoff_ts:
            trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
            bos_signals = self.htf_bos_detector.signals

            # HTF Alignment Check
            if df_4h is not None:
                bias_4h = self.htf_bos_detector.get_current_trend(df_4h)
                if trend_bias != bias_4h and bias_4h != BIAS.NEUTRAL:
                    return SignalSummary(direction="NEUTRAL")

            if trend_bias != BIAS.NEUTRAL and bos_signals:
                last_bos = bos_signals[-1]
                bos_time = last_bos.get("time")

                is_new_bos = (self.active_bos_timestamp is None or bos_time != self.active_bos_timestamp)

                # Check for new entry if no trade, OR if structure changed while pending
                should_search = (self.trade_status == TradeStatus.NONE) or \
                                (self.trade_status == TradeStatus.PENDING and is_new_bos)

                if should_search:
                    # REFACTORED: Find Zone instead of FVG
                    zone = self.find_nearest_zone(df_ltf, trend_bias, curr['close'])

                    if zone:
                        params = self.calculate_setup_params(zone, trend_bias, df_ltf)

                        if params['entry'] <= 0 or params['sl'] <= 0:
                            return SignalSummary(direction="NEUTRAL")

                        # Unique ID for this zone
                        z_id = f"{zone['start_time']}_{zone['top']}_{self.symbol}"

                        # Case 1: Enter New Trade
                        if self.trade_status == TradeStatus.NONE:
                            new_trade = TradeSignal(
                                action=TradeAction.ENTER,
                                symbol=self.symbol,
                                direction=params['direction'],
                                price=params['entry'], sl=params['sl'], tp=params['tp'], size=params['size'],
                                timestamp=curr.name,
                                metadata={"zone_id": z_id, "bos_id": bos_time}
                            )
                            self.active_trade = new_trade
                            self.trade_status = TradeStatus.PENDING
                            self.processed_zone_ids.add(z_id)
                            self.active_bos_timestamp = bos_time
                            self.log.info(f"🚀 SIGNAL: {params['direction'].name} Zone Limit @ {params['entry']:.2f}")
                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

                        # Case 2: Cancel & Replace (Structure Changed)
                        elif self.trade_status == TradeStatus.PENDING and is_new_bos:
                            current_zone_id = self.active_trade.metadata.get('zone_id')
                            if current_zone_id != z_id:
                                self.log.info(f"♻️ NEW STRUCTURE: Cancelling old order.")
                                cancel_sig = self.create_signal(TradeAction.CANCEL, ts_utc)
                                self.active_bos_timestamp = bos_time
                                return SignalSummary(direction="CANCELLED", trade_signal=cancel_sig)

        # 4. PENDING CANCEL (Cutoff)
        if self.trade_status == TradeStatus.PENDING:
            if ts_now >= cutoff_ts:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc))

        return SignalSummary(direction="NEUTRAL")
