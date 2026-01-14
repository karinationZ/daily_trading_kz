from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase, TradeAction, TradeSignal, \
    TradeStatus
from daily_trading.settings import logger


class SupplyDemandZoneTapStrategy(StrategyBase):
    """
    Institutional HTF Zone Tap Strategy.

    Logic Hierarchy:
        - Trigger: Monitors 5m TF for a 'Tap' (Touch) of the 1H Zone.
        - Confirmation: 5m candle must show 'Absorption' (wick) and not be a 'Falling Knife'.
        - Execution: Limit order placed at the 1H zone edge. (???fix it) sl?
         - Safety: Hard cutoff at session end and automatic mitigation of zones after one trade.
    """

    def __init__(self, symbol: str, timezone: ZoneInfo, main_tf="5m", risk: float = 150.0,
                 tp_multiple: float = 1.5,
                 wick_ratio: float = 0.10,
                 swing_length: int = 8,
                 internal_length: int = 2,
                 zone_precision_factor: float = 2.0,
                 fvg_based_only: bool = True,
                 fvg_offset: int = 3,
                 fvg_threshold=0.15,
                 sl_atr_multiplier=2.0,
                 eb_mult=2.0):
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
        self.enter_buffer = self.ENTRY_BUFFER[symbol]
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

        tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"1H TAP | {symbol}:{tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)

        # Unique IBKR Tag
        self.ext_tag = f"{symbol}_{tz_short}_1H_TAP"

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        logger.info(f"{self.symbol} | New Session: Resetting Strategy State and Cache.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.cached_zones_1h = []
        self.last_processed_hour = -1

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Trigger Logic: Restored to 'Striking Distance' (+ buffer for Longs)."""
        atr = self.sd_detector_1h.get_current_atr(ohlc)
        entry_buffer = self.enter_buffer * self.eb_mult

        curr = ohlc.iloc[-1]
        if direction == BIAS.LONG:
            touched = curr['low'] <= (zone['top'] - entry_buffer)
        else:
            touched = curr['high'] >= (zone['bottom'] + entry_buffer)

        if not touched:
            return False
        is_falling_knf = self.sd_detector_1h.is_falling_knife(curr, direction, atr)

        if is_falling_knf:
            self.log.info(f"Detected Falling knife. | {curr.name}")
            return False
        else:
            return True

    def calculate_trade_params(self, ohlc, zone: Dict, direction: BIAS) -> Dict:
        """Calculates entry at zone edge and SL beyond HTF zone boundary."""
        curr_candle = ohlc.iloc[-1]
        adjusted_buffer = self.enter_buffer * self.eb_mult
        if direction == BIAS.LONG:
            entry_price = curr_candle["low"] - adjusted_buffer
            sl = self.sd_detector_1h.calculate_atr_stop_loss(ohlc,
                                                             entry_price,
                                                             direction=direction,
                                                             atr_length=14,
                                                             atr_multiplier=self.sl_atr_multiplier)
            r = entry_price - sl
            tp = entry_price + (r * self.tp_multiple)
        else:
            entry_price = curr_candle["high"] + adjusted_buffer
            sl = self.sd_detector_1h.calculate_atr_stop_loss(ohlc,
                                                             entry_price,
                                                             direction=direction,
                                                             atr_length=14,
                                                             atr_multiplier=self.sl_atr_multiplier)

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
        max_contracts = 5
        size = max(1, min(raw_size, max_contracts))

        if raw_size > max_contracts:
            self.log.warning(
                f"⚠️ RISK CAP: Math wanted {raw_size} contracts, capped at {size}. "
                f"(Risk per contract: ${risk_per_contract:.2f})")

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
            self.log.info(f"1H Zones: {self.cached_zones_1h}")

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
                if z_id in self.processed_zone_ids:
                    continue

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
                        size=params['size'], timestamp=curr_5m.name, metadata={'zone_id': z_id,
                                                                               "sl_atr": self.sl_atr_multiplier,
                                                                               "eb_mult": self.eb_mult}
                    )
                    self.active_trade, self.trade_status = trade, TradeStatus.PENDING
                    self.processed_zone_ids.add(z_id)
                    return SignalSummary(direction=trade.direction.name, trade_signal=trade)

        return SignalSummary(direction="NEUTRAL")
