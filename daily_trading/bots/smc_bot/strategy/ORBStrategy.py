from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.ORBDetector import ORBDetector, SESSION_MAP
from daily_trading.bots.smc_bot.detectors.StrategyBase import BIAS, StrategyBase
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeAction, TradeSignal, TradeStatus
from daily_trading.settings import logger


class ORBStrategy(StrategyBase):
    """
    STRATEGY NAME: Institutional ORB Pro
    VERSION: 3.0 (Alpha Extraction)
    TYPE: Intraday Institutional Liquidity Retest

    OVERVIEW:
    A professional-grade strategy that identifies Institutional order blocks
    at the market open and enters only on high-conviction retracements.
    It avoids 'chasing' breakouts, waiting instead for the 'Value Area' retest.

    PHASES OF EXECUTION:

    PHASE 1: DAILY DIRECTION (THE 15M ORB)
    - Monitor the first 15 minutes of the NY/London session.
    - A 5m candle closing outside this range sets the directional BIAS (Long/Short).
    - No trades are taken against this bias for the duration of the session.

    PHASE 2: INSTITUTIONAL FOOTPRINT (SMC + FVG)
    - Identify Supply/Demand zones anchored by a Fair Value Gap (FVG).
    - Requirement: The move out of the zone must be 'Impulsive' (Institutional Presence).
    - Lookback: Zones must be fresh (created in current session or 1hr pre-market).
    - Support: Zones must align with ORB levels or major historical structure.

    PHASE 3: THE TRIGGER (THE RETRACE)
    - Wait for price to return ('retrace') to a valid zone.
    - Proximity: If multiple zones exist, the bot prioritizes the one closest
      to the current market price (Maximum Efficiency).

    PHASE 4: CONFIRMATION (ABSORPTION CHECK)
    - Wick Validation: Candle must show a rejection wick (Absorption) of >= {wick_ratio}%.
    - Knife Prevention: If a candle slams into a zone with high momentum and
      low/no wick, the bot skips the entry to avoid being 'run over.'

    PHASE 5: EXECUTION LOGIC
    - Aggressive (Swing Zones): Entry at Current Close + Buffer (Market-style).
    - Conservative (Internal Zones): Entry at Zone Edge + Buffer (Limit-style).
    - Target: 1.5x Reward-to-Risk (RR) multiple.

    PHASE 6: RISK & SAFETY
    - Fixed Dollar Risk: Each trade sized to lose exactly $150.
    - TP Invalidation: If price hits the TP level before the entry is triggered,
      the trade is cancelled (No 'chasing' the bus).
    - Hard Cutoff: 15 minutes before session end, all pendings are cancelled.
    - Final Exit: 5 minutes before session end, all active trades are market-closed.

    MITIGATION LOGIC:
    - Each unique Zone ID is 'Mitigated' (deleted from usable list) after one
      Trade Signal generation. One Zone = One Opportunity.
    """

    def __init__(self, symbol: str,
                 timezone: ZoneInfo,
                 main_tf="5m",
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
                 atr_length_buffer: int = 14):
        super().__init__(main_tf, atr_length_buffer=atr_length_buffer)
        self.orb_results = None
        self.symbol = symbol
        self.timezone = timezone

        tz_short = "NY" if timezone == ZoneInfo("America/New_York") else "LONDON"
        self.log_tag = f"ORB | {symbol}:{tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)

        # Unique IBKR Tag
        self.ext_tag = f"{symbol}_{tz_short}_ORB"

        self.risk = risk
        self.tp_multiple = tp_multiple
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = 5
        else:
            self.end_buffer_minutes = 0
        self.orb_detector = ORBDetector(orb_minutes=15, timezone=timezone)
        self.sd_detector = SupplyDemandZonesDetector(atr_length=atr_length,
                                                     fvg_based_only=fvg_based_only,
                                                     swing_length=swing_length,
                                                     internal_length=internal_length,
                                                     zone_precision_factor=zone_precision_factor,
                                                     fvg_offset=fvg_offset,
                                                     fvg_threshold=fvg_threshold)

        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[pd.Timestamp] = set()

        self.point_value = self.POINT_VALUE[symbol]
        self.entry_buffer = self.ENTRY_BUFFER[symbol]
        self.sl_buffer = self.SL_BUFFER[symbol]
        self.tick_buffer = self.ENTRY_BUFFER[symbol]
        self.wick_ratio = wick_ratio

        # New adaptive params
        self.entry_buffer_mult = entry_buffer_mult
        self.sl_buffer_mult = sl_buffer_mult
        self.tick_buffer_mult = tick_buffer_mult
        self.wick_ratio = wick_ratio
        self.zones = None

        # New method for current ATR (extracted from your calculate_atr_stop_loss)

    def reset_session_state(self):
        self.orb_results = None
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Trigger Logic: Restored to 'Striking Distance' (+ buffer for Longs)."""
        atr = self.get_current_atr(ohlc)
        tick_buffer = atr * self.tick_buffer_mult

        curr = ohlc.iloc[-1]
        if direction == BIAS.LONG:
            # Calculate Striking Distance for Longs
            touched = curr['low'] <= (zone['top'] + tick_buffer)

        else:
            # Calculate Striking Distance for Shorts
            touched = curr['high'] >= (zone['bottom'] - tick_buffer)

        if not touched:
            return False
        is_falling_knf = self.is_falling_knife(curr, direction, atr)

        if is_falling_knf:
            self.log.info(f"Detected Falling knife. | {curr.name}")
            self.log.info(f"ZONES: {self.zones}")
            return False
        else:
            return True

    def get_execution_levels(self, ohlc, zone: Dict, zone_type: str, direction: BIAS, current_close: float):
        # Compute dynamic buffers
        atr = self.get_current_atr(ohlc)
        entry_buffer = atr * self.entry_buffer_mult
        sl_buffer = atr * self.sl_buffer_mult
        tick_buffer = atr * self.tick_buffer_mult  # For London swing SL
        tz_key = str(self.timezone).lower()

        if "london" in tz_key:
            if zone_type == "SWING":
                # London Swing: Entry based on Close, SL based on tick_buffer
                if direction == BIAS.LONG:
                    entry = current_close + entry_buffer
                    sl = zone["bottom"] - tick_buffer
                else:
                    # BIAS.SHORT
                    entry = current_close - entry_buffer
                    sl = zone["top"] + tick_buffer
            else:
                # London Internal: Entry based on Zone Edge, SL based on sl_buffer
                if direction == BIAS.LONG:
                    entry = zone["top"] + entry_buffer
                    sl = zone["bottom"] - sl_buffer
                else:
                    # BIAS.SHORT
                    entry = zone["bottom"] - entry_buffer
                    sl = zone["top"] + sl_buffer
        else:
            # NY LOGIC: All entries based on Close, SL based on sl_buffer
            if direction == BIAS.LONG:
                entry = current_close + entry_buffer
                sl = zone["bottom"] - sl_buffer
            else:
                # BIAS.SHORT
                entry = current_close - entry_buffer
                sl = zone["top"] + sl_buffer

        return entry, sl

    def trade_setup(self, entry, sl, direction, current_close, min_dist, zone_id, zone_type, ts):
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

    def check_zone_retrace(self, ohlc: pd.DataFrame, swing_zones: List[Dict], internal_zones: List[Dict]):
        if not self.orb_results or not self.orb_results["signals"]: return None
        latest_orb_sig = max(self.orb_results["signals"], key=lambda d: d["time"])
        direction = latest_orb_sig["direction"]

        last_local = ohlc.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        session_start = last_local.replace(hour=s_info["start_hour"], minute=s_info["start_minute"], second=0,
                                           microsecond=0).tz_convert("UTC")
        lookback_start = session_start - timedelta(hours=2)

        current_close = ohlc.iloc[-1]["close"]
        best_signal, min_distance = None, float('inf')

        all_candidates = [(z, "SWING") for z in swing_zones] + [(z, "INTERNAL") for z in internal_zones]

        for zone, zone_type in all_candidates:
            zone_id = zone['start_time'].tz_convert("UTC") if zone['start_time'].tzinfo else zone[
                'start_time'].tz_localize("UTC")
            if zone["bias"] != direction or zone_id in self.processed_zone_ids or zone_id < lookback_start:
                continue

            if not self.validate_zone_reaction(ohlc, direction, zone):
                continue

            # Execution
            entry_p, sl_p = self.get_execution_levels(ohlc, zone, zone_type, direction, current_close)
            signal, updated_dist = self.trade_setup(entry_p, sl_p, direction, current_close, min_distance, zone_id,
                                                    zone_type, ohlc.index[-1])

            if signal:
                best_signal, min_distance = signal, updated_dist

        if best_signal:
            self.processed_zone_ids.add(best_signal.metadata['zone_id'])
        return best_signal

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """
        Calculates position size with a HARD CAP to prevent over-leveraging on tight stops.
        """
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
        MAX_CONTRACTS = 5
        size = max(1, min(raw_size, MAX_CONTRACTS))

        # 5. Calculate Take Profit
        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        # 6. Log Warning if Capped
        if raw_size > MAX_CONTRACTS:
            self.log.warning(
                f"⚠️ RISK CAP: Math wanted {raw_size} contracts, capped at {size}. "
                f"(Risk per contract: ${risk_per_contract:.2f})")

        return {"sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp) -> TradeSignal:
        sig = TradeSignal(symbol=self.symbol,
                          action=action,
                          direction=self.active_trade.direction,
                          price=self.active_trade.price,
                          sl=self.active_trade.sl,
                          tp=self.active_trade.tp,
                          size=self.active_trade.size,
                          timestamp=timestamp)

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
            return self.create_signal(TradeAction.CANCEL, ohlc.index[-1])
        if (self.active_trade.direction == BIAS.LONG and curr['high'] >= self.active_trade.tp) or \
                (self.active_trade.direction == BIAS.SHORT and curr['low'] <= self.active_trade.tp):
            return self.create_signal(TradeAction.CANCEL, ohlc.index[-1])
        return None

    def analyze(self, ohlc: pd.DataFrame) -> SignalSummary:
        ts_now = ohlc.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        curr = ohlc.iloc[-1]

        # 1. SETUP CUTOFFS
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16), minute=s_info.get("end_minute", 0), second=0,
                                     microsecond=0)
        pending_cutoff = session_end - timedelta(minutes=self.end_buffer_minutes)
        final_bar_for_entry_cutoff = session_end - timedelta(minutes=15)

        # 2. INDICATORS FIRST (Required to check fill validity)
        self.orb_results = self.orb_detector.calculate(ohlc)
        swing_zones, internal_zones = self.sd_detector.calculate(ohlc)
        self.zones = swing_zones + internal_zones

        # 3. FILL LOGIC (High Priority)
        # We check this before the cutoff logic to see if the 19:45 bar filled us
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
                    f"⚡ ORDER FILLED | {self.symbol}: {self.active_trade.direction.name} "
                    f"@ {self.active_trade.price:.2f} | Time: {curr.name}")

        # 4. SESSION EXIT LOGIC (Evaluate both Active and Just-Filled trades)
        if self.trade_status == TradeStatus.FILLED:
            # Check for SL/TP first
            is_long = self.active_trade.direction == BIAS.LONG
            hit_sl_tp = (is_long and (curr["low"] <= self.active_trade.sl or curr["high"] >= self.active_trade.tp)) or \
                        (not is_long and (curr["high"] >= self.active_trade.sl or curr["low"] <= self.active_trade.tp))

            # OR Check if we hit the Session Hard Exit (e.g., 19:45 or 19:55)
            # If we just filled but it's already cutoff time, we must EXIT at market price.
            is_cutoff_time = ts_now >= pending_cutoff

            if hit_sl_tp:
                self.log.info(
                    f"🎯 TRADE EXIT (HIT TP or SL) | {self.symbol}: {self.active_trade.direction.name} "
                    f"| Close: {curr['close']:.2f} | {curr.name}")
                return SignalSummary(direction="EXIT",
                                     trade_signal=self.create_signal(TradeAction.EXIT, ohlc.index[-1]))
            elif is_cutoff_time:
                self.log.info(
                    f"⏰ SESSION CUTOFF EXIT | Sending signal to close all orders |"
                    f" {self.symbol}: {self.active_trade.direction.name} | Close: {curr['close']:.2f}")
                return SignalSummary(direction="EXIT",
                                     trade_signal=self.create_signal(TradeAction.EXIT, ohlc.index[-1]))

            return SignalSummary(direction="FILLED" if filled else "IN_TRADE")

        # 5. PENDING ORDER CANCELLATION (Only if NOT filled)
        if self.trade_status == TradeStatus.PENDING:
            if ts_now >= pending_cutoff:
                self.log.info(f"🚫 CANCELLING PENDING (Session End) {self.symbol}")
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, ohlc.index[-1]))

            # Check zone validity
            cancel_sig = self.check_pending_validity(ohlc, internal_zones, swing_zones)
            if cancel_sig:
                self.log.info(f"⚠️ CANCELLING PENDING (Zone Invalidated or TP Hit) {self.symbol}")
                return SignalSummary(direction="CANCELLED", trade_signal=cancel_sig)
            return SignalSummary(direction="PENDING")

        # 6. NEW SIGNAL SEARCH
        if self.trade_status == TradeStatus.NONE and self.orb_results["signals"]:
            if ts_now < final_bar_for_entry_cutoff:  # Don't start new trades after cutoff
                new_trade = self.check_zone_retrace(ohlc, swing_zones, internal_zones)
                if new_trade:
                    self.active_trade, self.trade_status = new_trade, TradeStatus.PENDING
                    self.log.info(
                        f"🚀 NEW SIGNAL {self.symbol}: {new_trade.direction.name} "
                        f"@ {new_trade.price:.2f} | SL: {new_trade.sl:.2f} | TP: {new_trade.tp:.2f} | {curr.name}")
                    return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        self.log.info(f"No Signals | Zones: {self.zones}")
        return SignalSummary(direction="NEUTRAL")
