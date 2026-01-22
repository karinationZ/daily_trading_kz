from datetime import timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.ORBDetector import ORBDetector
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import ExitType, SESSION_MAP, SignalSummary, StrategyBase, \
    TradeAction, \
    TradeSignal, \
    TradeStatus
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
                 tick_buffer_mult: float = 0.05):
        super().__init__(main_tf)
        self.orb_results = None
        self.symbol = symbol
        self.timezone = timezone

        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"ORB | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)

        # Unique IBKR Tag
        self.ext_tag = f"{symbol}_{self.tz_short}_ORB"

        self.risk = risk
        self.tp_multiple = tp_multiple
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = 5
        else:
            self.end_buffer_minutes = 0
        self.orb_detector = ORBDetector(main_tf="5m", orb_minutes=15, timezone=timezone, wick_ratio=wick_ratio)
        self.sd_detector = SupplyDemandZonesDetector(atr_length=atr_length,
                                                     fvg_based_only=fvg_based_only,
                                                     swing_length=swing_length,
                                                     internal_length=internal_length,
                                                     zone_precision_factor=zone_precision_factor,
                                                     fvg_offset=fvg_offset,
                                                     fvg_threshold=fvg_threshold, wick_ratio=wick_ratio)
        # self.bos_detector = BosChochDetector(main_tf=main_tf, left_bars=5, right_bars=2,
        #                                      wick_ratio=wick_ratio, volume_thresh=5)

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

    def reset_session_state(self):
        self.orb_results = None
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    def validate_zone_reaction(self, ohlc: pd.DataFrame, direction: BIAS, zone: Dict) -> bool:
        """Trigger Logic: Restored to 'Striking Distance' (+ buffer for Longs)."""
        atr = self.sd_detector.get_current_atr(ohlc)
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
        is_falling_knf = self.sd_detector.is_falling_knife(curr, direction, atr)

        if is_falling_knf:
            self.log.info(f"Detected Falling knife. | {curr.name}")
            self.log.info(f"ZONES: {self.zones}")
            return False
        else:
            return True

    def get_execution_levels(self, ohlc, zone: Dict, zone_type: str, direction: BIAS, current_close: float):
        # 1. Get ATR and define Multipliers
        atr = self.sd_detector.get_current_atr(ohlc)

        # Standard Buffers
        entry_buffer = atr * self.entry_buffer_mult
        sl_buffer = atr * self.sl_buffer_mult
        tick_buffer = atr * self.tick_buffer_mult

        # MINIMUM STOP DISTANCE (Volatility Floor)
        # 1.0 is standard. 1.5 is very safe but might make TP hard to hit.
        min_atr_dist = atr * 1.0

        # 2. Determine Base Entry/SL based on Zone Logic (London vs NY)
        # (This is your existing logic)
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
            # NY LOGIC
            if direction == BIAS.LONG:
                entry = current_close + entry_buffer
                structural_sl = zone["bottom"] - sl_buffer
            else:
                entry = current_close - entry_buffer
                structural_sl = zone["top"] + sl_buffer

        # Calculate the "Safety SL" based on volatility
        if direction == BIAS.LONG:
            volatility_sl = entry - min_atr_dist
            # For Longs, SL is below price. We want the LOWER one (Further away).
            final_sl = min(structural_sl, volatility_sl)
        else:
            volatility_sl = entry + min_atr_dist
            # For Shorts, SL is above price. We want the HIGHER one (Further away).
            final_sl = max(structural_sl, volatility_sl)

        # Optional: Log if we modified the SL
        # if final_sl != structural_sl:
        #    print(f"DEBUG: SL Widened for Noise. Zone SL: {structural_sl} -> Volatility SL: {final_sl}")

        return entry, final_sl

    # def trade_setup(self, entry, sl, direction: BIAS, current_close, min_dist, zone_id, zone_type, ts):
    #     # Sanity Check Inputs ---
    #     if entry <= 0 or sl <= 0:
    #         self.log.warning(f"⚠️ SKIPPING INVALID SETUP: Entry {entry} or SL {sl} is invalid (Zero/Negative). "
    #                          f"Zone: {zone_id}")
    #         return None, min_dist
    #     try:
    #         pos = self.calculate_trade_position(entry, sl, direction)
    #         if (direction == BIAS.LONG and current_close >= pos["tp"]) or \
    #                 (direction == BIAS.SHORT and current_close <= pos["tp"]):
    #             return None, min_dist
    #
    #         dist = abs(entry - current_close)
    #         if dist < min_dist:
    #             signal = TradeSignal(
    #                 action=TradeAction.ENTER, symbol=self.symbol, direction=direction,
    #                 price=entry, sl=pos["sl"], tp=pos["tp"], size=pos["size"],
    #                 timestamp=ts, metadata={"zone_id": zone_id, "type": zone_type}
    #             )
    #             return signal, dist
    #     except ValueError:
    #         pass
    #     return None, min_dist

    def trade_setup(self, entry, sl, direction: BIAS, current_close, min_dist, zone_id, zone_type, ts):
        # Sanity Check Inputs ---
        if entry <= 0 or sl <= 0:
            return None, min_dist

        try:
            # 1. Calculate Standard Position (Limit)
            pos = self.calculate_trade_position(entry, sl, direction)

            # 2. CHECK: "Tap and Rip" Logic (Did we miss the Limit?)
            # If price is already running away (Current Close > Entry),
            # we check if it's still safe to enter at Market.

            use_market_entry = False
            final_entry = entry

            # --- LONG LOGIC ---
            if direction == BIAS.LONG:
                if current_close > entry:
                    # We missed the limit.
                    # Check Risk Tolerance: Allow up to 20% more risk than planned
                    original_risk = entry - pos['sl']
                    new_risk = current_close - pos['sl']

                    if new_risk <= (original_risk * 1.2):
                        use_market_entry = True
                        final_entry = current_close

            # --- SHORT LOGIC ---
            else:
                if current_close < entry:
                    # We missed the limit.
                    original_risk = pos['sl'] - entry
                    new_risk = pos['sl'] - current_close

                    if new_risk <= (original_risk * 1.2):
                        use_market_entry = True
                        final_entry = current_close

            # 3. Recalculate Size if entering at Market (Maintain $ Risk)
            if use_market_entry:
                pos = self.calculate_trade_position(final_entry, sl, direction)
                self.log.info(f"⚡ TAP & RIP: Switching to Market Entry @ {final_entry:.2f} (Missed {entry:.2f})")

            # 4. Filter: Don't enter if TP is already passed (Late Entry)
            if (direction == BIAS.LONG and final_entry >= pos["tp"]) or \
                    (direction == BIAS.SHORT and final_entry <= pos["tp"]):
                return None, min_dist

            # 5. Generate Signal
            signal = TradeSignal(
                action=TradeAction.ENTER,
                symbol=self.symbol,
                direction=direction,
                price=final_entry,  # <--- Updated Price
                sl=pos["sl"],
                tp=pos["tp"],
                size=pos["size"],
                timestamp=ts,
                metadata={"zone_id": zone_id, "type": zone_type,
                          "entry_type": "MARKET" if use_market_entry else "LIMIT"}
            )
            return signal, 0  # Distance is 0 because we are filling NOW

        except ValueError:
            pass
        return None, min_dist

    def check_zone_retrace(self, ohlc: pd.DataFrame, swing_zones: List[Dict], internal_zones: List[Dict]):
        # CHECK ORB CONFIRMATION (The Session "Momentum")
        if not self.orb_results or not self.orb_results["signals"]:
            # No ORB breakout yet -> We are inside the range -> Wait.
            return None

        latest_orb_sig = max(self.orb_results["signals"], key=lambda d: d["time"])
        orb_direction = latest_orb_sig["direction"]

        active_direction = orb_direction

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
            if zone["bias"] != active_direction or zone_id in self.processed_zone_ids or zone_id < lookback_start:
                continue

            if not self.validate_zone_reaction(ohlc, active_direction, zone):
                continue

            # Execution
            entry_p, sl_p = self.get_execution_levels(ohlc, zone, zone_type, active_direction, current_close)
            signal, updated_dist = self.trade_setup(entry_p, sl_p,
                                                    active_direction, current_close,
                                                    min_distance, zone_id,
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
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        # 5. Calculate Take Profit
        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        # 6. Log Warning if Capped
        if raw_size > max_contracts:
            self.log.warning(
                f"⚠️ RISK CAP: Math wanted {raw_size} contracts, capped at {size}. "
                f"(Risk per contract: ${risk_per_contract:.2f})")

        return {"sl": sl, "tp": tp, "size": size}

    def check_pending_validity(self, ohlc: pd.DataFrame):
        if not self.active_trade:
            return None

        curr = ohlc.iloc[-1]
        # Check: Target Reached Early (Missed the bus)
        # If price hits TP before filling our Entry, we cancel to avoid chasing.
        if (self.active_trade.direction == BIAS.LONG and curr['high'] >= self.active_trade.tp) or \
                (self.active_trade.direction == BIAS.SHORT and curr['low'] <= self.active_trade.tp):
            self.log.info(f"🚫 Cancelling Pending: Price hit TP before Entry.")
            return self.create_signal(TradeAction.CANCEL, ohlc.index[-1])

        return None

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        """
        Creates a signal.
        """
        # 1. Determine which price to report
        # If it's an EXIT, we want the execution price, not the entry price.
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(symbol=self.symbol,
                          action=action,
                          direction=self.active_trade.direction,
                          price=report_price,
                          sl=self.active_trade.sl,
                          tp=self.active_trade.tp,
                          size=self.active_trade.size,
                          timestamp=timestamp)

        # 2. Cleanup State
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            if self.active_trade:
                self.processed_zone_ids.add(self.active_trade.metadata['zone_id'])
            # Reset state after signal creation
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig

    def analyze(self, ohlc: pd.DataFrame) -> SignalSummary:
        ts_now = ohlc.index[-1].tz_convert(self.timezone)
        s_info = SESSION_MAP[self.timezone.key]
        curr = ohlc.iloc[-1]

        # 1. SETUP CUTOFFS
        session_end = ts_now.replace(hour=s_info.get("end_hour", 16),
                                     minute=s_info.get("end_minute", 0),
                                     second=0,
                                     microsecond=0)
        pending_cutoff = session_end - timedelta(minutes=self.end_buffer_minutes)
        if self.tz_short == "NY":
            final_bar_for_entry_cutoff = session_end - timedelta(hours=2)
        else:
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

            # Detailed check to see WHICH one was hit
            hit_sl = (is_long and curr["low"] <= self.active_trade.sl) or \
                     (not is_long and curr["high"] >= self.active_trade.sl)

            hit_tp = (is_long and curr["high"] >= self.active_trade.tp) or \
                     (not is_long and curr["low"] <= self.active_trade.tp)

            hit_sl_tp = hit_sl or hit_tp

            # OR Check if we hit the Session Hard Exit (e.g., 19:45 or 19:55)
            # If we just filled but it's already cutoff time, we must EXIT at market price.
            is_cutoff_time = ts_now >= pending_cutoff

            if hit_sl_tp:
                # Determine actual exit price
                exit_price = self.active_trade.sl if hit_sl else self.active_trade.tp

                self.log.info(
                    f"🎯 TRADE EXIT (HIT {'SL' if hit_sl else 'TP'}) | {self.symbol}: "
                    f"Close: {curr['close']:.2f} | Target: {exit_price:.2f}")

                signal = self.create_signal(TradeAction.EXIT, ohlc.index[-1], custom_price=exit_price)
                signal.metadata["exit_type"] = ExitType.SL_TP_HIT
                if hit_sl:
                    signal.trade_result = "LOSS"
                elif hit_tp:
                    signal.trade_result = "WIN"
                return SignalSummary(direction="EXIT", trade_signal=signal)

            elif is_cutoff_time:
                self.log.info(f"⏰ SESSION CUTOFF EXIT")
                exit_price = curr['close']
                # DETERMINE RESULT BASED ON PNL
                pnl = (exit_price - self.active_trade.price)
                if self.active_trade.direction == BIAS.SHORT:
                    pnl = -pnl

                res = "WIN" if pnl > 0 else "LOSS"
                sig = self.create_signal(TradeAction.EXIT, curr.name, custom_price=exit_price)
                sig.metadata["exit_type"] = ExitType.MARKET_FORCE
                sig.trade_result = res
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="FILLED" if filled else "IN_TRADE")

        # 5. PENDING ORDER CANCELLATION (Only if NOT filled)
        if self.trade_status == TradeStatus.PENDING:
            if ts_now >= pending_cutoff:
                self.log.info(f"🚫 CANCELLING PENDING (Session End) {self.symbol}")
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, ohlc.index[-1]))

            # Check zone validity
            cancel_sig = self.check_pending_validity(ohlc)
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
