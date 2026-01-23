from datetime import timedelta
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SESSION_MAP, SignalSummary, StrategyBase, TradeAction, \
    TradeSignal, \
    TradeStatus
from daily_trading.settings import logger

CONFIG = {
    # ==========================================================================
    # GROUP 1: BITCOIN (MBT) — "THE TREND BEAST"
    # Logic: Deep Entry (0.95) + Tight Stop (1.2) = Maximum PnL.
    # Bitcoin respects the deep FVG fill perfectly before continuing.
    # ==========================================================================

    # MBT NY (Row 3: $17,258 PnL | 58% WR)
    ("MBT", "NY"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2  # Validated Winner
    },

    # MBT LONDON/ASIA (Row 5 & 6: ~$11k PnL)
    ("MBT", "LONDON"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95, "tp_multiple": 1.2, "sl_buffer_mult": 1.2
    },
    ("MBT", "ASIA"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95, "tp_multiple": 1.2, "sl_buffer_mult": 1.2
    },

    # ==========================================================================
    # GROUP 2: SILVER (SIL) — "THE AGGRESSIVE RUNNER"
    # Logic: NY prefers shallower entry (0.75) than London/Asia (0.95).
    # All sessions prefer tight stops (1.2).
    # ==========================================================================

    # SIL NY (Row 4: $15,518 PnL)
    ("SIL", "NY"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.75,  # NY moves faster, 0.75 fills more often
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2
    },

    # SIL LONDON/ASIA (Row 1/2: ~$11k PnL)
    # Note: These had RR 2.0 in the screenshot, but 43% WR.
    # Row 16 (RR 1.2) had 55% WR and similar PnL. We prefer RR 1.2 for consistency.
    ("SIL", "LONDON"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2
    },
    ("SIL", "ASIA"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95, "tp_multiple": 1.2, "sl_buffer_mult": 1.2
    },

    # ==========================================================================
    # GROUP 3: MES (S&P 500) — "THE PRECISION SCALP"
    # Logic: Deep Entry (0.95) + Tight Stop (1.2) is the clear winner across board.
    # ==========================================================================

    # MES NY (Row 8: $14,094 PnL)
    ("MES", "NY"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2
    },

    # MES ASIA/LONDON (Row 10/11: $6,905 PnL)
    ("MES", "ASIA"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95, "tp_multiple": 1.2, "sl_buffer_mult": 1.5
    },
    ("MES", "LONDON"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95, "tp_multiple": 1.2, "sl_buffer_mult": 1.5
    },

    # ==========================================================================
    # GROUP 4: MNQ (NASDAQ) — "THE SPLIT PERSONALITY"
    # Logic: London/Asia need 0.75 Entry. NY allows 0.95 Deep Entry.
    # ==========================================================================

    # MNQ LONDON (Row 7: $6,528 PnL)
    ("MNQ", "LONDON"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.75,  # Speed required
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.5
    },

    # MNQ NY (Row 14: $9,159 PnL)
    ("MNQ", "NY"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,  # Deep value works in NY
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2  # Tight stop
    },

    # ==========================================================================
    # GROUP 5: MGC (GOLD) — "THE MOMENTUM REVERSAL"
    # Logic: Data shows Gold prefers SHALLOW entry (0.25) in London/Asia.
    # This contradicts the "Deep" logic of indices. Gold trends harder.
    # ==========================================================================

    # MGC LONDON/ASIA (Row 12/13: $9,214 PnL)
    ("MGC", "LONDON"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.25,  # Aggressive Entry
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2
    },
    ("MGC", "ASIA"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.25, "tp_multiple": 1.2, "sl_buffer_mult": 1.2
    },

    # MGC NY (Row 17: $6,320 PnL)
    ("MGC", "NY"): {
        "htf": "1h", "ltf": "5m",
        "fvg_mitigation_ratio": 0.95,  # NY Gold chops more, needs deep entry
        "tp_multiple": 1.2,
        "sl_buffer_mult": 1.2
    }
}


class BosFvgStrategy(StrategyBase):
    """
        STRATEGY: Multi-Timeframe Trend Following (SMC)

        OVERVIEW:
        A "Smart Money Concept" strategy that aligns Higher Timeframe (HTF) market structure
        with Lower Timeframe (LTF) entry signals. It seeks to join an established trend
        by placing Limit Orders inside fresh Fair Value Gaps (FVG).

        TRADING LOGIC:
        1. Trend Identification (HTF):
           - Uses Break of Structure (BOS) and Change of Character (CHoCH) on the HTF (e.g., 1H)
           - Determines the Bias (Long/Short).

        2. Entry Signal (LTF):
           - Scans the LTF (e.g., 5m) for "Fresh" (Unmitigated) Fair Value Gaps that align with the HTF Bias.
           - Places a Limit Order inside the FVG (defined by `fvg_mitigation_ratio`).
           - Stop Loss is placed slightly beyond the FVG boundary.

        3. Order Management:
           - 'Trailing' Entries: If a new BOS occurs while an order is pending, the old order
             is cancelled and a new one is calculated for the most recent FVG.
           - Session Cutoff: All pending orders are cancelled 5 minutes before session close.

        ROBUSTNESS & SAFETY MECHANISMS:
        ----------------------------------------------------------------------------
        [1] Restart Safety (Stateless Recovery):
            - On bot restart, it recalculates history to ensure it doesn't trade
              an FVG that was already invalidated/filled while the bot was offline.

        [2] "Zombie" Protection (Infinite Loop Prevention):
            - Actively detects invalid trade states (e.g., SL=0.00 due to bad data).
            - Force-clears internal state to prevent repeated entry/exit loops.

        [3] Math Gatekeepers:
            - Validates all calculated prices (Entry, SL, Size) before signal generation.
            - Prevents submission of invalid orders to the broker.

        [4] Atomic Updates:
            - Uses a "Cancel-then-Replace" logic for order updates rather than modification requests.
            - Prevents state desynchronization with the broker/database.
        ----------------------------------------------------------------------------
        """

    def __init__(self,
                 symbol: str,
                 timezone: ZoneInfo,
                 main_tf="5m",  # LTF for Entry
                 htf="1h",  # HTF for Trend
                 risk: float = 150.0,
                 tp_multiple: float = 2.0,
                 fvg_mitigation_ratio: float = 0.5,  # Entry at 50% of FVG
                 sl_buffer_mult: float = 0.1):

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

        # Detectors
        self.htf_bos_detector = BosChochDetector(main_tf=htf)
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.5)

        # Logging & Tags
        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS-FVG | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{self.tz_short}_BOS_FVG"

        # Cache & Config
        self.processed_zone_ids = set()
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

        # Session Cutoff
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        self.log.info("Resetting Session State.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.active_bos_timestamp = None

    # ==================================================================
    # 1. HELPERS: Validation
    # ==================================================================

    @staticmethod
    def is_fvg_invalidated(df_ltf: pd.DataFrame, fvg: Dict) -> bool:
        """
        Restart Safety: Checks if the FVG has been VIOLATED (Candle Closed beyond it)
        in the history between FVG creation and Now.
        """
        fvg_time = fvg['time']

        # 1. Robust Timezone Alignment
        try:
            if df_ltf.index.tz is None:
                fvg_time = fvg_time.replace(tzinfo=None)
            elif fvg_time.tzinfo is None:
                fvg_time = fvg_time.tz_localize(df_ltf.index.tz)
            else:
                fvg_time = fvg_time.tz_convert(df_ltf.index.tz)
        except Exception:
            fvg_time = fvg_time.replace(tzinfo=None)

        # 2. Get history strictly AFTER FVG creation
        # Exclude current candle to avoid self-invalidation during calc
        history = df_ltf[df_ltf.index > fvg_time].iloc[:-1]

        if history.empty:
            return False

        # 3. Check Invalidation (Close beyond the gap)
        if fvg['bias'] == 1:  # Bullish FVG (Demand)
            # Invalidated if a candle CLOSES below the bottom
            if history['close'].min() < fvg['bottom']:
                return True
        else:  # Bearish FVG (Supply)
            # Invalidated if a candle CLOSES above the top
            if history['close'].max() > fvg['top']:
                return True

        return False

    def find_valid_fvg(self, df_ltf: pd.DataFrame, direction: BIAS, current_price: float):
        """
        Finds the nearest FRESH FVG that matches the trend direction.
        """
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get('signals', [])

        if not fvg_signals: return None

        target_bias_val = BIAS.LONG.num_value if direction == BIAS.LONG else BIAS.SHORT.num_value

        # FvgDetector returns signals sorted by Time Descending (Newest First).
        # We iterate normally to find the most recent valid one.
        for fvg in fvg_signals:

            # 1. Bias Mismatch
            if fvg['bias'] != target_bias_val: continue

            # 2. Mitigation Check (Detector Flag)
            if fvg.get('mitigated', True): continue

            # 3. Memory Cache (Already traded/processed)
            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids: continue

            # 4. Historical Invalidation (Restart Safety)
            if self.is_fvg_invalidated(df_ltf, fvg):
                self.processed_zone_ids.add(fvg_id)
                continue

            # 5. Price Logic (Limit Order Validity)
            # We need the zone to be "behind" price so we can place a limit pullback order.
            if direction == BIAS.LONG:
                if fvg['top'] < current_price: return fvg
            elif direction == BIAS.SHORT:
                if fvg['bottom'] > current_price: return fvg

        return None

    # ==================================================================
    # 2. TRADE CALCULATION
    # ==================================================================

    def calculate_setup_params(self, fvg, direction: BIAS, df_ltf: pd.DataFrame):
        atr = self.ltf_fvg_detector.get_current_atr(df_ltf)
        top = fvg['top']
        bottom = fvg['bottom']
        height = top - bottom

        if direction == BIAS.LONG:
            # BUY LIMIT @ Depth % into the zone
            entry = top - (height * self.entry_depth)
            sl = bottom - (atr * self.sl_buffer_mult)
        else:
            # SELL LIMIT @ Depth % into the zone
            entry = bottom + (height * self.entry_depth)
            sl = top + (atr * self.sl_buffer_mult)

        pos_math = self.calculate_trade_position(entry, sl, direction)

        return {
            "entry": entry, "sl": pos_math['sl'], "tp": pos_math['tp'],
            "size": pos_math['size'], "direction": direction
        }

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """
        Calculates size with hard floors to prevent over-leveraging on tight stops.
        """
        # Calculate Chart Distance
        chart_dist = abs(entry_price - sl)
        if chart_dist == 0:
            chart_dist = 0.25

        # 3. Apply Safety Floor for Sizing Math
        # We size the position AS IF the stop was at least the Hard Floor distance
        hard_floor = self.HARD_MIN_DIST.get(self.symbol, 0.0)
        effective_dist_for_sizing = max(chart_dist, hard_floor)

        # Calculate Risk Per Contract
        risk_per_contract = effective_dist_for_sizing * self.point_value
        if risk_per_contract <= 0: risk_per_contract = 5.0

        # Calculate Size
        raw_size = int(self.risk / risk_per_contract)

        # 6. Safety Cap
        max_contracts = 10
        size = max(1, min(raw_size, max_contracts))

        if raw_size > max_contracts:
            self.log.warning(f"⚠️ RISK CAP: {raw_size} -> {size}")

        # Calculate TP based on ACTUAL Chart Distance
        # We use the real chart distance for TP to keep the RR ratio accurate on the chart
        if direction == BIAS.LONG:
            tp = entry_price + (chart_dist * self.tp_multiple)
        else:
            tp = entry_price - (chart_dist * self.tp_multiple)

        return {"sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        """Creates signal and auto-resets state on terminal actions."""
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(
            symbol=self.symbol, action=action, direction=self.active_trade.direction,
            price=report_price, sl=self.active_trade.sl, tp=self.active_trade.tp,
            size=self.active_trade.size, timestamp=timestamp
        )

        # Auto-Reset State on Exit/Cancel
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade = None
            self.trade_status = TradeStatus.NONE

        return sig

    def check_order_fill(self, candle) -> bool:
        if not self.active_trade:
            return False
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
        # 1. FILL CHECK (Limit Orders)
        if self.trade_status == TradeStatus.PENDING:
            if self.check_order_fill(curr):
                self.trade_status = TradeStatus.FILLED
                just_filled = True  # <--- MARK AS JUST FILLED

                self.log.info(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price:.2f}")

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:

            # --- ZOMBIE KILLER ---
            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.log.error(f"🧟 ZOMBIE TRADE DETECTED ({self.symbol}). Cleaning up.")
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")
            # ---------------------

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG

            # Initialize Exit Variables
            exit_type = None
            exit_price = None
            # ======================================================
            # Only check for Gaps if we didn't just fill this moment.
            # ======================================================
            if not just_filled:
                if is_long:
                    if curr['open'] <= sl:
                        exit_type, exit_price = 'LOSS', curr['open']
                    elif curr['open'] >= tp:
                        exit_type, exit_price = 'WIN', curr['open']
                else:
                    if curr['open'] >= sl:
                        exit_type, exit_price = 'LOSS', curr['open']
                    elif curr['open'] <= tp:
                        exit_type, exit_price = 'WIN', curr['open']
            # ======================================================
            # --- STANDARD CHECK (If no gap detected) ---
            if not exit_type:
                hit_sl = False
                hit_tp = False

                if just_filled:
                    # 🛑 STRICT MODE (Entry Candle)
                    # If we just filled, we CANNOT trust the High/Low for TP,
                    # because the High might have happened BEFORE the fill.
                    # We only take TP if the candle CLOSES past the target.
                    # We ALWAYS check SL because if Low <= SL, we definitely died.
                    if is_long:
                        hit_sl = curr['low'] <= sl
                        hit_tp = curr['close'] >= tp  # Strict (Close)
                    else:
                        hit_sl = curr['high'] >= sl
                        hit_tp = curr['close'] <= tp  # Strict (Close)
                else:
                    # ✅ STANDARD MODE (Subsequent Candles)
                    # We are already in. We trust the Wicks.
                    if is_long:
                        hit_sl = curr['low'] <= sl
                        hit_tp = curr['high'] >= tp
                    else:
                        hit_sl = curr['high'] >= sl
                        hit_tp = curr['low'] <= tp

                # --- LOGIC: CONFLICT RESOLUTION ---
                # If both hit in the same candle, assume SL was hit first (Pessimistic)
                if hit_sl and hit_tp:
                    hit_tp = False

                # --- LOGIC: ASSIGN EXIT TYPE ---
                if hit_sl:
                    exit_type = 'LOSS'
                    exit_price = sl
                elif hit_tp:
                    exit_type = 'WIN'
                    exit_price = tp

            # --- LOGIC: TIME CUTOFF ---
            # Only check cutoff if we haven't already hit SL or TP
            if not exit_type and ts_now >= cutoff_ts:
                exit_type = 'CUTOFF'
                exit_price = curr['close']

            # --- EXECUTION ---
            if exit_type:
                # Use custom_price to report the actual SL/TP/Close price
                sig = self.create_signal(TradeAction.EXIT, ts_utc, custom_price=exit_price)

                # Tag result for the Backtester
                if exit_type in ['WIN', 'LOSS']:
                    sig.trade_result = exit_type

                self.log.info(f"🎯 TRADE EXIT ({exit_type}) {self.symbol} @ {exit_price:.2f}")
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="IN_TRADE")

        # # ❌ Early NY session is hurting performance
        # if self.tz_short == "NY" and ts_now.hour < 11:
        #     logger.warning("Skipping High volatile NY hours")
        #     return SignalSummary(direction="NEUTRAL")

        if ts_now < cutoff_ts:
            trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
            bos_signals = self.htf_bos_detector.signals

            if df_4h is not None:
                bias_4h = self.htf_bos_detector.get_current_trend(df_4h)
                # If 1H is Long but 4H is Short -> SKIP
                if trend_bias != bias_4h and bias_4h != BIAS.NEUTRAL:
                    # self.log.info(f"⏳ FILTERED: 1H {trend_bias.name} vs 4H {bias_4h.name}")
                    return SignalSummary(direction="NEUTRAL")

            if trend_bias != BIAS.NEUTRAL and bos_signals:
                last_bos = bos_signals[-1]
                bos_time = last_bos.get("time")

                is_new_bos = (self.active_bos_timestamp is None or bos_time != self.active_bos_timestamp)

                should_search = (self.trade_status == TradeStatus.NONE) or \
                                (self.trade_status == TradeStatus.PENDING and is_new_bos)

                if should_search:
                    fvg = self.find_valid_fvg(df_ltf, trend_bias, curr['close'])

                    if fvg:
                        params = self.calculate_setup_params(fvg, trend_bias, df_ltf)

                        # --- GATEKEEPER ---
                        if params['entry'] <= 0 or params['sl'] <= 0 or params['size'] <= 0:
                            self.log.warning(f"⚠️ Bad Math in Setup. Skipping.")
                            return SignalSummary(direction="NEUTRAL")
                        # ------------------

                        fvg_id = f"{fvg['time']}_{self.symbol}"

                        # CASE 1: NEW TRADE
                        if self.trade_status == TradeStatus.NONE:
                            new_trade = TradeSignal(
                                action=TradeAction.ENTER,
                                symbol=self.symbol,
                                direction=params['direction'],
                                price=params['entry'], sl=params['sl'], tp=params['tp'], size=params['size'],
                                timestamp=curr.name, metadata={"fvg_id": fvg_id, "bos_id": bos_time,
                                                               "bos_type": last_bos.get('type')}
                            )
                            self.active_trade = new_trade
                            self.trade_status = TradeStatus.PENDING
                            self.processed_zone_ids.add(fvg_id)
                            self.active_bos_timestamp = bos_time
                            self.log.info(f"🚀 SIGNAL: {params['direction'].name} Limit @ {params['entry']:.2f}")
                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

                        # CASE 2: UPDATE VIA CANCEL (Clean Reset)
                        elif self.trade_status == TradeStatus.PENDING and is_new_bos:
                            current_fvg_id = self.active_trade.metadata.get('fvg_id')
                            if current_fvg_id != fvg_id:
                                self.log.info(f"♻️ NEW STRUCTURE: Cancelling old order to re-assess.")
                                cancel_sig = self.create_signal(TradeAction.CANCEL, ts_utc)
                                self.active_bos_timestamp = bos_time
                                return SignalSummary(direction="CANCELLED", trade_signal=cancel_sig)

        # 4. PENDING CANCEL (Cutoff)
        if self.trade_status == TradeStatus.PENDING:
            if ts_now >= cutoff_ts:
                return SignalSummary(direction="CANCELLED",
                                     trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc))

        return SignalSummary(direction="NEUTRAL")
