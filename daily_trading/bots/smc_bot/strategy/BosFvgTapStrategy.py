from typing import Optional, Dict
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase, TradeAction, TradeSignal, \
    TradeStatus
from daily_trading.settings import logger


class BosFvgStrategy(StrategyBase):
    """
    STRATEGY: HTF Trend + LTF FVG Entry (Limit Order)

    HIERARCHY:
    1. HTF (1H): Determine Trend via BOS/CHOCH signals.
    2. LTF (5m): Find the nearest Unmitigated FVG aligned with Trend.
    3. ENTRY: Place a Limit Order at the Equilibrium (50%) of the FVG.
    """

    def __init__(self,
                 symbol: str,
                 timezone: ZoneInfo,
                 main_tf="5m",  # LTF for Entry
                 htf="1h",  # HTF for Trend
                 risk: float = 150.0,
                 tp_multiple: float = 2.0,
                 fvg_mitigation_ratio: float = 0.5,  # 0.5 = Entry at 50% of FVG height
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

        # --- DETECTORS ---
        # 1. HTF Trend Detector (Now contains get_current_trend)
        self.htf_bos_detector = BosChochDetector(main_tf=htf)

        # 2. LTF FVG Detector
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.5)

        # Logging Setup
        tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS-FVG | {symbol}:{tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{tz_short}_BOS_FVG"

        # Cache processed zones to prevent re-entry on same FVG
        self.processed_zone_ids = set()
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.active_bos_timestamp = None

    # ==================================================================
    # 1. MARKET STRUCTURE ANALYSIS
    # ==================================================================

    def find_valid_fvg(self, df_ltf: pd.DataFrame, direction: BIAS, current_price: float):
        """
        Scans LTF for the best FVG that matches the HTF Trend.
        """
        # Run Detector
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get('signals', [])

        if not fvg_signals:
            return None

        # Map BIAS to FVG Target value (1=Long, 2=Short)
        target_bias_val = BIAS.LONG.num_value if direction == BIAS.LONG else BIAS.SHORT.num_value

        # BUG FIX: Iterate REVERSED to check the MOST RECENT FVG first
        for fvg in fvg_signals:

            # 1. Must match direction
            if fvg['bias'] != target_bias_val:
                continue

            # 2. Must be Unmitigated
            if fvg.get('mitigated', True):
                continue

            # 3. Must be Fresh (Not in processed list)
            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                continue

            # 4. Price Logic (Limit Order Logic)
            if direction == BIAS.LONG:
                # Buy Limit: FVG Top must be BELOW current price
                if fvg['top'] < current_price:
                    return fvg
            elif direction == BIAS.SHORT:
                # Sell Limit: FVG Bottom must be ABOVE current price
                if fvg['bottom'] > current_price:
                    return fvg

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
            "entry": entry,
            "sl": pos_math['sl'],
            "tp": pos_math['tp'],
            "size": pos_math['size'],
            "direction": direction
        }

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        r = abs(entry_price - sl)
        if r == 0: r = 0.25

        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0: risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)
        max_contracts = 5
        size = max(1, min(raw_size, max_contracts))

        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        if raw_size > max_contracts:
            self.log.warning(f"⚠️ RISK CAP: {raw_size} -> {size}")

        return {"sl": sl, "tp": tp, "size": size}

    # ==================================================================
    # 3. HELPERS (Signals & Checks)
    # ==================================================================

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None) -> TradeSignal:
        """
        Creates a signal.
        FIX: Added custom_price support for Exits and Auto-Cleanup.
        """
        # 1. Determine which price to report
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(
            symbol=self.symbol, action=action,
            direction=self.active_trade.direction,
            price=report_price,  # <--- Fix: Report actual exit price
            sl=self.active_trade.sl, tp=self.active_trade.tp,
            size=self.active_trade.size, timestamp=timestamp
        )

        # 2. Cleanup State Automatically
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig

    def check_order_fill(self, candle) -> bool:
        if not self.active_trade:
            return False
        # Buy Limit: Low <= Price
        # Sell Limit: High >= Price
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= self.active_trade.price
        else:
            return candle['high'] >= self.active_trade.price

    def check_exit(self, candle) -> dict:
        """Returns None if no exit, or {'type': 'SL'|'TP', 'price': float} if exit."""
        if not self.active_trade:
            return None

        sl, tp = self.active_trade.sl, self.active_trade.tp

        if self.active_trade.direction == BIAS.LONG:
            if candle['low'] <= sl: return {'type': 'LOSS', 'price': sl}
            if candle['high'] >= tp: return {'type': 'WIN', 'price': tp}
        else:
            if candle['high'] >= sl: return {'type': 'LOSS', 'price': sl}
            if candle['low'] <= tp: return {'type': 'WIN', 'price': tp}

        return None

    # ==================================================================
    # 4. MAIN EXECUTION LOOP
    # ==================================================================

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        """Called every 5 min"""
        curr = df_ltf.iloc[-1]

        # 1. FILL CHECK (High Priority)
        if self.trade_status == TradeStatus.PENDING:
            if self.check_order_fill(curr):
                self.trade_status = TradeStatus.FILLED
                self.log.info(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price:.2f}")
                self.active_trade.entry_timestamp = curr.name
                # Note: We return FILLED here so the bot knows state changed
                return SignalSummary(direction="FILLED", trend_signal=self.active_trade)

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:
            exit_info = self.check_exit(curr)

            if exit_info:
                # FIX: Use create_signal with custom_price
                signal = self.create_signal(TradeAction.EXIT, curr.name, custom_price=exit_info['price'])

                # Add Win/Loss Metadata
                signal.trade_result = exit_info['type']  # WIN/LOSS

                self.log.info(f"🎯 TRADE EXIT ({exit_info['type']}) {self.symbol} @ {exit_info['price']:.2f}")
                return SignalSummary(direction="EXIT", trade_signal=signal)

            return SignalSummary(direction="IN_TRADE")

        # 3. SIGNAL SEARCH & PENDING UPDATE
        if self.trade_status in [TradeStatus.NONE, TradeStatus.PENDING]:

            # A. Get HTF Trend (Uses Cache internally)
            trend_bias = self.htf_bos_detector.get_current_trend(df_htf)

            # B. Get Actual BOS Signal details for timestamp tracking
            # (We rely on the fact get_current_trend just calculated this)
            bos_results = self.htf_bos_detector.calculate_bos_choch(df_htf, plot=False)
            bos_signals = bos_results.get('signals', [])

            # Ensure we have a trend and at least one signal to reference
            if trend_bias != BIAS.NEUTRAL and bos_signals:

                last_bos_signal = bos_signals[-1]
                bos_timestamp = last_bos_signal.get('time')

                # Check if this is a NEW BOS signal
                is_new_bos = (self.active_bos_timestamp is None or
                              bos_timestamp != self.active_bos_timestamp)

                # LOGIC:
                # 1. If NO Trade: Search FVG
                # 2. If PENDING Trade AND New BOS: Search new FVG to update order (Trailing the trend)
                should_search_fvg = (self.trade_status == TradeStatus.NONE or
                                     (self.trade_status == TradeStatus.PENDING and is_new_bos))

                if should_search_fvg:
                    # Find FVG matching the trend
                    fvg = self.find_valid_fvg(df_ltf, trend_bias, curr['close'])

                    if fvg:
                        params = self.calculate_setup_params(fvg, trend_bias, df_ltf)
                        fvg_id = f"{fvg['time']}_{self.symbol}"

                        # CASE 1: New Trade
                        if self.trade_status == TradeStatus.NONE:
                            new_trade = TradeSignal(
                                action=TradeAction.ENTER,
                                symbol=self.symbol,
                                direction=params['direction'],
                                price=params['entry'],
                                sl=params['sl'],
                                tp=params['tp'],
                                size=params['size'],
                                timestamp=curr.name,
                                metadata={
                                    "fvg_id": fvg_id,
                                    "fvg_time": fvg["time"],
                                    "bos_time": bos_timestamp
                                }
                            )
                            self.active_trade = new_trade
                            self.trade_status = TradeStatus.PENDING
                            self.processed_zone_ids.add(fvg_id)
                            self.active_bos_timestamp = bos_timestamp

                            self.log.info(f"🚀 SIGNAL: {params['direction'].name} Limit @ {params['entry']:.2f}")
                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

                        # CASE 2: Update ONLY if NEW BOS (Move limit order to new fresh FVG)
                        elif self.trade_status == TradeStatus.PENDING and is_new_bos:
                            self.log.info(f"♻️ NEW BOS DETECTED: Updating order to fresh FVG ({fvg_id})")

                            new_trade = TradeSignal(
                                action=TradeAction.UPDATE,
                                symbol=self.symbol,
                                direction=params['direction'],
                                price=params['entry'],
                                sl=params['sl'],
                                tp=params['tp'],
                                size=params['size'],
                                timestamp=curr.name,
                                metadata={
                                    "fvg_id": fvg_id,
                                    "note": "Order Update - New BOS",
                                    "old_bos_time": self.active_bos_timestamp,
                                    "new_bos_time": bos_timestamp
                                }
                            )

                            self.active_trade = new_trade
                            self.processed_zone_ids.add(fvg_id)
                            self.active_bos_timestamp = bos_timestamp

                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

        return SignalSummary(direction="NEUTRAL")