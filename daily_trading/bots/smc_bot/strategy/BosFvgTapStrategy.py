from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS, Trend
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

        self.active_bos_timestamp = None
        self.symbol = symbol
        self.timezone = timezone
        self.htf = htf
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = fvg_mitigation_ratio
        self.sl_buffer_mult = sl_buffer_mult

        self.last_htf_timestamp = None
        self.cached_trend = Trend.NEUTRAL

        # --- DETECTORS ---
        # Note: We use the classes provided without modification
        self.htf_bos_detector = BosChochDetector(main_tf=htf)
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.5)

        # Logging Setup
        tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"BOS-FVG | {symbol}:{tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        # Unique IBKR Tag
        self.ext_tag = f"{symbol}_{tz_short}_BOS_FVG_TAP"

        # State
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE

        # Cache processed zones to prevent re-entry on same FVG
        self.processed_zone_ids = set()
        self.last_processed_hour: int = -1
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        logger.info(f"{self.symbol} | New Session: Resetting Strategy State and Cache.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()
        self.last_processed_hour = -1

    # ==================================================================
    # 1. MARKET STRUCTURE ANALYSIS
    # ==================================================================

    def get_htf_trend(self, df_htf: pd.DataFrame) -> tuple[Trend, Optional[any]]:
        """
        Analyzes 1H Structure. Returns (Trend, last_bos_signal).
        FIX: Now returns the actual BOS signal for tracking
        """
        if df_htf.empty:
            return Trend.NEUTRAL, None

        current_htf_ts = df_htf.index[-1]

        # If timestamp same as last check, return cached result
        if self.last_htf_timestamp == current_htf_ts:
            # Need to get the signal even from cache
            result = self.htf_bos_detector.calculate_bos_choch(df_htf, plot=False)
            signals = result.get('signals', [])
            last_sig = signals[-1] if signals else None
            return self.cached_trend, last_sig

        # Run Detector
        result = self.htf_bos_detector.calculate_bos_choch(df_htf, plot=False)
        signals = result.get('signals', [])

        if not signals:
            return Trend.NEUTRAL, None

        last_sig = signals[-1]
        bias = last_sig.get('bias')
        trend = Trend.NEUTRAL

        # Accept only BOS
        signal_type = last_sig.get("type", None)
        if signal_type == "BOS":
            if bias == BIAS.LONG.num_value:
                trend = Trend.BULLISH
            elif bias == BIAS.SHORT.num_value:
                trend = Trend.BEARISH

        # Cache
        self.last_htf_timestamp = current_htf_ts
        self.cached_trend = trend

        return trend, last_sig

    def find_valid_fvg(self, df_ltf: pd.DataFrame, trend: Trend, current_price: float):
        """
        Scans LTF for the best FVG that matches the HTF Trend.
        Optimized: Scans backwards to find the most recent valid FVG immediately.
        """
        # Run Detector
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get('signals', [])

        if not fvg_signals:
            return None

        # Map Trend to FVG Bias
        target_bias = BIAS.LONG.num_value if trend == Trend.BULLISH else BIAS.SHORT.num_value

        # OPTIMIZATION: Use reversed() to scan from Newest -> Oldest
        for fvg in fvg_signals:

            # 1. Must match direction
            if fvg['bias'] != target_bias:
                continue

            # 2. Must be Unmitigated
            if fvg.get('mitigated', True):
                continue

            # 3. Must be Fresh (Not in processed list)
            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                continue

            # 4. Price Logic (Discount/Premium)
            # If criteria met, return IMMEDIATELY. No need to check older ones.
            if trend == Trend.BULLISH:
                # Buy: FVG must be below current price
                if fvg['top'] < current_price:
                    return fvg
            elif trend == Trend.BEARISH:
                # Sell: FVG must be above current price
                if fvg['bottom'] > current_price:
                    return fvg

        return None

    # ==================================================================
    # 2. TRADE CALCULATION
    # ==================================================================

    def calculate_setup_params(self, fvg, trend: Trend, df_ltf: pd.DataFrame):
        atr = self.ltf_fvg_detector.get_current_atr(df_ltf)

        top = fvg['top']
        bottom = fvg['bottom']
        height = top - bottom

        if trend == Trend.BULLISH:
            # BUY LIMIT @ Depth % into the zone
            entry = top - (height * self.entry_depth)
            sl = bottom - (atr * self.sl_buffer_mult)
            direction = BIAS.LONG
        else:
            # SELL LIMIT @ Depth % into the zone
            entry = bottom + (height * self.entry_depth)
            sl = top + (atr * self.sl_buffer_mult)
            direction = BIAS.SHORT

        pos_math = self.calculate_trade_position(entry, sl, direction)

        return {
            "entry": entry,
            "sl": pos_math['sl'],
            "tp": pos_math['tp'],
            "size": pos_math['size'],
            "direction": direction
        }

    # ==================================================================
    # 3. HELPERS (Risk & State)
    # ==================================================================

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """
        Calculates position size with a HARD CAP (5 Contracts).
        """
        r = abs(entry_price - sl)
        if r == 0:
            r = 0.25

        # Calculate Raw Size
        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0:
            risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)

        # SAFETY CAP
        max_contracts = 5
        size = max(1, min(raw_size, max_contracts))

        # TP
        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        if raw_size > max_contracts:
            self.log.info(f"⚠️ RISK CAP: {raw_size} -> {size} (Risk/Cont: ${risk_per_contract:.2f})", level="warning")

        return {"sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp) -> TradeSignal:
        sig = TradeSignal(
            symbol=self.symbol, action=action,
            direction=self.active_trade.direction,
            price=self.active_trade.price,
            sl=self.active_trade.sl, tp=self.active_trade.tp,
            size=self.active_trade.size, timestamp=timestamp
        )
        self.active_trade, self.trade_status = None, TradeStatus.NONE
        return sig

    def check_order_fill(self, candle) -> bool:
        if not self.active_trade:
            return False
        # Buy Limit: Low must touch or go below Price
        # Sell Limit: High must touch or go above Price
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= self.active_trade.price
        else:
            return candle['high'] >= self.active_trade.price

    def check_exit(self, candle) -> bool:
        if not self.active_trade:
            return False
        sl, tp = self.active_trade.sl, self.active_trade.tp
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= sl or candle['high'] >= tp
        else:
            return candle['high'] >= sl or candle['low'] <= tp

    # ==================================================================
    # 4. MAIN EXECUTION LOOP
    # ==================================================================

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        """Called every 5 min"""
        curr = df_ltf.iloc[-1]

        # 1. FILL CHECK
        if self.trade_status == TradeStatus.PENDING:
            if self.check_order_fill(curr):
                self.trade_status = TradeStatus.FILLED
                self.log.info(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price}")
                self.active_trade.entry_timestamp = curr.name
                return SignalSummary(direction="FILLED", trend_signal=self.active_trade)

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:
            if self.check_exit(curr):
                self.log.info(f"🎯 TRADE EXIT (SL/TP) {self.symbol}")
                return SignalSummary(direction="EXIT", trade_signal=self.create_signal(TradeAction.EXIT, curr.name))
            self.log.info(f"IN_TRADE - {self.symbol}")
            return SignalSummary(direction="IN_TRADE")

        # 3. SIGNAL SEARCH & PENDING UPDATE
        if self.trade_status in [TradeStatus.NONE, TradeStatus.PENDING]:

            # FIX: Get trend AND the actual BOS signal
            trend, bos_signal = self.get_htf_trend(df_htf)

            if trend != Trend.NEUTRAL and bos_signal:
                bos_timestamp = bos_signal.get('time')

                # Check if this is a NEW BOS signal
                is_new_bos = (self.active_bos_timestamp is None or
                              bos_timestamp != self.active_bos_timestamp)

                # Only search for FVG if: (1) No trade, OR (2) New BOS detected
                should_search_fvg = (self.trade_status == TradeStatus.NONE or
                                     (self.trade_status == TradeStatus.PENDING and is_new_bos))

                if should_search_fvg:
                    fvg = self.find_valid_fvg(df_ltf, trend, curr['close'])

                    if fvg:
                        params = self.calculate_setup_params(fvg, trend, df_ltf)
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
                                    "structure": trend.name,
                                    "fvg_time": fvg["time"],
                                    "bos_time": bos_timestamp  # Track which BOS
                                }
                            )
                            self.active_trade = new_trade
                            self.trade_status = TradeStatus.PENDING
                            self.processed_zone_ids.add(fvg_id)
                            self.active_bos_timestamp = bos_timestamp  # FIX: Store BOS

                            self.log.info(
                                f"🚀 SIGNAL: {params['direction'].name} Limit @ {params['entry']:.2f} (BOS: {bos_timestamp})")
                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

                        # CASE 2: Update ONLY if NEW BOS
                        elif self.trade_status == TradeStatus.PENDING and is_new_bos:
                            self.log.info(f"♻️ NEW BOS DETECTED: Updating order with fresh FVG ({fvg_id})")

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
                                    "structure": trend.name,
                                    "old_zone_id": self.active_trade.metadata.get('fvg_id'),
                                    "old_bos_time": self.active_bos_timestamp,
                                    "new_bos_time": bos_timestamp,
                                    "note": "Order Update - New BOS",
                                    "fvg_time": fvg["time"]
                                }
                            )

                            self.active_trade = new_trade
                            self.processed_zone_ids.add(fvg_id)
                            self.active_bos_timestamp = bos_timestamp  # FIX: Update BOS tracking

                            return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

        return SignalSummary(direction="NEUTRAL")
