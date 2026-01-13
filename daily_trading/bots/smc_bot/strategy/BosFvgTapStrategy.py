from datetime import timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Detectors
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.StrategyBase import BIAS, StrategyBase, Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeAction, TradeSignal, TradeStatus
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
                 entry_depth: float = 0.5,  # 0.5 = Entry at 50% of FVG height
                 sl_buffer_mult: float = 0.1,
                 atr_length_buffer: int = 14):

        super().__init__(main_tf, atr_length_buffer=atr_length_buffer)

        self.symbol = symbol
        self.timezone = timezone
        self.htf = htf
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = entry_depth
        self.sl_buffer_mult = sl_buffer_mult

        # --- DETECTORS ---
        # Note: We use the classes provided without modification
        self.htf_bos_detector = BosChochDetector(main_tf=htf)
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.5)

        # Logging Setup
        tz_short = "NY" if "New_York" in str(timezone) else "LON"
        self.log_tag = f"BOS-FVG | {symbol}:{tz_short}"
        self._logger = logger.bind(bot_id=self.log_tag)

        # State
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE

        # Cache processed zones to prevent re-entry on same FVG
        self.processed_zone_ids = set()

        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

    # Logger Wrapper for compatibility
    def log(self, msg: str, level: str = "info"):
        if hasattr(self._logger, level):
            getattr(self._logger, level)(msg)
        else:
            self._logger.info(msg)

    # ==================================================================
    # 1. MARKET STRUCTURE ANALYSIS
    # ==================================================================

    def get_htf_trend(self, df_htf: pd.DataFrame) -> Trend:
        """
        Analyzes 1H Structure. Returns Trend.BULLISH, BEARISH, or NEUTRAL.
        """
        # Run Detector
        result = self.htf_bos_detector.calculate_bos_choch(df_htf, plot=False)
        signals = result.get('signals', [])

        if not signals:
            return Trend.NEUTRAL

        # Get the most recent structural event
        # Signals are usually appended chronologically, so take the last one
        last_sig = signals[-1]

        sig_type = last_sig.get('type', '')
        bias = last_sig.get('bias')  # BULLISH/BEARISH enum from SMCBase

        if bias == BIAS.LONG:  # Assuming SMCBase BULLISH maps to this
            return Trend.BULLISH
        elif bias == BIAS.SHORT:
            return Trend.BEARISH

        # Fallback if bias isn't explicitly set in signal dict
        if 'UP' in sig_type or 'BULL' in sig_type: return Trend.BULLISH
        if 'DOWN' in sig_type or 'BEAR' in sig_type: return Trend.BEARISH

        return Trend.NEUTRAL

    def find_valid_fvg(self, df_ltf: pd.DataFrame, trend: Trend, current_price: float):
        """
        Scans LTF for the best FVG that matches the HTF Trend.
        """
        # Run Detector
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get('signals', [])

        valid_fvgs = []

        # Map Trend to FVG Bias (Detector uses strings "BULLISH", "BEARISH")
        target_bias = BIAS.LONG if trend == Trend.BULLISH else BIAS.SHORT

        for fvg in fvg_signals:
            # 1. Must match direction
            if fvg['bias'] != target_bias:
                continue

            # 2. Must be Unmitigated
            if fvg.get('mitigated', True):
                continue

            # 3. Must be Fresh (Not in processed list)
            # Create unique ID based on time
            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                continue

            # 4. Price Logic (Discount/Premium)
            # Bullish: We want to buy LOWER than current price (Waiting for dip)
            if trend == Trend.BULLISH and fvg['top'] < current_price:
                valid_fvgs.append(fvg)
            # Bearish: We want to sell HIGHER than current price (Waiting for rally)
            elif trend == Trend.BEARISH and fvg['bottom'] > current_price:
                valid_fvgs.append(fvg)

        if not valid_fvgs:
            return None

        # Sort by Recency (Time descending) - We want the closest active FVG
        valid_fvgs.sort(key=lambda x: x['time'], reverse=True)
        return valid_fvgs[0]

    # ==================================================================
    # 2. TRADE CALCULATION
    # ==================================================================

    def calculate_setup_params(self, fvg, trend: Trend, df_ltf: pd.DataFrame):
        atr = self.get_current_atr(df_ltf)

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
    # 3. MAIN EXECUTION LOOP
    # ==================================================================

    def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> SignalSummary:
        """
        Called every 5 mins by MasterBot.
        """
        curr = df_ltf.iloc[-1]

        # 1. FILL CHECK
        if self.trade_status == TradeStatus.PENDING:
            if self.check_fill(curr):
                self.trade_status = TradeStatus.FILLED
                self.log(f"⚡ ORDER FILLED {self.symbol} @ {self.active_trade.price}")
                return SignalSummary(direction="FILLED")

            # Check Validity (Cancel if price ran away to TP?)
            # Simplified for now: Keep pending until session end or fill.

        # 2. EXIT CHECK
        if self.trade_status == TradeStatus.FILLED:
            if self.check_exit(curr):
                self.log(f"🎯 TRADE EXIT (SL/TP) {self.symbol}")
                return SignalSummary(direction="EXIT", trade_signal=self.create_signal(TradeAction.EXIT, curr.name))
            return SignalSummary(direction="IN_TRADE")

        # 3. NEW SIGNAL SEARCH
        if self.trade_status == TradeStatus.NONE:
            # A. Get Trend
            trend = self.get_htf_trend(df_htf)

            if trend != Trend.NEUTRAL:
                # B. Find FVG
                fvg = self.find_valid_fvg(df_ltf, trend, curr['close'])

                if fvg:
                    # C. Calculate Params
                    params = self.calculate_setup_params(fvg, trend, df_ltf)
                    fvg_id = f"{fvg['time']}_{self.symbol}"

                    # D. Generate Signal
                    new_trade = TradeSignal(
                        action=TradeAction.ENTER,
                        symbol=self.symbol,
                        direction=params['direction'],
                        price=params['entry'],
                        sl=params['sl'],
                        tp=params['tp'],
                        size=params['size'],
                        timestamp=curr.name,
                        metadata={"fvg_id": fvg_id, "structure": trend.name}
                    )

                    self.active_trade = new_trade
                    self.trade_status = TradeStatus.PENDING
                    self.processed_zone_ids.add(fvg_id)

                    self.log(
                        f"🚀 SIGNAL: {params['direction'].name} Limit @ {params['entry']:.2f} (Trend: {trend.name})")
                    return SignalSummary(direction=params['direction'].name, trade_signal=new_trade)

        return SignalSummary(direction="NEUTRAL")

    # ==================================================================
    # 4. HELPERS (Risk & State)
    # ==================================================================

    def calculate_trade_position(self, entry_price: float, sl: float, direction: BIAS) -> dict:
        """
        Calculates position size with a HARD CAP (5 Contracts).
        """
        r = abs(entry_price - sl)
        if r == 0: r = 0.25

        # Calculate Raw Size
        risk_per_contract = r * self.point_value
        if risk_per_contract <= 0: risk_per_contract = 5.0

        raw_size = int(self.risk / risk_per_contract)

        # SAFETY CAP
        MAX_CONTRACTS = 5
        size = max(1, min(raw_size, MAX_CONTRACTS))

        # TP
        if direction == BIAS.LONG:
            tp = entry_price + (r * self.tp_multiple)
        else:
            tp = entry_price - (r * self.tp_multiple)

        if raw_size > MAX_CONTRACTS:
            self.log(f"⚠️ RISK CAP: {raw_size} -> {size} (Risk/Cont: ${risk_per_contract:.2f})", level="warning")

        return {"sl": sl, "tp": tp, "size": size}

    def create_signal(self, action: TradeAction, timestamp) -> TradeSignal:
        sig = TradeSignal(
            symbol=self.symbol, action=action,
            direction=self.active_trade.direction,
            price=self.active_trade.price, sl=self.active_trade.sl, tp=self.active_trade.tp,
            size=self.active_trade.size, timestamp=timestamp
        )
        self.active_trade, self.trade_status = None, TradeStatus.NONE
        return sig

    def check_fill(self, candle) -> bool:
        if not self.active_trade: return False
        # Limit Logic:
        # Buy Limit: Low must touch or go below Price
        # Sell Limit: High must touch or go above Price
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= self.active_trade.price
        else:
            return candle['high'] >= self.active_trade.price

    def check_exit(self, candle) -> bool:
        if not self.active_trade: return False
        sl, tp = self.active_trade.sl, self.active_trade.tp
        if self.active_trade.direction == BIAS.LONG:
            return candle['low'] <= sl or candle['high'] >= tp
        else:
            return candle['high'] >= sl or candle['low'] <= tp

    def sync_state(self, current_positions: list, open_orders: list):
        """
        Recovers state from IBKR.
        """
        if self.trade_status != TradeStatus.NONE: return

        # Unique Tag: MNQ_NY_BOS_FVG
        session_short = "NY" if "New_York" in str(self.timezone) else "LON"
        my_tag = f"{self.symbol}_{session_short}_BOS_FVG"

        # 1. Position Check
        my_pos = next((p for p in current_positions if p.contract.symbol == self.symbol and p.position != 0), None)
        if my_pos:
            direction = BIAS.LONG if my_pos.position > 0 else BIAS.SHORT
            self.active_trade = TradeSignal(
                action=TradeAction.ENTER, symbol=self.symbol, direction=direction,
                price=my_pos.avgCost, sl=0, tp=0, size=abs(int(my_pos.position)),
                timestamp=pd.Timestamp.now(), metadata={"fvg_id": "RECOVERED"}
            )
            self.trade_status = TradeStatus.FILLED
            self.log(f"♻️ STATE RESTORED: ACTIVE POSITION {direction.name}")
            return

        # 2. Pending Check
        my_orders = [o for o in open_orders if o.order.orderRef == my_tag]
        parent = next((t for t in my_orders if t.order.parentId == 0), None)

        if parent:
            direction = BIAS.LONG if parent.order.action == 'BUY' else BIAS.SHORT
            self.active_trade = TradeSignal(
                action=TradeAction.ENTER, symbol=self.symbol, direction=direction,
                price=parent.order.lmtPrice, sl=0, tp=0, size=int(parent.order.totalQuantity),
                timestamp=pd.Timestamp.now(), metadata={"fvg_id": "RECOVERED"}
            )
            self.trade_status = TradeStatus.PENDING
            self.log(f"♻️ STATE RESTORED: PENDING ORDER {direction.name}")