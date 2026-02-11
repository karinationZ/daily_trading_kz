from datetime import timedelta
from typing import Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    ExitType,
)
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    SESSION_MAP,
    SignalSummary,
    StrategyBase,
    TradeAction,
    TradeSignal,
    TradeStatus,
)
from daily_trading.settings import logger
from daily_trading.technicals.AITradeAssistants.AiAssistant import (
    AITradeAssistant,
    AssetData,
    GEMINI_PRO,
    PromptType,
)


class BosFvgStrategy(StrategyBase):
    """
    STRATEGY: Multi-Timeframe Trend Following (SMC) - BOS + LTF FVG

    OVERVIEW:
    A "Smart Money Concept" strategy that aligns Higher Timeframe (HTF) market structure
    with Lower Timeframe (LTF) entry signals.

    LOGIC:
    1. HTF (e.g., 1H): Identify Trend Bias via Break of Structure (BOS).
    2. LTF (e.g., 5m): Identify "Fresh" Fair Value Gaps (FVG) matching the bias.
    3. ENTRY: Limit Order inside the FVG (at `fvg_mitigation_ratio`).
    4. EXIT: SL outside FVG, Fixed RR TP.

    ROBUSTNESS:
    - Stateless recovery (checks FVG invalidation on restart).
    - Zombie protection (bad math/state detection).
    - Session cutoffs.
    """

    def __init__(
        self,
        symbol: str,
        timezone: ZoneInfo,
        main_tf="5m",  # LTF for Entry execution
        risk: float = 150.0,
        tp_multiple: float = 2.0,
        fvg_mitigation_ratio: float = 0.5,  # 0.5 = Entry at 50% of FVG
        sl_buffer_mult: float = 0.1,
        priority: int = 1,
    ):

        super().__init__(main_tf, priority=priority)

        self.symbol = symbol
        self.timezone = timezone

        # Logging
        telegram_chat_id = self.get_telegram_session_channel_id(self.timezone)
        self.telegram_session_bot = TelegramBot(chat_id=telegram_chat_id)
        self.tz_short = self.get_tz_short(self.timezone)
        self.log_tag = f"1H-BOS-FVG | {symbol}:{self.tz_short}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{self.tz_short}_BOS_FVG"

        telegram_chat_id = self.get_telegram_session_channel_id(self.timezone)
        self.telegram_session_bot = TelegramBot(chat_id=telegram_chat_id)

        # Config
        self.risk = risk
        self.tp_multiple = tp_multiple
        self.entry_depth = fvg_mitigation_ratio
        self.sl_buffer_mult = sl_buffer_mult
        self.point_value = self.POINT_VALUE.get(symbol, 5.0)

        # Session Config
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

        # --- DETECTORS ---
        # HTF: For Trend Bias
        self.htf_bos_detector = BosChochDetector(main_tf="1h")
        # LTF: For Entry Signals
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.25)
        self.ob_detector = SupplyDemandZonesDetector(fvg_based_only=False, swing_length=50, internal_length=5)

        self.ai_throttle_minutes = 30
        self.last_ai_call_ts: Optional[pd.Timestamp] = None
        self.use_ai = symbol in ["MBT", "MNQ", "MGC"]

        # --- STATE ---
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[str] = set()
        self.restore_metadata_defaults = {"bos_ts": pd.Timestamp(0, tz="UTC")}

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        self.log.info("Resetting Session State.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    # -------------------------------------------------------------------------
    # HELPERS: Safety & Math
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # LOGIC: Find Setup
    # -------------------------------------------------------------------------
    async def find_ai_setup(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame):
        active_swing, active_internal = self.ob_detector.calculate(df_ltf)
        ob_signals = active_swing + active_internal
        fvg_results = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = fvg_results.get("signals", [])
        pivot_high_points, pivot_low_points = self.ltf_fvg_detector.calculate_pivots(df_ltf)
        pivots = {"5m_highs": pivot_high_points, "5m_lows": pivot_low_points}
        bos_choch_signals = self.htf_bos_detector.calculate_bos_choch(df_htf, plot=False).get("signals", [])

        asset_data = AssetData(
            symbol=self.symbol,
            current_price=df_ltf.iloc[-1]["close"],
            fvg_signals=fvg_signals,
            ob_signals=ob_signals,
            bos_choch_signals=bos_choch_signals,
            high_lows=pivots,
            ohlc_1h=df_htf.tail(100).to_dict(orient="records"),
            ohlc_5m=df_ltf.tail(50).to_dict(orient="records"),
        )

        ai_agent = AITradeAssistant(symbol=self.symbol, model=GEMINI_PRO, debug=self.debug)
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data(
            prompt_type=PromptType.BOS_FVG,
            asset_data=asset_data,
        )

        if not raw_message:
            return None, formatted_ai_message

        decision = raw_message.get("decision", "")
        params = raw_message.get("parameters", {})
        metadata = raw_message.get("metadata", {})

        if decision == "ENTER_LONG":
            direction = BIAS.LONG
        elif decision == "ENTER_SHORT":
            direction = BIAS.SHORT
        else:
            return None, formatted_ai_message

        entry = params.get("entry_price")
        sl = params.get("stop_loss")
        tp = params.get("take_profit")
        size = params.get("size")
        size = int(size)
        if size < 1:
            size = 1
        if entry is None or sl is None or tp is None:
            return None, formatted_ai_message

        new_trade = TradeSignal(
            action=TradeAction.ENTER,
            symbol=self.symbol,
            direction=direction,
            price=entry,
            sl=sl,
            tp=tp,
            size=size,
            timestamp=pd.Timestamp.now(tz=self.timezone),
            metadata={
                "fvg_id": metadata.get("fvg_id"),
                "bos_ts": metadata.get("bos_ts"),
                "ai_reasoning": raw_message.get("reasoning"),
            },
        )
        return new_trade, formatted_ai_message

    def find_setup(self, df_ltf: pd.DataFrame, htf_bias: BIAS, bos_time, ts_now) -> Optional[TradeSignal]:
        """
        Scans LTF for the MOST RECENT fresh FVG that aligns with HTF Bias.
        """
        # Get LTF FVGs (Already sorted by Time: Newest -> Oldest)
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get("signals", [])
        atr = self.ltf_fvg_detector.get_current_atr(df_ltf)

        if not fvg_signals:
            return None

        current_price = df_ltf.iloc[-1]["close"]
        target_bias_val = BIAS.LONG.num_value if htf_bias == BIAS.LONG else BIAS.SHORT.num_value

        # Iterate FVGs (Newest First)
        for fvg in fvg_signals:

            # A. Filter by Bias
            if fvg["bias"] != target_bias_val:
                continue

            # B. Filter by Mitigation (Must be Fresh)
            # We rely on the detector's geometry check.
            if fvg.get("mitigated", True):
                continue

            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                continue

            # D. Calculate Entry Params
            top = fvg["top"]
            bottom = fvg["bottom"]
            height = top - bottom

            if htf_bias == BIAS.LONG:
                entry = top - (height * self.entry_depth)

                # SL calculated from entry (like TradingView tool)
                sl_distance_from_atr = atr * self.sl_buffer_mult

                # Minimum SL distance from entry as percentage of entry price
                min_sl_distance_pct = 0.001  # 0.1% of entry price (adjustable)
                min_sl_distance = entry * min_sl_distance_pct

                # Use the LARGER distance
                sl_distance = max(sl_distance_from_atr, min_sl_distance)

                # SL is entry minus distance
                sl = entry - sl_distance
            else:
                entry = bottom + (height * self.entry_depth)

                # SL calculated from entry (like TradingView tool)
                sl_distance_from_atr = atr * self.sl_buffer_mult

                # Minimum SL distance from entry as percentage of entry price
                min_sl_distance_pct = 0.001  # 0.15% of entry price (adjustable)
                min_sl_distance = entry * min_sl_distance_pct

                # Use the LARGER distance
                sl_distance = max(sl_distance_from_atr, min_sl_distance)

                # SL is entry plus distance
                sl = entry + sl_distance

                # CRITICAL: For a Limit Sell, Entry must be >= Current Price
                if entry < current_price:
                    continue

            # E. Calculate Position Size
            pos_math = self.calculate_trade_position(
                symbol=self.symbol, tp_multiple=self.tp_multiple, risk=self.risk, entry_price=entry, sl=sl, direction=htf_bias
            )

            # Gatekeeper (Risk Management)
            if pos_math["size"] <= 0 or pos_math["sl"] <= 0:
                continue

            # F. Return the FIRST valid match (The Most Recent One)
            return TradeSignal(
                action=TradeAction.ENTER,
                symbol=self.symbol,
                direction=htf_bias,
                price=pos_math["entry"] if "entry" in pos_math else entry,
                sl=pos_math["sl"],
                tp=pos_math["tp"],
                size=pos_math["size"],
                timestamp=ts_now,
                metadata={
                    "fvg_id": fvg_id,
                    "bos_ts": bos_time,
                },
            )

        return None

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    def create_signal(self, action: TradeAction, timestamp, custom_price: float = None, reason=None) -> TradeSignal:
        """Helper to create signals and manage state transitions."""
        report_price = custom_price if custom_price is not None else self.active_trade.price

        sig = TradeSignal(
            symbol=self.symbol,
            action=action,
            direction=self.active_trade.direction,
            price=report_price,
            sl=self.active_trade.sl,
            tp=self.active_trade.tp,
            size=self.active_trade.size,
            reason=reason,
            timestamp=timestamp,
        )

        # Pass metadata
        sig.entry_timestamp = self.active_trade.entry_timestamp
        if self.active_trade and self.active_trade.metadata:
            sig.metadata.update(self.active_trade.metadata)

        # State Reset on Terminal Actions
        if action in [TradeAction.CANCEL, TradeAction.EXIT]:
            self.active_trade, self.trade_status = None, TradeStatus.NONE

        return sig

    async def analyze(self, df_ltf: pd.DataFrame, df_htf: pd.DataFrame, ib_client) -> SignalSummary:
        curr = df_ltf.iloc[-1]
        ts_now = curr.name.tz_convert(self.timezone)
        ts_utc = curr.name.tz_convert("UTC")

        # Session Cutoff Calculation
        s_info = SESSION_MAP[self.timezone.key]
        session_end = ts_now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0,
        )
        cutoff_ts = session_end - timedelta(minutes=self.end_buffer_minutes)
        entry_cutoff = session_end - timedelta(minutes=30)  # No new entries last 30m

        just_filled = False

        # ==================================================================
        # 1. FILL SIMULATION (DEBUG / BACKTEST ONLY)
        # ==================================================================
        if self.trade_status == TradeStatus.PENDING and self.debug:
            filled = False
            if self.active_trade.direction == BIAS.LONG:
                if curr["low"] <= self.active_trade.price:
                    filled = True
            else:
                if curr["high"] >= self.active_trade.price:
                    filled = True

            if filled:
                self.trade_status = TradeStatus.FILLED
                self.active_trade.entry_timestamp = ts_utc
                just_filled = True
                msg = f"⚡ ORDER FILLED BOS 1H-FVG {self.symbol} @ {self.active_trade.price:.2f}"
                self.log.info(msg)

        # ==================================================================
        # 2. EXIT CHECK (Stop Loss, Take Profit, Zombies, Cutoff)
        # ==================================================================
        if self.trade_status == TradeStatus.FILLED:
            self.processed_zone_ids.add(self.active_trade.metadata.get("fvg_id"))
            # --- ZOMBIE KILLER (Safety) ---
            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.log.error(f"🧟 ZOMBIE TRADE DETECTED ({self.symbol}). Cleaning up.")
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG

            exit_type = None
            exit_price = None
            exit_reason = "NORMAL"

            # SIMULATION -- GAP CHECK (If not just filled)
            if self.debug:
                if not just_filled:
                    if is_long:
                        if curr["open"] <= sl:
                            exit_type, exit_price = "LOSS", curr["open"]
                        elif curr["open"] >= tp:
                            exit_type, exit_price = "WIN", curr["open"]
                    else:
                        if curr["open"] >= sl:
                            exit_type, exit_price = "LOSS", curr["open"]
                        elif curr["open"] <= tp:
                            exit_type, exit_price = "WIN", curr["open"]

                # B. STANDARD CHECK (Trust Wicks)
                if not exit_type:
                    if is_long:
                        hit_sl = curr["low"] <= sl
                        hit_tp = curr["high"] >= tp
                    else:
                        hit_sl = curr["high"] >= sl
                        hit_tp = curr["low"] <= tp

                    # Pessimistic conflict resolution
                    if hit_sl and hit_tp:
                        hit_tp = False

                    if hit_sl:
                        exit_type = "LOSS"
                        exit_price = sl
                    elif hit_tp:
                        exit_type = "WIN"
                        exit_price = tp
            else:
                # -------------------------------------------------------------
                # REALITY CHECK (Live Trading Only)
                # -------------------------------------------------------------
                exit_type, exit_price, exit_reason = await self.check_broker_exit(ib_client, self.symbol, sl, tp)

                if not exit_type:
                    exit_type, exit_price, exit_reason = self.check_sl_tp_failure(curr, sl, tp, self.active_trade.direction)
                    if exit_type:
                        self.log.error(f"🚨 BROKER FAILURE DETECTED: Price ({exit_price}) passed limits. Forcing Exit.")

            # C. TIME CUTOFF
            is_cutoff = ts_now >= cutoff_ts
            if not exit_type and is_cutoff:
                exit_price = curr["close"]
                exit_reason = "CUTOFF"

                # Determine Result Manually
                pnl = (exit_price - self.active_trade.price) if is_long else (self.active_trade.price - exit_price)
                exit_type = "WIN" if pnl > 0 else "LOSS"

            # --- EXECUTION ---
            if exit_type:
                sig = self.create_signal(TradeAction.EXIT, ts_utc, custom_price=exit_price)
                sig.trade_result = exit_type
                sig.metadata["exit_reason"] = exit_reason

                if exit_reason == "CUTOFF":
                    sig.metadata["exit_type"] = ExitType.MARKET_FORCE
                    sig.metadata["exit_reason"] = "Session Cutoff"

                elif "BROKER_FAILED" in exit_reason:
                    sig.metadata["exit_type"] = ExitType.MARKET_FORCE
                    sig.metadata["exit_reason"] = "Broker Failed to Execute SL/TP"
                else:
                    sig.metadata["exit_type"] = ExitType.SL_TP_HIT
                    sig.metadata["exit_reason"] = "SL/TP Hit"

                self.log.info(f"🎯 TRADE EXIT ({exit_type}) {self.symbol} | Reason: {exit_reason}")
                return SignalSummary(direction="EXIT", trade_signal=sig)

            return SignalSummary(direction="IN_TRADE")

        # ==================================================================
        # 3. HTF TREND ANALYSIS
        # ==================================================================
        trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
        bos_signals = self.htf_bos_detector.signals

        # If no structure, no trade
        if trend_bias == BIAS.NEUTRAL or not bos_signals:
            return SignalSummary(direction="NEUTRAL")

        last_bos = bos_signals[-1]
        bos_time = last_bos.get("time")

        # ==================================================================
        # 4. NEW ENTRY SEARCH
        # ==================================================================
        if self.trade_status == TradeStatus.NONE:
            if ts_now < entry_cutoff:
                new_trade = self.find_setup(df_ltf, trend_bias, bos_time, ts_utc)
                if new_trade and self.use_ai and not self.debug:
                    new_trade = None
                    should_call_ai = True
                    if self.last_ai_call_ts is not None:
                        elapsed_min = (ts_utc - self.last_ai_call_ts).total_seconds() / 60.0
                        if elapsed_min < self.ai_throttle_minutes:
                            should_call_ai = False

                    if should_call_ai:
                        self.log.info("Calling AI to find trade setup...")
                        new_trade, ai_msg = await self.find_ai_setup(df_ltf, df_htf)
                        self.last_ai_call_ts = ts_utc
                        await self.telegram_session_bot.send_message(f"🤖 AI Analysis for {self.symbol}:\n\n{ai_msg}")
                        if new_trade and new_trade.metadata is not None:
                            new_trade.metadata["bos_ts"] = bos_time

                if new_trade:
                    self.active_trade = new_trade
                    self.trade_status = TradeStatus.PENDING

                    self.log.info(f"🚀 LIMIT ORDER PLACED: {new_trade.direction.name} @ {new_trade.price:.2f}")
                    return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)
                else:
                    self.log.info(f"No valid setup found for {self.symbol}")

        # ==================================================================
        # 5. PENDING MANAGEMENT (Cutoff & Structure Update)
        # ==================================================================
        if self.trade_status == TradeStatus.PENDING:
            should_cancel = False
            cancel_reason = ""

            # A. Time Cutoff
            if ts_now >= cutoff_ts:
                should_cancel = True
                cancel_reason = "End of Session"

            # B. Check for BOS Change (Trailing Entry Logic)
            # If HTF forms a NEW BOS, the old FVG might be outdated or trend accelerated.
            # Cancel old limit to look for a new one in the next tick.
            elif self.active_trade and self.active_trade.metadata:
                current_trade_bos_ts = self.active_trade.metadata.get("bos_ts")

                if current_trade_bos_ts and bos_time != current_trade_bos_ts:
                    should_cancel = True
                    cancel_reason = "New BOS Detected"

            if should_cancel:
                self.log.info(f"🚫 CANCELLING PENDING ORDER | {cancel_reason}")
                return SignalSummary(
                    direction="CANCELLED",
                    trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc, reason=cancel_reason),
                )

        return SignalSummary(direction="NEUTRAL")
