from datetime import timedelta
from typing import Optional, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.LevelsDetector import LevelsDetector
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    ExitType,
    SESSION_MAP,
    SignalSummary,
    StrategyBase,
    TradeAction,
    TradeSignal,
    TradeStatus,
)
from daily_trading.settings import logger


class DailySweepDeepStrategy(StrategyBase):
    """
    STRATEGY: Daily Sweep + Deep Mitigation (SMC)

    OVERVIEW:
    A multi-timeframe SMC strategy that:
    1) Maps daily/weekly/monthly external liquidity (PDH/PDL/PWH/PWL/PMH/PML).
    2) Waits for a 1H liquidity sweep and reclaim at those levels.
    3) Enters on a 5m FVG at 75% mitigation (deep pullback).

    GOAL:
    Align external liquidity sweep with internal displacement to capture high-probability reversals.
    """

    def __init__(
            self,
            symbol: str,
            timezone: ZoneInfo,
            main_tf: str = "5m",
            risk: float = 150.0,
            fvg_mitigation: float = 0.75,
            sl_buffer: float = 0.0,
            fvg_max_age_minutes: int = 120,
            priority: int = 1,
    ):
        super().__init__(main_tf, priority=priority)

        self.symbol = symbol
        self.timezone = timezone
        self.log_tag = f"1H-SWEEP-FVG | {symbol}:{self.get_tz_short(self.timezone)}"
        self.log = logger.bind(bot_id=self.log_tag)
        self.ext_tag = f"{symbol}_{self.get_tz_short(self.timezone)}_SWEEP_FVG"

        # Risk / Entry Config
        self.risk = risk
        self.entry_depth = fvg_mitigation  # 0.75 = 75% mitigation (deep pullback)
        self.sl_buffer_points = sl_buffer
        self.fvg_max_age_minutes = fvg_max_age_minutes

        # Session Config
        if timezone == ZoneInfo("America/New_York"):
            self.end_buffer_minutes = self.NY_END_BUFFER_MIN
        else:
            self.end_buffer_minutes = 0

        # Detectors
        self.htf_bos_detector = BosChochDetector(main_tf="1h")
        self.ltf_fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.25)
        self.levels_detector = LevelsDetector(main_tf="1h")

        # State
        self.active_trade: Optional[TradeSignal] = None
        self.trade_status: TradeStatus = TradeStatus.NONE
        self.processed_zone_ids: Set[str] = set()

    def reset_session_state(self):
        """Standard reset for start of trading day."""
        self.log.info("Resetting Session State.")
        self.active_trade = None
        self.trade_status = TradeStatus.NONE
        self.processed_zone_ids.clear()

    # -------------------------------------------------------------------------
    # HELPERS: Bias / Sweep / FVG
    # -------------------------------------------------------------------------

    def _get_level_map(self, df_htf: pd.DataFrame) -> dict:
        """
        Build a dict of major levels for quick access (PDH/PDL/PWH/PWL/PMH/PML).
        """
        levels = self.levels_detector.calculate_major_levels(df_htf)
        level_map = {}
        for lvl in levels:
            level_map[lvl["type"]] = lvl
        return level_map

    def _validate_sweep(
            self, df_htf: pd.DataFrame, level_map: dict, bias: BIAS
    ) -> Optional[Tuple[str, float, pd.Timestamp, float, float]]:
        """
        Validate 1H sweep:
        - Long: sweep below PDL/PWL and close back above.
        - Short: sweep above PDH/PWH and close back below.

        Returns: (level_type, level_price, sweep_ts, sweep_high, sweep_low)
        """
        if df_htf.empty:
            return None

        last = df_htf.iloc[-1]
        sweep_ts = df_htf.index[-1]

        if bias == BIAS.LONG:
            candidates = []
            for level_type in ["PDL", "PWL", "PML"]:
                level = level_map.get(level_type)
                if not level:
                    continue
                lvl_price = level["price"]
                if last["low"] < lvl_price and last["close"] > lvl_price:
                    candidates.append((level_type, lvl_price))
            if not candidates:
                return None
            # Choose the closest swept level to current price (most relevant reclaim)
            chosen = max(candidates, key=lambda x: x[1])
            return chosen[0], chosen[1], sweep_ts, last["high"], last["low"]

        if bias == BIAS.SHORT:
            candidates = []
            for level_type in ["PDH", "PWH", "PMH"]:
                level = level_map.get(level_type)
                if not level:
                    continue
                lvl_price = level["price"]
                if last["high"] > lvl_price and last["close"] < lvl_price:
                    candidates.append((level_type, lvl_price))
            if not candidates:
                return None
            # Choose the closest swept level to current price (most relevant reclaim)
            chosen = min(candidates, key=lambda x: x[1])
            return chosen[0], chosen[1], sweep_ts, last["high"], last["low"]

        return None

    def _select_fvg_after_sweep(self, df_ltf: pd.DataFrame, bias: BIAS, sweep_ts: pd.Timestamp) -> Optional[dict]:
        """
        Select the most recent unmitigated FVG after the sweep close.
        """
        result = self.ltf_fvg_detector.calculate_fvg(df_ltf, plot=False)
        fvg_signals = result.get("signals", [])
        if not fvg_signals:
            return None

        target_bias_val = BIAS.LONG.num_value if bias == BIAS.LONG else BIAS.SHORT.num_value
        max_age = timedelta(minutes=self.fvg_max_age_minutes)

        for fvg in fvg_signals:
            if fvg.get("bias") != target_bias_val:
                continue
            if fvg.get("mitigated", True):
                continue

            fvg_time = fvg.get("time")
            if fvg_time is None:
                continue

            # Only accept FVGs created during/after the sweep impulse window.
            if fvg_time < sweep_ts or fvg_time > sweep_ts + max_age:
                continue

            return fvg

        return None

    def _calc_entry_sl_tp(
            self,
            fvg: dict,
            bias: BIAS,
            sweep_high: float,
            sweep_low: float,
            level_map: dict,
            entry_current_price: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Compute entry (75% mitigation), SL beyond sweep swing, and TP1/TP2.
        Returns: (entry, sl, tp2)
        """
        top = fvg["top"]
        bottom = fvg["bottom"]
        height = top - bottom

        if bias == BIAS.LONG:
            entry = top - (height * self.entry_depth)
            if entry > entry_current_price:
                return None
            sl = sweep_low - self.sl_buffer_points
        else:
            entry = bottom + (height * self.entry_depth)
            if entry < entry_current_price:
                return None
            sl = sweep_high + self.sl_buffer_points

        # TP2: opposing major level
        tp2 = None
        if bias == BIAS.LONG:
            candidates = [level_map.get(k) for k in ["PDH", "PWH", "PMH"] if level_map.get(k)]
            above = [c["price"] for c in candidates if c["price"] > entry]
            tp2 = min(above) if above else None
        else:
            candidates = [level_map.get(k) for k in ["PDL", "PWL", "PML"] if level_map.get(k)]
            below = [c["price"] for c in candidates if c["price"] < entry]
            tp2 = max(below) if below else None

        # TP2 is optional; TP1 will be set by internal liquidity in analyze.
        return entry, sl, tp2

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

        sig.entry_timestamp = self.active_trade.entry_timestamp
        if self.active_trade and self.active_trade.metadata:
            sig.metadata.update(self.active_trade.metadata)

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
        entry_cutoff = session_end - timedelta(minutes=30)

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
                self.log.info(f"⚡ ORDER FILLED SWEEP-FVG {self.symbol} @ {self.active_trade.price:.2f}")

        # ==================================================================
        # 2. EXIT CHECK (Stop Loss, Take Profit, Zombies, Cutoff)
        # ==================================================================
        if self.trade_status == TradeStatus.FILLED:
            self.processed_zone_ids.add(self.active_trade.metadata.get("fvg_id"))

            if self.active_trade.sl <= 0 or self.active_trade.price <= 0:
                self.log.error(f"🧟 ZOMBIE TRADE DETECTED ({self.symbol}). Cleaning up.")
                self.active_trade, self.trade_status = None, TradeStatus.NONE
                return SignalSummary(direction="NEUTRAL")

            sl, tp = self.active_trade.sl, self.active_trade.tp
            is_long = self.active_trade.direction == BIAS.LONG

            exit_type = None
            exit_price = None
            exit_reason = "NORMAL"

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

                if not exit_type:
                    if is_long:
                        hit_sl = curr["low"] <= sl
                        hit_tp = curr["high"] >= tp
                    else:
                        hit_sl = curr["high"] >= sl
                        hit_tp = curr["low"] <= tp

                    if hit_sl and hit_tp:
                        hit_tp = False

                    if hit_sl:
                        exit_type = "LOSS"
                        exit_price = sl
                    elif hit_tp:
                        exit_type = "WIN"
                        exit_price = tp
            else:
                exit_type, exit_price, exit_reason = await self.check_broker_exit(ib_client, self.symbol, sl, tp)
                if not exit_type:
                    exit_type, exit_price, exit_reason = self.check_sl_tp_failure(curr, sl, tp,
                                                                                  self.active_trade.direction)
                    if exit_type:
                        self.log.error(f"🚨 BROKER FAILURE DETECTED: Price ({exit_price}) passed limits. Forcing Exit.")

            if not exit_type and ts_now >= cutoff_ts:
                exit_price = curr["close"]
                exit_reason = "CUTOFF"
                pnl = (exit_price - self.active_trade.price) if is_long else (self.active_trade.price - exit_price)
                exit_type = "WIN" if pnl > 0 else "LOSS"

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
        # 3. HTF TREND ANALYSIS + SWEEP VALIDATION
        # ==================================================================
        trend_bias = self.htf_bos_detector.get_current_trend(df_htf)
        if trend_bias == BIAS.NEUTRAL:
            return SignalSummary(direction="NEUTRAL")

        level_map = self._get_level_map(df_htf)
        sweep = self._validate_sweep(df_htf, level_map, trend_bias)
        if not sweep:
            return SignalSummary(direction="NEUTRAL")

        sweep_level, sweep_price, sweep_ts, sweep_high, sweep_low = sweep

        # ==================================================================
        # 4. NEW ENTRY SEARCH (FVG AFTER SWEEP)
        # ==================================================================
        if self.trade_status == TradeStatus.NONE and ts_now < entry_cutoff:
            fvg = self._select_fvg_after_sweep(df_ltf, trend_bias, sweep_ts)
            if not fvg:
                return SignalSummary(direction="NEUTRAL")

            fvg_id = f"{fvg['time']}_{self.symbol}"
            if fvg_id in self.processed_zone_ids:
                return SignalSummary(direction="NEUTRAL")

            # Entry / SL / TP logic
            entry_sl_tp = self._calc_entry_sl_tp(
                fvg=fvg,
                bias=trend_bias,
                sweep_high=sweep_high,
                sweep_low=sweep_low,
                level_map=level_map,
                entry_current_price=curr["close"],
            )
            if not entry_sl_tp:
                return SignalSummary(direction="NEUTRAL")

            entry, sl, tp2 = entry_sl_tp

            # TP1: nearest internal 1H liquidity (pivot highs/lows)
            pivot_highs, pivot_lows = self.htf_bos_detector.calculate_pivots(df_htf)
            if trend_bias == BIAS.LONG:
                above = [p["level"] for p in pivot_highs if p["level"] > entry]
                tp1 = min(above) if above else None
            else:
                below = [p["level"] for p in pivot_lows if p["level"] < entry]
                tp1 = max(below) if below else None

            # Final TP preference: TP1 (internal liquidity) then TP2 (external level)
            tp = tp1 if tp1 is not None else tp2
            if tp is None:
                return SignalSummary(direction="NEUTRAL")

            # Position sizing using existing risk engine
            pos_math = self.calculate_trade_position(
                symbol=self.symbol,
                tp_multiple=1.0,  # size only; TP is set explicitly below
                risk=self.risk,
                entry_price=entry,
                sl=sl,
                direction=trend_bias,
            )

            if pos_math["size"] <= 0 or pos_math["sl"] <= 0:
                return SignalSummary(direction="NEUTRAL")

            new_trade = TradeSignal(
                action=TradeAction.ENTER,
                symbol=self.symbol,
                direction=trend_bias,
                price=entry,
                sl=sl,
                tp=tp,
                size=pos_math["size"],
                timestamp=ts_utc,
                metadata={
                    "fvg_id": fvg_id,
                    "sweep_level": sweep_level,
                    "sweep_price": sweep_price,
                    "sweep_ts": sweep_ts,
                    "tp1": tp1,
                    "tp2": tp2,
                },
            )

            self.active_trade = new_trade
            self.trade_status = TradeStatus.PENDING
            self.log.info(f"🚀 LIMIT ORDER PLACED: {new_trade.direction.name} @ {new_trade.price:.2f}")
            return SignalSummary(direction=new_trade.direction.name, trade_signal=new_trade)

        # ==================================================================
        # 5. PENDING MANAGEMENT (Cutoff)
        # ==================================================================
        if self.trade_status == TradeStatus.PENDING and ts_now >= cutoff_ts:
            return SignalSummary(
                direction="CANCELLED",
                trade_signal=self.create_signal(TradeAction.CANCEL, ts_utc, reason="End of Session"),
            )

        return SignalSummary(direction="NEUTRAL")
