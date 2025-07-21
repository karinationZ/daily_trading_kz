from datetime import datetime
from typing import List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase
from daily_trading.settings import logger


class BosFvgFillStrategy(StrategyBase):
    """
    Analyzes OHLC data using Smart Money Concept (SMC) for continuation trade signals.
    Designed for scalping in volatile crypto markets with a 6-step process:
    1. Identify Trend 1H (last 24 candles)
    2. Detect Break of Structure (BOS) on 1H; reset on CHOCH.
    4. Find unmitigated FVG below (uptrend) or above (downtrend) liquidity pool on 15m.
    5. Confirm price touches the nearest FVG on 15m.
    6. Generate LONG/SHORT signal
    """

    def __init__(self, main_tf: str, debug: bool, telegram_chat_id):
        """
        Initialize SMC strategy with detectors.
        """
        super().__init__(main_tf, debug, telegram_chat_id=telegram_chat_id)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'bos_1h': BosChochDetector(main_tf),  # Detects BOS/CHOCH on 1H
            'fvg_15m': FvgDetector("15m")  # Detects FVGs on 15m
        }
        self.bos_signals = None
        self.recent_bos_signal = None
        self.bos_choch_fig = None
        self.fvg_fig = None
        self.last_bos_time = None  # Track last BOS time
        self.last_mitigated_fvg_level = None

    async def confirm_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime):
        """Step 1: Identify Trend on 1H (last 24 candles)."""
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        fig = swing_1h["fig"]
        trend, recent_smc_events = await self.get_trend(ohlc_1h, symbol, cache_key, current_time, fig)
        return trend, recent_smc_events, fig

    async def detect_bos_choch(self, ohlc_1h: pd.DataFrame, trend: str, symbol: str) -> List[str]:
        """Step 2: Detect BOS or CHOCH on 1H, reset on CHOCH."""
        recent_smc_events = []
        results = self.detectors['bos_1h'].calculate_bos_choch(ohlc_1h, plot=True)
        self.bos_signals = results["signals"]
        self.bos_choch_fig = results["fig"]
        bos_choch_signals = sorted(results["signals"], key=lambda x: x["time"], reverse=True)

        if not bos_choch_signals:
            event = "There are no BOS or ChoCH signals."
            recent_smc_events.append(event)
            return recent_smc_events

        recent_signal = bos_choch_signals[0]

        # Check if last smc event is ChoCH
        if recent_signal["type"] == "CHoCH":
            bos_event = f"Change of Character! Market trend may change ..."
            self.recent_bos_signal = None
            recent_smc_events.append(bos_event)
            logger.info(f"Step 2: {bos_event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {bos_event}.", self.bos_choch_fig)
            return recent_smc_events

        # Get recent BOS signal
        expected_bias = Trend.BULLISH.value if trend == Trend.BULLISH.name else Trend.BEARISH.value

        if recent_signal['bias'] != expected_bias:
            bos_event = f"Recent BOS opposes the trend direction. BOS bias {expected_bias}. Trend: {trend}"
            self.recent_bos_signal = None
            recent_smc_events.append(bos_event)
            logger.info(f"Step 2: {bos_event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {bos_event}.", self.bos_choch_fig)
            return recent_smc_events

        expected_bos_signals = [s for s in bos_choch_signals if s['type'] == 'BOS' and s['bias'] == expected_bias]

        if not expected_bos_signals:
            bos_event = f"No BOS signals detected. Expected Bias - {expected_bias}."
            recent_smc_events.append(bos_event)
            logger.debug(f"Step 2: {bos_event}")
        else:
            self.recent_bos_signal = expected_bos_signals[0]
            bos_event = f"{self.recent_bos_signal['type']} at " \
                        f"{self.recent_bos_signal['time'].strftime('%Y-%m-%d %H:%M')}"
            recent_smc_events.append(bos_event)
            logger.debug(f"Step 2: {bos_event}")

        await self.send_telegram_message(f"Step 2: {bos_event} for {symbol}.", self.bos_choch_fig)
        return recent_smc_events

    async def get_nearest_fvg(self, ohlc_15m: pd.DataFrame, trend: str):
        """Step 4: Confirm price touches the nearest unmitigated FVG on 15m."""
        recent_smc_events = []

        if trend == Trend.NEUTRAL.name:
            recent_smc_events.append(f"Trend is {trend}. Exiting FVG search step.")
            return None, recent_smc_events

        fvg_result = self.detectors["fvg_15m"].calculate_fvg(ohlc_15m, plot=True)
        self.fvg_fig = self.detectors["fvg_15m"].merge_bos_fvg_plots(ohlc_15m, self.bos_signals, fvg_result["signals"])

        if not fvg_result["signals"]:
            event = "No FVG signals!"
            self.fvg_fig = fvg_result["fig"]
            return None, [event]

        self.fvg_fig = self.detectors["fvg_15m"].merge_bos_fvg_plots(ohlc_15m, self.bos_signals, fvg_result["signals"])
        fvg_signals_sorted = sorted(fvg_result['signals'], key=lambda x: x['time'], reverse=True)

        bos_level = self.recent_bos_signal["level"]
        is_bullish_trend = trend == Trend.BULLISH.name

        # Find the nearest unmitigated FVG based on trend
        nearest_fvg = None

        for fvg_signal in fvg_signals_sorted:
            # Skip if mitigated
            if fvg_signal.get('mitigated', True):
                continue

            # Check if FVG bias matches trend
            expected_bias = Trend.BULLISH.value if is_bullish_trend else Trend.BEARISH.value
            if fvg_signal.get('bias') != expected_bias:
                continue

            # Check position relative to BOS level
            if is_bullish_trend:
                # For bullish trend: look for FVG below BOS level
                if fvg_signal.get('top', float('inf')) <= bos_level:
                    nearest_fvg = fvg_signal
                    break
            else:
                # For bearish trend: look for FVG above BOS level
                if fvg_signal.get('bottom', float('-inf')) >= bos_level:
                    nearest_fvg = fvg_signal
                    break

        if not nearest_fvg:
            event = "No valid FVG found."
            recent_smc_events.append(event)
            return None, recent_smc_events

        event = f"Found nearest {trend} FVG."
        recent_smc_events.append(event)
        return nearest_fvg, recent_smc_events

    async def confirm_fvg_mitigation(self, ohlc_15m: pd.DataFrame, trend: str, fvg_signal: dict):
        """
        Confirm if the current candle mitigates an FVG by 90% or more.

        Args:
            ohlc_15m (pd.DataFrame): OHLC data with at least one row.
            trend (str): Trend direction ('BULLISH' or 'BEARISH').
            fvg_signal (dict): FVG signal with 'top', 'bottom', 'time'.

        Returns:
            tuple[bool, list[str]]: (fvg_mitigated, recent_smc_events)
        """
        recent_smc_events = []
        fvg_top = fvg_signal['top']
        fvg_bottom = fvg_signal['bottom']
        fvg_range = fvg_top - fvg_bottom

        # Check current candle
        current_candle = ohlc_15m.iloc[-1]
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        current_time = ohlc_15m.index[-1]
        fvg_mitigated = False

        if candle_low <= fvg_top and candle_high >= fvg_bottom:
            recent_smc_events.append(f"Price touched {trend} FVG at {current_time}")

            if trend == Trend.BULLISH.name:
                # Update last_mitigated_fvg_level to min of candle_low and current level (if exists)
                if self.last_mitigated_fvg_level is None:
                    self.last_mitigated_fvg_level = candle_low  # or candle_high for bearish
                else:
                    self.last_mitigated_fvg_level = min(self.last_mitigated_fvg_level, candle_low)
                # Calculate mitigation percentage
                mitigation_depth = fvg_top - self.last_mitigated_fvg_level
                mitigation_percent = (mitigation_depth / fvg_range) * 100 if fvg_range > 0 else 0
                if mitigation_percent >= 90:
                    fvg_mitigated = True
                    recent_smc_events.append(f"Price mitigated {trend} FVG by {mitigation_percent:.2f}% (>= 90%)")
                    self.last_mitigated_fvg_level = None  # Reset on full mitigation
                else:
                    recent_smc_events.append(f"Price partially mitigated {trend} FVG by {mitigation_percent:.2f}%")
            else:  # BEARISH
                # Update last_mitigated_fvg_level to max of candle_high and current level (if exists)
                if self.last_mitigated_fvg_level is None:
                    self.last_mitigated_fvg_level = candle_high
                else:
                    self.last_mitigated_fvg_level = max(candle_high, self.last_mitigated_fvg_level)

                # Calculate mitigation percentage
                mitigation_depth = self.last_mitigated_fvg_level - fvg_bottom
                mitigation_percent = (mitigation_depth / fvg_range) * 100 if fvg_range > 0 else 0
                if mitigation_percent >= 90:
                    fvg_mitigated = True
                    recent_smc_events.append(f"Price mitigated {trend} FVG by {mitigation_percent:.2f}% (>= 90%)")
                    self.last_mitigated_fvg_level = None  # Reset on full mitigation
                else:
                    recent_smc_events.append(f"Price partially mitigated {trend} FVG by {mitigation_percent:.2f}%")
        else:
            recent_smc_events.append(f"Price did not touch {trend} FVG at {current_candle.name}")

        return fvg_mitigated, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC data for continuation signals based on Liquidity Pool sweep
        """
        cache_key = (symbol, self.main_tf)
        current_time = max(ohlc_15m.index.max(), ohlc_1h.index.max())
        recent_smc_events = []

        # Step 1: Identify market trend
        trend, trend_event, fig = await self.confirm_trend(ohlc_1h, symbol, cache_key, current_time)
        recent_smc_events.extend(trend_event)

        # Step 2: Detect BOS & ChoCH signals
        bos_events = await self.detect_bos_choch(ohlc_1h, trend, symbol)
        recent_smc_events.extend(bos_events)
        signal_summary = SignalSummary(
            direction=Trend.NEUTRAL.name,
            trend=trend,
            recent_smc_events=recent_smc_events,
            recent_bos_signal=self.recent_bos_signal,
            fig=self.bos_choch_fig
        )
        if self.recent_bos_signal is None or trend == Trend.NEUTRAL.name:
            return signal_summary

        # Update last BOS time
        self.last_bos_time = self.recent_bos_signal.get('time')

        # Step 3: Find nearest FVG
        nearest_fvg, step_3_events = await self.get_nearest_fvg(ohlc_15m, trend)
        recent_smc_events.extend(step_3_events)
        signal_summary.fig = self.fvg_fig
        if not nearest_fvg:
            signal_summary.recent_smc_events = recent_smc_events
            return signal_summary

        # Step 4: Confirm Pull Back - FVG mitigation
        fvg_mitigated, step_4_events = await self.confirm_fvg_mitigation(ohlc_15m, trend, nearest_fvg)
        recent_smc_events.extend(step_4_events)

        if not fvg_mitigated:
            signal_summary.recent_smc_events = recent_smc_events
            return signal_summary

        # # Check if this FVG was already mitigated and if a new BOS has occurred
        # if (self.last_mitigated_fvg and
        #         self.last_mitigated_fvg.get('time') == nearest_fvg.get('time') and
        #         (self.last_bos_time is None or
        #          self.last_bos_time <= self.last_mitigated_fvg.get('time'))):
        #     event = f"Skipping signal: FVG at {nearest_fvg['time']} already mitigated, no new BOS."
        #     recent_smc_events.append(event)
        #     signal_summary.recent_smc_events = recent_smc_events
        #     return signal_summary
        #
        # # Update last mitigated FVG
        # self.last_mitigated_fvg = nearest_fvg

        # Step 5: Generate final signal
        signal = "LONG" if trend == Trend.BULLISH.name else "SHORT"
        event = f"Generated {signal} signal for {symbol} based on FVG mitigation."
        recent_smc_events.append(event)
        signal_summary.direction = signal
        signal_summary.recent_smc_events = recent_smc_events
        return signal_summary
