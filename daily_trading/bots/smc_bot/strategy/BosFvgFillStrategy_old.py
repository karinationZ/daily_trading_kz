import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.StrategyBase import GET_READY_ALERT, BIAS, Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase


class BosFvgFillStrategy(StrategyBase):
    """
    Analyzes OHLC mgc using Smart Money Concept (SMC) for continuation trade signals.
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
            'fvg_15m': FvgDetector(main_tf="15m", pivot_period=5, left_bars=5)  # Detects FVGs on 15m
        }
        self.bos_signals = None
        self.recent_bos_signal = None
        self.bos_choch_fig = None
        self.fvg_fig = None
        self.last_bos_time = None  # Track last BOS time
        self.last_mitigated_fvg_level_map = {}  # Initialize as empty dict
        self._current_fvg_id_map = {}  # Initialize as empty dict

    async def get_nearest_fvg(self, ohlc_lower_tf: pd.DataFrame, trend: str):
        """Step 4: Confirm price touches the nearest unmitigated FVG on 15m."""
        recent_smc_events = []

        if trend == Trend.NEUTRAL.name:
            recent_smc_events.append(f"Trend is {trend}. Exiting FVG search step.")
            return None, recent_smc_events

        fvg_result = self.detectors["fvg_15m"].calculate_fvg(ohlc_lower_tf, plot=True)
        self.fvg_fig = self.detectors["fvg_15m"].merge_bos_fvg_plots(ohlc_lower_tf, self.bos_signals, fvg_result["signals"])

        if not fvg_result["signals"]:
            event = "No FVG signals!"
            self.fvg_fig = fvg_result["fig"]
            return None, [event]

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

    async def confirm_fvg_mitigation(self, symbol, ohlc_lower_tf: pd.DataFrame, trend: str, fvg_signal: dict):
        """Confirm if the current candle mitigates an FVG by 60% or more."""
        recent_smc_events = []

        fvg_top = fvg_signal['top']
        fvg_bottom = fvg_signal['bottom']
        fvg_range = fvg_top - fvg_bottom

        # Check current candle
        current_candle = ohlc_lower_tf.iloc[-1]
        candle_high = current_candle['high']
        candle_low = current_candle['low']
        fvg_mitigated = False

        # Reset state for new FVG
        if symbol not in self._current_fvg_id_map or self._current_fvg_id_map[symbol] != id(fvg_signal):
            self.last_mitigated_fvg_level_map[symbol] = None
            self._current_fvg_id_map[symbol] = id(fvg_signal)

        if fvg_bottom <= current_candle['close'] <= fvg_top:
            recent_smc_events.append(f"Current price is in {trend} FVG.")

        if candle_low <= fvg_top and candle_high >= fvg_bottom:
            # Update last_mitigated_fvg_level based on trend
            current_level = self.last_mitigated_fvg_level_map.get(symbol)
            if trend == Trend.BULLISH.name:
                if current_level is None or candle_low < current_level:
                    self.last_mitigated_fvg_level_map[symbol] = max(candle_low, fvg_bottom)
                mitigation_depth = fvg_top - self.last_mitigated_fvg_level_map[symbol]
            else:  # BEARISH
                if current_level is None or candle_high > current_level:
                    self.last_mitigated_fvg_level_map[symbol] = min(candle_high, fvg_top)
                mitigation_depth = self.last_mitigated_fvg_level_map[symbol] - fvg_bottom

            mitigation_percent = int((mitigation_depth / fvg_range) * 100) if fvg_range > 0 else 0
            if mitigation_percent >= 60:
                fvg_mitigated = True
                msg = f"{symbol} - Price mitigated {trend} FVG by {mitigation_percent}% (>= 60%)" if trend == Trend.BULLISH.name else f"Price mitigated {trend} FVG by {mitigation_percent}% (>= 60%)"
                recent_smc_events.append(msg)
                self.last_mitigated_fvg_level_map[symbol] = None
                self._current_fvg_id_map[symbol] = None
            elif mitigation_percent > 0:
                recent_smc_events.append(
                    f"{GET_READY_ALERT} - Price partially mitigated ({mitigation_percent}%) {trend} FVG.")
        else:
            recent_smc_events.append(f"Current price is outside of the {trend} FVG.")

        return fvg_mitigated, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_lower_tf: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC mgc for continuation signals based on Liquidity Pool sweep
        """
        cache_key = (symbol, self.main_tf)
        current_time = max(ohlc_lower_tf.index.max(), ohlc_1h.index.max())
        recent_smc_events = []

        # Step 1: Identify market trend
        trend, trend_event, fig = await self.confirm_trend(self.detectors['swing_1h'], ohlc_1h, symbol, cache_key,
                                                           current_time)
        recent_smc_events.extend(trend_event)

        # Step 2: Detect BOS & ChoCH signals
        bos_events = await self.detect_bos_choch(self.detectors['bos_1h'], ohlc_1h, trend, symbol)
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
        nearest_fvg, step_3_events = await self.get_nearest_fvg(ohlc_lower_tf, trend)
        recent_smc_events.extend(step_3_events)
        signal_summary.fig = self.fvg_fig
        if not nearest_fvg:
            signal_summary.recent_smc_events = recent_smc_events
            return signal_summary

        # Step 4: Confirm Pull Back - FVG mitigation
        fvg_mitigated, step_4_events = await self.confirm_fvg_mitigation(symbol, ohlc_lower_tf, trend, nearest_fvg)
        recent_smc_events.extend(step_4_events)
        if GET_READY_ALERT in step_4_events[-1]:
            signal_summary.direction = GET_READY_ALERT

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
        signal = BIAS.LONG.value if trend == Trend.BULLISH.name else BIAS.SHORT.value
        event = f"Generated {signal} signal for {symbol} based on FVG mitigation."
        recent_smc_events.append(event)
        signal_summary.direction = signal
        signal_summary.recent_smc_events = recent_smc_events
        return signal_summary
