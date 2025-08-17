import pandas as pd

from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.LevelsDetector import LevelsDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import GET_READY_ALERT, Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase


class SupportResistanceStrategy(StrategyBase):
    """
    Analyzes OHLC data using Smart Money Concept (SMC) with Support and Resistance levels.
    Designed for scalping in volatile crypto markets with a 5-step process:
    1. Identify Trend on 1H (last 24 candles).
    2. Detect key Support/Resistance levels on 1H.
    3. Confirm price approaches a key level on 15m.
    4. Validate level strength and price interaction.
    5. Generate LONG/SHORT signal.
    """

    def __init__(self, main_tf: str, debug: bool, telegram_chat_id):
        super().__init__(main_tf, debug, telegram_chat_id=telegram_chat_id)
        self.strength = 4
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'levels_detector': LevelsDetector(main_tf=main_tf, pivot_length=3, strength=self.strength)
        }
        self.sr_signals = None
        self.recent_sr_signal = None
        self.sr_fig = None
        self.last_level_time = None
        self.last_tested_level_map = {}

    async def get_nearest_level(self, ohlc_15m: pd.DataFrame, trend: str):
        """Step 3: Find the nearest untested Support/Resistance level on 1H."""
        recent_smc_events = []

        if trend == Trend.NEUTRAL.name:
            recent_smc_events.append(f"Trend is {trend}. Exiting level search step.")
            return None, recent_smc_events

        sr_result = self.detectors["levels_detector"].calculate_support_resistance_levels(ohlc_15m, plot=True)
        self.sr_fig = sr_result["fig"]

        support_levels = sr_result["support_levels"]
        resistance_levels = sr_result["resistance_levels"]
        if not (support_levels or resistance_levels):
            event = "No Support/Resistance levels found!"
            recent_smc_events.append(event)
            return None, recent_smc_events

        current_price = ohlc_15m.iloc[-1]['close']
        nearest_level = None
        min_distance = float('inf')
        is_bullish_trend = trend == Trend.BULLISH.name

        for level in support_levels + resistance_levels:
            if level['strength'] < self.strength or level.get('tested', False):
                continue
            distance = abs(level['price'] - current_price)
            level_type = level['type']
            if is_bullish_trend and level_type == 'Support' and level['price'] < current_price:
                if distance < min_distance:
                    nearest_level = level
                    min_distance = distance
            elif not is_bullish_trend and level_type == 'Resistance' and level['price'] > current_price:
                if distance < min_distance:
                    nearest_level = level
                    min_distance = distance

        if not nearest_level:
            event = "No valid Support/Resistance level found."
            recent_smc_events.append(event)
            return None, recent_smc_events

        event = f"Found nearest {trend} {nearest_level['type']} level at {nearest_level['price']}."
        recent_smc_events.append(event)
        return nearest_level, recent_smc_events

    async def confirm_level_interaction(self, symbol, ohlc_15m: pd.DataFrame, level_signal: dict):
        """Step 4: Confirm price interacts with the level (within 0.5% of level)."""
        recent_smc_events = []
        level_price = level_signal['price']
        level_type = level_signal['type']
        current_candle = ohlc_15m.iloc[-1]
        current_price = current_candle['close']
        threshold = self.detectors["levels_detector"].calculate_atr(ohlc_15m, period=14).iloc[-1] / 2

        if symbol not in self.last_tested_level_map or self.last_tested_level_map[symbol] != id(level_signal):
            self.last_tested_level_map[symbol] = None

        if abs(current_price - level_price) <= threshold:
            event = f"Price near {level_type} level at {level_price} (within {threshold:.4f})."
            recent_smc_events.append(event)
            level_signal['tested'] = True
            self.last_tested_level_map[symbol] = id(level_signal)
            return True, recent_smc_events
        else:
            event = f"Price not within {threshold:.4f} of {level_type} level at {level_price}."
            recent_smc_events.append(event)
            return False, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_htf: pd.DataFrame, ohlc_for_trend_detection: pd.DataFrame,
                          ohlc_ltf: pd.DataFrame) -> SignalSummary:
        """Analyze OHLC data for continuation signals based on Support/Resistance levels."""
        cache_key = (symbol, self.main_tf)
        current_time = max(ohlc_ltf.index.max(), ohlc_htf.index.max())
        recent_smc_events = []

        # Step 1: Identify market trend
        trend, trend_event, fig = await self.confirm_trend(self.detectors['swing_1h'], ohlc_for_trend_detection, symbol,
                                                           cache_key,
                                                           current_time)
        recent_smc_events.extend(trend_event)

        # Step 2: Detect Support/Resistance levels
        sr_result = self.detectors["levels_detector"].calculate_support_resistance_levels(ohlc_htf, plot=True)
        self.sr_signals = sr_result
        self.sr_fig = sr_result["fig"]
        self.recent_sr_signal = (sr_result["support_levels"] + sr_result["resistance_levels"])[-1] if (
                sr_result["support_levels"] or sr_result["resistance_levels"]) else None
        recent_smc_events.append(
            f"Detected {len(sr_result['support_levels'])} support and {len(sr_result['resistance_levels'])} resistance levels.")

        signal_summary = SignalSummary(
            direction=Trend.NEUTRAL.name,
            trend=trend,
            recent_smc_events=recent_smc_events,
            recent_bos_signal=self.recent_sr_signal,
            fig=self.sr_fig
        )
        if not self.recent_sr_signal:
            return signal_summary

        self.last_level_time = self.recent_sr_signal.get('start_time')

        # Step 3: Find nearest level
        nearest_level, step_3_events = await self.get_nearest_level(ohlc_ltf, trend)
        recent_smc_events.extend(step_3_events)
        signal_summary.fig = self.sr_fig
        if not nearest_level:
            signal_summary.recent_smc_events = recent_smc_events
            return signal_summary

        # Step 4: Confirm price interaction with level
        level_interacted, step_4_events = await self.confirm_level_interaction(symbol, ohlc_ltf, nearest_level)
        recent_smc_events.extend(step_4_events)

        if not level_interacted:
            signal_summary.recent_smc_events = recent_smc_events
            return signal_summary

        # Step 5: Generate final signal
        event = f"Generated signal for {symbol} based on {nearest_level['type']} level interaction."
        recent_smc_events.append(event)
        signal_summary.direction = GET_READY_ALERT
        signal_summary.fig = self.detectors["levels_detector"].plot_htf_levels_on_ltf(ohlc_ltf, self.sr_signals)
        signal_summary.recent_smc_events = recent_smc_events
        return signal_summary
