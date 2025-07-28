import json
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.detectors.CandlePatternDetector import CandlePatternDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import Signal, Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase
from daily_trading.settings import logger


class OrderBlockSweepStrategy(StrategyBase):
    """
    Analyzes OHLC data for continuation trade signals based on block order sweeps.
    Designed for scalping in volatile crypto markets with a 4-step process:
    1. Identify Uptrend or Downtrend on 1H (last 24 candles).
    2. Detect Block Orders (OB) on 1H.
    3. For bullish trend, wait for price to pull back and tap into bullish OB below current price;
       for bearish trend, wait for price to tap into bearish OB above current price.
    4. Confirm OB mitigation with 15m candlestick rejection patterns, generate Buy (bullish) or Sell (bearish) signal
       with entry, SL, and TP (1:2 RR).
    Sends Telegram messages with plots (OHLC.tail(100)) for each step, including FVGs in final plot for visibility.
    """

    def __init__(self, main_tf: str, debug: bool, telegram_chat_id=None, ):
        """
        Initialize strategy with detectors for trend, OB, and candlestick patterns.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for trend and OB detection.
            telegram_chat_id (str): Telegram channel ID for message delivery.
        """
        super().__init__(main_tf, debug, telegram_chat_id=telegram_chat_id)
        self.ob_tap_cache = {}  # Cache: {symbol: tap_ob_timestamp}
        self.ob_fig = None
        self.debug = debug

        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),
            'fvg_1h': FvgDetector(main_tf),
            'candle_1h': CandlePatternDetector(self.main_tf),
            'ob_1h': OrderBlocksDetector(main_tf)
        }
        self.nearest_ob = None
        logger.info(f"ContinuationBlockOrderSweepStrategy initialized with detectors")

    async def detect_ob(self, ohlc_1h: pd.DataFrame, symbol: str, trend: str, cache_key: Tuple[str, str]):
        """Step 2: Detect Block Orders on 1H."""
        current_time = ohlc_1h.index.max()
        ob_result = self.detectors['ob_1h'].calculate_order_blocks(ohlc_1h, plot=True)
        recent_smc_events = []
        fig = ob_result["fig"]

        # Normalize OB type to uppercase
        for ob in ob_result['order_blocks']:
            ob['type'] = ob['type'].upper()

        ob_signals = [ob for ob in ob_result['order_blocks'] if not ob['breaker']]

        new_ob_event = f"Detected {len(ob_signals)} unmitigated Order Blocks."
        if await self.manage_cache_event(cache_key, new_ob_event, current_time):
            recent_smc_events.append(new_ob_event)
            logger.info(f"Step 2: {new_ob_event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {new_ob_event}.", fig)
        self.ob_fig = fig
        logger.info(f"Found OB: {ob_signals}")
        logger.info(recent_smc_events)
        expected_bias = Trend.BULLISH.name if trend == Trend.BULLISH.name else Trend.BEARISH.name
        relevant_obs = [ob for ob in ob_signals if ob['type'] == expected_bias]
        self.nearest_ob = max(relevant_obs, key=lambda x: x['time'], default=None)
        return relevant_obs, recent_smc_events

    async def confirm_ob_tap(self, ohlc_1h: pd.DataFrame, trend: str, symbol: str,
                             relevant_obs: List[Dict], cache_key: Tuple[str, str],
                             current_time: datetime):
        """Step 3: Confirm if the current candle taps into the most recent OB, formed at least 5 1H candles ago."""
        recent_smc_events = []

        if not relevant_obs or (current_time - self.nearest_ob["time"]).total_seconds() / 3600 < 5:
            new_ob_event = f"No valid {trend} OB found (formed at least 5 candles ago)."
            if await self.manage_cache_event(cache_key, new_ob_event, current_time):
                recent_smc_events.append(new_ob_event)
                logger.info(f"{symbol} | Step 3: {new_ob_event}.")
                await self.send_telegram_message(f"{symbol} | Step 3: {new_ob_event}.", self.ob_fig)
            return recent_smc_events

        # Check if the current candle taps the OB
        tap_timestamp = None
        if trend == Trend.BULLISH.name:
            if (ohlc_1h['low'].iloc[-1] <= self.nearest_ob['top'] and
                    ohlc_1h['high'].iloc[-1] >= self.nearest_ob['bottom']):
                tap_timestamp = current_time
        else:  # BEARISH
            if (ohlc_1h['high'].iloc[-1] >= self.nearest_ob['bottom'] and
                    ohlc_1h['low'].iloc[-1] <= self.nearest_ob['top']):
                tap_timestamp = current_time

        if not tap_timestamp:
            new_ob_event = f"Current candle did not tap {trend} OB."
            self.ob_tap_cache.update({symbol: None})
            if await self.manage_cache_event(cache_key, new_ob_event, current_time):
                recent_smc_events.append(new_ob_event)
                logger.info(f"{symbol} | Step 3: {new_ob_event}")
                await self.send_telegram_message(f"Step 3: {new_ob_event} for {symbol}.", self.ob_fig)
            return recent_smc_events

        self.nearest_ob['tap_timestamp'] = tap_timestamp
        # Update cache
        self.ob_tap_cache.update({symbol: tap_timestamp})

        new_ob_event = f"Current candle tapped {trend} OB at {tap_timestamp}."
        if await self.manage_cache_event(cache_key, new_ob_event, current_time):
            recent_smc_events.append(new_ob_event)
            logger.info(f"Step 3: {new_ob_event} for {symbol}")
            await self.send_telegram_message(f"Step 3: {new_ob_event} for {symbol}.", self.ob_fig)

        return recent_smc_events

    async def confirm_candle_pattern(self, ohlc_1h: pd.DataFrame, trend: str,
                                     symbol: str, cache_key: Tuple[str, str],
                                     current_time: datetime):
        """Step 4: Generate trade signal after OB mitigation and rejection confirmed by 15m candlestick patterns."""
        recent_smc_events = []

        pattern_type, pattern_name = self.detectors['candle_1h'].detect_single_candle_pattern(ohlc_1h.tail(1))
        is_bullish = pattern_type == Trend.BULLISH
        is_bearish = pattern_type == Trend.BEARISH

        rejection_confirmed = False
        if trend == Trend.BULLISH.name and is_bullish:
            rejection_confirmed = True
        elif trend == Trend.BEARISH.name and is_bearish:
            rejection_confirmed = True

        if not rejection_confirmed:
            new_signal_event = f"No {trend} rejection pattern after OB tap."
            if await self.manage_cache_event(cache_key, new_signal_event, current_time):
                recent_smc_events.append(new_signal_event)
                logger.info(f"Step 4: {new_signal_event} for {symbol}")
                await self.send_telegram_message(f"Step 4: {new_signal_event} for {symbol}.", self.ob_fig)
            return recent_smc_events

        new_signal_event = f"{trend} signal generated at {current_time} with {pattern_name}."
        if await self.manage_cache_event(cache_key, new_signal_event, current_time):
            recent_smc_events.append(new_signal_event)
            message = f"{symbol} | Step 4: Rejection Candle: {pattern_name} - {current_time}."
            logger.info(message)
            await self.send_telegram_message(message, self.ob_fig)

        return rejection_confirmed, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC data for continuation signals based on block order sweeps.
        """
        current_time = ohlc_1h.index.max()
        logger.info(f"Starting Block Order Sweep analysis for {symbol} - {current_time}")

        cache_key = (symbol, self.main_tf)
        recent_smc_events = []

        # Step 1: Confirm trend alignment
        trend, step1_events, high_low_fig = await self.confirm_trend(self.detectors['swing_1h'], ohlc_1h, symbol,
                                                                     cache_key, current_time)
        recent_smc_events.extend(step1_events)
        signal_summary = SignalSummary(direction=Trend.NEUTRAL.name,
                                       trend=trend,
                                       recent_smc_events=recent_smc_events,
                                       nearest_ob=self.nearest_ob)
        # if trend == Trend.NEUTRAL.name:
        #     logger.info(f"{symbol} | Step 1: Neutral trend detected, stopping analysis.")
        #     signal_summary.fig = high_low_fig
        #     return signal_summary

        # Step 2: Detect OBs
        relevant_obs, step2_events = await self.detect_ob(ohlc_1h, symbol, trend, cache_key)
        recent_smc_events.extend(step2_events)
        signal_summary.recent_smc_events = recent_smc_events
        signal_summary.fig = self.ob_fig
        if not relevant_obs:
            logger.info(f"{symbol} | Step 2: No unmitigated OBs detected, stopping analysis.")
            return signal_summary

        # Step 3: Confirm price taps into OB
        if symbol not in self.ob_tap_cache or self.ob_tap_cache[symbol] is None:
            step3_events = await self.confirm_ob_tap(ohlc_1h, trend, symbol, relevant_obs, cache_key, current_time)
            recent_smc_events.extend(step3_events)
            signal_summary.recent_smc_events = recent_smc_events

            if self.ob_tap_cache[symbol] is None:
                logger.info("Current candle did not tap OB.")
                return signal_summary
            else:
                logger.info("Moving on to STEP 4.")
        else:
            # Validate OB tap TTL — must be within 3H
            last_ob_tap = self.ob_tap_cache[symbol]
            if int((current_time - last_ob_tap).total_seconds() / 3600) > 3:
                logger.info("OB Tap expired after 3H without confirmation. Resetting.")
                self.ob_tap_cache.update({symbol: None})
                recent_smc_events.extend("OB Tap Expired – No Rejection Signal within 3H after OB Tap.")
                return signal_summary
            else:
                logger.info("Moving on to Step 4 - candle pattern confirmation.")

        # Step 4: Confirm candle pattern
        rejection_confirmed, step4_events = await self.confirm_candle_pattern(ohlc_1h, trend, symbol,
                                                                              cache_key, current_time)
        recent_smc_events.extend(step4_events)
        signal_summary.recent_smc_events = recent_smc_events
        if rejection_confirmed:
            if trend == Trend.BULLISH.name:
                signal_summary.direction = Signal.LONG.value
            else:
                signal_summary.direction = Signal.SHORT.value
        else:
            logger.info(f"No trading signal. Current SMC events: {json.dumps(recent_smc_events, indent=1)}")
            return signal_summary

        # Step 5:  Generate trade
        # Reset OB tapping time
        ob_tap_time = self.ob_tap_cache[symbol]
        self.ob_tap_cache[symbol] = None
        message = f"""
                    📊 SMC Final Signal for {symbol}
                    Time: {current_time}
                    Direction: {signal_summary.direction}
                    Trend: {trend}
                    SMC Last Event: {signal_summary.recent_smc_events[-1]}
                    Tapped at: {ob_tap_time}
                    """
        logger.info(message)
        fvg_result = self.detectors['fvg_1h'].calculate_fvg(ohlc_1h, plot=True)
        await self.send_telegram_message(message, self.ob_fig)
        return signal_summary
