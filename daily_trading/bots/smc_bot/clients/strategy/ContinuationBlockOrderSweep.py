import json
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.clients.CandlePatternDetector import CandlePatternDetector
from daily_trading.bots.smc_bot.clients.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.clients.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.clients.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.clients.SMCBase import Trend
from daily_trading.bots.smc_bot.clients.strategy.StrategyBase import StrategyBase
from daily_trading.settings import logger


class ContinuationBlockOrderSweepStrategy(StrategyBase):
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

    def __init__(self, main_tf: str):
        """
        Initialize strategy with detectors for trend, OB, and candlestick patterns.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for trend and OB detection.
            telegram_token (str): Telegram bot token for sending signals.
            channel_id (str): Telegram channel ID for message delivery.
        """
        super().__init__(main_tf)
        self.ob_tap_cache = {}  # Cache: {symbol: tap_ob_timestamp}
        self.ob_fig = None

        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),
            'fvg_1h': FvgDetector(main_tf),
            'candle_1h': CandlePatternDetector(self.main_tf),
            'ob_1h': OrderBlocksDetector(main_tf)
        }
        self.nearest_ob = None
        logger.info(f"ContinuationBlockOrderSweepStrategy initialized with detectors")

    async def confirm_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime):
        """Step 1: Identify Trend on 1H (last 24 candles)."""
        recent_smc_events = []
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        fig = swing_1h["fig"]
        trend = self.detectors['swing_1h'].detect_trend(swing_1h)

        if trend != Trend.NEUTRAL.name:
            trend_event = f"{trend} trend confirmed."
            if await self.manage_cache_event(cache_key, trend_event, current_time):
                recent_smc_events.append(trend_event)
                await self.send_telegram_message(f"{symbol} | Step 1: {trend_event}", fig)
        else:
            logger.info(f"Trend is {trend}.")
            if cache_key in self.event_cache:
                self.event_cache[cache_key]['events'] = []
            await self.send_telegram_message(f"{symbol} | Step 1: Neutral trend detected.", fig)

        return trend, recent_smc_events, fig

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

        new_ob_event = f"Detected {len(ob_signals)} unmitigated Order Blocks"
        if await self.manage_cache_event(cache_key, new_ob_event, current_time):
            recent_smc_events.append(new_ob_event)
            logger.info(f"Step 2: {new_ob_event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {new_ob_event}.", fig)
        ob_fig = ob_result["fig"]
        logger.info(f"Found OB: {ob_signals}")
        logger.info(recent_smc_events)
        self.ob_fig = ob_fig
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
            new_ob_event = f"No valid {'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} OB found (formed at least 5 candles ago)"
            if await self.manage_cache_event(cache_key, new_ob_event, current_time):
                recent_smc_events.append(new_ob_event)
                logger.info(f"{symbol} | Step 3: {new_ob_event}.")
                await self.send_telegram_message(f"{symbol} | Step 3: {new_ob_event}.", self.ob_fig)
            return None, recent_smc_events

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
            new_ob_event = f"Current candle did not tap {'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} OB"
            if await self.manage_cache_event(cache_key, new_ob_event, current_time):
                recent_smc_events.append(new_ob_event)
                logger.info(f"{symbol} | Step 3: {new_ob_event}")
                await self.send_telegram_message(f"Step 3: {new_ob_event} for {symbol}.", self.ob_fig)
            return None, recent_smc_events

        self.nearest_ob['tap_timestamp'] = tap_timestamp
        # Update cache
        self.ob_tap_cache.update({symbol: tap_timestamp})

        new_ob_event = f"Current candle tapped {'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} OB at {tap_timestamp}"
        if await self.manage_cache_event(cache_key, new_ob_event, current_time):
            recent_smc_events.append(new_ob_event)
            logger.info(f"Step 3: {new_ob_event} for {symbol}")
            await self.send_telegram_message(f"Step 3: {new_ob_event} for {symbol}.", self.ob_fig)

        return recent_smc_events

    async def confirm_candle_pattern(self, ohlc_1h: pd.DataFrame, trend: str, nearest_ob: Dict,
                                     symbol: str, cache_key: Tuple[str, str],
                                     current_time: datetime):
        """Step 4: Generate trade signal after OB mitigation and rejection confirmed by 15m candlestick patterns."""
        trade_signal = None
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
            new_signal_event = f"No {'Bullish' if trend == Trend.BULLISH.name else 'Bearish'}" \
                               f" rejection pattern after OB tap"
            if await self.manage_cache_event(cache_key, new_signal_event, current_time):
                recent_smc_events.append(new_signal_event)
                logger.info(f"Step 4: {new_signal_event} for {symbol}")
                await self.send_telegram_message(f"Step 4: {new_signal_event} for {symbol}.")
            return trade_signal, recent_smc_events

        if trend == Trend.BULLISH.name:
            direction = "LONG"
        else:
            direction = "SHORT"

        ob_tap_time = self.ob_tap_cache.get(symbol)
        trade_signal = {
            'type': 'TradeSignal',
            'direction': direction,
            'time': current_time,
            'rejection_pattern': pattern_name,
            'ob_tap_time': ob_tap_time,
            'fig': None
        }

        new_signal_event = f"{'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} signal generated at {current_time} with {pattern_name}"
        if await self.manage_cache_event(cache_key, new_signal_event, current_time):
            recent_smc_events.append(new_signal_event)
            message = f"{symbol} | Step 4: {direction} signal.. Liquidity swept at {ob_tap_time}," \
                      f" Rejection Candle: {pattern_name} - {current_time}."
            logger.info(message)
            await self.send_telegram_message(message, self.ob_fig)

        return trade_signal, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame) -> Dict:
        """
        Analyze OHLC data for continuation signals based on block order sweeps.
        """
        logger.info(f"Starting Block Order Sweep analysis for {symbol}")
        current_time = ohlc_1h.index.max()
        cache_key = (symbol, self.main_tf)
        recent_smc_events = []

        # Step 1: Confirm trend alignment
        trend, step1_events, high_low_fig = await self.confirm_trend(ohlc_1h, symbol, cache_key, current_time)
        recent_smc_events.extend(step1_events)

        trade_signal = {
            'direction': Trend.NEUTRAL.name,
            'trend': trend,
            'trend_signal': None,
            'recent_smc_events': recent_smc_events,
            'fig': None,
            'nearest_ob': self.nearest_ob
        }
        if trend == Trend.NEUTRAL.name:
            logger.info(f"{symbol} | Step 1: Neutral trend detected, stopping analysis.")
            trade_signal["fig"] = high_low_fig
            return trade_signal

        # Step 2: Detect OBs
        relevant_obs, step2_events = await self.detect_ob(ohlc_1h, symbol, trend, cache_key)
        recent_smc_events.extend(step2_events)
        if not relevant_obs:
            logger.info(f"{symbol} | Step 2: No unmitigated OBs detected, stopping analysis.")
            trade_signal["fig"] = self.ob_fig
            return trade_signal

        # Step 3: Confirm price taps into OB (t must happen at least 3H later after the OB formed)
        if symbol in self.ob_tap_cache and self.ob_tap_cache[symbol] is not None:
            last_ob_tap = self.ob_tap_cache[symbol]
            if int((current_time - last_ob_tap).total_seconds() / 3600) <= 3:
                logger.info(f"Skipping Step 3. OB Tab has happened less than 1h ago - {last_ob_tap}")
            else:
                self.ob_tap_cache.update({symbol: None})
                logger.info(
                    f"Step 3: {symbol} | Resetting OB Tap. "
                    f"Previous OB tap was not confirmed by a rejection pattern in Step 4.")
                return trade_signal

        else:
            self.ob_tap_cache.update({symbol: None})
            step3_events = await self.confirm_ob_tap(ohlc_1h, trend, symbol, relevant_obs, cache_key, current_time)
            recent_smc_events.extend(step3_events)
            # If OB tap is still None, exit and start again.
            if not self.ob_tap_cache[symbol]:
                logger.info(f"Step 3: {symbol} | No OB tap confirmed, waiting for the next candle.")
                return trade_signal

        # Step 4: Generate trade signal after OB mitigation and rejection
        trade_signal, step4_events = await self.confirm_candle_pattern(ohlc_1h, trend, self.nearest_ob, symbol,
                                                                       cache_key, current_time)
        recent_smc_events.extend(step4_events)

        if not trade_signal:
            logger.info(f"No trading signal. Current SMC events: {json.dumps(recent_smc_events, indent=1)}")
            return {
                'direction': Trend.NEUTRAL.name,
                'trend': trend,
                'trend_signal': trade_signal,
                'recent_smc_events': recent_smc_events,
                'fig': self.ob_fig,
                'nearest_ob': self.nearest_ob
            }

        # Reset OB tapping time
        self.ob_tap_cache[symbol] = None

        message = f"""
                    📊 SMC Final Signal for {symbol}
                    Time: {trade_signal['time']}
                    Direction: {trade_signal['direction']}
                    Trend: {trend}
                    Candle Pattern: {trade_signal['rejection_pattern']}
                    Tapped at: {trade_signal['ob_tap_time']}
                    """
        logger.info(message)
        fvg_result = self.detectors['fvg_1h'].calculate_fvg(ohlc_1h, plot=True)
        await self.send_telegram_message(message, self.ob_fig)

        return {
            'direction': {trade_signal['direction']},
            'trend': trend,
            'trend_signal': trade_signal,
            'recent_smc_events': recent_smc_events,
            'fig': self.ob_fig,
            'nearest_ob': self.nearest_ob
        }
