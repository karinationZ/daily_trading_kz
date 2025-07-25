from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase
from daily_trading.settings import logger


class BosOrderBlockTapStrategy(StrategyBase):
    """
    Analyzes OHLC data for continuation trade signals based on BOS and 15m OB.
    Designed for scalping in volatile crypto markets with a 4-step process:
    1. Determine Market Trend on 1H (last 24 candles).
    2. Identify the most recent BOS level in the direction of the trend.
    3. Find the nearest unmitigated 15m OB below (bullish) or above (bearish) current price, with OB top/bottom relative to BOS.
    4. Confirm price taps OB, generate Buy (bullish) or Sell (bearish) signal.
    Sends Telegram messages with plots (OHLC.tail(100)) for each step.
    Executed every 15 minutes.
    """

    def __init__(self, main_tf: str = '1h', debug: bool = False, telegram_chat_id=None):
        """
        Initialize strategy with detectors for trend, BOS, and OB.

        Args:
            main_tf (str): Main timeframe for trend and BOS detection (default: '1h').
            debug (bool): Enable debug mode.
            telegram_chat_id (str): Telegram channel ID for message delivery.
        """
        super().__init__(main_tf, debug, telegram_chat_id=telegram_chat_id)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'bos_1h': BosChochDetector(main_tf),  # Detects BOS/CHOCH on 1H
            'ob_15m': OrderBlocksDetector('15m')  # Detects OBs on 15m
        }
        self.ob_fig = None
        self.ob_tap_cache = {}  # Cache: {symbol: tap_ob_timestamp}
        logger.info("BosOrderBlockStrategy initialized with detectors")

    async def find_nearest_ob(self, ohlc_15m: pd.DataFrame, trend: str, symbol: str, current_price: float):
        """Step 3: Find the nearest unmitigated 15m OB in the trend direction relative to BOS."""
        recent_smc_events = []
        ob_result = self.detectors['ob_15m'].calculate_order_blocks(ohlc_15m, plot=True)
        self.ob_fig = self.detectors['ob_15m'].merge_bos_ob_plots(ohlc_15m=ohlc_15m,
                                                                  bos_signals=self.bos_signals,
                                                                  ob_signals=ob_result['order_blocks'])
        ob_signals = [ob for ob in ob_result['order_blocks'] if not ob['breaker']]

        if not ob_signals:
            event = "No unmitigated 15m OBs detected."
            recent_smc_events.append(event)
            logger.info(f"Step 3: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 3: {event}.", self.ob_fig)
            return None, recent_smc_events

        expected_bias = Trend.BULLISH.name if trend == Trend.BULLISH.name else Trend.BEARISH.name
        relevant_obs = [ob for ob in ob_signals if ob['type'].upper() == expected_bias]
        if not relevant_obs:
            event = f"No {expected_bias} OBs detected on 15m."
            recent_smc_events.append(event)
            logger.info(f"Step 3: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 3: {event}.", self.ob_fig)
            return None, recent_smc_events

        bos_level = self.recent_bos_signal["level"]
        nearest_ob = None
        for ob in sorted(relevant_obs, key=lambda x: x['time'], reverse=True):
            if trend == Trend.BULLISH.name:
                if ob['top'] > bos_level and ob['top'] < current_price:
                    nearest_ob = ob
                    break
            else:  # BEARISH
                if ob['bottom'] < bos_level and ob['bottom'] > current_price:
                    nearest_ob = ob
                    break

        if not nearest_ob:
            event = f"No valid {trend} OB found relative to BOS level."
            recent_smc_events.append(event)
            logger.info(f"Step 3: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 3: {event}.", self.ob_fig)
            return None, recent_smc_events

        event = f"Found nearest {trend} 15m OB at {nearest_ob['time'].strftime('%Y-%m-%d %H:%M')}."
        recent_smc_events.append(event)
        logger.info(f"Step 3: {event} for {symbol}")
        await self.send_telegram_message(f"{symbol} | Step 3: {event}.", self.ob_fig)
        return nearest_ob, recent_smc_events

    async def confirm_ob_tap(self, ohlc_15m: pd.DataFrame, trend: str, symbol: str, nearest_ob: Dict,
                             cache_key: Tuple[str, str], current_time: datetime):
        """Step 4: Confirm price taps the nearest 15m OB and generate trade signal."""
        recent_smc_events = []
        if not nearest_ob:
            event = "No valid OB to confirm tap."
            recent_smc_events.append(event)
            logger.info(f"Step 4: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 4: {event}.", self.ob_fig)
            return recent_smc_events

        current_candle = ohlc_15m.iloc[-1]
        tap_timestamp = None
        if trend == Trend.BULLISH.name:
            if current_candle['low'] <= nearest_ob['top'] and current_candle['high'] >= nearest_ob['bottom']:
                tap_timestamp = current_time
        else:  # BEARISH
            if current_candle['high'] >= nearest_ob['bottom'] and current_candle['low'] <= nearest_ob['top']:
                tap_timestamp = current_time

        if not tap_timestamp:
            event = f"Current 15m candle did not tap {trend} OB."
            self.ob_tap_cache.update({symbol: None})
            recent_smc_events.append(event)
            logger.info(f"Step 4: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 4: {event}.", self.ob_fig)
            return recent_smc_events

        self.ob_tap_cache.update({symbol: tap_timestamp})
        event = f"Current 15m candle tapped {trend} OB at {tap_timestamp}."
        recent_smc_events.append(event)
        logger.info(f"Step 4: {event} for {symbol}")
        await self.send_telegram_message(f"{symbol} | Step 4: {event}.", self.ob_fig)
        return recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC data for continuation signals based on BOS and 15m OB taps.
        """
        current_time = max(ohlc_1h.index.max(), ohlc_15m.index.max())
        current_price = ohlc_15m['close'].iloc[-1]
        cache_key = (symbol, self.main_tf)
        recent_smc_events = []

        # Step 1: Confirm trend
        trend, step1_events, trend_fig = await self.confirm_trend(self.detectors['swing_1h'], ohlc_1h, symbol,
                                                                  cache_key, current_time)
        recent_smc_events.extend(step1_events)
        signal_summary = SignalSummary(
            direction=Trend.NEUTRAL.name,
            trend=trend,
            recent_smc_events=recent_smc_events,
            fig=trend_fig
        )
        if trend == Trend.NEUTRAL.name:
            logger.info(f"{symbol} | Step 1: Neutral trend detected, stopping analysis.")
            return signal_summary

        # Step 2: Detect BOS
        step2_events = await self.detect_bos_choch(self.detectors['bos_1h'], ohlc_1h, trend, symbol)
        recent_smc_events.extend(step2_events)
        signal_summary.recent_smc_events = recent_smc_events
        signal_summary.fig = self.bos_choch_fig
        if self.recent_bos_signal is None:
            logger.info(f"{symbol} | Step 2: No valid BOS detected, stopping analysis.")
            return signal_summary

        # Step 3: Find nearest 15m OB
        nearest_ob, step3_events = await self.find_nearest_ob(ohlc_15m, trend, symbol, current_price)
        recent_smc_events.extend(step3_events)
        signal_summary.recent_smc_events = recent_smc_events
        signal_summary.fig = self.ob_fig
        if not nearest_ob:
            logger.info(f"{symbol} | Step 3: No valid OB found, stopping analysis.")
            return signal_summary

        # Step 4: Confirm OB tap and generate signal
        step4_events = await self.confirm_ob_tap(ohlc_15m, trend, symbol, nearest_ob, cache_key, current_time)
        recent_smc_events.extend(step4_events)
        signal_summary.recent_smc_events = recent_smc_events
        signal_summary.fig = self.ob_fig

        if self.ob_tap_cache.get(symbol):
            signal_summary.direction = "LONG" if trend == Trend.BULLISH.name else "SHORT"
            event = f"Generated {signal_summary.direction} signal for {symbol} at {current_time}."
            recent_smc_events.append(event)
            logger.info(f"Step 4: {event}")
            await self.send_telegram_message(f"{symbol} | Step 4: {event}.", self.ob_fig)
            self.ob_tap_cache[symbol] = None  # Reset cache after signal

        return signal_summary
