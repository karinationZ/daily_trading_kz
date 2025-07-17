from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.clients.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.clients.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.clients.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.clients.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.clients.SMCBase import BEARISH, BULLISH, Trend
from daily_trading.bots.smc_bot.clients.strategy.StrategyBase import StrategyBase
from daily_trading.settings import logger


class ContinuationInducementStrategy(StrategyBase):
    """
    Analyzes OHLC data using Smart Money Concept (SMC) for continuation trade signals.
    Designed for scalping in volatile crypto markets with a 6-step process:
    1. Identify Trend 1H (last 24 candles)
    2. Detect Break of Structure (BOS) on 1H; reset on CHOCH.
    3. Detect liquidity pool on 15m, post-BOS, between HH/HL (uptrend) or LH/LL (downtrend) on 1H.
    4. Find unmitigated FVG below (uptrend) or above (downtrend) liquidity pool on 15m.
    5. Confirm price touches the nearest FVG on 15m.
    6. Generate LONG/SHORT signal with entry, SL, and TP (1:2 RR).
    """

    def __init__(self, main_tf: str, telegram_token: str, channel_id: str):
        """
        Initialize SMC strategy with detectors.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for trend and BOS/CHOCH.
            higher_tf (str): Higher timeframe (e.g., '4h') for trend confirmation.
            telegram_token (str): Telegram bot token for sending signals.
            channel_id (str): Telegram channel ID for message delivery.
        """
        super().__init__(main_tf, telegram_token, channel_id)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'bos_1h': BosChochDetector(main_tf),  # Detects BOS/CHOCH on 1H
            'liquidity_pool_15m': LiquidityPoolDetector(self.lower_tf_15m),  # Detects liquidity pools on 15m
            'fvg_15m': FvgDetector(self.lower_tf_15m)  # Detects FVGs on 15m
        }

    async def process_trend(self, ohlc_1h: pd.DataFrame, symbol: str, plot: bool, cache_key: Tuple[str, str],
                            current_time: datetime):
        """Step 1: Identify Trend - 1H"""
        recent_smc_events = []

        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=False)

        trend = self.detectors['swing_1h'].detect_trend(swing_1h)

        if trend != Trend.NEUTRAL.name:
            trend_event = f"{trend} trend confirmed."
            recent_smc_events.append(f"{trend} trend confirmed.")
            if await self.manage_cache_event(cache_key, trend_event, current_time):
                recent_smc_events.append(trend_event)
                if plot:
                    signals_1h = (swing_1h['signals']['HH'] + swing_1h['signals']['LH'] +
                                  swing_1h['signals']['HL'] + swing_1h['signals']['LL'])
                    combined_signals = signals_1h[-5:]
                    fig = self.plot_step(ohlc_1h, combined_signals, f"{symbol} | Step 1: Trend Detection")
                    await self.send_telegram_message(f"Step 1: {trend_event} for {symbol}.", fig)

        else:
            logger.info(f"Trend is {trend}.")
            if cache_key in self.event_cache:
                self.event_cache[cache_key]['events'] = []  # Clear cache for neutral trend
                return trend, recent_smc_events

    async def detect_bos_choch(self, ohlc_1h: pd.DataFrame, trend: str, symbol: str,
                               plot: bool, cache_key: Tuple[str, str],
                               current_time: datetime) -> Tuple[Optional[datetime], List[str]]:
        """Step 2: Detect BOS or CHOCH on 1H, reset on CHOCH."""
        bos_result = self.detectors['bos_1h'].calculate_bos_choch(ohlc_1h, plot=False)
        recent_smc_events = []

        # Map numeric biases to strings
        bias_map = {1: 'BULLISH', 0: 'BEARISH', -1: 'NEUTRAL'}
        for s in bos_result['signals']:
            s['bias'] = bias_map.get(s['bias'], s['bias'])

        # Check CHOCH
        choch_signals = [s for s in bos_result['signals'] if
                         s['type'] in ['Major BULLISH_CHoCH', 'Major BEARISH_CHoCH']]
        if choch_signals:
            choch_type = 'BULLISH_CHoCH' if any(
                s['type'] == 'Major BULLISH_CHoCH' for s in choch_signals) else 'BEARISH_CHoCH'
            choch_time = max(s['time'] for s in choch_signals)
            new_bos_event = f"{'Bullish' if choch_type == 'Major BULLISH_CHoCH' else 'Bearish'} CHOCH at {choch_time}"
            self.event_cache.update({cache_key: {'events': []}})
            recent_smc_events.append(new_bos_event)
            logger.info(f"Step 2: {new_bos_event}, resetting cache")
            if plot:
                fig = self.plot_step(ohlc_1h, bos_result['signals'][-5:], f"{symbol} | Step 2: BOS/CHOCH Detection")
                await self.send_telegram_message(f"Step 2: {new_bos_event} for {symbol}. Strategy invalidated.", fig)
            return None, recent_smc_events

        # Check BOS
        expected_bias = [BEARISH, 'BEARISH'] if trend == Trend.BULLISH.name else [BULLISH, 'BULLISH']
        bos_signals = [s for s in bos_result['signals'] if s['type'] == 'Major BOS' and s['bias'] in expected_bias]
        if bos_signals:
            bos_time = max(s['time'] for s in bos_signals)
            new_bos_event = f"{'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} BOS at {bos_time}"
            logger.debug(f"Step 2: {new_bos_event}, signals: {bos_result['signals'][-5:]}")
        else:
            logger.debug(f"Step 2: No BOS signals detected")
            return None, recent_smc_events

        if await self.manage_cache_event(cache_key, new_bos_event, current_time):
            recent_smc_events.append(new_bos_event)
            if plot:
                fig = self.plot_step(ohlc_1h, bos_result['signals'][-5:], f"{symbol} | Step 2: BOS/CHOCH Detection")
                await self.send_telegram_message(f"Step 2: {new_bos_event} for {symbol}.", fig)

        return bos_time, recent_smc_events

    async def detect_liquidity_pool(self, ohlc_15m: pd.DataFrame, ohlc_1h: pd.DataFrame,
                                    trend: str, bos_time: datetime, symbol: str,
                                    plot: bool, cache_key: Tuple[str, str],
                                    current_time: datetime) -> Tuple[Optional[Dict], List[str]]:
        """Step 3: Detect liquidity pool after BOS on 15m, between HH/HL or LH/LL on 1H."""
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=False)
        liquidity_result = self.detectors['liquidity_pool_15m'].calculate_liquidity_pools(ohlc_15m, plot=False)
        recent_smc_events = []

        logger.debug(
            f"Step 3: bos_time={bos_time}, swing_1h_signals={swing_1h['signals']}, liquidity_signals={liquidity_result['signals']}")

        new_liquidity_event = None
        if trend == Trend.BULLISH.name:
            hh = max((s['level'] for s in swing_1h['signals']['HH'] if s['time'] >= bos_time), default=None)
            hl = max((s['level'] for s in swing_1h['signals']['HL'] if s['time'] >= bos_time), default=None)
            liquidity_pool = next(
                (lp for lp in liquidity_result['signals']
                 if lp['type'] == 'LiquidityPool' and lp['bias'] == BEARISH and lp['time'] >= bos_time and
                 hh is not None and hl is not None and hl <= lp['high'] <= hh),
                None
            )
            if liquidity_pool:
                new_liquidity_event = f"Sell-side liquidity pool at {liquidity_pool['time']}"
        else:
            ll = min((s['level'] for s in swing_1h['signals']['LL'] if s['time'] >= bos_time), default=None)
            lh = min((s['level'] for s in swing_1h['signals']['LH'] if s['time'] >= bos_time), default=None)
            liquidity_pool = next(
                (lp for lp in liquidity_result['signals']
                 if lp['type'] == 'LiquidityPool' and lp['bias'] == BULLISH and lp['time'] >= bos_time and
                 ll is not None and lh is not None and ll <= lp['low'] <= lh),
                None
            )
            if liquidity_pool:
                new_liquidity_event = f"Buy-side liquidity pool at {liquidity_pool['time']}"

        if not liquidity_pool:
            new_liquidity_event = "No valid liquidity pool after BOS"
            if await self.manage_cache_event(cache_key, new_liquidity_event, current_time):
                recent_smc_events.append(new_liquidity_event)
                logger.info(f"Step 3: No valid liquidity pool detected for {symbol}")
                if plot:
                    fig = self.plot_step(ohlc_15m,
                                         [s for s in liquidity_result['signals'] if s['type'] == 'LiquidityPool'],
                                         f"{symbol} | Step 3: Liquidity Pool Detection")
                    await self.send_telegram_message(f"Step 3: No valid liquidity pool detected for {symbol}.", fig)
        elif await self.manage_cache_event(cache_key, new_liquidity_event, current_time):
            recent_smc_events.append(new_liquidity_event)
            if plot:
                fig = self.plot_step(ohlc_15m, [liquidity_pool], f"{symbol} | Step 3: Liquidity Pool Detection")
                await self.send_telegram_message(f"Step 3: {new_liquidity_event} for {symbol}.", fig)

        return liquidity_pool, recent_smc_events

    async def detect_fvg(self, ohlc_15m: pd.DataFrame, trend: str, inducement_level: float,
                         symbol: str, plot: bool, cache_key: Tuple[str, str],
                         current_time: datetime) -> Tuple[bool, List[str]]:
        """Step 4: Detect unmitigated FVG below (uptrend) or above (downtrend) liquidity pool on 15m."""
        fvg_result = self.detectors['fvg_15m'].calculate_fvg(ohlc_15m, plot=False)
        recent_smc_events = []

        fvg_confirmed = any(
            s['bias'] == (BULLISH if trend == Trend.BULLISH.name else BEARISH) and not s['mitigated'] and
            (s['top'] <= inducement_level if trend == Trend.BULLISH.name else s['bottom'] >= inducement_level)
            for s in fvg_result['signals']
        )
        if fvg_confirmed:
            new_fvg_event = f"{'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} FVG at {max(s['time'] for s in fvg_result['signals'] if s['bias'] == (BULLISH if trend == Trend.BULLISH.name else BEARISH) and not s['mitigated'])}"
        else:
            new_fvg_event = "No valid FVG below/above inducement level"
            if await self.manage_cache_event(cache_key, new_fvg_event, current_time):
                recent_smc_events.append(new_fvg_event)
                logger.info(f"Step 4: No valid FVG detected for {symbol}")
                if plot:
                    fig = self.plot_step(ohlc_15m, fvg_result['signals'], f"{symbol} | Step 4: FVG Detection")
                    await self.send_telegram_message(f"Step 4: No valid FVG for {symbol}.", fig)

        if new_fvg_event and await self.manage_cache_event(cache_key, new_fvg_event, current_time):
            recent_smc_events.append(new_fvg_event)
            if plot:
                fig = self.plot_step(ohlc_15m, fvg_result['signals'], f"{symbol} | Step 4: FVG Detection")
                await self.send_telegram_message(f"Step 4: {new_fvg_event} for {symbol}.", fig)

        return fvg_confirmed, recent_smc_events

    async def confirm_price_touch(self, ohlc_15m: pd.DataFrame, trend: str, inducement_level: float,
                                  symbol: str, plot: bool, cache_key: Tuple[str, str],
                                  current_time: datetime) -> Tuple[Optional[Dict], List[str]]:
        """Step 5: Confirm price touches nearest unmitigated FVG on 15m."""
        fvg_result = self.detectors['fvg_15m'].calculate_fvg(ohlc_15m, plot=False)
        recent_smc_events = []

        nearest_fvg = next(
            (s for s in sorted(fvg_result['signals'], key=lambda x: x['time'], reverse=True)
             if s['bias'] == (BULLISH if trend == Trend.BULLISH.name else BEARISH) and not s['mitigated'] and
             (s['top'] <= inducement_level if trend == Trend.BULLISH.name else s['bottom'] >= inducement_level)),
            None
        )
        if not nearest_fvg:
            new_touch_event = "No valid FVG found"
            if await self.manage_cache_event(cache_key, new_touch_event, current_time):
                recent_smc_events.append(new_touch_event)
                logger.info(f"Step 5: No valid FVG for {symbol}")
                if plot:
                    fig = self.plot_step(ohlc_15m, fvg_result['signals'], f"{symbol} | Step 5: Price Touches FVG")
                    await self.send_telegram_message(f"Step 5: No valid FVG for {symbol}.", fig)
            return None, recent_smc_events

        price_touched_fvg = any(
            (ohlc_15m['low'].iloc[-i] <= nearest_fvg['top'] and ohlc_15m['high'].iloc[-i] >= nearest_fvg['bottom'])
            for i in range(1, 6)
        )
        new_touch_event = None
        if not price_touched_fvg:
            new_touch_event = "Price did not touch nearest FVG"
            if await self.manage_cache_event(cache_key, new_touch_event, current_time):
                recent_smc_events.append(new_touch_event)
                logger.info(f"Step 5: Price did not touch nearest FVG for {symbol}")
                if plot:
                    fig = self.plot_step(ohlc_15m, [nearest_fvg], f"{symbol} | Step 5: Price Touches FVG")
                    await self.send_telegram_message(f"Step 5: Price did not touch nearest FVG for {symbol}.", fig)
        else:
            new_touch_event = f"Price touched {'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} FVG at {nearest_fvg['time']}"
            if await self.manage_cache_event(cache_key, new_touch_event, current_time):
                recent_smc_events.append(new_touch_event)
                if plot:
                    fig = self.plot_step(ohlc_15m, [nearest_fvg], f"{symbol} | Step: Price Touches FVG")
                    await self.send_telegram_message(f"Step 5: {new_touch_event} for {symbol}.", fig)

        return nearest_fvg, recent_smc_events

    async def generate_trade_signal(self, ohlc_15m: pd.DataFrame, trend: str, nearest_fvg: Dict,
                                    symbol: str, plot: bool, cache_key: Tuple[str, str],
                                    current_time: datetime) -> Tuple[str, float, float, float, List[Dict], List[str]]:
        """Step 6: Generate trade signal with entry, SL, and TP."""
        signals = []
        recent_smc_events = []

        entry = ohlc_15m['close'].iloc[-1]
        if trend == Trend.BULLISH.name:
            signal = "LONG"
            sl = nearest_fvg['bottom'] * 0.998
            tp = entry + 2 * (entry - sl)
        else:
            signal = "SHORT"
            sl = nearest_fvg['top'] * 1.002
            tp = entry - 2 * (sl - entry)

        new_signal_event = f"{'Bullish' if trend == Trend.BULLISH.name else 'Bearish'} signal generated at {current_time}"
        if await self.manage_cache_event(cache_key, new_signal_event, current_time):
            recent_smc_events.append(new_signal_event)
            logger.info(f"Step 6: Signal generated for {symbol}: {signal}, Entry={entry:.6f}, SL={sl:.6f}, TP={tp:.6f}")
            if plot:
                fig = self.plot_step(ohlc_15m, signals, f"{symbol} | Step 6: Signal Generation", entry, sl, tp)
                await self.send_telegram_message(
                    f"Step 6: {signal} signal for {symbol}. Entry={entry:.6f}, SL={sl:.6f}, TP={tp:.6f}", fig)

        return signal, entry, sl, tp, signals, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame, plot: bool = True) -> Dict:
        """
        Analyze OHLC data for SMC continuation signals using a 6-step process.
        """
        logger.info(f"Starting SMC analysis for {symbol}")
        current_time = datetime.now()
        cache_key = (symbol, self.lower_tf_15m)
        signal = Trend.NEUTRAL.name
        signals = []
        recent_smc_events = []

        # Step 1: Confirm trend alignment
        trend, step1_events = await self.process_trend(ohlc_1h, symbol, plot, cache_key, current_time)
        recent_smc_events.extend(step1_events)
        if trend == Trend.NEUTRAL.name:
            return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot)

        # Step 2: Detect BOS or CHOCH
        bos_time, step2_events = await self.detect_bos_choch(ohlc_1h, trend, symbol, plot, cache_key, current_time)
        recent_smc_events.extend(step2_events)
        if not bos_time:
            return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot)

        # Step 3: Detect liquidity pool
        liquidity_pool, step3_events = await self.detect_liquidity_pool(ohlc_15m, ohlc_1h, trend, bos_time,
                                                                        symbol, plot, cache_key, current_time)
        recent_smc_events.extend(step3_events)
        if not liquidity_pool:
            return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot)

        # Step 4: Detect unmitigated FVG
        inducement_level = liquidity_pool['high'] if trend == Trend.BULLISH.name else liquidity_pool['low']
        fvg_confirmed, step4_events = await self.detect_fvg(ohlc_15m, trend, inducement_level,
                                                            symbol, plot, cache_key, current_time)
        recent_smc_events.extend(step4_events)
        if not fvg_confirmed:
            return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot)

        # Step 5: Confirm price touches FVG
        nearest_fvg, step5_events = await self.confirm_price_touch(ohlc_15m, trend, inducement_level,
                                                                   symbol, plot, cache_key, current_time)
        recent_smc_events.extend(step5_events)
        if not nearest_fvg:
            return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot)

        # Step 6: Generate trade signal
        signal, entry, sl, tp, step6_signals, step6_events = await self.generate_trade_signal(
            ohlc_15m, trend, nearest_fvg, symbol, plot, cache_key, current_time)
        signals.extend(step6_signals)
        recent_smc_events.extend(step6_events)

        return await self.compile_results(ohlc_1h, signal, trend, signals, recent_smc_events, symbol, plot, entry, sl,
                                          tp)
