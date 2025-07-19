from datetime import datetime
from typing import List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.clients.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.clients.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.clients.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.clients.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.clients.SMCBase import Trend
from daily_trading.bots.smc_bot.clients.strategy.StrategyBase import SignalSummary, StrategyBase
from daily_trading.settings import logger


class BreakOfStructureStrategy(StrategyBase):
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

    def __init__(self, main_tf: str, debug: bool, telegram_chat_id):
        """
        Initialize SMC strategy with detectors.
        """
        super().__init__(main_tf, debug, telegram_chat_id=telegram_chat_id)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'bos_1h': BosChochDetector(main_tf),  # Detects BOS/CHOCH on 1H
            'liquidity_pool_15m': LiquidityPoolDetector("15m"),  # Detects liquidity pools on 15m
            'fvg_15m': FvgDetector(self.lower_tf_15m)  # Detects FVGs on 15m
        }
        self.recent_bos_time = None
        self.bos_choch_fig = None
        self.liquidity_fig = None

    async def confirm_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime):
        """Step 1: Identify Trend on 1H (last 24 candles)."""
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        fig = swing_1h["fig"]
        trend, recent_smc_events = self.get_trend(ohlc_1h, symbol, cache_key, current_time, fig)
        return trend, recent_smc_events, fig

    async def detect_bos_choch(self, ohlc_1h: pd.DataFrame, trend: str, symbol: str) -> List[str]:
        """Step 2: Detect BOS or CHOCH on 1H, reset on CHOCH."""
        recent_smc_events = []
        results = self.detectors['bos_1h'].calculate_bos_choch(ohlc_1h, plot=True)
        self.bos_choch_fig = results["fig"]
        bos_choch_result = sorted(results["signals"], key=lambda x: x["time"], reverse=True)

        # Check if last smc event is ChoCH
        last_type = bos_choch_result[0]["type"] == "CHoCH"
        if last_type == "CHoCH":
            bos_event = f"Change of Character! Market trend may change ..."
            recent_smc_events.append(bos_event)
            self.recent_bos_time = None
            logger.info(f"Step 2: {bos_event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {bos_event}.", self.bos_choch_fig)
            return recent_smc_events

        # Get recent BOS signal
        expected_bias = Trend.BULLISH.value if trend == Trend.BULLISH.name else Trend.BEARISH.value
        bos_signals = [s for s in bos_choch_result['signals'] if
                       s['type'] == 'BOS' and s['bias'] == expected_bias]

        if not bos_signals:
            self.recent_bos_time = None
            bos_event = "No BOS signals detected"
            logger.debug(f"Step 2: {bos_event}")
        else:
            latest_bos = max(results["signals"], key=lambda x: x["time"])
            bos_time = latest_bos["time"]
            bos_event = f"{latest_bos['type']} at {bos_time}"
            self.recent_bos_time = bos_time
            recent_smc_events.append(bos_event)
            logger.debug(f"Step 2: {bos_event}")

        await self.send_telegram_message(f"Step 2: {bos_event} for {symbol}.", self.bos_choch_fig)
        return recent_smc_events

    async def detect_liquidity_pool(self, ohlc_15m: pd.DataFrame, trend: str, bos_time: datetime, symbol: str,
                                    current_time: datetime):
        """Step 3: Detect liquidity pool after BOS on 15m, between HH/HL or LH/LL on 1H."""
        liquidity_result = self.detectors['liquidity_pool_15m'].calculate_liquidity_pools(ohlc_15m, plot=True)
        self.liquidity_fig = liquidity_result["fig"]
        swing_signals = liquidity_result['swing_signals']
        recent_smc_events = []

        # Filter swing highs and lows after bos_time
        swing_highs = [s for s in swing_signals if s['type'] == 'SwingHigh' and s['time'] >= bos_time]
        swing_lows = [s for s in swing_signals if s['type'] == 'SwingLow' and s['time'] >= bos_time]

        liquidity_event = None
        if trend == Trend.BULLISH.name:
            # Find highest SwingHigh (HH) and highest SwingLow (HL) after bos_time
            hh = max((s['level'] for s in swing_highs), default=None)
            hl = max((s['level'] for s in swing_lows), default=None)

            if hh is None or hl is None:
                liquidity_event = "No valid HH or HL after BOS for bullish liquidity pool. No liquidity pool after BOS"
                liquidity_pool = None
            else:
                # Look for buy-side liquidity pool (below HL)
                liquidity_pool = next(
                    (lp for lp in liquidity_result['signals']
                     if lp['type'] == 'LiquidityPool' and lp['bias'] == Trend.BULLISH.value and
                     bos_time <= lp['time'] <= current_time and lp['low'] <= hl), None
                )
                if liquidity_pool:
                    liquidity_event = f"Buy-side liquidity pool at {liquidity_pool['time']}"

        else:  # Trend.BEARISH
            # Find lowest SwingLow (LL) and lowest SwingHigh (LH) after bos_time
            ll = min((s['level'] for s in swing_lows), default=None)
            lh = min((s['level'] for s in swing_highs), default=None)
            if ll is None or lh is None:
                liquidity_pool = None
                liquidity_event = "No valid LL or LH after BOS for bearish liquidity pool. No liquidity pool after BOS"
            else:
                # Look for buy-side liquidity pool (above LL)
                liquidity_pool = next(
                    (lp for lp in liquidity_result['signals']
                     if lp['type'] == 'LiquidityPool' and lp['bias'] == Trend.BULLISH.value and
                     bos_time <= lp['time'] <= current_time and lp['low'] >= ll),
                    None
                )
                if liquidity_pool:
                    liquidity_event = f"Buy-side liquidity pool at {liquidity_pool['time']}"

        if liquidity_pool is None:
            liquidity_event = "No valid liquidity pool after BOS."

        await self.send_telegram_message(f"Step 3: {liquidity_event} for {symbol}.", self.liquidity_fig)
        recent_smc_events.append(liquidity_event)
        return liquidity_pool, recent_smc_events

    async def confirm_price_touch(self, ohlc_15m: pd.DataFrame, trend: str,
                                  inducement_level: float, symbol: str,
                                  current_time: datetime):
        """Step 4: Confirm price touches the nearest unmitigated FVG on 15m."""
        fvg_result = self.detectors['fvg_15m'].calculate_fvg(ohlc_15m, plot=False)
        recent_smc_events = []

        nearest_fvg = next(
            (s for s in sorted(fvg_result['signals'], key=lambda x: x['time'], reverse=True)
             if s['bias'] == (Trend.BULLISH.value if trend == Trend.BULLISH.name else Trend.BEARISH.value) and not s['mitigated'] and
             (s['top'] <= inducement_level if trend == Trend.BULLISH.name else s['bottom'] >= inducement_level)),
            None
        )
        if not nearest_fvg:
            fvg_tap_event = "No valid FVG found"
            recent_smc_events.append(fvg_tap_event)
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

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC data for continuation signals based on Liquidity Pool sweep
        """
        cache_key = (symbol, self.main_tf)
        current_time = max(ohlc_15m.index.max(), ohlc_1h.index.max())
        logger.info(f"Starting Inducement Level Hunt | {symbol} - {current_time}")
        recent_smc_events = []

        # Step 1: Identify market trend
        trend, trend_event, fig = await self.confirm_trend(ohlc_1h, symbol, cache_key, current_time)
        recent_smc_events.extend(trend_event)

        # Step 2: Detect BOS & ChoCH signals
        bos_events = await self.detect_bos_choch(ohlc_1h, trend, symbol)
        recent_smc_events.extend(bos_events)
        signal_summary = SignalSummary(direction=Trend.NEUTRAL.name,
                                       trend=trend,
                                       recent_smc_events=recent_smc_events,
                                       recent_bos_time=self.recent_bos_time,
                                       fig=self.bos_choch_fig)
        if self.recent_bos_time is None:
            return signal_summary

        # Step 3: Detect Liquidity Pool (Inducement Level)
        liquidity_pool, liquidity_pool_event = await self.detect_liquidity_pool(ohlc_15m,
                                                                                trend,
                                                                                self.recent_bos_time, symbol,
                                                                                current_time)
        recent_smc_events.extend(liquidity_pool_event)
        if not liquidity_pool:
            signal_summary.recent_smc_events = recent_smc_events
            signal_summary.fig = self.liquidity_fig
            return signal_summary

        # Step 4:
