import datetime
from typing import List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.detectors.CandlePatternDetector import CandlePatternDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import Trend
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, StrategyBase


class MajorHHLLRejectionStrategy(StrategyBase):
    """
    Strategy to detect Major HH/LL on 1H timeframe with rejection candle on 15m timeframe.
    Uses CandlePatternDetector for identifying rejection candles.
    """

    def __init__(self, main_tf: str, debug: bool, telegram_chat_id: str):
        super().__init__(main_tf, debug, telegram_chat_id)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf=main_tf),
            'candle_15m': CandlePatternDetector(main_tf="15m")
        }
        self.last_hh_time = None
        self.last_ll_time = None
        self.wait_window = 3  # Wait for 3 candles after HH/LL on 15m
        self.current_wait_count = 0
        self.waiting_for_rejection = False
        self.current_trend = Trend.NEUTRAL.name

    async def confirm_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime):
        """Step 1: Identify Trend on 1H (last 24 candles)."""
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        fig = swing_1h["fig"]
        trend, recent_smc_events = await self.get_trend(ohlc_1h, symbol, cache_key, current_time, fig)
        return trend, recent_smc_events, fig

    async def detect_hh_ll(self, pivot_high_points: List[dict], pivot_low_points: List[dict]) -> List[str]:
        """
        Step 2: Detect new HH/LL on 1H timeframe.
        """
        recent_smc_events = []
        latest_high = max(pivot_high_points, key=lambda x: x['time']) if pivot_high_points else None
        latest_low = max(pivot_low_points, key=lambda x: x['time']) if pivot_low_points else None

        if self.current_trend == Trend.BULLISH.name and latest_high and latest_high['type'] == 'HH':
            if not self.waiting_for_rejection or (self.last_hh_time and latest_high['time'] > self.last_hh_time):
                self.waiting_for_rejection = True
                self.current_wait_count = 0
                self.last_hh_time = latest_high['time']
                recent_smc_events.append(f"New HH detected at {self.last_hh_time}")

        elif self.current_trend == Trend.BEARISH.name and latest_low and latest_low['type'] == 'LL':
            if not self.waiting_for_rejection or (self.last_ll_time and latest_low['time'] > self.last_ll_time):
                self.waiting_for_rejection = True
                self.current_wait_count = 0
                self.last_ll_time = latest_low['time']
                recent_smc_events.append(f"New LL detected at {self.last_ll_time}")

        if not self.waiting_for_rejection:
            recent_smc_events.append("No new HH/LL detected. Waiting for pivot.")

        return recent_smc_events

    async def check_rejection_candle(self, symbol: str, ohlc_15m: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Step 3: Check for rejection candle on 15m timeframe within wait window.
        """
        recent_smc_events = []
        signal = Trend.NEUTRAL.name

        if not self.waiting_for_rejection:
            return signal, recent_smc_events

        recent_candles = ohlc_15m.iloc[-self.wait_window:] if len(ohlc_15m) >= self.wait_window else ohlc_15m
        for idx, candle in recent_candles.iterrows():
            if self.current_wait_count >= self.wait_window:
                self.waiting_for_rejection = False
                self.current_wait_count = 0
                recent_smc_events.append("Wait window expired. Resetting for next HH/LL.")
                break

            candle_time = pd.Timestamp(candle.name)
            is_bullish = self.current_trend == Trend.BULLISH.name
            is_bearish = self.current_trend == Trend.BEARISH.name
            is_after_hh = is_bullish and self.last_hh_time and candle_time > pd.Timestamp(self.last_hh_time)
            is_after_ll = is_bearish and self.last_ll_time and candle_time > pd.Timestamp(self.last_ll_time)

            if is_after_hh or is_after_ll:
                self.current_wait_count += 1
                candle_df = pd.DataFrame([candle], columns=['open', 'high', 'low', 'close'])
                trend, pattern = self.detectors['candle_15m'].detect_single_candle_pattern(candle_df)

                if is_bullish and pattern in ['Shooting Star', 'Inverted Hammer'] and trend == Trend.BEARISH:
                    signal = "SHORT"
                    event = f"{pattern} detected on 15m after HH. Generating {signal} signal."
                    recent_smc_events.append(event)
                    self.waiting_for_rejection = False
                    self.current_wait_count = 0
                    return signal, recent_smc_events
                if is_bearish and pattern in ['Hanging Man', 'Hammer'] and trend == Trend.BULLISH:
                    signal = "LONG"
                    event = f"{pattern} detected on 15m after LL. Generating {signal} signal."
                    recent_smc_events.append(event)
                    self.waiting_for_rejection = False
                    self.current_wait_count = 0
                    return signal, recent_smc_events
                recent_smc_events.append(
                    f"Candle {self.current_wait_count}/{self.wait_window} on 15m after {'HH' if is_bullish else 'LL'}: No rejection pattern ({pattern}).")

        return signal, recent_smc_events

    async def analyze_smc(self, symbol: str, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame) -> SignalSummary:
        """
        Analyze OHLC data for Major HH/LL rejection signals. HH/LL on 1H, rejection candle on 15m.
        """
        recent_smc_events = []
        current_time = max(ohlc_1h.index.max(), ohlc_15m.index.max())
        cache_key = (symbol, self.main_tf)

        # Step 1: Identify trend
        trend, trend_events, fig = await self.confirm_trend(ohlc_1h, symbol, cache_key, current_time)
        recent_smc_events.extend(trend_events)

        # Get pivot points for HH/LL detection
        swing_result = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        pivot_high_points = swing_result['pivot_high_points']
        pivot_low_points = swing_result['pivot_low_points']

        # Step 2: Detect HH/LL
        hh_ll_events = await self.detect_hh_ll(pivot_high_points, pivot_low_points)
        recent_smc_events.extend(hh_ll_events)

        signal_summary = SignalSummary(
            direction=Trend.NEUTRAL.name,
            trend=self.current_trend,
            recent_smc_events=recent_smc_events,
            fig=fig
        )

        # Step 3: Check rejection candle
        signal, rejection_events = await self.check_rejection_candle(symbol, ohlc_15m)
        recent_smc_events.extend(rejection_events)

        if signal != Trend.NEUTRAL.name:
            signal_summary.direction = signal
            await self.send_telegram_message(f"{symbol} | {rejection_events[-1]}", fig)

        signal_summary.recent_smc_events = recent_smc_events
        return signal_summary
