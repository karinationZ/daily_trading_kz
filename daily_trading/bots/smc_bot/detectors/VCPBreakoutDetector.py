import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.LevelsDetector import LevelsDetector
from daily_trading.bots.smc_bot.detectors.SMCBase import SMCBase, Trend
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfiler
from daily_trading.settings import logger

# TODO: FIX / REFACTOR
class VCPBreakoutDetector(SMCBase):
    """
    Detector for Volatility Contraction Pattern (VCP) and breakouts with plotting for debugging.

    The VCPBreakoutDetector identifies Volatility Contraction Patterns by detecting a series of
    contracting swing highs and lows, decreasing volume, and a breakout when price crosses key
    support or resistance levels. It uses a 1-hour timeframe for trend and volume analysis, a
    4-hour timeframe for support/resistance levels, and a 15-minute timeframe for breakout
    confirmation. Plotting is included for debugging, showing candlesticks, levels, and breakout signals.

    Attributes:
        min_swings (int): Minimum number of swings required to detect a VCP (default: 3).
        contraction_ratio (float): Threshold for swing range contraction (e.g., 0.7 means each range is 70% of the previous).
        volume_decrease_ratio (float): Threshold for volume decrease (e.g., 0.8 means each volume is 80% of the previous).
        plot_tail (int): Number of candles to plot for visualization (default: 200).
        swing_detector (HighLowDetector): Detector for swing highs and lows on 1H timeframe.
        levels_detector (LevelsDetector): Detector for support and resistance levels on 4H timeframe.
        volume_profiler (VolumeProfiler): Analyzer for volume trends on 1H timeframe.
    """

    def __init__(self, main_tf: str, min_swings=3, contraction_ratio=0.7, volume_decrease_ratio=0.8, plot_tail=200):
        super().__init__(main_tf=main_tf)
        self.min_swings = min_swings
        self.contraction_ratio = contraction_ratio
        self.volume_decrease_ratio = volume_decrease_ratio
        self.plot_tail = plot_tail
        self.swing_detector = HighLowDetector('1h')  # 1H for swing highs/lows
        self.levels_detector = LevelsDetector(main_tf='4h', pivot_length=5, strength=3)  # 4H for levels
        self.volume_profiler = VolumeProfiler('1h')  # 1H for volume analysis
        logger.debug(f"Initialized VCPBreakoutDetector: tf={main_tf}, min_swings={min_swings}, "
                     f"contraction_ratio={contraction_ratio}, volume_decrease_ratio={volume_decrease_ratio}")

    def detect_contracting_swings(self, ohlc_1h: pd.DataFrame):
        """
        Detects a series of contracting swing highs and lows on 1-hour timeframe data.

        Args:
            ohlc_1h (pd.DataFrame): OHLCV data for 1-hour timeframe.

        Returns:
            tuple[bool, list]: (Whether VCP is detected, list of events).
        """
        recent_events = []
        swings = self.swing_detector.calculate_high_low_swings(ohlc_1h)
        highs = swings['pivot_high_points']
        lows = swings['pivot_low_points']

        # Combine and sort swings by time to ensure temporal alignment
        all_swings = highs + lows
        all_swings = sorted(all_swings, key=lambda x: x['time'])

        # Select the most recent swings, ensuring alternating high-low pairs
        recent_swings = all_swings[-self.min_swings * 2:] if len(all_swings) >= self.min_swings * 2 else all_swings
        if len(recent_swings) < self.min_swings * 2:
            event = f"Insufficient swings for VCP detection: {len(recent_swings)} swings available."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events

        # Extract alternating highs and lows
        high_levels = [s['level'] for s in recent_swings if s['type'] in ['HH', 'LH']]
        low_levels = [s['level'] for s in recent_swings if s['type'] in ['HL', 'LL']]

        # Ensure we have enough highs and lows
        if len(high_levels) < self.min_swings or len(low_levels) < self.min_swings:
            event = f"Insufficient highs ({len(high_levels)}) or lows ({len(low_levels)}) for VCP detection."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events

        # Calculate ranges between consecutive high-low pairs
        ranges = []
        for i in range(min(len(high_levels), len(low_levels)) - 1):
            # Use the most recent min_swings pairs
            high = high_levels[-(self.min_swings - i)]
            low = low_levels[-(self.min_swings - i)]
            range_val = high - low
            if range_val <= 0:
                event = f"Invalid range ({range_val}) at index {i}: high={high}, low={low}."
                recent_events.append(event)
                logger.info(event)
                return False, recent_events
            ranges.append(range_val)

        # Log the selected swings and ranges for debugging
        logger.debug(f"Selected swings: {[s for s in recent_swings[-self.min_swings * 2:]]}")
        logger.debug(f"Calculated ranges: {ranges}")

        # Check if ranges are contracting
        if len(ranges) < 2:
            event = "Insufficient ranges for VCP detection."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events

        if all(ranges[i + 1] < ranges[i] * self.contraction_ratio for i in range(len(ranges) - 1)):
            event = f"VCP detected with contracting ranges: {ranges}."
            recent_events.append(event)
            logger.info(event)
            return True, recent_events
        else:
            event = "No volatility contraction detected."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events

    def confirm_volume_contraction(self, ohlc_1h: pd.DataFrame):
        """
        Confirms decreasing volume during the contraction phase on 1-hour timeframe.

        Args:
            ohlc_1h (pd.DataFrame): OHLCV data for 1-hour timeframe.

        Returns:
            tuple[bool, list]: (Whether volume contraction is confirmed, list of events).
        """
        recent_events = []
        volume_profile = self.volume_profiler.calculate_volume_profile(ohlc_1h)
        recent_volumes = volume_profile['volumes'][-self.min_swings:]
        if len(recent_volumes) < self.min_swings:
            event = "Insufficient volume data for confirmation."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events
        if all(recent_volumes[i + 1] < recent_volumes[i] * self.volume_decrease_ratio for i in
               range(len(recent_volumes) - 1)):
            event = f"Volume contraction confirmed: {recent_volumes}."
            recent_events.append(event)
            logger.info(event)
            return True, recent_events
        else:
            event = "No volume decrease during contraction."
            recent_events.append(event)
            logger.info(event)
            return False, recent_events

    def detect_breakout(self, ohlc_15m: pd.DataFrame, levels: dict, trend: str):
        """
        Detects breakout signals on 15-minute timeframe when price crosses key levels.

        Args:
            ohlc_15m (pd.DataFrame): OHLCV data for 15-minute timeframe.
            levels (dict): Dictionary containing support and resistance levels.
            trend (str): Current market trend (BULLISH, BEARISH, or NEUTRAL).

        Returns:
            tuple[str, list]: (Breakout direction ('LONG', 'SHORT', or None), list of events).
        """
        recent_events = []
        current_price = ohlc_15m.iloc[-1]['close']
        resistance_levels = levels['resistance_levels']
        support_levels = levels['support_levels']
        resistance = max(r['price'] for r in resistance_levels) if resistance_levels else None
        support = min(s['price'] for s in support_levels) if support_levels else None
        threshold = self.levels_detector.calculate_atr(ohlc_15m, period=14).iloc[-1] / 2

        is_bull_break = trend == Trend.BULLISH.name and resistance is not None and abs(
            current_price - resistance) <= threshold
        is_bear_break = trend == Trend.BEARISH.name and support is not None and abs(
            current_price - support) <= threshold

        if is_bull_break:
            event = f"Bullish breakout detected at resistance {resistance} (within {threshold:.4f})."
            recent_events.append(event)
            logger.info(event)
            return 'LONG', recent_events
        elif is_bear_break:
            event = f"Bearish breakout detected at support {support} (within {threshold:.4f})."
            recent_events.append(event)
            logger.info(event)
            return 'SHORT', recent_events
        else:
            event = "No breakout detected."
            recent_events.append(event)
            logger.info(event)
            return None, recent_events

    def _plot_vcp(self, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame, levels: dict, breakout_dir: str, is_vcp: bool):
        """
        Plots VCP analysis with candlesticks, levels, and breakout signals for debugging.

        Args:
            ohlc_1h (pd.DataFrame): OHLCV data for 1-hour timeframe.
            ohlc_15m (pd.DataFrame): OHLCV data for 15-minute timeframe.
            levels (dict): Support and resistance levels.
            breakout_dir (str): Breakout direction ('LONG', 'SHORT', or None).
            is_vcp (bool): Whether a VCP was detected.

        Returns:
            go.Figure: Plotly figure with 1H and 15m candlesticks, levels, and breakout annotations.
        """
        ohlc = ohlc_1h.tail(self.plot_tail).sort_index()
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"{symbol} | VCP Analysis (1H)", f"{symbol} | Breakout (15m)"],
                            vertical_spacing=0.1)

        # Plot 1H candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC 1H"
        ), row=1, col=1)

        # Plot support/resistance levels
        for level in levels['support_levels']:
            fig.add_shape(type="line", x0=ohlc.index[0], x1=ohlc.index[-1], y0=level['price'], y1=level['price'],
                          line=dict(color="green", width=2, dash="dash"), row=1, col=1)
        for level in levels['resistance_levels']:
            fig.add_shape(type="line", x0=ohlc.index[0], x1=ohlc.index[-1], y0=level['price'], y1=level['price'],
                          line=dict(color="red", width=2, dash="dash"), row=1, col=1)

        # Plot 15m candlesticks
        ohlc_15m = ohlc_15m.tail(self.plot_tail).sort_index()
        fig.add_trace(go.Candlestick(
            x=ohlc_15m.index,
            open=ohlc_15m['open'],
            high=ohlc_15m['high'],
            low=ohlc_15m['low'],
            close=ohlc_15m['close'],
            name="OHLC 15m"
        ), row=2, col=1)

        # Plot breakout annotation
        if breakout_dir:
            last_candle = ohlc_15m.iloc[-1]
            fig.add_annotation(x=ohlc_15m.index[-1], y=last_candle['close'], text=f"{breakout_dir} Breakout",
                               showarrow=True, arrowhead=2, ax=20, ay=-30, row=2, col=1)

        # Update layout
        last_candle_time = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        fig.update_layout(
            title=f"{symbol} | VCP {'Detected' if is_vcp else 'Not Detected'} | Last Candle: {last_candle_time} (PT)",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800
        )
        fig.update_xaxes(title_text="Time", type='date', tickformat="%H:%M", tickmode='linear', dtick=7200000,
                         tickangle=45, row=1, col=1)
        fig.update_xaxes(title_text="Time", type='date', tickformat="%H:%M", tickmode='linear', dtick=900000,
                         tickangle=45, row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=2, col=1)
        logger.debug("VCP plot generated successfully")
        return fig

    def calculate_vcp_breakout(self, ohlc_4h: pd.DataFrame, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame, plot=True):
        """
        Calculates VCP and breakout signals, returning results and plots.

        Args:
            ohlc_1h (pd.DataFrame): OHLCV data for 1-hour timeframe (for swings and volume).
            ohlc_15m (pd.DataFrame): OHLCV data for 15-minute timeframe (for breakout).
            ohlc_4h (pd.DataFrame): OHLCV data for 4-hour timeframe (for levels).
            trend (str): Current market trend (BULLISH, BEARISH, or NEUTRAL).
            plot (bool): Whether to generate plots for debugging.

        Returns:
            dict: Contains is_vcp (bool), breakout_dir (str or None), events (list), fig (Plotly figure), ltf_fig (Plotly figure).
        """
        trend = self.detect_trend(ohlc_1h)
        events = []
        if trend == Trend.NEUTRAL.name:
            events.append("Neutral trend, skipping VCP detection.")
            return {'is_vcp': False, 'breakout_dir': None, 'events': events, 'fig': None, 'ltf_fig': None}
        is_contracting, contr_events = self.detect_contracting_swings(ohlc_1h)
        events.extend(contr_events)
        if not is_contracting:
            fig = self._plot_vcp(ohlc_1h, ohlc_15m, {'support_levels': [], 'resistance_levels': []}, None,
                                 False) if plot else None
            return {'is_vcp': False, 'breakout_dir': None, 'events': events, 'fig': fig, 'ltf_fig': None}
        levels_result = self.levels_detector.calculate_support_resistance_levels(ohlc_4h, plot=plot)
        event = f"Detected {len(levels_result['support_levels'])} support and {len(levels_result['resistance_levels'])} resistance levels."
        events.append(event)
        logger.info(event)
        is_vol_contract, vol_events = self.confirm_volume_contraction(ohlc_1h)
        events.extend(vol_events)
        if not is_vol_contract:
            fig = self._plot_vcp(ohlc_1h, ohlc_15m, levels_result, None, True) if plot else None
            return {'is_vcp': False, 'breakout_dir': None, 'events': events, 'fig': fig, 'ltf_fig': None}
        breakout_dir, break_events = self.detect_breakout(ohlc_15m, levels_result, trend)
        events.extend(break_events)
        fig = self._plot_vcp(ohlc_1h, ohlc_15m, levels_result, breakout_dir, True) if plot else None
        ltf_fig = self.levels_detector.plot_htf_levels_on_ltf(ohlc_15m, levels_result) if plot else None
        return {'is_vcp': True, 'breakout_dir': breakout_dir, 'events': events, 'fig': fig, 'ltf_fig': ltf_fig}
