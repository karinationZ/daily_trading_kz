import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.SMCBase import SMCBase
from daily_trading.settings import logger


class LevelsBreakDetector(SMCBase):
    """
    Class to detect support and resistance breaks, inheriting from SMCBase.
    Implements logic to identify pivot highs/lows and their breaks with volume confirmation.
    """

    def __init__(self, main_tf, left_bars=6, right_bars=6, toggle_breaks=True, volume_thresh=10):
        """
        Initialize the LevelsBreakDetector with shared configuration.

        Parameters:
        - left_bars (int): Number of bars to look back for pivot calculation.
        - right_bars (int): Number of bars to look forward for pivot calculation.
        - volume_thresh (float): Volume oscillator threshold for breaks (default 10 for 15-minute crypto).
        - toggle_breaks (bool): Enable/disable break and wick detection.
        """
        super().__init__(main_tf, left_bars, right_bars)
        self.toggle_breaks = toggle_breaks
        self.volume_thresh = volume_thresh

    def calculate_support_resistance_breaks(self, ohlc, plot=True):
        """
        Calculate support and resistance levels with breaks.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
        - fig (go.Figure): Plotly figure with candlesticks, levels, breaks, and volume oscillator.
        - support_breaks (list): List of support break events.
        - resistance_breaks (list): List of resistance break events.
        - bull_wicks (list): List of bullish wick events.
        - bear_wicks (list): List of bearish wick events.
        - alert_conditions (dict): Dictionary of alert conditions.
        """
        # Validate input data
        ohlc = self.validate_data(ohlc)

        # Initialize pivot series
        pivot_high = pd.Series(np.nan, index=ohlc.index)
        pivot_low = pd.Series(np.nan, index=ohlc.index)
        pivot_high_points = []
        pivot_low_points = []

        # Calculate pivots, excluding the pivot point itself
        for i in range(self.left_bars, len(ohlc) - self.right_bars):
            left_start = max(0, i - self.left_bars)
            left_window_high = ohlc['high'].iloc[left_start:i]
            left_window_low = ohlc['low'].iloc[left_start:i]
            right_window_high = ohlc['high'].iloc[i + 1:i + self.right_bars + 1]
            right_window_low = ohlc['low'].iloc[i + 1:i + self.right_bars + 1]

            # Pivot high
            if (left_window_high.empty or ohlc['high'].iloc[i] > left_window_high.max()) and \
                    (right_window_high.empty or ohlc['high'].iloc[i] > right_window_high.max()):
                pivot_high.iloc[i] = ohlc['high'].iloc[i]
                pivot_high_points.append({"time": ohlc.index[i], "level": ohlc['high'].iloc[i]})
                logger.debug(f"Pivot High at index {i}, time {ohlc.index[i]}, level {ohlc['high'].iloc[i]:.6f}")

            # Pivot low
            if (left_window_low.empty or ohlc['low'].iloc[i] < left_window_low.min()) and \
                    (right_window_low.empty or ohlc['low'].iloc[i] < right_window_low.min()):
                pivot_low.iloc[i] = ohlc['low'].iloc[i]
                pivot_low_points.append({"time": ohlc.index[i], "level": ohlc['low'].iloc[i]})
                logger.debug(f"Pivot Low at index {i}, time {ohlc.index[i]}, level {ohlc['low'].iloc[i]:.6f}")

        # Apply forward-fill
        pivot_high_shifted = pivot_high.ffill()
        pivot_low_shifted = pivot_low.ffill()

        # Calculate volume oscillator
        short_ema = self.calculate_ema(ohlc['volume'], 5)
        long_ema = self.calculate_ema(ohlc['volume'], 10)
        osc = 100 * (short_ema - long_ema) / long_ema

        # Log oscillator values at key indices (e.g., 65, 84)
        for i in range(max(10, self.left_bars), len(ohlc)):
            if i in [65, 84]:
                logger.debug(f"Volume Oscillator at index {i}, time {ohlc.index[i]}: "
                             f"volume={ohlc['volume'].iloc[i]:.2f}, "
                             f"short_ema={short_ema.iloc[i]:.2f}, "
                             f"long_ema={long_ema.iloc[i]:.2f}, "
                             f"osc={osc.iloc[i]:.2f}")

        # Detect breaks and wicks
        support_breaks = []
        resistance_breaks = []
        bull_wicks = []
        bear_wicks = []

        for i in range(1, len(ohlc)):
            # Log support break conditions at key indices
            if i in [65, 84]:
                logger.debug(f"Checking support break at index {i}, time {ohlc.index[i]}: "
                             f"close[{i - 1}]={ohlc['close'].iloc[i - 1]:.6f}, "
                             f"close[{i}]={ohlc['close'].iloc[i]:.6f}, "
                             f"pivot_low_shifted[{i - 1}]={pivot_low_shifted.iloc[i - 1]:.6f}, "
                             f"osc[{i}]={osc.iloc[i]:.2f}, "
                             f"wick_condition={not (ohlc['open'].iloc[i] - ohlc['close'].iloc[i] < ohlc['high'].iloc[i] - ohlc['open'].iloc[i])}")

            # Support break condition
            if (self.toggle_breaks and not pd.isna(pivot_low_shifted.iloc[i - 1]) and
                    ohlc['close'].iloc[i - 1] >= pivot_low_shifted.iloc[i - 1] and
                    ohlc['close'].iloc[i] < pivot_low_shifted.iloc[i - 1] and
                    not (ohlc['open'].iloc[i] - ohlc['close'].iloc[i] < ohlc['high'].iloc[i] - ohlc['open'].iloc[i]) and
                    osc.iloc[i] > self.volume_thresh):
                support_breaks.append({
                    'time': ohlc.index[i],
                    'level': pivot_low_shifted.iloc[i - 1],
                    'type': 'Support Break',
                    'text': 'B'
                })
                logger.debug(
                    f"Support Break at index {i}, time {ohlc.index[i]}, level {pivot_low_shifted.iloc[i - 1]:.6f}")

            # Resistance break condition
            if (self.toggle_breaks and not pd.isna(pivot_high_shifted.iloc[i - 1]) and
                    ohlc['close'].iloc[i - 1] <= pivot_high_shifted.iloc[i - 1] and
                    ohlc['close'].iloc[i] > pivot_high_shifted.iloc[i - 1] and
                    not (ohlc['open'].iloc[i] - ohlc['low'].iloc[i] > ohlc['close'].iloc[i] - ohlc['open'].iloc[i]) and
                    osc.iloc[i] > self.volume_thresh):
                resistance_breaks.append({
                    'time': ohlc.index[i],
                    'level': pivot_high_shifted.iloc[i - 1],
                    'type': 'Resistance Break',
                    'text': 'B'
                })
                logger.debug(
                    f"Resistance Break at index {i}, time {ohlc.index[i]}, level {pivot_high_shifted.iloc[i - 1]:.6f}")

            # Bull wick condition
            if (self.toggle_breaks and not pd.isna(pivot_high_shifted.iloc[i - 1]) and
                    ohlc['close'].iloc[i - 1] <= pivot_high_shifted.iloc[i - 1] and
                    ohlc['close'].iloc[i] > pivot_high_shifted.iloc[i - 1] and
                    ohlc['open'].iloc[i] - ohlc['low'].iloc[i] > ohlc['close'].iloc[i] - ohlc['open'].iloc[i]):
                bull_wicks.append({
                    'time': ohlc.index[i],
                    'level': pivot_high_shifted.iloc[i - 1],
                    'type': 'Bull Wick',
                    'text': 'Bull Wick'
                })
                logger.debug(
                    f"Bull Wick at index {i}, time {ohlc.index[i]}, level {pivot_high_shifted.iloc[i - 1]:.6f}")

            # Bear wick condition
            if (self.toggle_breaks and not pd.isna(pivot_low_shifted.iloc[i - 1]) and
                    ohlc['close'].iloc[i - 1] >= pivot_low_shifted.iloc[i - 1] and
                    ohlc['close'].iloc[i] < pivot_low_shifted.iloc[i - 1] and
                    ohlc['open'].iloc[i] - ohlc['close'].iloc[i] < ohlc['high'].iloc[i] - ohlc['open'].iloc[i]):
                bear_wicks.append({
                    'time': ohlc.index[i],
                    'level': pivot_low_shifted.iloc[i - 1],
                    'type': 'Bear Wick',
                    'text': 'Bear Wick'
                })
                logger.debug(f"Bear Wick at index {i}, time {ohlc.index[i]}, level {pivot_low_shifted.iloc[i - 1]:.6f}")

        fig = None
        if plot:
            fig = self.plot_levels_and_breaks(ohlc, support_breaks,
                                              resistance_breaks,
                                              bull_wicks,
                                              bear_wicks,
                                              pivot_high_points,
                                              pivot_low_points)

        results = {"support_breaks": support_breaks,
                   "resistance_breaks": resistance_breaks,
                   "bull_wicks": bull_wicks,
                   "bear_wicks": bear_wicks,
                   "pivot_high_points": pivot_high_points,
                   "pivot_low_points": pivot_low_points,
                   "fig": fig}
        return results

    def _plot_support_resistance(self, fig, pivot_high_points, pivot_low_points, ohlc):
        """
        Plot support and resistance levels as horizontal lines.

        Parameters:
        - fig (go.Figure): Plotly figure object.
        - pivot_high_points (list): List of pivot high points with time, level.
        - pivot_low_points (list): List of pivot low points with time, level.
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        """
        ohlc_reset = ohlc.reset_index()  # Create copy with integer indices
        # Filter pivots and recompute indices
        pivot_high_points = [
            {**p, "index": ohlc_reset.index[ohlc_reset['time'] == p['time']].tolist()[0]}
            for p in pivot_high_points if p['time'] in ohlc.index
        ]
        pivot_low_points = [
            {**p, "index": ohlc_reset.index[ohlc_reset['time'] == p['time']].tolist()[0]}
            for p in pivot_low_points if p['time'] in ohlc.index
        ]

        for pivot in pivot_high_points:
            if pivot["index"] < len(ohlc):
                start_time = ohlc.index[pivot["index"]]
                end_idx = min(pivot["index"] + self.right_bars, len(ohlc) - 1)
                end_time = ohlc.index[end_idx]
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time],
                    y=[pivot['level'], pivot['level']],
                    mode='lines+text',
                    text=['', "Resistance"],
                    textposition='top center',
                    line=dict(color='#FF0000', width=1, dash='solid'),
                    name='Resistance',
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid pivot index {pivot['index']} for high pivot at {pivot['time']}")

        for pivot in pivot_low_points:
            if pivot["index"] < len(ohlc):
                start_time = ohlc.index[pivot["index"]]
                end_idx = min(pivot["index"] + self.right_bars, len(ohlc) - 1)
                end_time = ohlc.index[end_idx]
                fig.add_trace(go.Scatter(
                    x=[start_time, end_time],
                    y=[pivot['level'], pivot['level']],
                    mode='lines+text',
                    text=['', "Support"],
                    textposition='bottom center',
                    line=dict(color='#233dee', width=1, dash='solid'),
                    name='Support',
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid pivot index {pivot['index']} for low pivot at {pivot['time']}")

    def _plot_support_resistance_breaks(self, fig, support_breaks, resistance_breaks, bull_wicks, bear_wicks, ohlc):
        """
        Plot support and resistance breaks and wicks.

        Parameters:
        - fig (go.Figure): Plotly figure object.
        - support_breaks (list): List of support break events with time, level, text, type.
        - resistance_breaks (list): List of resistance break events with time, level, text, type.
        - bull_wicks (list): List of bullish wick events with time, level, text, type.
        - bear_wicks (list): List of bearish wick events with time, level, text, type.
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        """
        ohlc_reset = ohlc.reset_index()  # Create copy with integer indices
        # Filter events and recompute indices
        support_breaks = [
            {**b, "index": ohlc_reset.index[ohlc_reset['time'] == b['time']].tolist()[0]}
            for b in support_breaks if b['time'] in ohlc.index
        ]
        resistance_breaks = [
            {**b, "index": ohlc_reset.index[ohlc_reset['time'] == b['time']].tolist()[0]}
            for b in resistance_breaks if b['time'] in ohlc.index
        ]
        bull_wicks = [
            {**w, "index": ohlc_reset.index[ohlc_reset['time'] == w['time']].tolist()[0]}
            for w in bull_wicks if w['time'] in ohlc.index
        ]
        bear_wicks = [
            {**w, "index": ohlc_reset.index[ohlc_reset['time'] == w['time']].tolist()[0]}
            for w in bear_wicks if w['time'] in ohlc.index
        ]

        for break_event in support_breaks:
            if break_event["index"] < len(ohlc):
                fig.add_trace(go.Scatter(
                    x=[ohlc.index[break_event["index"]]],
                    y=[break_event['level'] * 1.001],
                    mode='markers+text',
                    text=[break_event['text']],
                    textposition='top center',
                    marker=dict(symbol='triangle-down', size=8, color='red'),
                    name=break_event['type'],
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid index {break_event['index']} for support break at {break_event['time']}")

        for break_event in resistance_breaks:
            if break_event["index"] < len(ohlc):
                fig.add_trace(go.Scatter(
                    x=[ohlc.index[break_event["index"]]],
                    y=[break_event['level'] * 0.999],
                    mode='markers+text',
                    text=[break_event['text']],
                    textposition='bottom center',
                    marker=dict(symbol='triangle-up', size=8, color='green'),
                    name=break_event['type'],
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid index {break_event['index']} for resistance break at {break_event['time']}")

        for wick in bull_wicks:
            if wick["index"] < len(ohlc):
                end_idx = min(wick["index"] + 1, len(ohlc) - 1)
                end_time = ohlc.index[end_idx]
                fig.add_trace(go.Scatter(
                    x=[ohlc.index[wick["index"]], end_time],
                    y=[wick['level'] * 0.999, wick['level'] * 0.999],
                    mode='lines+text',
                    text=['', wick['text']],
                    textposition='bottom center',
                    line=dict(color='green', width=2, dash='solid'),
                    name=wick['type'],
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid index {wick['index']} for bull wick at {wick['time']}")

        for wick in bear_wicks:
            if wick["index"] < len(ohlc):
                end_idx = min(wick["index"] + 1, len(ohlc) - 1)
                end_time = ohlc.index[end_idx]
                fig.add_trace(go.Scatter(
                    x=[ohlc.index[wick["index"]], end_time],
                    y=[wick['level'] * 1.001, wick['level'] * 1.001],
                    mode='lines+text',
                    text=['', wick['text']],
                    textposition='top center',
                    line=dict(color='red', width=2, dash='solid'),
                    name=wick['type'],
                    showlegend=False
                ))
            else:
                logger.warning(f"Invalid index {wick['index']} for bear wick at {wick['time']}")

    def plot_levels_and_breaks(self, ohlc, support_breaks, resistance_breaks, bull_wicks, bear_wicks, pivot_high_points,
                               pivot_low_points):
        """
        Generate Plotly figure with candlesticks, support/resistance levels, breaks, and wicks.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - support_breaks (list): List of support break events.
        - resistance_breaks (list): List of resistance break events.
        - bull_wicks (list): List of bullish wick events.
        - bear_wicks (list): List of bearish wick events.
        - pivot_high_points (list): List of pivot high points.
        - pivot_low_points (list): List of pivot low points.

        Returns:
        - go.Figure: Plotly figure with candlesticks, support/resistance levels, breaks, and wicks.
        """
        logger.debug("Generating Plotly figure for support and resistance levels")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.debug(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 200 rows
        self.plot_tail = 200  # Ensure consistent 200 rows
        ohlc = ohlc.tail(self.plot_tail)
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        # Create Plotly figure
        fig = go.Figure()

        # Plot candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ))

        # Plot support and resistance levels
        self._plot_support_resistance(fig, pivot_high_points, pivot_low_points, ohlc)

        # Plot breaks and wicks
        self._plot_support_resistance_breaks(fig, support_breaks, resistance_breaks, bull_wicks, bear_wicks, ohlc)

        # Update layout
        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Support and Resistance Levels | {candle_time_str} | TF: {self.main_tf}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=45)])
        )
        fig.update_xaxes(
            title_text="Time",
            type='date',
            tickformat="%H:%M",
            tickmode='linear',
            dtick=7200000,
            tickangle=45
        )
        logger.debug("Plotly figure generated successfully")
        return fig
