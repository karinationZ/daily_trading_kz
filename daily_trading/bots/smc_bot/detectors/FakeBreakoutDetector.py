import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.DetectorsBase import DetectorsBase
from daily_trading.settings import logger


class FakeBreakoutDetector(DetectorsBase):
    """
    Class to detect false breakouts based on Smart Money Concepts (SMC),
    inheriting from SMCBase. It identifies instances where price moves beyond
    a significant high or low but fails to sustain the move, often indicating
    liquidity grabs.
    """

    def __init__(self, main_tf):
        """
        Initialize the FalseBreakoutDetector with timeframe-dependent configuration.

        Parameters:
        - lower_tf (str): Lower timeframe for analysis (e.g., '15m', '1h', '4h').
        """
        super().__init__(main_tf=main_tf)
        logger.debug(f"Initialized FalseBreakoutDetector with lower_tf={main_tf}, length={self.length}")

    def calculate_false_breakouts(self, ohlc, plot=True):
        """
        Detect false breakouts based on identified pivot highs and lows.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and labeled false breakouts.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and labeled false breakouts (if plot=True).
            - false_breakouts (list): List of detected false breakout points.
            - alert_conditions (dict): Dictionary of alert conditions for false breakouts (bullish/bearish).
        """
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
        false_breakouts = self.identify_false_breakouts_logic(ohlc, pivot_high_points, pivot_low_points)

        # Create Plotly figure if requested
        fig = None
        if plot:
            fig = self.plot_false_breakouts(ohlc, false_breakouts)

        # Prepare alert conditions
        alert_conditions = {
            "bullish_false_breakout": [fb for fb in false_breakouts if fb["type"] == "bullish_false_breakout"],
            "bearish_false_breakout": [fb for fb in false_breakouts if fb["type"] == "bearish_false_breakout"]
        }

        results = {
            "fig": fig,
            "false_breakouts": false_breakouts,
            "alert_conditions": alert_conditions
        }
        logger.debug(f"Completed false breakout detection: {len(false_breakouts)} false breakouts found.")
        return results

    def identify_false_breakouts_logic(self, ohlc, pivot_high_points, pivot_low_points):
        """
        Optimized false breakout detection for 1-hour timeframe with long wick confirmation and minimum candle range.
        """
        false_breakouts = []
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        # Convert pivot points to DataFrames with proper datetime indices
        high_pivots = pd.DataFrame(pivot_high_points)
        high_pivots['time'] = pd.to_datetime(high_pivots['time'])
        high_pivots = high_pivots.set_index('time')

        low_pivots = pd.DataFrame(pivot_low_points)
        low_pivots['time'] = pd.to_datetime(low_pivots['time'])
        low_pivots = low_pivots.set_index('time')

        # Parameters optimized for 1h timeframe
        LOOKAHEAD_CANDLES = 24  # Look ahead max 24 hours (1 day) for breakouts
        BREAKOUT_THRESHOLD = 0.0005  # % beyond pivot level for valid breakout
        WICK_THRESHOLD = 0.4  # Wick must be at least 50% of candle range
        MIN_CANDLE_RANGE = 0.07  # Candle range must be at least 0.1% of pivot level

        # For bearish false breakouts (failed break of highs)
        for pivot_time, high_pivot in high_pivots.iterrows():
            pivot_level = high_pivot['level']

            # Get the position of the pivot in the OHLC mgc
            try:
                pivot_idx = ohlc.index.get_loc(pivot_time)
                window = ohlc.iloc[pivot_idx + 1:pivot_idx + 1 + LOOKAHEAD_CANDLES]
            except KeyError:
                logger.debug(f"Pivot time {pivot_time} not found in OHLC data, skipping")
                continue

            breakout_found = False
            for candle_time, candle in window.iterrows():
                # Calculate candle range and minimum required range
                candle_range = candle['high'] - candle['low']
                min_range = pivot_level * MIN_CANDLE_RANGE

                # Skip if candle range is too small
                if candle_range < min_range:
                    logger.debug(
                        f"Skipping candle at time {candle_time}, range={candle_range:.6f} < min_range={min_range:.6f}")
                    continue

                # Check for valid breakout (price exceeds pivot level + threshold)
                if candle['high'] > pivot_level * (1 + BREAKOUT_THRESHOLD):
                    # Calculate wick and wick ratio
                    upper_wick = candle['high'] - candle['close']
                    wick_ratio = upper_wick / candle_range if candle_range > 0 else 0

                    # Check for false breakout with long wick
                    if candle['close'] < pivot_level and wick_ratio >= WICK_THRESHOLD:
                        false_breakouts.append({
                            "time": candle_time,
                            "level": pivot_level,
                            "type": "bearish_false_breakout",
                            "candle_close": candle['close'],
                            "breakout_size": candle['high'] - pivot_level,
                            "wick_ratio": wick_ratio,
                            "candle_range": candle_range
                        })
                        logger.debug(f"Detected bearish false breakout at time {candle_time}, "
                                     f"high={candle['high']:.6f}, close={candle['close']:.6f}, "
                                     f"pivot_level={pivot_level:.6f}, wick_ratio={wick_ratio:.2f}, "
                                     f"range={candle_range:.6f}")
                        breakout_found = True
                        break
                    # If it closed above, it's a valid breakout - move to next pivot
                    elif candle['close'] >= pivot_level:
                        logger.debug(f"Valid bearish breakout at time {candle_time}, "
                                     f"high={candle['high']:.6f}, close={candle['close']:.6f}, "
                                     f"pivot_level={pivot_level:.6f}")
                        breakout_found = True
                        break

        # For bullish false breakouts (failed break of lows)
        for pivot_time, low_pivot in low_pivots.iterrows():
            pivot_level = low_pivot['level']

            try:
                pivot_idx = ohlc.index.get_loc(pivot_time)
                window = ohlc.iloc[pivot_idx + 1:pivot_idx + 1 + LOOKAHEAD_CANDLES]
            except KeyError:
                logger.debug(f"Pivot time {pivot_time} not found in OHLC data, skipping")
                continue

            breakout_found = False
            for candle_time, candle in window.iterrows():
                # Calculate candle range and minimum required range
                candle_range = candle['high'] - candle['low']
                min_range = pivot_level * MIN_CANDLE_RANGE

                # Skip if candle range is too small
                if candle_range < min_range:
                    logger.debug(
                        f"Skipping candle at time {candle_time}, range={candle_range:.6f} < min_range={min_range:.6f}")
                    continue

                # Check for valid breakout (price falls below pivot level - threshold)
                if candle['low'] < pivot_level * (1 - BREAKOUT_THRESHOLD):
                    # Calculate wick and wick ratio
                    lower_wick = candle['close'] - candle['low']
                    wick_ratio = lower_wick / candle_range if candle_range > 0 else 0

                    # Check for false breakout with long wick
                    if candle['close'] > pivot_level and wick_ratio >= WICK_THRESHOLD:
                        false_breakouts.append({
                            "time": candle_time,
                            "level": pivot_level,
                            "type": "bullish_false_breakout",
                            "candle_close": candle['close'],
                            "breakout_size": pivot_level - candle['low'],
                            "wick_ratio": wick_ratio,
                            "candle_range": candle_range
                        })
                        logger.debug(f"Detected bullish false breakout at time {candle_time}, "
                                     f"low={candle['low']:.6f}, close={candle['close']:.6f}, "
                                     f"pivot_level={pivot_level:.6f}, wick_ratio={wick_ratio:.2f}, "
                                     f"range={candle_range:.6f}")
                        breakout_found = True
                        break
                    # If it closed below, it's a valid breakout - move to next pivot
                    elif candle['close'] <= pivot_level:
                        logger.debug(f"Valid bullish breakout at time {candle_time}, "
                                     f"low={candle['low']:.6f}, close={candle['close']:.6f}, "
                                     f"pivot_level={pivot_level:.6f}")
                        breakout_found = True
                        break

        logger.debug(f"Identified {len(false_breakouts)} false breakouts.")
        return false_breakouts

    def plot_false_breakouts(self, ohlc, false_breakouts):
        """
        Plot candlesticks and labeled false breakout points.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame.
        - false_breakouts (list): List of detected false breakout points.

        Returns:
        - go.Figure: Plotly figure with candlesticks and labeled false breakouts.
        """
        logger.debug("Generating Plotly figure for false breakouts")
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with False Breakouts"])

        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        for fb in false_breakouts:
            marker_symbol = 'circle'
            marker_color = 'purple'
            text_position = 'top center'
            if fb['type'] == 'bearish_false_breakout':
                marker_color = 'orange'
                text_position = 'top center'
            elif fb['type'] == 'bullish_false_breakout':
                marker_color = 'cyan'
                text_position = 'bottom center'

            fig.add_trace(go.Scatter(
                x=[fb['time']],
                y=[fb['level']],
                mode='markers',
                marker=dict(symbol=marker_symbol, size=10, color=marker_color,
                            line=dict(width=1, color='DarkSlateGrey')),
                name=fb['type'].replace('_', ' ').title(),
                showlegend=True
            ), row=1, col=1)

            # Optional: Add text label
            fig.add_trace(go.Scatter(
                x=[fb['time']],
                y=[fb['level'] * 1.002 if fb['type'] == 'bearish_false_breakout' else fb['level'] * 0.998],
                mode='text',
                text=[f"FB {fb['type'].split('_')[0].capitalize()}"],
                textposition=text_position,
                showlegend=False,
                textfont=dict(color=marker_color, size=10)
            ), row=1, col=1)

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | False Breakouts [SMC]\n{candle_time_str} | TF: {self.main_tf}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(type='category', showticklabels=False),
            yaxis=dict(title="Price")
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        logger.debug("Plotly figure for false breakouts generated successfully")
        return fig
