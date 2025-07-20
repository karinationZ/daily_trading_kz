import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.SMCBase import SMCBase
from daily_trading.settings import logger


class HighLowDetector(SMCBase):
    """
    Class to detect Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL)
    pivot points, inheriting from SMCBase. Implements logic to identify swing highs and lows
    and label them based on Smart Money Concepts (SMC).
    """

    def __init__(self, main_tf, length=6):
        """
        Initialize the HighLowDetector with timeframe-dependent configuration.

        Parameters:
        - lower_tf (str): Lower timeframe for analysis (e.g., '15m', '1h', '4h').
        """
        super().__init__(main_tf=main_tf)
        self.length = length
        logger.debug(f"Initialized HighLowDetector with lower_tf={main_tf}, length={self.length}")

    def calculate_high_low_swings(self, ohlc, plot=True):
        """
        Calculate swing highs and lows and label them as HH, HL, LH, or LL.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and labeled pivots.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and labeled pivots (if plot=True).
            - pivot_high_points (list): List of pivot high points with labels (HH, LH).
            - pivot_low_points (list): List of pivot low points with labels (HL, LL).
            - alert_conditions (dict): Dictionary of alert conditions for HH, HL, LH, LL.
        """
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)

        # Create Plotly figure if requested
        fig = None
        if plot:
            fig = self._plot_pivots(ohlc, pivot_high_points, pivot_low_points)

        # Prepare alert conditions
        signals = {
            "HH": [p for p in pivot_high_points if p["type"] == "HH"],
            "LH": [p for p in pivot_high_points if p["type"] == "LH"],
            "HL": [p for p in pivot_low_points if p["type"] == "HL"],
            "LL": [p for p in pivot_low_points if p["type"] == "LL"]
        }

        results = {
            "fig": fig,
            "pivot_high_points": pivot_high_points,
            "pivot_low_points": pivot_low_points,
            "signals": signals,
        }
        logger.debug(f"Completed pivot calculation: {len(pivot_high_points)} highs, {len(pivot_low_points)} lows")
        return results

    def _plot_pivots(self, ohlc, pivot_high_points, pivot_low_points):
        """
        Plot candlesticks and labeled pivot points (HH, HL, LH, LL) for the last 200 candles.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - pivot_high_points (list): List of pivot high points with labels.
        - pivot_low_points (list): List of pivot low points with labels.

        Returns:
        - go.Figure: Plotly figure with candlesticks and labeled pivots.
        """
        logger.debug("Generating Plotly figure for pivot points")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.debug(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 200 rows
        ohlc = ohlc.tail(self.plot_tail)

        # Filter pivots to ensure timestamps are within ohlc.index
        pivot_high_points = [p for p in pivot_high_points if p['time'] in ohlc.index]
        pivot_low_points = [p for p in pivot_low_points if p['time'] in ohlc.index]
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        # Create Plotly figure
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with HH/HL/LH/LL"])

        # Plot candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        # Plot pivot highs
        for pivot in pivot_high_points:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 1.001],
                mode='markers+text',
                text=[pivot['type']],
                textposition='top center',
                marker=dict(symbol='triangle-down', size=8, color='red'),
                name=pivot['type'],
                showlegend=False
            ), row=1, col=1)

        # Plot pivot lows
        for pivot in pivot_low_points:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 0.999],
                mode='markers+text',
                text=[pivot['type']],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=8, color='green'),
                name=pivot['type'],
                showlegend=False
            ), row=1, col=1)

        # Update layout
        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Higher Highs, Higher Lows, Lower Highs, Lower Lows | {candle_time_str}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            yaxis=dict(title="Price")
        )
        fig.update_xaxes(
            title_text="Time",
            type='date',  # Ensure datetime handling
            tickformat="%H:%M",  # Show only time (HH:MM)
            ticklabelmode="period",  # Align ticks with period start
            dtick=7200000,  # 2-hour intervals
            tickangle=45,  # Rotate for readability
            row=1, col=1
        )
        logger.debug("Plotly figure generated successfully")
        return fig