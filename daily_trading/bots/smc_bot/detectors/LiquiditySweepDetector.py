import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.SMCBase import BEARISH, BULLISH, SMCBase
from daily_trading.settings import logger


class LiquiditySweepDetector(SMCBase):
    def __init__(self, main_tf, sweep_window=1, volume_threshold=10, left_bars=5, right_bars=1):
        """
        Liquidity Sweep
        Definition: A price movement that breaks a swing high/low (capturing liquidity, e.g., triggering stop-losses) and reverses,
        closing on the opposite side of the pivot level.
        Purpose: Signals potential market manipulation or absorption of liquidity by institutional traders.

        Parameters:
        - lower_tf (str): Lower timeframe for analysis (e.g., '15m', '1h', '4h').
        - sweep_window (int): Number of bars to look forward for sweep detection.
        - volume_threshold (float): Minimum volume required to confirm a sweep.
        - length (int): Length for pivot calculation (inherited from SMCBase).
        - left_bars (int): Number of left bars for pivot detection.
        - right_bars (int): Number of right bars for pivot detection.
        """
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.sweep_window = sweep_window
        self.volume_threshold = volume_threshold
        logger.debug(f"Initialized LiquiditySweepDetector with lower_tf={main_tf}, sweep_window={sweep_window}, "
                    f"volume_threshold={volume_threshold}, length={self.length}, left_bars={left_bars}, right_bars={right_bars}")

    def get_liquidity_count(self, ohlc, top, btm, start_idx, end_idx):
        """Calculate liquidity interactions within a given range."""
        count = 0
        volume = 0.0
        for j in range(start_idx, min(end_idx, len(ohlc))):
            if ohlc['low'].iloc[j] < top and ohlc['high'].iloc[j] > btm:
                count += 1
                volume += ohlc['volume'].iloc[j] if 'volume' in ohlc.columns else 0
        return count, volume

    def detect_sweeps(self, ohlc, upper, lower, i, sweep_window=None, volume_threshold=None):
        """Detect liquidity sweeps."""
        signals = []
        sweep_window = sweep_window if sweep_window is not None else self.sweep_window
        volume_threshold = volume_threshold if volume_threshold is not None else self.volume_threshold

        if lower["value"] is not None and i < len(ohlc) - sweep_window:
            break_indices = [j for j in range(i + 1, i + sweep_window + 1) if ohlc['low'].iloc[j] < lower["value"]]
            if break_indices and ohlc['close'].iloc[i + sweep_window] > lower["value"]:
                count, volume = self.get_liquidity_count(ohlc, ohlc['high'].iloc[i], lower["value"], i,
                                                         i + sweep_window)
                if volume > volume_threshold:
                    signals.append({
                        "type": "Sweep",
                        "bias": BULLISH,
                        "time": ohlc.index[i + sweep_window],
                        "level": lower["value"],
                        "fractal_time": lower["loc"],
                        "volume": volume
                    })
                    logger.debug(
                        f"Bullish Sweep detected: level={lower['value']:.6f}, time={ohlc.index[i + sweep_window]}, volume={volume:.2f}")

        if upper["value"] is not None and i < len(ohlc) - sweep_window:
            break_indices = [j for j in range(i + 1, i + sweep_window + 1) if ohlc['high'].iloc[j] > upper["value"]]
            if break_indices and ohlc['close'].iloc[i + sweep_window] < upper["value"]:
                count, volume = self.get_liquidity_count(ohlc, upper["value"], ohlc['low'].iloc[i], i, i + sweep_window)
                if volume > volume_threshold:
                    signals.append({
                        "type": "Sweep",
                        "bias": BEARISH,
                        "time": ohlc.index[i + sweep_window],
                        "level": upper["value"],
                        "fractal_time": upper["loc"],
                        "volume": volume
                    })
                    logger.debug(
                        f"Bearish Sweep detected: level={upper['value']:.6f}, time={ohlc.index[i + sweep_window]}, volume={volume:.2f}")

        return signals

    def calculate_liquidity_sweeps(self, ohlc, plot=True):
        """
        Calculate liquidity sweeps based on pivot points and generate signals.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and sweep signals.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and sweep signals (if plot=True).
            - signals (list): List of detected sweep signals.
        """
        logger.debug(f"Starting liquidity sweep calculation with sweep_window={self.sweep_window}, "
                    f"volume_threshold={self.volume_threshold}, length={self.length}")
        if len(ohlc) < self.length * 2 + 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for length={self.length}")
            return {"fig": None, "signals": []}

        ohlc = self.validate_data(ohlc)
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
        signals = []

        for i in range(self.left_bars, len(ohlc) - self.sweep_window):
            upper = {"value": None, "loc": None}
            lower = {"value": None, "loc": None}

            # Find the most recent pivot high and low before index i
            for pivot in reversed(pivot_high_points):
                if pivot["index"] < i:
                    upper = {"value": pivot["level"], "loc": pivot["time"]}
                    break
            for pivot in reversed(pivot_low_points):
                if pivot["index"] < i:
                    lower = {"value": pivot["level"], "loc": pivot["time"]}
                    break

            # Detect sweeps for the current index
            sweep_signals = self.detect_sweeps(ohlc, upper, lower, i)
            signals.extend(sweep_signals)

        fig = None
        if plot:
            fig = self._plot_sweeps(ohlc, signals, pivot_high_points, pivot_low_points)

        logger.debug(
            f"Detected {len(signals)} liquidity sweeps: {sum(1 for s in signals if s['bias'] == BULLISH)} bullish, "
            f"{sum(1 for s in signals if s['bias'] == BEARISH)} bearish")
        return {"fig": fig, "signals": signals}

    def _plot_sweeps(self, ohlc, signals, pivot_high_points, pivot_low_points):
        """
        Plot candlesticks and liquidity sweep signals with improved visibility, excluding pivots and text labels.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame.
        - signals (list): List of sweep signals.
        - pivot_high_points (list): List of pivot high points (unused).
        - pivot_low_points (list): List of pivot low points (unused).

        Returns:
        - go.Figure: Plotly figure with candlesticks and sweep signals.
        """
        ohlc = ohlc.sort_index()
        ohlc = ohlc.tail(self.plot_tail)
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        logger.debug("Generating Plotly figure for liquidity sweeps")
        ohlc = self.validate_data(ohlc)
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with Liquidity Sweeps"])

        # Plot candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        # Plot sweep signals with horizontal lines
        for signal in signals:
            marker_color = '#1E90FF' if signal['bias'] == BULLISH else '#DC143C'
            marker_symbol = 'triangle-up' if signal['bias'] == BULLISH else 'triangle-down'
            end_time = ohlc.index[min(ohlc.index.get_loc(signal['time']) + 5, len(ohlc) - 1)]
            fig.add_trace(go.Scatter(
                x=[signal['time']],
                y=[signal['level'] * (1.003 if signal['bias'] == BULLISH else 0.997)],
                mode='markers',
                marker=dict(symbol=marker_symbol, size=16, color=marker_color, line=dict(width=2, color='black')),
                name='Bull Sweep' if signal['bias'] == BULLISH else 'Bear Sweep',
                showlegend=True
            ), row=1, col=1)
            fig.add_shape(
                type="line",
                x0=signal['time'], x1=end_time,
                y0=signal['level'], y1=signal['level'],
                line=dict(color=marker_color, width=2, dash="dot"),
                opacity=0.7
            )

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Liquidity Sweeps | {candle_time_str}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M', showticklabels=True),
            yaxis=dict(title="Price"),
            margin=dict(t=100, b=100, l=50, r=50)
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        logger.debug("Plotly figure generated successfully")
        return fig
