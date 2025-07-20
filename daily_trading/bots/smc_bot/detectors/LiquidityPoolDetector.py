import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.SMCBase import BEARISH, BULLISH, SMCBase
from daily_trading.settings import logger


class LiquidityPoolDetector(SMCBase):
    def __init__(self, main_tf, left_bars=4, right_bars=4, show_swing=True, show_liquidity=True,
                 threshold=1.0001, threshold_lower=0.9999):
        """
        Finds Swing Points: Spots high and low price points (called swing highs and lows) on a chart using 4 candles before and after each point (15-minute timeframe by default).
        Creates Liquidity Pools:
            Buy-Side Pool: A zone just above a swing high where buy orders might sit.
            Sell-Side Pool: A zone just below a swing low where sell orders might sit.

        Parameters:
        - lower_tf (str): Timeframe for analysis (e.g., '15m').
        - left_bars (int): Number of left bars for pivot detection (default 4).
        - right_bars (int): Number of right bars for pivot detection (default 4).
        - show_swing (bool): Show swing high/low labels (default True).
        - show_liquidity (bool): Show liquidity pool lines and fills (default True).
        - threshold (float): Multiplier for buy-side liquidity pool upper bound (default 1.0001).
        - threshold_lower (float): Multiplier for sell-side liquidity pool lower bound (default 0.9999).
        """
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.show_swing = show_swing
        self.show_liquidity = show_liquidity
        self.threshold = threshold
        self.threshold_lower = threshold_lower
        logger.debug(f"Initialized LiquidityPoolDetector with lower_tf={main_tf}, left_bars={left_bars}, "
                    f"right_bars={right_bars}, show_swing={show_swing}, threshold={threshold}, "
                    f"threshold_lower={threshold_lower}")

    def detect_liquidity_raids(self, ohlc, pivot_high_points, pivot_low_points, i):
        """Detect liquidity raids when price crosses swing high/low levels."""
        signals = []
        # Find the most recent swing high and low before or at index i
        upper = {"value": None, "loc": None}
        lower = {"value": None, "loc": None}
        for pivot in reversed(pivot_high_points):
            if pivot["index"] <= i:
                upper = {"value": pivot["level"], "loc": pivot["time"]}
                break
        for pivot in reversed(pivot_low_points):
            if pivot["index"] <= i:
                lower = {"value": pivot["level"], "loc": pivot["time"]}
                break

        # Buy-side liquidity raid (price crosses above swing high)
        if upper["value"] is not None and ohlc['high'].iloc[i] > upper["value"]:
            signals.append({
                "type": "LiquidityRaid",
                "bias": BEARISH,
                "time": ohlc.index[i],
                "level": upper["value"],
                "fractal_time": upper["loc"]
            })
            logger.debug(f"Buy-side Liquidity Raid detected: level={upper['value']:.6f}, time={ohlc.index[i]}")

        # Sell-side liquidity raid (price crosses below swing low)
        if lower["value"] is not None and ohlc['low'].iloc[i] < lower["value"]:
            signals.append({
                "type": "LiquidityRaid",
                "bias": BULLISH,
                "time": ohlc.index[i],
                "level": lower["value"],
                "fractal_time": lower["loc"]
            })
            logger.debug(f"Sell-side Liquidity Raid detected: level={lower['value']:.6f}, time={ohlc.index[i]}")

        return signals

    def calculate_liquidity_pools(self, ohlc, plot=True):
        """
        Calculate liquidity pools and raids based on swing points.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks, pools, and raids.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks, pools, and raids (if plot=True).
            - signals (list): List of detected liquidity pool and raid signals.
            - swing_signals (list): List of swing high/low creation signals.
        """
        logger.debug(f"Starting liquidity pool calculation with lower_tf={self.lower_tf}, left_bars={self.left_bars}, "
                    f"right_bars={self.right_bars}, show_swing={self.show_swing}, show_liquidity={self.show_liquidity}")
        if len(ohlc) < self.left_bars * 2 + 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for left_bars={self.left_bars}")
            return {"fig": None, "signals": [], "swing_signals": []}

        ohlc = self.validate_data(ohlc)
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
        signals = []
        swing_signals = []

        # Generate swing creation signals
        for pivot in pivot_high_points:
            swing_signals.append({
                "type": "SwingHigh",
                "time": pivot["time"],
                "level": pivot["level"]
            })
            logger.debug(f"Swing High detected: level={pivot['level']:.6f}, time={pivot['time']}")
        for pivot in pivot_low_points:
            swing_signals.append({
                "type": "SwingLow",
                "time": pivot["time"],
                "level": pivot["level"]
            })
            logger.debug(f"Swing Low detected: level={pivot['level']:.6f}, time={pivot['time']}")

        # Generate liquidity pool signals
        for pivot in pivot_high_points:
            # Buy-side liquidity pool (LSH to LSH * threshold)
            signals.append({
                "type": "LiquidityPool",
                "bias": BULLISH,
                "time": pivot["time"],
                "high": pivot["level"] * self.threshold,
                "low": pivot["level"],
                "index": pivot["index"]
            })
            logger.debug(
                f"Buy-side Liquidity Pool detected: high={pivot['level'] * self.threshold:.6f}, low={pivot['level']:.6f}, time={pivot['time']}")
        for pivot in pivot_low_points:
            # Sell-side liquidity pool (LSL * threshold_lower to LSL)
            signals.append({
                "type": "LiquidityPool",
                "bias": BEARISH,
                "time": pivot["time"],
                "high": pivot["level"],
                "low": pivot["level"] * self.threshold_lower,
                "index": pivot["index"]
            })
            logger.debug(
                f"Sell-side Liquidity Pool detected: high={pivot['level']:.6f}, low={pivot['level'] * self.threshold_lower:.6f}, time={pivot['time']}")

        # Detect liquidity raids
        for i in range(self.left_bars, len(ohlc)):
            raid_signals = self.detect_liquidity_raids(ohlc, pivot_high_points, pivot_low_points, i)
            signals.extend(raid_signals)

        fig = None
        if plot:
            fig = self._plot_liquidity_pools(ohlc, pivot_high_points, pivot_low_points, signals)

        logger.debug(
            f"Detected {len(signals)} signals: {sum(1 for s in signals if s['bias'] == BULLISH)} bullish, "
            f"{sum(1 for s in signals if s['bias'] == BEARISH)} bearish; "
            f"{len(swing_signals)} swing signals")
        return {"fig": fig, "signals": signals, "swing_signals": swing_signals}

    def _plot_liquidity_pools(self, ohlc, pivot_high_points, pivot_low_points, signals):
        """
        Generate Plotly figure with candlesticks, swing points, liquidity pools, and signals.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - pivot_high_points (list): List of pivot high points with time, level, index.
        - pivot_low_points (list): List of pivot low points with time, level, index.
        - signals (list): List of signals with type, bias, time, high, low.

        Returns:
        - go.Figure: Plotly figure with candlesticks, pivots, liquidity pools, and signals.
        """
        logger.debug("Generating Plotly figure for liquidity pools")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.debug(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 200 rows
        ohlc = ohlc.tail(self.plot_tail)
        ohlc_reset = ohlc.reset_index()  # Create a copy with integer indices for lookup

        # Filter pivots and recompute indices to match sliced ohlc
        pivot_high_points = [
            {**p, "index": ohlc_reset.index[ohlc_reset['time'] == p['time']].tolist()[0]}
            for p in pivot_high_points if p['time'] in ohlc.index
        ]
        pivot_low_points = [
            {**p, "index": ohlc_reset.index[ohlc_reset['time'] == p['time']].tolist()[0]}
            for p in pivot_low_points if p['time'] in ohlc.index
        ]
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name='OHLC'
        ))

        # Plot swing highs/lows
        if self.show_swing:
            for pivot in pivot_high_points:
                if pivot["index"] < len(ohlc):
                    fig.add_trace(go.Scatter(
                        x=[ohlc.index[pivot["index"]]],
                        y=[pivot["level"]],
                        mode="markers+text",
                        text=["SH"],
                        textposition="top center",
                        marker=dict(size=8, color="gray", symbol="triangle-down"),
                        showlegend=False
                    ))
                else:
                    logger.warning(f"Invalid pivot index {pivot['index']} for high pivot at {pivot['time']}")
            for pivot in pivot_low_points:
                if pivot["index"] < len(ohlc):
                    fig.add_trace(go.Scatter(
                        x=[ohlc.index[pivot["index"]]],
                        y=[pivot["level"]],
                        mode="markers+text",
                        text=["SL"],
                        textposition="bottom center",
                        marker=dict(size=8, color="gray", symbol="triangle-up"),
                        showlegend=False
                    ))
                else:
                    logger.warning(f"Invalid pivot index {pivot['index']} for low pivot at {pivot['time']}")

        # Plot liquidity pools
        if self.show_liquidity:
            line_width = 5
            line_style = 'dot'
            buy_side_color = 'rgba(255,82,82,0.35)'
            sell_side_color = 'rgba(14,243,14,0.35)'
            for pivot in pivot_high_points:
                if pivot["index"] < len(ohlc):
                    x0 = ohlc.index[max(0, pivot["index"] - self.left_bars)]
                    x1 = ohlc.index[pivot["index"]]
                    y_lower = pivot["level"]
                    y_upper = pivot["level"] * self.threshold
                    fig.add_shape(
                        type="line",
                        x0=x0, y0=y_lower, x1=x1, y1=y_lower,
                        line=dict(color=buy_side_color, width=line_width, dash=line_style)
                    )
                    fig.add_shape(
                        type="line",
                        x0=x0, y0=y_upper, x1=x1, y1=y_upper,
                        line=dict(color=buy_side_color, width=line_width, dash=line_style)
                    )
                    fig.add_shape(
                        type="rect",
                        x0=x0, y0=y_lower, x1=x1, y1=y_upper,
                        fillcolor=buy_side_color, opacity=0.35, line_width=0
                    )
                else:
                    logger.warning(
                        f"Skipping liquidity pool for high pivot at {pivot['time']} due to invalid index {pivot['index']}")
            for pivot in pivot_low_points:
                if pivot["index"] < len(ohlc):
                    x0 = ohlc.index[max(0, pivot["index"] - self.left_bars)]
                    x1 = ohlc.index[pivot["index"]]
                    y_lower = pivot["level"] * self.threshold_lower
                    y_upper = pivot["level"]
                    fig.add_shape(
                        type="line",
                        x0=x0, y0=y_lower, x1=x1, y1=y_lower,
                        line=dict(color=sell_side_color, width=line_width, dash=line_style)
                    )
                    fig.add_shape(
                        type="line",
                        x0=x0, y0=y_upper, x1=x1, y1=y_upper,
                        line=dict(color=sell_side_color, width=line_width, dash=line_style)
                    )
                    fig.add_shape(
                        type="rect",
                        x0=x0, y0=y_lower, x1=x1, y1=y_upper,
                        fillcolor=sell_side_color, opacity=0.35, line_width=0
                    )
                else:
                    logger.warning(
                        f"Skipping liquidity pool for low pivot at {pivot['time']} due to invalid index {pivot['index']}")

        # Plot liquidity pools and raids as markers
        for signal in signals:
            if signal["type"] == "LiquidityPool":
                color = "red" if signal["bias"] == BEARISH else "green"
                text = "Buy Pool" if signal["bias"] == BEARISH else "Sell Pool"
                y_level = (signal["high"] + signal["low"]) / 2
                fig.add_trace(go.Scatter(
                    x=[signal["time"]],
                    y=[y_level],
                    mode="markers+text",
                    text=[text],
                    textposition="top center" if signal["bias"] == BEARISH else "bottom center",
                    marker=dict(size=10, color=color, symbol="square"),
                    showlegend=False
                ))

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Liquidity Pools | {candle_time_str}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=15)])
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