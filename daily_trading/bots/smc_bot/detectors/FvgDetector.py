import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your constants
from daily_trading.bots.smc_bot.detectors.DetectorsBase import (
    BEARISH,
    BULLISH,
    DetectorsBase,
)
from daily_trading.settings import logger


class FvgDetector(DetectorsBase):
    def __init__(
        self,
        main_tf,
        pivot_period=5,
        left_bars=5,
        right_bars=1,
        show_bull=True,
        show_bear=True,
        fvg_threshold=0.25,
        mode="Present",
        back=180,
        show_labels=False,
        show_fulfilled=True,
    ):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.pivot_period = pivot_period
        self.show_bull = show_bull
        self.show_bear = show_bear
        self.fvg_threshold = fvg_threshold
        self.mode = mode
        self.back = back
        self.show_labels = show_labels
        self.show_fulfilled = show_fulfilled

    def calculate_fvg(self, ohlc, plot=True):
        """
        Vectorized FVG Calculation (100x Faster).
        """
        if len(ohlc) < 5:
            return {"fig": None, "signals": [], "alert_conditions": {}}

        # Ensure index is datetime for plotting/logic
        if ohlc.index.dtype == "int64":  # Handling numeric index if necessary
            times = ohlc.index
        else:
            times = ohlc.index

        # 1. PRE-CALCULATE COLUMNS (Vectorized)
        high = ohlc["high"].values
        low = ohlc["low"].values
        close = ohlc["close"].values
        volume = ohlc["volume"].values

        # Calculate ATR (Range) for threshold - vectorized
        # Simple Approx ATR: High - Low
        ranges = high - low
        # Efficient Rolling Mean using Pandas
        thresholds = pd.Series(ranges).rolling(window=144, min_periods=1).mean().values * self.fvg_threshold

        signals = []

        # 2. VECTORIZED DETECTION
        # We look at i, i-1, i-2.
        # In numpy shifting: array[:-2] is (i-2), array[1:-1] is (i-1), array[2:] is (i)

        # Align arrays for vector logic
        # Current Candle (i)
        curr_low = low[2:]
        curr_high = high[2:]
        curr_time = times[2:]
        curr_idx = np.arange(len(ohlc))[2:]
        curr_vol = volume[2:]
        curr_thresh = thresholds[2:]

        # Candle i-1
        prev_close = close[1:-1]

        # Candle i-2
        prev2_high = high[:-2]
        prev2_low = low[:-2]

        # --- BULLISH LOGIC ---
        if self.show_bull:
            # Condition: Current Low > Prev2 High AND Prev Close > Prev2 High
            bull_cond = (curr_low > prev2_high) & (prev_close > prev2_high)

            # Filter by Threshold
            gap_sizes = curr_low - prev2_high
            valid_bull = bull_cond & (gap_sizes >= curr_thresh)

            # Get indices where True
            indices = np.where(valid_bull)[0]

            for idx in indices:
                # Real index in original dataframe = idx + 2 (due to slicing)
                real_idx = curr_idx[idx]
                top = curr_low[idx]
                bottom = prev2_high[idx]

                # Check Mitigation (Numpy Slicing - Fast)
                # Look at all lows AFTER this candle
                future_lows = low[real_idx + 1 :]

                # Find if any future low <= bottom
                # np.argmax returns index of first True, or 0 if none
                mitigation_mask = future_lows <= bottom
                mitigated = False
                mit_time = None

                if mitigation_mask.any():
                    mitigated = True
                    # Get relative index of first trigger
                    first_mit_idx = np.argmax(mitigation_mask)
                    # Absolute index
                    abs_mit_idx = real_idx + 1 + first_mit_idx
                    mit_time = times[abs_mit_idx]

                signals.append(
                    {
                        "type": "Major Bullish FVG",
                        "bias": BULLISH,
                        "top": top,
                        "bottom": bottom,
                        "time": curr_time[idx],
                        "index": real_idx,
                        "volume": curr_vol[idx],
                        "mitigated": mitigated,
                        "mitigated_time": mit_time,
                    }
                )

        # --- BEARISH LOGIC ---
        if self.show_bear:
            # Condition: Current High < Prev2 Low AND Prev Close < Prev2 Low
            bear_cond = (curr_high < prev2_low) & (prev_close < prev2_low)

            # Filter by Threshold
            gap_sizes = prev2_low - curr_high
            valid_bear = bear_cond & (gap_sizes >= curr_thresh)

            # Get indices
            indices = np.where(valid_bear)[0]

            for idx in indices:
                real_idx = curr_idx[idx]
                top = prev2_low[idx]
                bottom = curr_high[idx]

                # Check Mitigation
                # Look at all highs AFTER this candle
                future_highs = high[real_idx + 1 :]

                mitigation_mask = future_highs >= top
                mitigated = False
                mit_time = None

                if mitigation_mask.any():
                    mitigated = True
                    first_mit_idx = np.argmax(mitigation_mask)
                    abs_mit_idx = real_idx + 1 + first_mit_idx
                    mit_time = times[abs_mit_idx]

                signals.append(
                    {
                        "type": "Major Bearish FVG",
                        "bias": BEARISH,
                        "top": top,
                        "bottom": bottom,
                        "time": curr_time[idx],
                        "index": real_idx,
                        "volume": curr_vol[idx],
                        "mitigated": mitigated,
                        "mitigated_time": mit_time,
                    }
                )

        # Sort signals by time
        signals.sort(key=lambda x: x["time"], reverse=True)

        # --- ALERTS & PLOTTING ---
        alert_conditions = {
            "BULLISH_FVG": [s for s in signals if s["type"] == "Major Bullish FVG" and not s["mitigated"]],
            "BEARISH_FVG": [s for s in signals if s["type"] == "Major Bearish FVG" and not s["mitigated"]],
            "BULLISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bullish FVG" and s["mitigated"]],
            "BEARISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bearish FVG" and s["mitigated"]],
        }

        fig = None
        if plot:
            # Calculate pivots only if plotting is needed (saves time)
            pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
            fig = self._plot_fvg(ohlc, signals, pivot_high_points, pivot_low_points)

        return {"fig": fig, "signals": signals, "alert_conditions": alert_conditions}

    def _plot_fvg(self, ohlc, signals, pivot_high_points, pivot_low_points):
        """
        Plot candlesticks, pivot points, and FVG signals for the last 200 candles.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - signals (list): List of FVG signals with bias, time, top, bottom, mitigated, mitigated_time.
        - pivot_high_points (list): List of pivot high points with labels.
        - pivot_low_points (list): List of pivot low points with labels.

        Returns:
        - go.Figure: Plotly figure with candlesticks, pivots, and FVG rectangles.
        """
        logger.info("Generating Plotly figure for FVG signals")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.info(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 200 rows
        ohlc = ohlc.tail(self.plot_tail)

        # Filter pivots to ensure timestamps are within ohlc.index
        pivot_high_points = [p for p in pivot_high_points if p["time"] in ohlc.index]
        pivot_low_points = [p for p in pivot_low_points if p["time"] in ohlc.index]
        symbol = ohlc["symbol"].iloc[0] if "symbol" in ohlc.columns else "Unknown"

        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{symbol} | Price Chart with Liquidity Voids (FVG)"],
        )

        # Plot candlesticks
        fig.add_trace(
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Plot pivot highs
        for pivot in pivot_high_points:
            fig.add_trace(
                go.Scatter(
                    x=[pivot["time"]],
                    y=[pivot["level"] * 1.001],
                    mode="markers+text",
                    text=[pivot["type"]],
                    textposition="top center",
                    marker=dict(symbol="triangle-down", size=8, color="#8E2043"),
                    name=pivot["type"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Plot pivot lows
        for pivot in pivot_low_points:
            fig.add_trace(
                go.Scatter(
                    x=[pivot["time"]],
                    y=[pivot["level"] * 0.999],
                    mode="markers+text",
                    text=[pivot["type"]],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-up", size=8, color="#027C5E"),
                    name=pivot["type"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Plot FVG rectangles
        for signal in signals:
            if (signal["bias"] == BULLISH and self.show_bull) or (signal["bias"] == BEARISH and self.show_bear):
                if not signal["mitigated"] or self.show_fulfilled:
                    color = "rgba(0, 255, 0, 0.5)" if signal["bias"] == BULLISH else "rgba(255, 0, 0, 0.5)"
                    if signal["mitigated"]:
                        color = "rgba(169, 169, 169, 0.6)"
                    end_time = signal["mitigated_time"] if signal["mitigated"] else ohlc.index[-1]
                    if signal["mitigated"] and end_time not in ohlc.index:
                        end_time = ohlc.index[ohlc.index.get_indexer([end_time], method="nearest")[0]]

                    fig.add_shape(
                        type="rect",
                        x0=signal["time"],
                        x1=end_time,
                        y0=signal["bottom"],
                        y1=signal["top"],
                        fillcolor=color,
                        line=dict(
                            width=1,
                            color=color.replace("0.5", "0.8").replace("0.6", "0.8"),
                        ),
                        layer="below",
                        opacity=0.5 if not signal["mitigated"] else 0.6,
                        name=signal["type"],
                    )
                    if self.show_labels:
                        text_position = "bottom center" if signal["bias"] == BULLISH else "top center"
                        fig.add_trace(
                            go.Scatter(
                                x=[signal["time"]],
                                y=[(signal["top"] + signal["bottom"]) / 2],
                                mode="text",
                                text=["Liquidity Void"],
                                textposition=text_position,
                                textfont=dict(
                                    color=color.replace("0.5", "1.0").replace("0.6", "1.0"),
                                    size=12,
                                ),
                                showlegend=False,
                            )
                        )

        # Update layout
        last_candle = ohlc.index.max().tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M")
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | FVG | {candle_time_str}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            yaxis=dict(title="Price"),
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=60)]),
            # Extend range by 15 minutes
        )
        fig.update_xaxes(
            title_text="Time",
            type="date",  # Ensure datetime handling
            tickformat="%H:%M",  # Show only time (HH:MM)
            tickmode="linear",  # Enforce strict dtick intervals
            dtick=7200000,  # 2-hour intervals
            tickangle=45,  # Rotate for readability
            row=1,
            col=1,
        )
        logger.info("Plotly figure generated successfully")
        return fig
