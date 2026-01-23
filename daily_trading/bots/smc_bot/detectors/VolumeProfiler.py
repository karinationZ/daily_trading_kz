import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.settings import logger


class VolumeProfiler:
    """
    Class to collect and analyze volume profile mgc, calculating up/down volumes, POC, and Value Area.
    Follows Pine Script logic for fixed-range volume profile.
    """

    def __init__(self, main_tf, num_bars=150, price_bins=24, va_percent=70.0):
        """
        Initialize the VolumeProfileCollector with timeframe and configuration.

        Parameters:
        - main_tf (str): Timeframe for analysis (e.g., '15m', '1h', '4h').
        - num_bars (int): Number of bars to analyze (default 150).
        - price_bins (int): Number of price levels for volume profile (default 24).
        - va_percent (float): Value Area volume percentage (default 70.0).
        """
        self.main_tf = main_tf
        self.num_bars = num_bars
        self.price_bins = price_bins
        self.va_percent = va_percent
        self.plot_tail = 200
        logger.debug(
            f"Initialized VolumeProfileCollector with main_tf={main_tf}, num_bars={num_bars}, price_bins={price_bins}, va_percent={va_percent}"
        )

    def calculate_volume_profile(self, ohlc, plot=True):
        """
        Calculate volume profile for given OHLCV mgc, with up/down volumes, POC, and Value Area.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and volume profile.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and volume profile (if plot=True).
            - volume_profile (pd.Series): Total volume at each price level.
            - up_volume (pd.Series): Up (bullish) volume at each price level.
            - down_volume (pd.Series): Down (bearish) volume at each price level.
            - price_levels (np.ndarray): Price levels for volume profile.
            - poc_level (float): Price level of the Point of Control.
            - va_high (float): Upper bound of Value Area.
            - va_low (float): Lower bound of Value Area.
        """
        # Limit to last num_bars
        ohlc = ohlc.tail(self.num_bars).sort_index()

        # Calculate price range and bins
        price_min = ohlc["low"].min()
        price_max = ohlc["high"].max()
        price_levels = np.linspace(price_min, price_max, self.price_bins + 1)
        bin_width = (price_max - price_min) / self.price_bins

        # Initialize volume arrays
        up_volumes = np.zeros(self.price_bins)
        down_volumes = np.zeros(self.price_bins)

        def get_vol(y11, y12, y21, y22, height, vol):
            """Calculate volume for intersection of two price ranges."""
            return (
                max(
                    min(max(y11, y12), max(y21, y22)) - max(min(y11, y12), min(y21, y22)),
                    0,
                )
                * vol
                / height
                if height > 0
                else 0
            )

        # Distribute volume for each candle
        for _, row in ohlc.iterrows():
            body_top = max(row["close"], row["open"])
            body_bot = min(row["close"], row["open"])
            is_green = row["close"] >= row["open"]
            top_wick = row["high"] - body_top
            bottom_wick = body_bot - row["low"]
            body = body_top - body_bot

            # Calculate volume proportions
            total = 2 * top_wick + 2 * bottom_wick + body
            body_vol = body * row["volume"] / total if total > 0 else row["volume"]
            top_wick_vol = 2 * top_wick * row["volume"] / total if total > 0 else 0
            bottom_wick_vol = 2 * bottom_wick * row["volume"] / total if total > 0 else 0

            # Distribute volumes across bins
            for x in range(self.price_bins):
                level_low = price_levels[x]
                level_high = price_levels[x + 1]
                # Body volume
                body_vol_bin = get_vol(level_low, level_high, body_bot, body_top, body, body_vol)
                # Wick volumes (split equally between up/down)
                top_wick_vol_bin = (
                    get_vol(
                        level_low,
                        level_high,
                        body_top,
                        row["high"],
                        top_wick,
                        top_wick_vol,
                    )
                    / 2
                )
                bottom_wick_vol_bin = (
                    get_vol(
                        level_low,
                        level_high,
                        body_bot,
                        row["low"],
                        bottom_wick,
                        bottom_wick_vol,
                    )
                    / 2
                )
                # Assign to up or down volumes
                up_volumes[x] += body_vol_bin if is_green else 0
                up_volumes[x] += top_wick_vol_bin + bottom_wick_vol_bin
                down_volumes[x] += body_vol_bin if not is_green else 0
                down_volumes[x] += top_wick_vol_bin + bottom_wick_vol_bin

        # Total volume per bin
        total_volumes = up_volumes + down_volumes
        bin_midpoints = price_levels[:-1] + bin_width / 2
        volume_profile = pd.Series(total_volumes, index=bin_midpoints.round(6))
        up_volume = pd.Series(up_volumes, index=bin_midpoints.round(6))
        down_volume = pd.Series(down_volumes, index=bin_midpoints.round(6))

        # Calculate POC
        poc_index = np.argmax(total_volumes)
        poc_level = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2

        # Calculate Value Area
        total_max = total_volumes.sum() * self.va_percent / 100
        va_total = total_volumes[poc_index]
        up_idx, down_idx = poc_index, poc_index
        while va_total < total_max and (up_idx < self.price_bins - 1 or down_idx > 0):
            upper_vol = total_volumes[up_idx + 1] if up_idx < self.price_bins - 1 else 0
            lower_vol = total_volumes[down_idx - 1] if down_idx > 0 else 0
            if upper_vol == 0 and lower_vol == 0:
                break
            if upper_vol >= lower_vol:
                va_total += upper_vol
                up_idx += 1
            else:
                va_total += lower_vol
                down_idx -= 1
        va_high = price_levels[up_idx + 1]
        va_low = price_levels[down_idx]

        # Normalize volumes for plotting (as in Pine Script: volumes * bbars / (3 * maxvol))
        max_vol = total_volumes.max()
        if max_vol > 0:
            up_volume *= self.num_bars / (3 * max_vol)
            down_volume *= self.num_bars / (3 * max_vol)
            volume_profile *= self.num_bars / (3 * max_vol)

        # Create Plotly figure if requested
        fig = None
        if plot:
            fig = self._plot_volume_profile(
                ohlc,
                volume_profile,
                up_volume,
                down_volume,
                price_levels,
                poc_level,
                va_high,
                va_low,
            )

        results = {
            "fig": fig,
            "volume_profile": volume_profile,
            "up_volume": up_volume,
            "down_volume": down_volume,
            "price_levels": price_levels,
            "poc_level": poc_level,
            "va_high": va_high,
            "va_low": va_low,
        }
        logger.debug(f"Completed volume profile calculation: {len(volume_profile)} price levels, POC at {poc_level}")
        return results

    def _plot_volume_profile(
        self,
        ohlc,
        volume_profile,
        up_volume,
        down_volume,
        price_levels,
        poc_level,
        va_high,
        va_low,
    ):
        """
        Plot candlesticks and volume profile with up/down volumes, POC, and Value Area.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - volume_profile (pd.Series): Total volume at each price level.
        - up_volume (pd.Series): Up (bullish) volume at each price level.
        - down_volume (pd.Series): Down (bearish) volume at each price level.
        - price_levels (np.ndarray): Price levels for volume profile.
        - poc_level (float): Price level of the Point of Control.
        - va_high (float): Upper bound of Value Area.
        - va_low (float): Lower bound of Value Area.

        Returns:
        - go.Figure: Plotly figure with candlesticks and volume profile.
        """
        logger.debug("Generating Plotly figure for volume profile")
        ohlc = ohlc.sort_index().tail(self.plot_tail)
        symbol = ohlc["symbol"].iloc[0] if "symbol" in ohlc.columns else "Unknown"

        # Create Plotly figure with two subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=[f"{symbol} | Price Chart", "Volume Profile"],
            shared_yaxes=True,
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

        # Plot up volume (blue)
        fig.add_trace(
            go.Bar(
                y=up_volume.index,
                x=up_volume.values,
                orientation="h",
                name="Up Volume",
                marker=dict(color="rgba(0, 0, 255, 0.75)"),
                width=0.4 * (price_levels[1] - price_levels[0]),
            ),
            row=1,
            col=2,
        )

        # Plot down volume (orange, offset slightly)
        fig.add_trace(
            go.Bar(
                y=down_volume.index,
                x=down_volume.values,
                orientation="h",
                name="Down Volume",
                marker=dict(color="rgba(255, 165, 0, 0.75)"),
                width=0.4 * (price_levels[1] - price_levels[0]),
                base=up_volume.values,
            ),
            row=1,
            col=2,
        )

        # Plot POC line
        fig.add_hline(
            y=poc_level,
            line_color="red",
            line_width=2,
            annotation_text=f"POC: {poc_level:.6f}",
            annotation_position="right",
            row=1,
            col=2,
        )

        # Plot Value Area bounds
        fig.add_hrect(
            y0=va_low,
            y1=va_high,
            fillcolor="blue",
            opacity=0.3,
            line_width=0,
            annotation_text="Value Area",
            annotation_position="right",
            row=1,
            col=2,
        )

        # Update layout
        last_candle = ohlc.index.max().tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M")
        fig.update_layout(
            title=f"{symbol} | Volume Profile | Last Candle: {last_candle} (PT) | TF: {self.main_tf}",
            showlegend=True,
            yaxis=dict(title="Price"),
            xaxis2=dict(title="Normalized Volume"),
        )
        fig.update_xaxes(
            title_text="Time",
            type="date",
            tickformat="%H:%M",
            ticklabelmode="period",
            dtick=7200000,
            tickangle=45,
            row=1,
            col=1,
        )
        logger.debug("Plotly figure generated successfully")
        return fig
