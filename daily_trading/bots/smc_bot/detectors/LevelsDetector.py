import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.DetectorsBase import DetectorsBase


class LevelsDetector(DetectorsBase):
    def __init__(self, main_tf="1d", pivot_length=5, strength=1):
        super().__init__(main_tf, pivot_length, pivot_length)
        self.pivot_length = pivot_length
        self.strength = strength
        self.atr_length = 20
        self.too_close_atr = 1.0 / 8.0
        self.max_sr_info_size = 10
        self.atr = None

    def calculate_support_resistance_levels(self, ohlc, plot=True):
        ohlc = self.validate_data(ohlc)
        atr = self.calculate_atr(ohlc, self.atr_length)

        pivot_high = pd.Series(np.nan, index=ohlc.index)
        pivot_low = pd.Series(np.nan, index=ohlc.index)
        sr_info_list = []

        for i in range(self.pivot_length, len(ohlc) - self.pivot_length):
            left_window = ohlc.iloc[max(0, i - self.pivot_length) : i]
            right_window = ohlc.iloc[i + 1 : i + self.pivot_length + 1]
            if (left_window["high"].empty or ohlc["high"].iloc[i] >= left_window["high"].max()) and (
                right_window["high"].empty or ohlc["high"].iloc[i] >= right_window["high"].max()
            ):
                pivot_high.iloc[i] = ohlc["high"].iloc[i]
                if self._is_valid_sr(ohlc["high"].iloc[i], sr_info_list, atr.iloc[i]):
                    sr_info_list.insert(
                        0,
                        {
                            "start_time": ohlc.index[i],
                            "price": ohlc["high"].iloc[i],
                            "type": "Resistance",
                            "strength": 1,
                        },
                    )
                    if len(sr_info_list) > self.max_sr_info_size:
                        sr_info_list.pop()

            if (left_window["low"].empty or ohlc["low"].iloc[i] <= left_window["low"].min()) and (
                right_window["low"].empty or ohlc["low"].iloc[i] <= right_window["low"].min()
            ):
                pivot_low.iloc[i] = ohlc["low"].iloc[i]
                if self._is_valid_sr(ohlc["low"].iloc[i], sr_info_list, atr.iloc[i]):
                    sr_info_list.insert(
                        0,
                        {
                            "start_time": ohlc.index[i],
                            "price": ohlc["low"].iloc[i],
                            "type": "Support",
                            "strength": 1,
                        },
                    )
                    if len(sr_info_list) > self.max_sr_info_size:
                        sr_info_list.pop()

        for i in range(1, len(ohlc)):
            for sr in sr_info_list:
                if sr["type"] == "Resistance" and ohlc["high"].iloc[i] >= sr["price"] and ohlc["close"].iloc[i] <= sr["price"]:
                    sr["strength"] += 1
                elif sr["type"] == "Support" and ohlc["low"].iloc[i] <= sr["price"] and ohlc["close"].iloc[i] >= sr["price"]:
                    sr["strength"] += 1

        fig = None
        if plot:
            fig = self.plot_levels(ohlc, sr_info_list)

        pivot_high_points = [{"time": idx, "level": pivot_high.loc[idx]} for idx in pivot_high[pivot_high.notna()].index]
        pivot_low_points = [{"time": idx, "level": pivot_low.loc[idx]} for idx in pivot_low[pivot_low.notna()].index]

        return {
            "support_levels": [sr for sr in sr_info_list if sr["type"] == "Support"],
            "resistance_levels": [sr for sr in sr_info_list if sr["type"] == "Resistance"],
            "pivot_high_points": pivot_high_points,
            "pivot_low_points": pivot_low_points,
            "fig": fig,
        }

    def calculate_major_levels(self, ohlc_1h):
        # 1. Standardize Data
        df = self.validate_data(ohlc_1h).copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure index is sorted
        df = df.sort_index()

        major_levels = []

        # Get the EXACT timestamp of the very last candle in your data
        # We use this to define "NOW"
        last_candle_time = df.index[-1]

        # Helper Function to get the truly closed previous period
        def get_closed_level(resampled_df, period_name, strength):
            if resampled_df.empty:
                return

            # CRITICAL LOGIC:
            # 1. Determine the start time of the period the LAST candle belongs to.
            #    (e.g., if last candle is 23:00, period_start is 00:00 today)
            # 2. Filter the resampled dataframe to only keep rows BEFORE that start time.

            # Find the bin in resampled_df that corresponds to last_candle_time
            # 'asof' finds the nearest index label equal to or less than the value
            current_period_idx = resampled_df.index.asof(last_candle_time)

            # Filter: Keep only fully completed periods (Strictly Less Than current period)
            completed_periods = resampled_df[resampled_df.index < current_period_idx]

            if not completed_periods.empty:
                # The last item in this filtered list is the actual PREVIOUS period
                prev_candle = completed_periods.iloc[-1]
                start_time = completed_periods.index[-1]

                major_levels.append({"type": f"P{period_name}H", "price": prev_candle['high'], "start_time": start_time, "strength": strength})
                major_levels.append({"type": f"P{period_name}L", "price": prev_candle['low'], "start_time": start_time, "strength": strength})

        # --- 1. Daily Levels ---
        if len(df) >= 24:
            # Resample Daily
            daily = df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
            get_closed_level(daily, "D", 2)

        # --- 2. Weekly Levels ---
        if len(df) >= 120:
            # Resample Weekly (Ends Sunday)
            weekly = df.resample('W').agg({'high': 'max', 'low': 'min'}).dropna()
            get_closed_level(weekly, "W", 3)

        # --- 3. Monthly Levels ---
        if len(df) >= 700:
            # Resample Monthly (Month End)
            monthly = df.resample('ME').agg({'high': 'max', 'low': 'min'}).dropna()
            get_closed_level(monthly, "M", 4)

        return major_levels

    def _is_valid_sr(self, price, sr_info_list, atr):
        for sr in sr_info_list:
            if abs(sr["price"] - price) < atr * self.too_close_atr:
                return False
        return True

    def plot_levels(self, ohlc, sr_info_list):
        self.get_time_frame_name(ohlc)
        ohlc = ohlc.sort_index().tail(self.plot_tail)
        symbol = ohlc["symbol"].iloc[0] if "symbol" in ohlc.columns else "Unknown"
        fig = make_subplots(rows=1, cols=1)

        # 1. Define Color/Style Map for different levels
        style_map = {
            "Resistance": {"color": "blue", "width": 1, "dash": "solid"},
            "Support": {"color": "blue", "width": 1, "dash": "solid"},
            "PDH": {"color": "green", "width": 2, "dash": "solid"},  # Daily High
            "PDL": {"color": "green", "width": 2, "dash": "solid"},  # Daily Low
            "PWH": {"color": "orange", "width": 3, "dash": "dash"},  # Weekly
            "PWL": {"color": "orange", "width": 3, "dash": "dash"},
            "PMH": {"color": "red", "width": 3, "dash": "longdash"},  # Monthly
            "PML": {"color": "red", "width": 3, "dash": "longdash"},
        }

        # Plot Candlesticks
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

        for sr in sr_info_list:
            # Filter by strength (Major levels usually have strength >= 2, so they pass)
            if sr["strength"] >= self.strength:

                # Handle Start Time (Snap to left edge if older than plot history)
                start_time = sr["start_time"]
                if start_time not in ohlc.index:
                    # Find nearest index
                    idx_loc = ohlc.index.get_indexer([start_time], method="nearest")[0]
                    # If the actual start time is BEFORE the first visible candle, snap to first candle
                    if start_time < ohlc.index[0]:
                        start_time = ohlc.index[0]
                    else:
                        start_time = ohlc.index[idx_loc]

                end_time = ohlc.index[-1]

                # Get Style based on Type (Default to Blue if unknown)
                level_type = sr["type"]
                style = style_map.get(level_type, {"color": "gray", "width": 1, "dash": "dot"})

                # Plot the Line
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[sr["price"], sr["price"]],
                        mode="lines",
                        line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                        name=level_type,
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

                # Add Annotation (Label + Price)
                # Logic: If it's a Major Level, show Type + Price. If standard, just Price.
                if level_type in ["PDH", "PDL", "PWH", "PWL", "PMH", "PML"]:
                    label_text = f"<b>{level_type}</b>: {sr['price']:.4f}"
                else:
                    label_text = f"{sr['price']:.4f}"

                fig.add_annotation(
                    x=end_time,
                    y=sr["price"],
                    text=label_text,
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    font=dict(size=10, color=style["color"]),
                    xshift=5,
                )

        # ... (Rest of layout setup remains the same) ...
        last_candle = ohlc.index.max().tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M")
        x_range_end = ohlc.index[-1] + pd.Timedelta(minutes=120)

        fig.update_layout(
            title=f"{symbol} | Key Levels | Last: {last_candle}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=[ohlc.index[0], x_range_end]),
            margin=dict(l=50, r=50, t=100, b=100),
        )
        fig.update_xaxes(
            type="date",
            tickformat="%H:%M",
            dtick=7200000,
            tickangle=45,
            row=1,
            col=1,
        )
        return fig

    def calculate_atr(self, ohlc, period):
        high_low = ohlc["high"] - ohlc["low"]
        high_close = (ohlc["high"] - ohlc["close"].shift(1)).abs()
        low_close = (ohlc["low"] - ohlc["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.atr = true_range.rolling(window=period).mean()
        return self.atr
