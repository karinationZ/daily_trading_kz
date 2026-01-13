import math
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.StrategyBase import BIAS

# ==========================
# Session map (start, end hours)
# ==========================
SESSION_MAP = {
    # NY: 9:30 AM - 4:00 PM EST
    "America/New_York": {"start_hour": 9, "start_minute": 30, "end_hour": 16, "end_minute": 0},
    # LONDON: 8:00 AM - 2:30 PM London Time (Captures the Euro open + Pre-NY)
    "Europe/London": {"start_hour": 8, "start_minute": 0, "end_hour": 14, "end_minute": 30},
    # ASIA (Tokyo): 9:00 AM - 3:00 PM JST
    "Asia/Tokyo": {"start_hour": 9, "start_minute": 0, "end_hour": 15, "end_minute": 0},
}


# ==========================
# ORB Detector
# ==========================
class ORBDetector:
    def __init__(self, timezone: ZoneInfo, orb_minutes: int = 15, target_percent: float = 0.5, plot_tail: int = 500):
        self.tz = timezone
        self.orb_minutes = orb_minutes
        self.t_per = target_percent
        self.plot_tail = plot_tail
        self.UTC = ZoneInfo("UTC")
        # session times
        self.session_info = SESSION_MAP[timezone.key]

    # --------------------------
    # Utility
    # --------------------------
    @staticmethod
    def _get_1up(val: float) -> int:
        return int(math.floor(val) + 1) if val % 1 > 0 else int(val)

    @staticmethod
    def _crossover(prev, curr, level) -> bool:
        return prev < level <= curr

    @staticmethod
    def _crossunder(prev, curr, level) -> bool:
        return prev > level >= curr

    # --------------------------
    # Calculate ORB
    # --------------------------
    def calculate(self, ohlc: pd.DataFrame) -> dict:
        df = ohlc.copy().sort_index()

        if df.index.tz is None:
            df.index = df.index.tz_localize(self.UTC)

        # 1. Convert to local time (Automatic DST Handling)
        tz_col = "ts_local"
        df[tz_col] = df.index.tz_convert(self.tz)

        # 2. Find the most recent session start relative to the LATEST bar in 'ohlc'
        # We don't want the last one in the whole file, we want the one relevant to now.
        last_bar_local = df[tz_col].iloc[-1]
        start_h = self.session_info["start_hour"]
        start_m = self.session_info["start_minute"]
        end_h = self.session_info["end_hour"]

        # Determine what the "Current Day's Open" is
        current_open_local = last_bar_local.replace(hour=start_h, minute=start_m, second=0, microsecond=0)

        # If the latest bar is BEFORE today's open, we should look at yesterday's open
        if last_bar_local < current_open_local:
            # We are in the overnight session waiting for the open
            # professional strategy: return empty so we don't trade yesterday's levels
            return {"boxes": [], "signals": [], "up_targets": [], "down_targets": []}

        # If the latest bar is AFTER market close, clear levels (No overnight trading)
        current_close_local = last_bar_local.replace(hour=end_h, minute=0, second=0, microsecond=0)
        if last_bar_local > current_close_local:
            return {"boxes": [], "signals": [], "up_targets": [], "down_targets": []}

        # 3. Slice data from today's open to now
        # Logic Fix: Find the first bar that is >= 09:30 (handles missing 09:30 bar)
        session_df = df[df[tz_col] >= current_open_local].copy()

        if session_df.empty:
            return {"boxes": [], "signals": [], "up_targets": [], "down_targets": []}

        # ===============================
        # From here, your ORB state logic is mostly fine,
        # but apply it to the 'session_df' slice only.
        # ===============================
        orb_boxes = []
        signals = []

        # Calculate the end of the 15-min range
        orb_end_local = current_open_local + pd.Timedelta(minutes=self.orb_minutes)

        # Get bars that form the range
        orb_range_bars = session_df[session_df[tz_col] < orb_end_local]
        if orb_range_bars.empty:
            return {"boxes": [], "signals": [], "up_targets": [], "down_targets": []}

        or_high = orb_range_bars['high'].max()
        or_low = orb_range_bars['low'].min()

        # If we haven't finished the 15 mins yet, we can't send signals
        if last_bar_local < orb_end_local:
            return {"boxes": [], "signals": [], "up_targets": [], "down_targets": []}

        # Validated ORB
        orb_boxes.append((orb_range_bars.index[0], orb_range_bars.index[-1], or_high, or_low))

        # Check for signals in the bars AFTER the ORB is formed
        post_orb_bars = session_df[session_df[tz_col] >= orb_end_local]

        for i in range(1, len(post_orb_bars)):
            row = post_orb_bars.iloc[i]
            prev_row = post_orb_bars.iloc[i - 1]

            if self._crossover(prev_row.close, row.close, or_high):
                signals.append({
                    "time": post_orb_bars.index[i],
                    "direction": BIAS.LONG,
                    "orb_high": or_high, "orb_low": or_low
                })
            elif self._crossunder(prev_row.close, row.close, or_low):
                signals.append({
                    "time": post_orb_bars.index[i],
                    "direction": BIAS.SHORT,
                    "orb_high": or_high, "orb_low": or_low
                })

        return {"boxes": orb_boxes, "signals": signals, "upper": or_high, "lower": or_low}

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    def plot(self, ohlc: pd.DataFrame, result: dict) -> go.Figure:
        ohlc = ohlc.tail(self.plot_tail)
        fig = go.Figure()

        x_start = ohlc.index[0]
        x_end = ohlc.index[-1]

        # --- Candles
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc.open,
            high=ohlc.high,
            low=ohlc.low,
            close=ohlc.close,
            name="OHLC"
        ))

        # --- ORB boxes and lines
        for start, end, or_high, or_low in result["boxes"]:
            # Box extended to chart end
            fig.add_shape(
                type="rect",
                x0=start,
                x1=x_end,
                y0=or_low,
                y1=or_high,
                fillcolor="rgba(120,123,134,0.25)",
                line=dict(width=1, color="rgba(120,123,134,0.7)"),
                layer="below"
            )

            # OR high/low lines
            fig.add_trace(go.Scatter(
                x=[start, x_end],
                y=[or_high, or_high],
                mode="lines",
                line=dict(color="gray", width=1, dash="solid"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[start, x_end],
                y=[or_low, or_low],
                mode="lines",
                line=dict(color="gray", width=1, dash="solid"),
                showlegend=False
            ))

            # --- Targets
            step = (or_high - or_low) / 2

            # Upper targets
            upper1 = or_high + step
            upper2 = upper1 + step
            for target in [upper1, upper2]:
                fig.add_trace(go.Scatter(
                    x=[start, x_end],
                    y=[target, target],
                    mode="lines",
                    line=dict(color="green", width=1, dash="dot"),
                    showlegend=False
                ))

            # Lower targets
            lower1 = or_low - step
            lower2 = lower1 - step
            for target in [lower1, lower2]:
                fig.add_trace(go.Scatter(
                    x=[start, x_end],
                    y=[target, target],
                    mode="lines",
                    line=dict(color="red", width=1, dash="dot"),
                    showlegend=False
                ))

        # --- Layout
        fig.update_layout(
            title="ORB 15m",
            xaxis_rangeslider_visible=False,
            xaxis=dict(type="date", tickformat="%H:%M"),
            yaxis_title="Price",
            template="plotly_white",
            height=600
        )

        return fig
