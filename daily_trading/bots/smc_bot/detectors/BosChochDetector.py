import pandas as pd
import plotly.graph_objects as go

# Imports from your project structure
from daily_trading.bots.smc_bot.detectors.DetectorsBase import (
    BEARISH,
    BIAS,
    BULLISH,
    DetectorsBase,
)
from daily_trading.settings import logger


class BosChochDetector(DetectorsBase):
    def __init__(
        self,
        main_tf,
        pivot_period=None,
        left_bars=8,
        right_bars=3,
        show_bull=True,
        show_bear=True,
        volume_thresh=5,
        min_price_move=0.001,
        min_signal_gap=1,
        wick_ratio: float = 0.10,
    ):
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH).
        Recommended for 1H: left_bars=5, right_bars=2.
        """
        super().__init__(
            main_tf=main_tf,
            left_bars=left_bars,
            right_bars=right_bars,
            wick_ratio=wick_ratio,
        )

        self.signals = None
        self.last_trend_timestamp = None
        self.cached_trend_bias = BIAS.NEUTRAL
        self.pivot_period = pivot_period if pivot_period is not None else self.length

        # Configuration
        self.show_bull = show_bull
        self.show_bear = show_bear
        self.volume_thresh = volume_thresh
        self.min_price_move = min_price_move
        self.min_signal_gap = min_signal_gap

        # Internal State
        self.ExternalTrend = "No Trend"
        self.LockBreak_M = None
        self.LastSignalIndex = None
        self.bull_ms_count = 0
        self.bear_ms_count = 0

        # Level Tracking
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None

        logger.info(f"Initialized BosChochDetector: {main_tf} | L:{left_bars} R:{right_bars}")

    def detect_signals_optimized(
        self,
        i,
        closes,
        highs,
        lows,
        volumes,
        times,
        signals,
        pivot_high_points,
        pivot_low_points,
    ):
        """
        Optimized core logic using Numpy arrays.
        Implements 'Break and Reset' to eliminate signal noise.
        """
        close = closes[i]
        high = highs[i]
        low = lows[i]
        volume = volumes[i]
        time = times[i]

        # 0. Min Signal Gap Check
        if self.LastSignalIndex is not None and i - self.LastSignalIndex < self.min_signal_gap:
            return signals

        # 1. Update Trend based on Pivot Sequence (Fractals)
        if len(self.ArrayType) >= 3:
            last = self.ArrayType[-1]
            prev = self.ArrayType[-2]
            prev2 = self.ArrayType[-3]

            if last == "LH" and prev in ["HH", "H"] and prev2 in ["L", "LL", "HL"]:
                self.ExternalTrend = "Down Trend"
            elif last == "LL" and prev == "LH" and prev2 in ["H", "HH", "LH"]:
                self.ExternalTrend = "Down Trend"
            elif last == "HH" and prev in ["L", "LL", "HL"] and prev2 in ["H", "HH", "LH"]:
                self.ExternalTrend = "Up Trend"
            elif last == "HL" and prev == "HH" and prev2 in ["L", "LL", "HL"]:
                self.ExternalTrend = "Up Trend"

        # 2. Bullish Major BOS/CHoCH Check
        if self.Major_HighLevel is not None and high > self.Major_HighLevel and self.show_bull and volume >= self.volume_thresh:

            # Lookahead: Ensure close confirms the break
            next_close = closes[i + 1] if i < len(closes) - 1 else close

            if next_close > self.Major_HighLevel or close > self.Major_HighLevel:
                signal_type = "BOS" if self.ExternalTrend in ["No Trend", "Up Trend"] else "CHoCH"

                signals.append(
                    {
                        "type": f"{signal_type}",
                        "bias": BULLISH,
                        "time": time,
                        "level": self.Major_HighLevel,
                        "fractal_time": (times[self.Major_HighIndex] if self.Major_HighIndex is not None else time),
                        "index": i,
                        "volume": volume,
                    }
                )

                self.bull_ms_count += 1
                self.ExternalTrend = "Up Trend"
                self.LastSignalIndex = i

                # --- NOISE FIX: RESET HIGH ---
                # We stop looking for bullish breaks until a NEW Pivot High forms.
                self.Major_HighLevel = None
                self.Major_HighIndex = None
                self.LockBreak_M = None
                # -----------------------------

                # Trailing: Update Major Low to the most recent significant low
                valid_lows = [p for p in pivot_low_points if p["index"] <= i]
                if valid_lows:
                    latest_low = valid_lows[-1]
                    self.Major_LowLevel = latest_low["level"]
                    self.Major_LowIndex = latest_low["index"]
                    self.Major_LowType = latest_low["type"]

        # 3. Bearish Major BOS/CHoCH Check
        if self.Major_LowLevel is not None and low < self.Major_LowLevel and self.show_bear and volume >= self.volume_thresh:

            next_close = closes[i + 1] if i < len(closes) - 1 else close

            if next_close < self.Major_LowLevel or close < self.Major_LowLevel:
                signal_type = "BOS" if self.ExternalTrend in ["No Trend", "Down Trend"] else "CHoCH"

                signals.append(
                    {
                        "type": f"{signal_type}",
                        "bias": BEARISH,
                        "time": time,
                        "level": self.Major_LowLevel,
                        "fractal_time": (times[self.Major_LowIndex] if self.Major_LowIndex is not None else time),
                        "index": i,
                        "volume": volume,
                    }
                )

                self.bear_ms_count += 1
                self.ExternalTrend = "Down Trend"
                self.LastSignalIndex = i

                # --- NOISE FIX: RESET LOW ---
                # We stop looking for bearish breaks until a NEW Pivot Low forms.
                self.Major_LowLevel = None
                self.Major_LowIndex = None
                self.LockBreak_M = None
                # ----------------------------

                # Trailing: Update Major High to the most recent significant high
                valid_highs = [p for p in pivot_high_points if p["index"] <= i]
                if valid_highs:
                    latest_high = valid_highs[-1]
                    self.Major_HighLevel = latest_high["level"]
                    self.Major_HighIndex = latest_high["index"]
                    self.Major_HighType = latest_high["type"]

        # 4. Update Major Levels based on NEW PIVOTS
        # This re-enables the BOS check after the reset above.
        if i in self.ArrayIndex:
            pivot_idx = self.ArrayIndex.index(i)
            pivot_type = self.ArrayType[pivot_idx]
            pivot_value = self.ArrayValue[pivot_idx]

            if pivot_type in ["H", "HH", "LH"]:
                # Set High if it was Reset (None) OR if price moved significantly
                if self.Major_HighLevel is None or abs(pivot_value - self.Major_HighLevel) / self.Major_HighLevel >= self.min_price_move:
                    self.Major_HighLevel = pivot_value
                    self.Major_HighIndex = i
                    self.Major_HighType = pivot_type

            elif pivot_type in ["L", "LL", "HL"]:
                # Set Low if it was Reset (None) OR if price moved significantly
                if self.Major_LowLevel is None or abs(pivot_value - self.Major_LowLevel) / self.Major_LowLevel >= self.min_price_move:
                    self.Major_LowLevel = pivot_value
                    self.Major_LowIndex = i
                    self.Major_LowType = pivot_type

        return signals

    def calculate_bos_choch(self, ohlc, plot=True):
        # 1. Validation
        if len(ohlc) < self.left_bars + self.right_bars + 1:
            return {"fig": None, "signals": [], "alert_conditions": {}}

        # 2. OPTIMIZATION: Prepare Numpy Arrays
        closes = ohlc["close"].values
        highs = ohlc["high"].values
        lows = ohlc["low"].values
        volumes = ohlc["volume"].values
        times = ohlc.index

        # Reset State
        self.ArrayType = []
        self.ArrayValue = []
        self.ArrayIndex = []
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None
        self.ExternalTrend = "No Trend"
        self.LastSignalIndex = None
        self.bull_ms_count = 0
        self.bear_ms_count = 0

        signals = []

        # Calculate Pivots (Populates self.ArrayType, etc.)
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)

        # 3. Main Loop
        # Iterate through the entire dataframe using optimized arrays
        for i in range(self.left_bars, len(ohlc)):
            signals = self.detect_signals_optimized(
                i,
                closes,
                highs,
                lows,
                volumes,
                times,
                signals,
                pivot_high_points,
                pivot_low_points,
            )

        # 4. Result Building
        fig = None
        if plot:
            fig = self._plot_signals(ohlc, signals, pivot_high_points, pivot_low_points)

        alert_conditions = {
            "BULLISH_BOS": [s for s in signals if s["type"] == "BOS" and s["bias"] == BULLISH],
            "BEARISH_BOS": [s for s in signals if s["type"] == "BOS" and s["bias"] == BEARISH],
            "BULLISH_CHoCH": [s for s in signals if s["type"] == "CHoCH" and s["bias"] == BULLISH],
            "BEARISH_CHoCH": [s for s in signals if s["type"] == "CHoCH" and s["bias"] == BEARISH],
        }

        return {"fig": fig, "signals": signals, "alert_conditions": alert_conditions}

    def get_current_trend(self, ohlc: pd.DataFrame) -> BIAS:
        """
        Determines current market bias based on the LAST confirmed BOS/CHoCH.
        """
        current_ts = ohlc.index[-1]

        if self.last_trend_timestamp == current_ts:
            return self.cached_trend_bias

        result = self.calculate_bos_choch(ohlc, plot=False)
        signals = result.get("signals", [])
        self.signals = signals

        new_bias = BIAS.NEUTRAL

        if signals:
            last_sig = signals[-1]
            sig_bias = last_sig.get("bias")

            if sig_bias == BIAS.LONG or sig_bias == BIAS.LONG.num_value:
                new_bias = BIAS.LONG
            elif sig_bias == BIAS.SHORT or sig_bias == BIAS.SHORT.num_value:
                new_bias = BIAS.SHORT

        self.last_trend_timestamp = current_ts
        self.cached_trend_bias = new_bias

        return new_bias

    def _plot_signals(self, ohlc, signals, pivot_high_points, pivot_low_points):
        """Standard Plotly visualization."""
        ohlc = ohlc.sort_index()
        ohlc = ohlc.tail(self.plot_tail)

        # Filter points to visible range
        visible_highs = [p for p in pivot_high_points if p["time"] in ohlc.index]
        visible_lows = [p for p in pivot_low_points if p["time"] in ohlc.index]

        symbol = ohlc["symbol"].iloc[0] if "symbol" in ohlc.columns else "Unknown"
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                name="OHLC",
            )
        )

        # Pivots
        fig.add_trace(
            go.Scatter(
                x=[p["time"] for p in visible_highs],
                y=[p["level"] for p in visible_highs],
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color="#8E2043"),
                name="Swing High",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[p["time"] for p in visible_lows],
                y=[p["level"] for p in visible_lows],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="#027C5E"),
                name="Swing Low",
            )
        )

        # Signals
        for signal in signals:
            if signal["time"] in ohlc.index or signal["fractal_time"] in ohlc.index:
                color = "#01AADD" if signal["bias"] == BULLISH else "#EFB465"
                fig.add_trace(
                    go.Scatter(
                        x=[signal["fractal_time"], signal["time"]],
                        y=[signal["level"], signal["level"]],
                        mode="lines+text",
                        text=[signal["type"], ""],
                        textposition=("bottom center" if signal["bias"] == BULLISH else "top center"),
                        line=dict(color=color, dash="solid"),
                        name=signal["type"],
                        showlegend=False,
                    )
                )

        last_ts = ohlc.index[-1].strftime("%Y-%m-%d %H:%M")
        fig.update_layout(
            title=f"{symbol} | Structure {self.main_tf} | {last_ts}",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=30)]),
        )
        return fig
