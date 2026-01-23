from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.DetectorsBase import DetectorsBase


class SupplyDemandZonesDetector(DetectorsBase):
    BULLISH = BIAS.LONG
    BEARISH = BIAS.SHORT

    def __init__(self, main_tf="5m", swing_length: int = 50, internal_length: int = 5, atr_length: int = 200,
                 zone_precision_factor: float = 2.0, volatility_filter: str = "atr", max_display_zones: int = 5,
                 plot_tail: int = 300, fvg_based_only: bool = True, fvg_offset: int = 3, fvg_threshold=0.15,
                 wick_ratio: float = 0.10, internal_confluence_filter: bool = True):
        super().__init__(main_tf, plot_tail, wick_ratio=wick_ratio)
        self.swing_length = swing_length
        self.internal_length = internal_length
        self.atr_length = atr_length
        self.atr_multiplier = zone_precision_factor
        self.volatility_filter = volatility_filter.lower()
        self.max_display_zones = max_display_zones
        self.plot_tail = plot_tail
        self.fvg_based_only = fvg_based_only
        self.fvg_offset = fvg_offset
        self.fvg_threshold = fvg_threshold
        self.zone_timeframe = None
        self.internal_confluence_filter = internal_confluence_filter

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        if self.volatility_filter == 'atr':
            return tr.rolling(self.atr_length).mean().bfill()
        return tr.expanding().mean().fillna(0)

    def calculate(self, ohlc: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        self.zone_timeframe = self.get_time_frame_name(ohlc)
        df = ohlc.copy()
        n = len(df)
        if n < self.swing_length * 2:
            return [], []

        # Volatility parsing
        vol_measure = self._calculate_volatility(df)
        high_vol = (df['high'] - df['low']) >= (self.atr_multiplier * vol_measure)
        self.parsed_highs = np.where(high_vol, df["low"], df["high"])
        self.parsed_lows = np.where(high_vol, df["high"], df["low"])

        self.raw_highs = df["high"].values
        self.raw_lows = df["low"].values
        self.raw_closes = df["close"].values
        self.raw_opens = df["open"].values
        self.df_index = df.index

        # ATR for FVG Threshold
        tr = pd.Series(self.raw_highs - self.raw_lows)
        self.atr_series = tr.rolling(20).mean().bfill().values

        # Run logic
        swing_zones = self._find_zones(self.swing_length, use_confluence=False)
        internal_zones = self._find_zones(self.internal_length, use_confluence=self.internal_confluence_filter)

        return swing_zones[:self.max_display_zones], internal_zones[:self.max_display_zones]

    def _find_zones(self, length: int, use_confluence=False) -> List[Dict]:
        """Finds zones using a robust pivot lookback/lookahead window."""
        zones = []
        n = len(self.raw_closes)
        p_high = {"price": None, "index": 0, "crossed": False}
        p_low = {"price": None, "index": 0, "crossed": False}

        # Start at 2*length to ensure a full window is available for the first pivot check
        for i in range(length * 2, n):
            idx = i - length

            # Pivot High check
            window_h = self.raw_highs[idx - length: idx + length + 1]
            if len(window_h) > 0 and self.raw_highs[idx] == np.max(window_h):
                p_high = {"price": self.raw_highs[idx], "index": idx, "crossed": False}

            # Pivot Low check
            window_l = self.raw_lows[idx - length: idx + length + 1]
            if len(window_l) > 0 and self.raw_lows[idx] == np.min(window_l):
                p_low = {"price": self.raw_lows[idx], "index": idx, "crossed": False}

            # --- PRE-CALCULATE CONFLUENCE (Wick Logic) ---
            # Replicating Pine lines 431-432:
            # bullishBar := high - max(close, open) > min(close, open) - low
            valid_bullish_confluence = True
            valid_bearish_confluence = True

            if use_confluence:
                open_price = self.raw_opens[i]
                close_price = self.raw_closes[i]
                high_price = self.raw_highs[i]
                low_price = self.raw_lows[i]

                upper_wick = high_price - max(close_price, open_price)
                lower_wick = min(close_price, open_price) - low_price

                # Pine Script Logic:
                # Bullish BOS requires Upper Wick > Lower Wick
                # Bearish BOS requires Upper Wick < Lower Wick
                valid_bullish_confluence = upper_wick > lower_wick
                valid_bearish_confluence = upper_wick < lower_wick

            # Breaks
            close_i = self.raw_closes[i]
            if p_high["price"] and close_i > p_high["price"] and not p_high["crossed"]:
                p_high["crossed"] = True
                ob = self._store_ob(p_high["index"], i, self.BULLISH)
                if ob: zones.insert(0, ob)

            if p_low["price"] and close_i < p_low["price"] and not p_low["crossed"]:
                p_low["crossed"] = True
                ob = self._store_ob(p_low["index"], i, self.BEARISH)
                if ob: zones.insert(0, ob)

            self._mitigate(zones, i)
        return zones

    def _store_ob(self, pivot_idx: int, break_idx: int, bias: BIAS) -> Optional[Dict]:
        """Finds the OB candle and performs SUSTAINED displacement check."""
        if bias == self.BULLISH:
            seg = self.parsed_lows[pivot_idx: break_idx]
            actual_idx = pivot_idx + np.argmin(seg)
        else:
            seg = self.parsed_highs[pivot_idx: break_idx]
            actual_idx = pivot_idx + np.argmax(seg)

        top, bottom = float(self.raw_highs[actual_idx]), float(self.raw_lows[actual_idx])
        threshold = self.atr_series[break_idx] * self.fvg_threshold

        # --- SUSTAINED FVG DISPLACEMENT CHECK ---
        if self.fvg_based_only:
            fvg_confirmed = False
            # Scan offsets within the OB candle for FVG
            for offset in range(self.fvg_offset):
                i0 = actual_idx + offset
                i2 = i0 + 2
                if i2 >= len(self.raw_closes): break

                if bias == self.BULLISH:
                    gap = self.raw_lows[i2] > (self.raw_highs[i0] + threshold)
                    if gap:
                        # Persistence: Ensure next 3 bars don't close the gap
                        is_filled = False
                        for k in range(i2 + 1, min(i2 + 4, len(self.raw_closes))):
                            if self.raw_lows[k] <= self.raw_highs[i0]:
                                is_filled = True
                                break
                        if not is_filled:
                            fvg_confirmed = True
                            break
                else:
                    gap = self.raw_highs[i2] < (self.raw_lows[i0] - threshold)
                    if gap:
                        is_filled = False
                        for k in range(i2 + 1, min(i2 + 4, len(self.raw_closes))):
                            if self.raw_highs[k] >= self.raw_lows[i0]:
                                is_filled = True;
                                break
                        if not is_filled:
                            fvg_confirmed = True
                            break

            if not fvg_confirmed: return None

        # Pre-break Mitigation Check (Matches Pine Script 'crossed' logic)
        for j in range(actual_idx + 1, break_idx):
            if bias == self.BULLISH and self.raw_closes[j] < bottom: return None
            if bias == self.BEARISH and self.raw_closes[j] > top: return None

        return {"start_time": self.df_index[actual_idx],
                "top": top,
                "bottom": bottom,
                "bias": bias,
                "active": True,
                "timeframe": self.zone_timeframe}

    def _mitigate(self, zones: List[Dict], current_idx: int):
        close, high, low = self.raw_closes[current_idx], self.raw_highs[current_idx], self.raw_lows[current_idx]
        i = 0
        while i < len(zones):
            z = zones[i]
            invalidated = False
            if z["bias"] == self.BEARISH:  # Supply
                source = high
                if source > z["top"]: invalidated = True
            else:  # Demand
                source = low
                if source < z["bottom"]: invalidated = True

            if invalidated:
                zones.pop(i)
            else:
                i += 1

    def plot(self, ohlc: pd.DataFrame) -> go.Figure:
        active_swing, active_internal = self.calculate(ohlc)
        plot_df = ohlc.tail(self.plot_tail)
        fig = go.Figure(data=[
            go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'],
                           close=plot_df['close'])])
        for zones in [active_swing, active_internal]:
            for z in zones:
                if z["start_time"] not in ohlc.index: continue
                color = "rgba(33, 87, 243, 0.1)" if z["bias"] == self.BULLISH else "rgba(242, 54, 69, 0.1)"
                fig.add_shape(type="rect", x0=z["start_time"], x1=ohlc.index[-1], y0=z["bottom"], y1=z["top"],
                              fillcolor=color, line_width=0, layer="below")
        fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False)
        return fig
