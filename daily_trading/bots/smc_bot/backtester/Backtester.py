from datetime import datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction, TradeSignal, TradeStatus
from daily_trading.settings import logger


class Backtester:
    """
    Unified Backtester for Single-TF (ORB) and Multi-TF (SupplyDemand) Strategies.

    Features:
    - Auto-detects mode based on provided data.
    - Robust Exchange Simulation (Gaps, Slippage, Independent Fills).
    - Phantom Fill Protection.
    """

    def __init__(self, strategy_instance, timezone: ZoneInfo, plot_tail: int = 300):
        self.strategy = strategy_instance
        self.timezone = timezone
        self.plot_tail = plot_tail
        self.trades: List[dict] = []
        self.pending_trade: Optional[dict] = None

    @staticmethod
    def load_csv(file_path: str, symbol: str, csv_tz: str = "UTC") -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        if df[time_col].dtype in [np.int64, np.float64]:
            df['timestamp'] = pd.to_datetime(df[time_col], unit='s', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df[time_col]).dt.tz_localize(csv_tz).dt.tz_convert("UTC")
        df = df.set_index('timestamp').sort_index()
        df['symbol'] = symbol
        return df[~df.index.duplicated(keep='first')]

    @staticmethod
    def is_win(trade: dict, close_price: float) -> bool:
        if trade["direction"] == "LONG":
            return close_price > trade["entry"]
        return close_price < trade["entry"]

    def run(self,
            ohlc_5m: pd.DataFrame,
            ohlc_1h: Optional[pd.DataFrame] = None,
            lookback_5m: int = 200,
            lookback_1h: int = 200):
        """
        Universal Run Loop.
        - If ohlc_1h is passed -> MTF Mode (passes 2 DFs to strategy).
        - If ohlc_1h is None   -> STF Mode (passes 1 DF to strategy).
        """
        mode = "MTF" if ohlc_1h is not None else "STF"
        logger.info(f"{mode} Backtest Started | Symbol: {self.strategy.symbol}")

        # 1. Setup Data
        ohlc_5m = ohlc_5m.sort_index()
        if ohlc_5m.index.tz is None: ohlc_5m.index = ohlc_5m.index.tz_localize("UTC")

        if ohlc_1h is not None:
            ohlc_1h = ohlc_1h.sort_index()
            if ohlc_1h.index.tz is None: ohlc_1h.index = ohlc_1h.index.tz_localize("UTC")

        self.strategy.reset_session_state()
        self.trades, self.pending_trade = [], None

        # 2. Session Setup
        s_info = SESSION_MAP[self.timezone.key]
        sess_start_wall = time(s_info["start_hour"], s_info["start_minute"])
        sess_end_wall = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))
        dummy_dt = datetime.combine(datetime.min, sess_start_wall) - timedelta(hours=1)
        pre_market_wall = dummy_dt.time()

        start_idx = max(lookback_5m, 1)

        # MAIN LOOP
        for i in range(start_idx, len(ohlc_5m)):
            ts_utc = ohlc_5m.index[i]
            ts_local = ts_utc.astimezone(self.timezone)
            ts_local_time = ts_local.time()

            # Session Filter
            is_interesting = False
            if pre_market_wall < sess_end_wall:
                if pre_market_wall <= ts_local_time <= sess_end_wall: is_interesting = True
            else:
                if ts_local_time >= pre_market_wall or ts_local_time <= sess_end_wall: is_interesting = True
            if not is_interesting: continue

            # 3. Prepare Windows
            curr_bar = ohlc_5m.iloc[i]
            window_5m = ohlc_5m.iloc[i - lookback_5m: i + 1]

            # 4. Strategy Analysis (Generic Dispatch)
            if mode == "MTF":
                # Slice 1H Context: Bars closed BEFORE current 5m time
                window_1h = ohlc_1h[ohlc_1h.index < ts_utc].tail(lookback_1h)
                if len(window_1h) < 8: continue  # Not enough context
                result = self.strategy.analyze(window_5m, window_1h)
            else:
                # Single Timeframe (ORB)
                result = self.strategy.analyze(window_5m)

            # ====================================================================
            # PHASE 1: EXCHANGE SIMULATOR (Fill Validation)
            # ====================================================================
            if self.pending_trade and self.pending_trade["status"] == "PENDING":
                entry_price = self.pending_trade["entry"]
                is_long = self.pending_trade["direction"] == "LONG"

                filled = False
                if is_long:
                    if curr_bar["low"] <= entry_price: filled = True
                else:
                    if curr_bar["high"] >= entry_price: filled = True

                if filled:
                    self.pending_trade["status"] = "FILLED"
                    self.pending_trade["fill_time"] = ts_utc
                    self.strategy.trade_status = TradeStatus.FILLED

            # ====================================================================
            # PHASE 2: EXCHANGE SIMULATOR (Auto TP/SL Exit)
            # ====================================================================
            if self.pending_trade and self.pending_trade["status"] == "FILLED":
                t_rec = self.pending_trade
                is_long = t_rec["direction"] == "LONG"
                sl_price, tp_price = t_rec["sl"], t_rec["tp"]

                bar_open, b_high, b_low = curr_bar["open"], curr_bar["high"], curr_bar["low"]

                exit_triggered = False
                exit_price, res, comment = 0.0, "", ""

                # Gap Check
                if is_long:
                    if bar_open <= sl_price:
                        exit_triggered, exit_price, res, comment = True, bar_open, "LOSS", "Gap SL"
                    elif bar_open >= tp_price:
                        exit_triggered, exit_price, res, comment = True, bar_open, "WIN", "Gap TP"
                else:
                    if bar_open >= sl_price:
                        exit_triggered, exit_price, res, comment = True, bar_open, "LOSS", "Gap SL"
                    elif bar_open <= tp_price:
                        exit_triggered, exit_price, res, comment = True, bar_open, "WIN", "Gap TP"

                # Touch Check
                if not exit_triggered:
                    if is_long:
                        if b_low <= sl_price:
                            exit_triggered, exit_price, res, comment = True, sl_price, "LOSS", "Hard SL Hit"
                        elif b_high >= tp_price:
                            exit_triggered, exit_price, res, comment = True, tp_price, "WIN", "Hard TP Hit"
                    else:
                        if b_high >= sl_price:
                            exit_triggered, exit_price, res, comment = True, sl_price, "LOSS", "Hard SL Hit"
                        elif b_low <= tp_price:
                            exit_triggered, exit_price, res, comment = True, tp_price, "WIN", "Hard TP Hit"

                if exit_triggered:
                    self._calculate_and_close(t_rec, exit_price, ts_utc, res, comment)
                    self.trades.append(t_rec)
                    self.pending_trade = None
                    self.strategy.trade_status = TradeStatus.NONE
                    continue

            # ====================================================================
            # PHASE 3: SIGNAL PROCESSING (Enter/Cancel/Exit)
            # ====================================================================
            if not result.trade_signal: continue
            sig: TradeSignal = result.trade_signal

            # 1. ENTER
            if sig.action == TradeAction.ENTER:
                if self.pending_trade and self.pending_trade["status"] == "FILLED":
                    continue

                self.pending_trade = {
                    "timestamp": sig.timestamp,
                    "utc_signal": ts_utc,
                    "window_indices_5m": (i - lookback_5m, i),
                    "direction": sig.direction.name,
                    "entry": sig.price,
                    "sl": sig.sl,
                    "tp": sig.tp,
                    "size": sig.size,
                    "status": "PENDING",
                    "metadata": sig.metadata,
                    # Optional: Store raw params if needed for debugging
                    "sl_atr": sig.metadata.get("sl_atr", 0) if sig.metadata else 0,
                    "fill_time": None,
                    "comments": ""
                }

            # 2. CANCEL
            elif sig.action == TradeAction.CANCEL and self.pending_trade:
                t_rec = self.pending_trade
                if t_rec["status"] == "PENDING":
                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": 0.0,
                        "result": "CANCELLED",
                        "pnl": 0.0,
                        "status": "CLOSED",
                        "comments": "Signal Cancelled"
                    })
                else:
                    self._calculate_and_close(t_rec, curr_bar["close"], ts_utc, None, "Cancelled After Fill")

                self.trades.append(t_rec)
                self.pending_trade = None
                self.strategy.trade_status = TradeStatus.NONE

            # 3. EXIT (Manual/Session)
            elif sig.action == TradeAction.EXIT and self.pending_trade:
                t_rec = self.pending_trade
                if t_rec["status"] == "PENDING":
                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": 0.0,
                        "result": "CANCELLED",
                        "pnl": 0.0,
                        "status": "CLOSED",
                        "comments": "Session Cutoff (Never Filled)"
                    })
                else:
                    price = curr_bar["close"]
                    t_rec.update({"entry_time": sig.entry_timestamp})
                    res = "WIN" if self.is_win(t_rec, price) else "LOSS"
                    self._calculate_and_close(t_rec, price, ts_utc, res, "Session Cutoff / Manual Exit")

                self.trades.append(t_rec)
                self.pending_trade = None
                self.strategy.trade_status = TradeStatus.NONE

            # 4. UPDATE (Cancel Old + Place New)
            elif sig.action == TradeAction.UPDATE:
                if self.pending_trade:
                    # A. "Cancel" the old trade stats
                    t_rec = self.pending_trade
                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": 0.0,
                        "result": "CANCELLED",
                        "pnl": 0.0,
                        "status": "CLOSED",
                        "comments": "Replaced by Better Setup"
                    })
                    self.trades.append(t_rec)

                # B. Place the New Pending Order
                self.pending_trade = {
                    "timestamp": sig.timestamp,
                    "utc_signal": ts_utc,
                    "window_indices_5m": (i - lookback_5m, i),
                    "direction": sig.direction.name,
                    "entry": sig.price,
                    "sl": sig.sl,
                    "tp": sig.tp,
                    "size": sig.size,
                    "status": "PENDING",
                    "metadata": sig.metadata,
                    "sl_atr": sig.metadata.get("sl_atr", 0) if sig.metadata else 0,
                    "fill_time": None,
                    "comments": ""
                }
                # Strategy status remains PENDING
                self.strategy.trade_status = TradeStatus.PENDING

        return pd.DataFrame(self.trades)

    def _calculate_and_close(self, t_rec, exit_price, exit_time, result, comment):
        """Helper to close trade and calc PnL."""
        is_long = t_rec["direction"] == "LONG"
        # Handle cases where strategy might not expose POINT_VALUE directly (fallback to 1 if missing)
        pv = getattr(self.strategy, "POINT_VALUE", {}).get(self.strategy.symbol,
                                                           getattr(self.strategy, "point_value", 1.0))

        if is_long:
            dollar_pnl = (exit_price - t_rec["entry"]) * t_rec["size"] * pv
        else:
            dollar_pnl = (t_rec["entry"] - exit_price) * t_rec["size"] * pv

        t_rec.update({
            "exit_time": exit_time,
            "exit_price": exit_price,
            "result": result,
            "pnl": round(dollar_pnl, 2),
            "status": "CLOSED",
            "comments": comment
        })

    def plot_trade(self, trade_idx: int, ohlc_5m: pd.DataFrame, ohlc_1h: Optional[pd.DataFrame] = None) -> Optional[
        go.Figure]:
        """
        Generic Plotter.
        Detects if strategy has its own plotting logic (like ORB), otherwise draws generic boxes.
        """
        if trade_idx >= len(self.trades): return None
        trade = self.trades[trade_idx]
        if trade["status"] == "CANCELLED": return None

        # 1. Window Indices
        start_5m, signal_5m = trade["window_indices_5m"]
        exit_ts = trade.get("exit_time")

        if exit_ts and exit_ts in ohlc_5m.index:
            end_idx = ohlc_5m.index.get_loc(exit_ts) + 20
        else:
            end_idx = signal_5m + 50

        end_idx = min(len(ohlc_5m) - 1, end_idx)
        view_5m = ohlc_5m.iloc[start_5m: end_idx + 1]

        # 2. Base Figure
        # Attempt to use Strategy's specific plotter first (for ORB lines etc)
        if hasattr(self.strategy, "orb_detector") and hasattr(self.strategy.orb_detector, "plot"):
            # For ORB, we can just pass the window to its plotter
            fig = self.strategy.orb_detector.plot(view_5m)
        else:
            # Default Generic Plot
            fig = go.Figure(data=[go.Candlestick(
                x=view_5m.index, open=view_5m['open'], high=view_5m['high'],
                low=view_5m['low'], close=view_5m['close'], name="Price"
            )])

        # 3. Supply/Demand Specific Overlays (If 1H data provided)
        if ohlc_1h is not None and hasattr(self.strategy, "sd_detector_1h"):
            view_1h = ohlc_1h[ohlc_1h.index <= trade["utc_signal"]].tail(100)
            swing_1h, internal_1h = self.strategy.sd_detector_1h.calculate(view_1h)
            for z in (swing_1h + internal_1h):
                color = "rgba(0, 200, 0, 0.15)" if z['bias'] == BIAS.LONG else "rgba(200, 0, 0, 0.15)"
                fig.add_shape(type="rect", x0=view_5m.index[0], x1=view_5m.index[-1],
                              y0=z['bottom'], y1=z['top'], fillcolor=color, line_width=0, layer="below")

        # 4. Standard Trade Boxes
        entry_ts = trade["utc_signal"]
        exit_ts = trade.get("exit_time", view_5m.index[-1])
        is_win = trade["result"] == "WIN"

        # TP Box
        fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts, y0=trade["entry"], y1=trade["tp"],
                      fillcolor="rgba(0, 255, 100, 0.15)", line=dict(width=1, color="green"), layer="below")
        # SL Box
        fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts, y0=trade["entry"], y1=trade["sl"],
                      fillcolor="rgba(255, 50, 50, 0.15)", line=dict(width=1, color="red"), layer="below")

        # Annotation
        res_color = "green" if is_win else "red"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
            text=f"<b>{trade['result']}</b><br>PnL: ${trade['pnl']}",
            bgcolor=res_color, font=dict(color="white"), borderpad=5
        )

        fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_white")
        return fig
