import uuid
# Import constants
from datetime import datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction
from daily_trading.settings import logger


class Backtester:
    """
    THE ACCOUNTANT.

    Role:
    1. Feeds Data (STF/MTF).
    2. Records Signals emitted by the Strategy.
    3. Calculates PnL based on Strategy's reported prices.

    It contains NO trading logic. It trusts the Strategy implicitly.
    """

    def __init__(self, strategy, timezone: ZoneInfo):
        self.strategy = strategy
        self.timezone = timezone
        self.journal: List[dict] = []  # Completed trade records
        self.active_record: Optional[dict] = None  # Currently open/pending ledger entry

    def run(self,
            ohlc_5m: pd.DataFrame,
            ohlc_1h: Optional[pd.DataFrame] = None,
            ohlc_4h: Optional[pd.DataFrame] = None,  # <--- ADDED
            lookback_5m: int = 200,
            lookback_1h: int = 200,
            lookback_4h: int = 200):

        # 1. SETUP
        mode = "MTF" if ohlc_1h is not None else "STF"
        logger.info(f"🏁 {mode} Backtest | {self.strategy.symbol}")

        # Standardize Data
        ohlc_5m = self._prepare_data(ohlc_5m)
        if ohlc_1h is not None:
            ohlc_1h = self._prepare_data(ohlc_1h)

        if ohlc_4h is not None:
            ohlc_4h = self._prepare_data(ohlc_4h)

        self.strategy.reset_session_state()
        self.journal = []
        self.active_record = None

        # Session Times for pre-filtering (Optimization)
        s_info = SESSION_MAP[self.timezone.key]
        sess_start = time(s_info["start_hour"], s_info["start_minute"])
        sess_end = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))
        pre_market = (datetime.combine(datetime.min, sess_start) - timedelta(hours=1)).time()

        start_idx = max(lookback_5m, 1)

        # 2. MAIN DATA FEED LOOP
        for i in range(start_idx, len(ohlc_5m)):
            ts_utc = ohlc_5m.index[i]

            # A. Session Filter (Skip overnight processing to save speed)
            # Note: Strategy also handles cutoffs, this is just to skip "dead" data
            t_local = ts_utc.astimezone(self.timezone).time()
            is_active = False
            if pre_market < sess_end:
                if pre_market <= t_local <= sess_end: is_active = True
            else:  # Overnight session (Asia)
                if t_local >= pre_market or t_local <= sess_end: is_active = True

            if not is_active: continue

            # B. Prepare Windows
            window_5m = ohlc_5m.iloc[i - lookback_5m: i + 1]

            # C. Execute Strategy
            if mode == "MTF":
                # Strict lookahead protection: 1H candle must be closed before 5m time
                window_1h = ohlc_1h[ohlc_1h.index < ts_utc].tail(lookback_1h)
                if len(window_1h) < 8:
                    continue
                # Prepare Arguments
                strategy_args = {'df_ltf': window_5m, 'df_htf': window_1h}

                # 4H Slicing Logic
                if ohlc_4h is not None:
                    # Strict lookahead protection: 4H candle must be closed before 5m time
                    window_4h = ohlc_4h[ohlc_4h.index < ts_utc].tail(lookback_4h)

                    # Ensure we have enough 4H data (at least 2 bars for trend calc)
                    if len(window_4h) < 2:
                        continue

                    strategy_args['df_4h'] = window_4h
                # -----------------------------
                analysis = self.strategy.analyze(**strategy_args)
            else:
                analysis = self.strategy.analyze(window_5m)

            # 3. RECORD KEEPING (The Accountant)
            if analysis.trade_signal:
                self._process_signal(analysis.trade_signal, ts_utc)

        return pd.DataFrame(self.journal)

    def _process_signal(self, sig, ts_utc):
        """Updates the Ledger based on Strategy Signals."""

        # --- NEW TRADE ---
        if sig.action == TradeAction.ENTER:
            if self.active_record is None:
                # Create new ledger entry
                self.active_record = {
                    "timestamp": ts_utc,
                    "trade_id": str(uuid.uuid4())[:8],  # Short unique ID
                    "direction": sig.direction.name,
                    "entry_time": ts_utc,
                    "entry_price": sig.price,
                    "sl": sig.sl,
                    "tp": sig.tp,
                    "result": None,
                    "metadata": sig.metadata,
                    "size": sig.size,
                    "exit_time": None,
                    "exit_price": None,
                    "pnl": 0.0,
                    "comment": None,
                    "status": "OPEN",
                    "symbol": self.strategy.symbol,
                }

        # --- UPDATE ORDER ---
        # (Used in Confirmation Strategies to cancel/replace pending orders)
        elif sig.action == TradeAction.UPDATE and self.active_record:
            # Just update the price/sl/tp/size of the active pending record
            self.active_record.update({
                "entry_price": sig.price,
                "sl": sig.sl,
                "tp": sig.tp,
                "size": sig.size,
                "metadata": sig.metadata  # Update metadata (e.g., new FVG ID)
            })

        # --- EXIT TRADE ---
        elif sig.action == TradeAction.EXIT and self.active_record:
            # Calculate PnL
            exit_price = sig.price
            entry_price = self.active_record["entry_price"]
            size = self.active_record["size"]
            direction = self.active_record["direction"]
            point_value = getattr(self.strategy, "point_value", 1.0)

            # Logic provided by Strategy (WIN/LOSS tag)
            result_tag = getattr(sig, 'trade_result', 'UNKNOWN')

            if direction == "LONG":
                pnl = (exit_price - entry_price) * size * point_value
            else:
                pnl = (entry_price - exit_price) * size * point_value

            self.active_record.update({
                "status": "CLOSED",
                "exit_time": ts_utc,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "result": result_tag,
                "comment": "Strategy Exit"
            })

            # Commit to Journal and Clear Active
            self.journal.append(self.active_record)
            self.active_record = None

        # --- CANCEL ORDER ---
        elif sig.action == TradeAction.CANCEL and self.active_record:
            self.active_record.update({
                "status": "CANCELLED",
                "exit_time": ts_utc,
                "exit_price": 0.0,
                "pnl": 0.0,
                "result": "CANCELLED",
                "comment": "Signal Cancelled"
            })
            self.journal.append(self.active_record)
            self.active_record = None

    @staticmethod
    def _prepare_data(df):
        """Helper to ensure UTC index."""
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    @staticmethod
    def load_csv(file_path: str, symbol: str) -> pd.DataFrame:
        """Utility for loading data."""
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'

        # Robust Time Parsing
        if df[time_col].dtype in [np.int64, np.float64]:
            df['timestamp'] = pd.to_datetime(df[time_col], unit='s', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df[time_col], utc=True)

        df = df.set_index('timestamp').sort_index()
        df['symbol'] = symbol
        return df[~df.index.duplicated(keep='first')]

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
            view_1h = ohlc_1h[ohlc_1h.index <= trade["timestamp"]].tail(100)
            swing_1h, internal_1h = self.strategy.sd_detector_1h.calculate(view_1h)
            for z in (swing_1h + internal_1h):
                color = "rgba(0, 200, 0, 0.15)" if z['bias'] == BIAS.LONG else "rgba(200, 0, 0, 0.15)"
                fig.add_shape(type="rect", x0=view_5m.index[0], x1=view_5m.index[-1],
                              y0=z['bottom'], y1=z['top'], fillcolor=color, line_width=0, layer="below")

        # 4. Standard Trade Boxes
        entry_ts = trade["timestamp"]
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
