from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction
from daily_trading.settings import logger


class ORBBacktester:
    """
    Institutional Backtester for ORB SMC Strategy.
    - DST-Aware: Programmatically determines session times regardless of time of year.
    - Performance Optimized: Skips overnight data.
    - QA Ready: Tracks SL % and CANCELLATIONS.
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

    def run(self, ohlc: pd.DataFrame, bar_range: Optional[Tuple[int, int]] = None):
        # 1. Setup & Logging
        if bar_range is not None:
            start_i, end_i = bar_range
            ohlc = ohlc.iloc[start_i:end_i].copy()
            logger.warning(f"⚠️ Backtest restricted to bars [{start_i}:{end_i}]")

        # 2. Date Handling
        if ohlc.index.tz is None:
            ohlc.index = ohlc.index.tz_localize("UTC")

        start_dt = ohlc.index.min().strftime('%Y-%m-%d')
        end_dt = ohlc.index.max().strftime('%Y-%m-%d')
        logger.info(f"Backtest Started | Symbol: {self.strategy.symbol} | Range: {start_dt} to {end_dt}")

        self.strategy.reset_session_state()
        self.trades = []
        self.pending_trade = None

        # 3. Session Constraints
        s_info = SESSION_MAP[self.timezone.key]
        sess_start_wall = time(s_info["start_hour"], s_info["start_minute"])
        sess_end_wall = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))
        dummy_dt = datetime.combine(datetime.min, sess_start_wall) - timedelta(hours=1)
        pre_market_wall = dummy_dt.time()

        # --- WINDOW CONFIGURATION ---
        # 200 is correct for ORB/Fresh Zone strategies.
        window_size = 1000

        # Start the loop at window_size to ensure we always have 200 bars of data
        loop_start = window_size

        for i in range(loop_start, len(ohlc)):
            ts_utc = ohlc.index[i]
            ts_local = ts_utc.astimezone(self.timezone)
            ts_local_time = ts_local.time()

            # --- DST-AWARE SESSION CHECK ---
            is_interesting = False
            if pre_market_wall < sess_end_wall:
                if pre_market_wall <= ts_local_time <= sess_end_wall: is_interesting = True
            else:
                if ts_local_time >= pre_market_wall or ts_local_time <= sess_end_wall: is_interesting = True

            if not is_interesting: continue

            # --- DEFINE WINDOW (Fixed NameError) ---
            # We calculate start_idx based on the fixed window_size
            start_idx = i - window_size
            curr_bar = ohlc.iloc[i]

            # Slice exactly 200 bars ending at i
            window = ohlc.iloc[start_idx: i + 1]

            # Run Strategy
            result = self.strategy.analyze(window)

            # --- FILL DETECTION ---
            if self.pending_trade and self.strategy.trade_status.name == "FILLED":
                if self.pending_trade["status"] == "PENDING":
                    self.pending_trade["status"] = "FILLED"
                    self.pending_trade["fill_time"] = ts_utc

            # --- SIGNAL PROCESSING ---
            if result.trade_signal:
                sig = result.trade_signal

                # 1. ENTER
                if sig.action == TradeAction.ENTER:
                    sl_dist = abs(sig.price - sig.sl)
                    sl_pct = (sl_dist / sig.price) * 100 if sig.price != 0 else 0

                    self.pending_trade = {
                        "timestamp": sig.timestamp,
                        "utc_signal": ts_utc,
                        "window_indices": (start_idx, i),
                        "direction": sig.direction.name,
                        "entry": sig.price,
                        "sl": sig.sl,
                        "sl_pct": round(sl_pct, 4),
                        "tp": sig.tp,
                        "size": sig.size,
                        "status": "PENDING",
                        "metadata": sig.metadata,
                        "fig": sig.fig,
                        "comments": "Has not been filled",
                        "fill_time": None
                    }

                # 2. CANCEL
                elif sig.action == TradeAction.CANCEL and self.pending_trade:
                    t_rec = self.pending_trade
                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": curr_bar["close"],
                        "result": "CANCELLED",
                        "pnl": 0.0,
                        "status": "CLOSED",
                        "comments": "Signal Cancelled"
                    })
                    self.trades.append(t_rec)
                    self.pending_trade = None

                # 3. EXIT (With Gap Fix & Immediate Fill Fix)
                elif sig.action == TradeAction.EXIT and self.pending_trade:
                    t_rec = self.pending_trade

                    # FIX: If trade filled and exited in same bar, record fill time
                    if t_rec["fill_time"] is None:
                        t_rec["fill_time"] = ts_utc
                        t_rec["status"] = "FILLED"

                    is_long = t_rec["direction"] == "LONG"

                    # --- GAP / SLIPPAGE LOGIC ---
                    bar_open = curr_bar["open"]
                    bar_close = curr_bar["close"]

                    # Determine hits (using epsilon for float precision)
                    tp_hit = curr_bar["high"] >= (t_rec["tp"] - 0.0001) if is_long else curr_bar["low"] <= (
                            t_rec["tp"] + 0.0001)
                    sl_hit = curr_bar["low"] <= (t_rec["sl"] + 0.0001) if is_long else curr_bar["high"] >= (
                            t_rec["sl"] - 0.0001)

                    final_res = "WIN"  # Default

                    if tp_hit:
                        final_res = "WIN"
                        exit_comment = "Hard TP Hit"
                        # Gap Benefit: If market opens past TP, take the better price
                        if is_long:
                            final_price = max(bar_open, t_rec["tp"]) if bar_open > t_rec["tp"] else t_rec["tp"]
                        else:
                            final_price = min(bar_open, t_rec["tp"]) if bar_open < t_rec["tp"] else t_rec["tp"]

                    elif sl_hit:
                        final_res = "LOSS"
                        exit_comment = "Hard SL Hit"
                        # Gap Penalty: If market opens past SL, fill at Open (Slippage)
                        if is_long:
                            final_price = min(bar_open, t_rec["sl"]) if bar_open < t_rec["sl"] else t_rec["sl"]
                        else:
                            final_price = max(bar_open, t_rec["sl"]) if bar_open > t_rec["sl"] else t_rec["sl"]

                    else:
                        # Session Cutoff or Manual Exit
                        final_price = bar_close
                        is_win = (final_price > t_rec["entry"]) if is_long else (final_price < t_rec["entry"])
                        final_res = "WIN" if is_win else "LOSS"
                        exit_comment = "Session Cutoff (Partial)"

                    # Calc PnL
                    pv = self.strategy.POINT_VALUE[self.strategy.symbol]
                    if is_long:
                        dollar_pnl = (final_price - t_rec["entry"]) * t_rec["size"] * pv
                    else:
                        dollar_pnl = (t_rec["entry"] - final_price) * t_rec["size"] * pv

                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": final_price,
                        "result": final_res,
                        "pnl": round(dollar_pnl, 2),
                        "status": "CLOSED",
                        "comments": exit_comment,
                    })
                    self.trades.append(t_rec)
                    self.pending_trade = None

        return pd.DataFrame(self.trades)

    def plot_trade(self, trade_idx: int, ohlc: pd.DataFrame) -> Optional[go.Figure]:
        """Generates Plot on-demand with 50 bars of future context."""
        if trade_idx >= len(self.trades):
            logger.error("Trade index out of range.")
            return None

        trade = self.trades[trade_idx]

        if "window_indices" not in trade:
            logger.error("Missing window_indices. Run backtest with updated code.")
            return None

        # 1. Window Calculation
        start_idx, signal_idx = trade["window_indices"]
        if trade.get("exit_time") and trade["exit_time"] in ohlc.index:
            exit_idx = ohlc.index.get_loc(trade["exit_time"])
            plot_end_idx = min(len(ohlc) - 1, exit_idx + 50)
        else:
            plot_end_idx = min(len(ohlc) - 1, signal_idx + 50)

        window = ohlc.iloc[start_idx: plot_end_idx + 1]

        # 2. Base Strategy Plot
        fig = self.strategy.sd_detector.plot(window)

        # 3. Execution Visuals
        entry_ts = trade["utc_signal"]
        exit_ts = trade.get("exit_time", window.index[-1])

        # Risk/Reward Boxes
        fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts, y0=trade["entry"], y1=trade["tp"],
                      fillcolor="rgba(0, 255, 100, 0.15)", line=dict(width=1, color="green"), layer="below")
        fig.add_shape(type="rect", x0=entry_ts, x1=exit_ts, y0=trade["entry"], y1=trade["sl"],
                      fillcolor="rgba(255, 50, 50, 0.15)", line=dict(width=1, color="red"), layer="below")

        # Result Annotation
        result_color = "green" if trade["result"] == "WIN" else "red"
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,  # Left side
            y=0.99,  # Top
            xanchor="left",
            yanchor="top",
            text=f"<b>{trade['result']}</b><br>PnL: {trade['size']} units",
            showarrow=False,
            bgcolor=result_color,
            font=dict(color="white", size=13),
            bordercolor="white",
            borderwidth=2,
            borderpad=6,
            opacity=0.95,
        )

        fig.update_layout(xaxis_rangeslider_visible=False, height=700)
        return fig
