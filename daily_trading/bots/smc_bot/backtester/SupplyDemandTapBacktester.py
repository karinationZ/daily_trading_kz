from datetime import datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.StrategyBase import BIAS
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction, TradeStatus
from daily_trading.settings import logger


class SupplyDemandBacktester:
    """
    MTF Backtester for SupplyDemandZoneTapStrategy.
    Trusts Strategy for State (Pending/Filled), handles Accounting (PnL/Slippage).
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

    def run(self, ohlc_5m: pd.DataFrame, ohlc_1h: pd.DataFrame, lookback_5m: int = 200, lookback_1h: int = 200):
        logger.info(f"MTF Backtest Started | Symbol: {self.strategy.symbol}")

        # 1. Setup
        ohlc_5m = ohlc_5m.sort_index()
        ohlc_1h = ohlc_1h.sort_index()
        if ohlc_5m.index.tz is None: ohlc_5m.index = ohlc_5m.index.tz_localize("UTC")
        if ohlc_1h.index.tz is None: ohlc_1h.index = ohlc_1h.index.tz_localize("UTC")

        self.strategy.reset_session_state()
        self.trades, self.pending_trade = [], None

        s_info = SESSION_MAP[self.timezone.key]
        sess_start_wall = time(s_info["start_hour"], s_info["start_minute"])
        sess_end_wall = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))
        dummy_dt = datetime.combine(datetime.min, sess_start_wall) - timedelta(hours=1)
        pre_market_wall = dummy_dt.time()

        for i in range(lookback_5m, len(ohlc_5m)):
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

            # Windows
            window_5m = ohlc_5m.iloc[i - lookback_5m: i + 1]
            window_1h = ohlc_1h[ohlc_1h.index < ts_utc].tail(lookback_1h)
            if len(window_1h) < 8: continue

            # Run Strategy
            result = self.strategy.analyze(window_5m, window_1h)
            curr_bar = ohlc_5m.iloc[i]

            # ====================================================================
            # STATE SYNCHRONIZATION (Trust the Strategy)
            # ====================================================================
            # If Strategy says "FILLED", we update our record.
            if self.pending_trade and self.strategy.trade_status == TradeStatus.FILLED:
                if self.pending_trade["status"] == "PENDING":
                    self.pending_trade["status"] = "FILLED"
                    self.pending_trade["fill_time"] = ts_utc

            if result.trade_signal:
                sig = result.trade_signal

                # --- 1. ENTER ---
                if sig.action == TradeAction.ENTER:
                    self.pending_trade = {
                        "timestamp": sig.timestamp,
                        "utc_signal": ts_utc,
                        "window_indices_5m": (i - lookback_5m, i),
                        "direction": sig.direction.name,
                        "entry": sig.price,
                        "sl": sig.sl,
                        "tp": sig.tp,
                        "size": sig.size,
                        "status": "PENDING",  # Strategy starts as Pending
                        "metadata": sig.metadata,
                        "sl_atr": sig.metadata.get("sl_atr", 0),
                        "eb_mult": sig.metadata.get("eb_mult", 0),
                        "fill_time": None,
                        "comments": ""
                    }

                # --- 2. CANCEL (Manual or Timeout) ---
                elif sig.action == TradeAction.CANCEL and self.pending_trade:
                    t_rec = self.pending_trade

                    if t_rec["status"] == "PENDING":
                        # True Cancellation (Never filled)
                        t_rec.update({
                            "exit_time": ts_utc,
                            "exit_price": 0.0,
                            "result": "CANCELLED",
                            "pnl": 0.0,
                            "status": "CLOSED",
                            "comments": "Signal Cancelled"
                        })
                    else:
                        # Trade was active -> Force Close
                        self._calculate_and_close(t_rec, curr_bar, ts_utc, "Cancelled After Fill")

                    self.trades.append(t_rec)
                    self.pending_trade = None

                # --- 3. EXIT (Session Cutoff / SL / TP) ---
                elif sig.action == TradeAction.EXIT and self.pending_trade:
                    t_rec = self.pending_trade

                    # === PHANTOM FILL FIX ===
                    # If Strategy says we are still PENDING when Exit arrives,
                    # it means the entry was never hit. We CANCEL, not CLOSE.
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
                        # Trade is active. Calculate PnL (including Gaps).
                        self._calculate_and_close(t_rec, curr_bar, ts_utc, "Trade Exit")

                    self.trades.append(t_rec)
                    self.pending_trade = None

        return pd.DataFrame(self.trades)

    def _calculate_and_close(self, t_rec, curr_bar, ts_utc, default_comment):
        """Calculates PnL with Gap/Slippage detection."""
        is_long = t_rec["direction"] == "LONG"
        bar_open = curr_bar["open"]
        final_price = curr_bar["close"]
        final_res = "WIN"  # Default placeholder
        exit_comment = default_comment

        # --- GAP DETECTION ---
        # Did market open BEYOND our levels?
        sl_price, tp_price = t_rec["sl"], t_rec["tp"]

        if is_long:
            gap_sl = bar_open <= sl_price
            gap_tp = bar_open >= tp_price
        else:
            gap_sl = bar_open >= sl_price
            gap_tp = bar_open <= tp_price

        if gap_sl:
            final_price = bar_open
            final_res = "LOSS"
            exit_comment = "Gap SL (Slippage)"
        elif gap_tp:
            final_price = bar_open
            final_res = "WIN"
            exit_comment = "Gap TP (Bonus)"
        else:
            # --- NORMAL HIT ---
            # If not a gap, did we hit SL/TP during the bar?
            if is_long:
                hit_sl = curr_bar["low"] <= sl_price
                hit_tp = curr_bar["high"] >= tp_price
            else:
                hit_sl = curr_bar["high"] >= sl_price
                hit_tp = curr_bar["low"] <= tp_price

            if hit_sl:
                final_price = sl_price
                final_res = "LOSS"
                exit_comment = "Hard SL Hit"
            elif hit_tp:
                final_price = tp_price
                final_res = "WIN"
                exit_comment = "Hard TP Hit"
            else:
                # Manual / Session Exit
                final_res = "WIN" if self.is_win(t_rec, final_price) else "LOSS"

        # --- PNL CALC ---
        pv = self.strategy.point_value
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
            "comments": exit_comment
        })

    def plot_trade(self, trade_idx: int, ohlc_5m: pd.DataFrame, ohlc_1h: pd.DataFrame) -> Optional[go.Figure]:
        # (Your plot logic remains valid and can be kept as is)
        if trade_idx >= len(self.trades): return None
        trade = self.trades[trade_idx]
        if trade["status"] == "CANCELLED": return None

        start_5m, signal_5m = trade["window_indices_5m"]
        exit_ts = trade.get("exit_time")

        if exit_ts and exit_ts in ohlc_5m.index:
            end_idx = ohlc_5m.index.get_loc(exit_ts) + 20
        else:
            end_idx = signal_5m + 50

        end_idx = min(len(ohlc_5m) - 1, end_idx)
        view_5m = ohlc_5m.iloc[start_5m: end_idx + 1]
        view_1h = ohlc_1h[ohlc_1h.index <= trade["utc_signal"]].tail(100)

        fig = go.Figure(data=[go.Candlestick(
            x=view_5m.index, open=view_5m['open'], high=view_5m['high'],
            low=view_5m['low'], close=view_5m['close'], name="5m Trigger TF"
        )])

        swing_1h, internal_1h = self.strategy.sd_detector_1h.calculate(view_1h)
        for z in (swing_1h + internal_1h):
            color = "rgba(0, 200, 0, 0.15)" if z['bias'] == BIAS.LONG else "rgba(200, 0, 0, 0.15)"
            fig.add_shape(type="rect", x0=view_5m.index[0], x1=view_5m.index[-1],
                          y0=z['bottom'], y1=z['top'], fillcolor=color, line_width=0, layer="below")

        entry_ts = trade["utc_signal"]
        chart_end = view_5m.index[-1]
        is_win = trade["result"] == "WIN"

        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end, y0=trade["entry"], y1=trade["tp"],
                      fillcolor="rgba(0, 255, 100, 0.2)" if is_win else "rgba(0, 255, 100, 0.05)",
                      line=dict(width=1, color="green"), layer="below")
        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end, y0=trade["entry"], y1=trade["sl"],
                      fillcolor="rgba(255, 0, 0, 0.05)" if is_win else "rgba(255, 0, 0, 0.2)",
                      line=dict(width=1, color="red"), layer="below")

        border_color = "green" if is_win else "red"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=f"<b>{trade['result']}</b><br>PnL: ${trade['pnl']}<br>Entry: {trade['entry']}",
            showarrow=False, bgcolor=border_color, font=dict(color="white", size=12),
            bordercolor="white", borderwidth=2, borderpad=6, opacity=0.9
        )

        fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_white", margin=dict(r=50))
        return fig
