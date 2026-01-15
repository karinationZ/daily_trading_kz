from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.backtester.common_utils import (
    parse_params_from_results_csv_filename,
    monte_carlo_stress,
)
from daily_trading.settings import logger


class BacktestResultsViewer:
    def __init__(self, html_path: Path, risk_amount: float = 150.0, strategy_type: str = "ORB"):
        self.html_path = html_path
        self.risk_amount = risk_amount
        self.strategy_type = strategy_type

        self.colors = {
            "bg": "#0F172A",
            "card": "#1E293B",
            "header": "#334155",
            "primary": "#A05AFF",
            "secondary": "#1BCFB4",
            "accent_blue": "#4BCBEB",
            "accent_pink": "#FE9496",
            "text_main": "#F1F5F9",
            "text_dim": "#94A3B8",
        }

    # ==========================================================
    # MAIN DASHBOARD
    # ==========================================================
    def generate_dashboard(self, csv_results_file: Path):
        meta = parse_params_from_results_csv_filename(csv_results_file.name)
        df = pd.read_csv(csv_results_file)

        if df.empty:
            logger.error(f"{csv_results_file.name} is empty.")
            return None

        # --------------------------------------------------
        # DATA NORMALIZATION
        # --------------------------------------------------
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        filled_df = df[df["result"] != "CANCELLED"].copy()

        # --------------------------------------------------
        # CORE METRICS
        # --------------------------------------------------
        equity_curve = filled_df["pnl"].cumsum()

        total_pnl = filled_df["pnl"].sum()
        total_trades = len(filled_df)

        wins_df = filled_df[filled_df["pnl"] > 0]
        losses_df = filled_df[filled_df["pnl"] < 0]

        wins = len(wins_df)
        losses = len(losses_df)
        cancelled = len(df) - total_trades

        win_rate = (wins / total_trades * 100) if total_trades else 0.0
        expectancy = filled_df["pnl"].mean() if total_trades else 0.0

        gross_profit = wins_df["pnl"].sum()
        gross_loss = abs(losses_df["pnl"].sum())

        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

        max_drawdown = (equity_curve.cummax() - equity_curve).max()

        # --------------------------------------------------
        # MONTE CARLO STRESS TEST
        # --------------------------------------------------
        mc = monte_carlo_stress(
            filled_df["pnl"],
            runs=2000,
            initial_capital=0.0,
        )

        mc_return_5 = mc["return"].quantile(0.05)
        mc_return_50 = mc["return"].quantile(0.50)
        mc_return_95 = mc["return"].quantile(0.95)

        mc_dd_50 = mc["max_drawdown"].quantile(0.50)
        mc_dd_95 = mc["max_drawdown"].quantile(0.95)

        # --------------------------------------------------
        # MONTHLY STATS
        # --------------------------------------------------
        filled_df["Month"] = filled_df["timestamp"].dt.to_period("M")

        monthly = (
            filled_df.groupby("Month")
            .agg(
                Trades=("pnl", "count"),
                WinRate=("pnl", lambda x: (x > 0).mean() * 100),
                PnL=("pnl", "sum"),
            )
            .reset_index()
        )

        monthly["Month"] = monthly["Month"].astype(str)

        # --------------------------------------------------
        # FIGURE LAYOUT
        # --------------------------------------------------
        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"colspan": 3, "type": "scatter"}, None, None],
                [{"type": "table"}, {"colspan": 2, "type": "table"}, None],
            ],
            row_heights=[0.22, 0.38, 0.40],
            vertical_spacing=0.08,
        )

        # --------------------------------------------------
        # TOP INDICATORS
        # --------------------------------------------------
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_pnl,
                number={"prefix": "$", "font": {"size": 48, "color": self.colors["secondary"]}},
                title={"text": "Net PnL", "font": {"color": self.colors["text_dim"]}},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=win_rate,
                number={"suffix": "%", "font": {"size": 48, "color": self.colors["primary"]}},
                gauge={
                    "axis": {"range": [0, 100], "visible": False},
                    "bar": {"color": self.colors["primary"]},
                },
                title={"text": "Win Rate", "font": {"color": self.colors["text_dim"]}},
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=expectancy,
                number={"prefix": "$", "font": {"size": 48, "color": self.colors["accent_blue"]}},
                title={"text": "Expectancy / Trade", "font": {"color": self.colors["text_dim"]}},
            ),
            row=1,
            col=3,
        )

        # --------------------------------------------------
        # EQUITY CURVE
        # --------------------------------------------------
        fig.add_trace(
            go.Scatter(
                x=filled_df["timestamp"],
                y=equity_curve,
                mode="lines",
                line=dict(width=3, color=self.colors["secondary"]),
                fill="tozeroy",
                fillcolor="rgba(27, 207, 180, 0.08)",
                name="Equity",
            ),
            row=2,
            col=1,
        )

        # --------------------------------------------------
        # AUDIT TABLE
        # --------------------------------------------------
        audit_metrics = {
            "Filled Trades": total_trades,
            "Wins": wins,
            "Losses": losses,
            "Cancelled Trades": cancelled,
            "Win Rate": f"{win_rate:.1f}%",
            "Net PnL": f"${total_pnl:,.0f}",
            "Profit Factor": profit_factor,
            "Max Drawdown": f"${max_drawdown:,.0f}",
            "Expectancy": f"${expectancy:,.1f}",
            "MC Median Return": f"${mc_return_50:,.0f}",
            "MC Worst 5%": f"${mc_return_5:,.0f}",
            "MC Best 95%": f"${mc_return_95:,.0f}",
            "MC Median DD": f"${mc_dd_50:,.0f}",
            "MC Worst DD (95%)": f"${mc_dd_95:,.0f}",
        }

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["METRIC", "VALUE"],
                    fill_color=self.colors["header"],
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[list(audit_metrics.keys()), list(audit_metrics.values())],
                    fill_color=self.colors["card"],
                    font=dict(color="white", size=12),
                    align="left",
                ),
            ),
            row=3,
            col=1,
        )

        # --------------------------------------------------
        # MONTHLY TABLE
        # --------------------------------------------------
        pnl_colors = [
            self.colors["secondary"] if v >= 0 else self.colors["accent_pink"]
            for v in monthly["PnL"]
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["MONTH", "TRADES", "WIN RATE", "PNL"],
                    fill_color=self.colors["header"],
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        monthly["Month"],
                        monthly["Trades"],
                        monthly["WinRate"].map(lambda x: f"{x:.1f}%"),
                        monthly["PnL"].map(lambda x: f"${x:,.0f}"),
                    ],
                    font=dict(color=["white", "white", "white", pnl_colors]),
                    fill_color=self.colors["card"],
                    align="left",
                ),
            ),
            row=3,
            col=2,
        )

        # --------------------------------------------------
        # LAYOUT
        # --------------------------------------------------
        fig.update_layout(
            title=dict(
                text=f"<b>{self.strategy_type} Strategy Audit</b><br><sup>{meta['description']}</sup>",
                font=dict(size=22, color=self.colors["text_main"]),
                x=0.02,
            ),
            template="plotly_dark",
            paper_bgcolor=self.colors["bg"],
            plot_bgcolor=self.colors["bg"],
            height=1100,
            showlegend=False,
        )

        self._save_to_html(fig)
        return win_rate, total_pnl

    # ==========================================================
    # SAVE
    # ==========================================================
    def _save_to_html(self, fig):
        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            str(self.html_path),
            include_plotlyjs="cdn",
            config={"displayModeBar": False},
        )
