import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.settings import logger


class BacktestResultsViewer:
    def __init__(self, html_path: Path, risk_amount: float = 150.0, strategy_type="ORB"):
        self.html_path = html_path
        self.risk_amount = risk_amount
        self.strategy_type = strategy_type

        # FIXED: Added 'header' and unified color names
        self.colors = {
            'bg': '#0F172A',  # Deep Dark Background
            'card': '#1E293B',  # Card Surface
            'header': '#334155',  # Table Header / Divider
            'primary': '#A05AFF',  # Purple (Gauge/Monthly Header)
            'secondary': '#1BCFB4',  # Teal (Profit)
            'accent_blue': '#4BCBEB',  # Light Blue (Expectancy)
            'accent_pink': '#FE9496',  # Coral (Losses)
            'text_main': '#F1F5F9',  # High contrast white
            'text_dim': '#94A3B8',  # Muted slate
        }

    def generate_dashboard(self, csv_results_file: Path):
        """
        Refactored High-End Dashboard for Single Setup Analysis.
        Analyst Fixes: Corrected PnL/DD logic, added conditional formatting,
        and included visibility for Canceled Trades.
        """
        meta = self._parse_meta_from_filename(csv_results_file.name)
        df = pd.read_csv(csv_results_file)

        if df.empty:
            logger.error(f"Error: {csv_results_file.name} is empty.")
            return

        # --- 1. DATA PROCESSING (FIXED LOGIC) ---
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Cumulative Equity (starting from 0)
        equity_curve = df['pnl'].cumsum()

        # Monthly Stats
        df['Month_Sort'] = df['timestamp'].dt.to_period('M')
        monthly_groups = df.groupby('Month_Sort')

        monthly_data = []
        for month, group in monthly_groups:
            # Filter out cancelled for WR calculation
            filled_group = group[group['result'] != 'CANCELLED']
            m_wins = len(filled_group[filled_group['result'] == 'WIN'])
            m_total = len(filled_group)
            m_wr = (m_wins / m_total * 100) if m_total > 0 else 0
            m_pnl = group['pnl'].sum()

            monthly_data.append({
                'Month': month.strftime('%Y - %B'),
                'Trades': m_total,
                'Win Rate': m_wr,
                'PnL': m_pnl
            })
        m_df = pd.DataFrame(monthly_data)

        # Performance Audit Metrics (FIXED)
        wins = len(df[df['result'] == 'WIN'])
        losses = len(df[df['result'].str.contains('LOSS', na=False)])
        canceled = len(df[df['result'] == 'CANCELLED'])  # <--- New Logic

        total_filled = wins + losses
        wr = (wins / total_filled * 100) if total_filled > 0 else 0
        pnl_val = df['pnl'].sum()
        expectancy = pnl_val / total_filled if total_filled > 0 else 0

        # Correct Max Drawdown on realized PnL
        max_drawdown = (equity_curve.cummax() - equity_curve).max()

        # --- 2. GRID CONFIGURATION ---
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"colspan": 3, "type": "scatter"}, None, None],
                [{"type": "table"}, {"colspan": 2, "type": "table"}, None]
            ],
            vertical_spacing=0.08,
            row_heights=[0.2, 0.35, 0.45],
            subplot_titles=(" ", " ", " ", "Realized Equity Growth", "Performance Audit", "Monthly Snapshot")
        )

        # --- 3. TOP ROW: INDICATORS ---
        fig.add_trace(go.Indicator(
            mode="number", value=pnl_val,
            number={'prefix': "$", 'font': {'size': 50, 'color': self.colors['secondary']}},
            title={'text': "Total Net Profit", 'font': {'color': self.colors['text_dim'], 'size': 14}},
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="gauge+number", value=wr,
            number={'suffix': "%", 'font': {'size': 50, 'color': self.colors['primary']}},
            gauge={'bar': {'color': self.colors['primary']}, 'axis': {'range': [0, 100], 'visible': False},
                   'bgcolor': self.colors['card']},
            title={'text': "Win Rate", 'font': {'color': self.colors['text_dim'], 'size': 14}},
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="number", value=expectancy,
            number={'prefix': "$", 'font': {'size': 50, 'color': self.colors['accent_blue']}},
            title={'text': "Expectancy/Trade", 'font': {'color': self.colors['text_dim'], 'size': 14}},
        ), row=1, col=3)

        # --- 4. MIDDLE ROW: EQUITY CURVE ---
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))), y=equity_curve,
            mode='lines+markers', line=dict(color=self.colors['secondary'], width=3),
            fill='tozeroy', fillcolor='rgba(27, 207, 180, 0.05)',
            marker=dict(size=6, color=self.colors['primary']),
            name="Equity"
        ), row=2, col=1)

        # --- 5. BOTTOM ROW: TABLES ---
        # Table Left: Audit Metrics
        audit_metrics = {
            "Filled Trades": total_filled,
            "Wins": wins,
            "Losses": losses,
            "Canceled Trades": canceled,  # <--- Added as requested
            "Win Rate": f"{wr:.1f}%",
            "Net PnL": f"${pnl_val:,.0f}",
            "Profit Factor": f"{round((wins * meta['rr_val']) / max(losses, 1), 2)}",
            "Max Drawdown": f"${max_drawdown:,.0f}",
            "Expectancy": f"${expectancy:,.1f}"
        }

        fig.add_trace(go.Table(
            header=dict(values=["<b>METRIC</b>", "<b>VALUE</b>"], fill_color=self.colors['header'], align='left',
                        font=dict(color='white', size=12), height=30),
            cells=dict(values=[list(audit_metrics.keys()), list(audit_metrics.values())],
                       fill_color=self.colors['card'], align='left',
                       font=dict(color=[self.colors['text_dim'], 'white'], size=12), height=25)
        ), row=3, col=1)

        # Table Right: Monthly Breakdown
        m_pnl_colors = [self.colors['secondary'] if v >= 0 else self.colors['accent_pink'] for v in m_df['PnL']]
        m_wr_colors = [self.colors['primary'] if v >= 50 else self.colors['accent_pink'] for v in m_df['Win Rate']]

        fig.add_trace(go.Table(
            header=dict(values=["<b>MONTH</b>", "<b>TRADES</b>", "<b>WIN RATE</b>", "<b>NET PNL</b>"],
                        fill_color=self.colors['header'], align='left', font=dict(color='white', size=12), height=30),
            cells=dict(values=[
                m_df['Month'],
                m_df['Trades'],
                m_df['Win Rate'].apply(lambda x: f"{x:.1f}%"),
                m_df['PnL'].apply(lambda x: f"${x:,.0f}")
            ],
                fill_color=self.colors['card'], align='left',
                font=dict(color=[
                    'white',
                    'white',
                    m_wr_colors,
                    m_pnl_colors
                ], size=12), height=25)
        ), row=3, col=2)

        # --- 6. LAYOUT ---
        fig.update_layout(
            title=dict(
                text=f"<b>Institutional {self.strategy_type} Strategy Audit</b><br><sup>{meta['description']}</sup>",
                font=dict(size=24, color=self.colors['text_main']), x=0.02, y=0.97
            ),
            template="plotly_dark", paper_bgcolor=self.colors['bg'], plot_bgcolor=self.colors['bg'],
            height=1100, margin=dict(t=120, b=50, l=50, r=50), showlegend=False
        )

        fig.update_xaxes(showgrid=True, gridcolor='#334155', zeroline=False, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#334155', zeroline=False, row=2, col=1)

        self._save_to_html(fig, self.html_path)
        return wr, pnl_val

    def _save_to_html(self, fig, path: Path):
        """Single centralized method to handle HTML exports."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            str(path),
            include_plotlyjs='cdn',
            config={'displayModeBar': False}
        )

    @staticmethod
    def _parse_meta_from_filename(filename: str) -> Optional[dict]:
        """Extract setup details from filename using a single flexible pattern."""

        # Combined pattern with optional groups for sl and eb_mult
        # The _? before results handles filenames both with and without that underscore
        pattern = r"""
            (?P<symbol>[A-Z0-9]+)_
            (?P<session>[A-Z]+)
            _sw(?P<sw>\d+)
            _int(?P<int>\d+)
            _off(?P<off>\d+)
            _atr(?P<atr>[\d.]+)
            _thr(?P<thr>[\d.]+)
            _rr(?P<rr>[\d.]+)
            (?:_sl(?P<sl_atr>[\d.]+))?           # Optional SL group
            (?:_eb(?:_mult)?(?P<eb_mult>[\d.]+))? # Optional EB or EB_MULT group
            _?results\.csv                        # Flexible results suffix
        """

        match = re.search(pattern, filename, re.IGNORECASE | re.VERBOSE)
        if not match:
            return None

        gd = match.groupdict()
        symbol = gd["symbol"].upper()
        session = gd["session"].upper()

        # Build dynamic description
        parts = [
            f"Symbol: {symbol} | {session}",
            f"SW:{gd['sw']}",
            f"INT:{gd['int']}",
            f"FVG Offset:{gd['off']}",
            f"Zone Precision:{gd['atr']}",
            f"FVG THR:{gd['thr']}",
            f"RR:{gd['rr']}"
        ]

        if gd.get("sl_atr"):
            parts.append(f"SL_ATR:{gd['sl_atr']}")

        if gd.get("eb_mult"):
            parts.append(f"EB:{gd['eb_mult']}")

        return {
            "symbol": symbol,
            "session": session,
            "description": " | ".join(parts),
            "rr_val": float(gd["rr"]),
            # Optional: include raw values for calculations
            "sl_atr": float(gd["sl_atr"]) if gd.get("sl_atr") else None,
            "eb_mult": float(gd["eb_mult"]) if gd.get("eb_mult") else None
        }

    def format_trade_plot(self, fig, setup_name: str, description: str, trade_idx: int, result: str):
        """Stylizes a technical chart into the modern UI theme."""
        fig.update_layout(
            title=dict(
                text=f"<b>{setup_name}</b> <br><sup>Trade {trade_idx} • {result.upper()} • {description}</sup>",
                font=dict(size=20, color=self.colors['text_main'], family="Inter, sans-serif"),
                x=0.02, y=0.95
            ),
            template="plotly_dark",
            paper_bgcolor=self.colors['bg'],
            plot_bgcolor=self.colors['bg'],
            margin=dict(t=100, b=40, l=40, r=40)
        )
        return fig
