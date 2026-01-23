from enum import Enum
from pathlib import Path

import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots


class LeadershipBy(Enum):
    SCORE = "Score"
    PNL = "PnL"
    WIN_RATE = "WinRate"


class MasterLeaderboardViewer:
    """
    Institutional Master Leaderboard.
    Aggregates granular CSVs and filters for the single Highest Scoring setup
    per Symbol/Session combination (Robustness > Raw PnL).
    """

    def __init__(
        self,
        leaderboard_dir: Path,
        report_name: str = "Master_Strategy_Leaderboard.html",
        strategy_type="ORB",
        sort_by: LeadershipBy = LeadershipBy.SCORE,
    ):
        self.leaderboard_dir = leaderboard_dir
        self.report_name = report_name
        self.strategy_type = strategy_type
        self.colors = {
            "bg": "#0F172A",
            "card": "#1E293B",
            "header": "#334155",
            "primary": "#A05AFF",  # Purple
            "secondary": "#1BCFB4",  # Teal
            "accent": "#06B6D4",  # Cyan
            "text_main": "#F1F5F9",
            "text_dim": "#94A3B8",
            "positive": "#22c55e",
            "negative": "#ef4444",
        }
        self.sort_by = sort_by

    def generate_master_dashboard(self):
        # 1. FIND AND LOAD ALL AGGREGATED FILES
        csv_files = list(self.leaderboard_dir.glob("*_Leaderboard.csv"))
        if not csv_files:
            print(f"Error: No Leaderboard CSV files found in {self.leaderboard_dir}")
            return

        master_list = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    master_list.append(df)
            except Exception as e:
                print(f"Skipping bad CSV {f}: {e}")

        if not master_list:
            return

        master_df = pd.concat(master_list, ignore_index=True)

        # 3. FILTER: BEST SETUP PER (SYMBOL + SESSION)
        # We take the row with the highest Score/PnL
        winners_df = (
            master_df.sort_values(by=self.sort_by.value, ascending=False)
            .groupby(["symbol", "session"], as_index=False)
            .head(1)
            .sort_values(by=self.sort_by.value, ascending=False)
        )

        # 4. CONSTRUCT CONFIG STRING FOR DISPLAY
        # Re-construct readable config string if needed
        winners_df["config_display"] = winners_df["config"]
        # winners_df['config_display'] = (
        #         "SW:" + winners_df['sw'].astype(str) + " | " +
        #         "INT:" + winners_df['int'].astype(str) + " | " +
        #         "OFF:" + winners_df['off'].astype(str) + " | " +
        #         "ATR:" + winners_df['atr'].astype(str) + " | " +
        #         "THR:" + winners_df['thr'].astype(str)
        # )

        # 5. CREATE DASHBOARD
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "bar", "colspan": 2}, None],
                [{"type": "table", "colspan": 2}, None],
            ],
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6],
            subplot_titles=(
                f"Top Champions by {self.sort_by.value}",
                "Detailed Configuration of Champions",
            ),
        )

        winners_df["Label"] = winners_df["symbol"] + " (" + winners_df["session"] + ")"

        # --- BAR CHART (PnL with Score Color) ---
        # We show PnL on the Y-axis because it's tangible, but color by Score (Robustness)
        y_val = winners_df["PnL"]
        color_val = winners_df["Score"] if "Score" in winners_df.columns else winners_df["WinRate"]

        fig.add_trace(
            go.Bar(
                x=winners_df["Label"],
                y=y_val,
                marker=dict(
                    color=color_val,
                    # Custom Gradient based on your image: Red (Bad) -> Orange -> Yellow -> Green -> Blue (Best)
                    colorscale=[
                        [0.0, "#E74236"],  # 1. Red (Lowest Score)
                        [0.25, "#F68529"],  # 2. Orange
                        [0.5, "#FBBF24"],  # 3. Golden Yellow
                        [0.75, "#268D56"],  # 4. Green
                        [1.0, "#2B5889"],  # 5. Deep Blue (Highest Score)
                    ],
                    showscale=True,
                    colorbar=dict(title="Score", len=0.4, y=0.8),
                ),
                text=winners_df["PnL"].apply(lambda x: f"${x:,.0f}"),
                textposition="outside",
                cliponaxis=False,
                name="PnL",
            ),
            row=1,
            col=1,
        )

        # --- TABLE ---
        # Prepare columns safely
        score_vals = winners_df["Score"].apply(lambda x: f"{x:.1f}")

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "<b>SYMBOL/SESSION</b>",
                        "<b>SCORE</b>",
                        "<b>PNL</b>",
                        "<b>WIN RATE</b>",
                        "<b>TRADES</b>",
                        "<b>CONFIGURATION</b>",
                        "<b>RR</b>",
                    ],
                    fill_color=self.colors["header"],
                    align="left",
                    font=dict(color="white", size=13),
                    height=40,
                ),
                cells=dict(
                    values=[
                        winners_df["Label"],
                        score_vals,
                        winners_df["PnL"].apply(lambda x: f"${x:,.0f}"),
                        winners_df["WinRate"].apply(lambda x: f"{x:.1f}%"),
                        winners_df["Trades"],
                        winners_df["config_display"],
                        winners_df["rr"].apply(lambda x: f"{x}R"),
                    ],
                    fill_color=self.colors["card"],
                    align="left",
                    font=dict(
                        color=[
                            self.colors["text_main"],
                            self.colors["accent"],  # Score highlight
                            self.colors["secondary"],  # PnL highlight
                            "white",
                            "white",
                            self.colors["text_dim"],
                            "white",
                        ],
                        size=12,
                    ),
                    height=35,
                ),
            ),
            row=2,
            col=1,
        )

        # --- LAYOUT ---
        fig.update_layout(
            title=dict(
                text=f"<b>{self.strategy_type} Champions Board</b><br><sup>Selected by highest {self.sort_by.value}</sup>",
                font=dict(size=28, color=self.colors["text_main"]),
                x=0.02,
                y=0.97,
            ),
            template="plotly_dark",
            paper_bgcolor=self.colors["bg"],
            plot_bgcolor=self.colors["bg"],
            height=1000,
            margin=dict(t=150, b=50, l=50, r=50),
            showlegend=False,
        )

        fig.update_yaxes(title="Net PnL ($)", gridcolor="#334155", row=1, col=1)

        output_path = self.leaderboard_dir / self.report_name
        fig.write_html(str(output_path), include_plotlyjs="cdn", config={"displayModeBar": False})

        print(f"\n🏆 Master Leaderboard Generated: {output_path}")
