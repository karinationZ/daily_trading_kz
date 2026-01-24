import glob
import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_trading.bots.smc_bot.backtester.BacktestResultsViewer import (
    BacktestResultsViewer,
)
from daily_trading.bots.smc_bot.backtester.common_utils import (
    parse_params_from_results_csv_filename,
)
from daily_trading.bots.smc_bot.backtester.MasterLeaderboardViewer import (
    LeadershipBy,
    MasterLeaderboardViewer,
)
from daily_trading.settings import logger

RANKING_COLUMN = "Score"
TOP_N = 20


# ========================================================================
# OPTIONAL: Penalize High RR with Low Sample Size
# ========================================================================
# High RR strategies need more trades to prove themselves


def rr_sample_penalty(row):
    """
    If RR > 2.0 and Trades < 50, apply penalty
    - Prevents "got lucky with 3 big wins" scenarios
    """
    if row["rr"] > 2.0 and row["Trades"] < 50:
        penalty = 1 - (0.01 * (50 - row["Trades"]))  # Max 50% penalty
        return max(penalty, 0.5)
    return 1.0


# ========================================================================
# Calculate True Edge (most important metric)
# ========================================================================
# Edge = (Win% × Avg_Win) - (Loss% × Avg_Loss)
# Normalized by risk (RR-aware breakeven calculation)


def calculate_edge_margin(row):
    """
    Edge Margin = Actual Win% - Breakeven Win%
    Breakeven Win% = 1 / (1 + RR)

    Example:
    - 60% WR at 1.2 RR → BE = 45.5% → Edge = +14.5%
    - 52% WR at 2.5 RR → BE = 28.6% → Edge = +23.4%
    """
    breakeven_wr = 100 / (1 + row["rr"])
    edge_margin = row["WinRate"] - breakeven_wr
    return edge_margin


SESSION_TARGETS = {
    "ASIA": 30,  # Quieter sessions, fewer setups
    "LONDON": 40,  # Medium activity
    "NY": 50,  # Highest activity, most trades
}


def statistical_confidence(trades, session):
    """
    Logarithmic confidence score
    - 20 trades = ~0.75 confidence
    - 50 trades = 1.00 confidence
    - 100 trades = 1.10 confidence (diminishing returns)
    """
    target = SESSION_TARGETS.get(session, 50)
    if trades < 10:
        return 0.5  # Heavy penalty for tiny samples

    confidence = np.log(trades + 1) / np.log(target + 1)
    return min(confidence, 1.15)  # Cap at 1.15 (slight bonus for huge samples)


def rank_results(df_results):
    # 1. Hard Filters
    df = df_results[df_results["Trades"] >= 25].copy()
    df = df[df["WinRate"] >= 48.0].copy()

    # 2. Normalize Columns
    max_pnl = df["PnL"].max()

    # 3. Calculate Score
    # 40% Weight on Win Rate (Psychological ease)
    # 40% Weight on PnL (Profitability)
    # 20% Weight on RR (Risk profile - Higher RR is generally more robust to slippage)

    df["Score"] = (df["WinRate"] / 100.0 * 0.4) + (df["PnL"] / max_pnl * 0.4) + (df["rr"] / 2.0 * 0.2)

    return df.sort_values(by="Score", ascending=False)


def merge_leaderboards(input_folder, output_file):
    # 1. Find all CSV files in the folder (excluding the output file if it exists)
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    all_files = [f for f in all_files if f != os.path.join(input_folder, output_file)]

    if not all_files:
        print("❌ No CSV files found in the specified folder.")
        return

    master_list = []

    print(f"🔍 Found {len(all_files)} files. Processing...")

    for filename in all_files:
        # Read the CSV
        df = pd.read_csv(filename)

        # Check if required columns exist
        if RANKING_COLUMN not in df.columns:
            print(f"⚠️  Skipping {filename}: Column '{RANKING_COLUMN}' not found.")
            continue

        # Sort by Score (Descending) just in case the file wasn't sorted
        df_sorted = df.sort_values(by=RANKING_COLUMN, ascending=False)

        # Take the Top N
        top_rows = df_sorted.head(TOP_N)

        # Optional: Add a column to track which file it came from
        top_rows["Source_File"] = os.path.basename(filename)

        master_list.append(top_rows)
        print(f"✅ Added top {len(top_rows)} rows from: {os.path.basename(filename)}")

    # 2. Concatenate all dataframes
    if master_list:
        final_df = pd.concat(master_list, ignore_index=True)

        # 3. Final Sort (Optional: Sort the master list by Score as well)
        final_df = final_df.sort_values(by=RANKING_COLUMN, ascending=False)

        # 4. Save to CSV
        final_df.to_csv(input_folder / output_file, index=False)
        print("\n" + "=" * 50)
        print(f"🚀 SUCCESS! Merged {len(master_list)} files.")
        print(f"📄 Saved {len(final_df)} total rows to: {output_file}")
        print("=" * 50)
    else:
        print("⚠️  No valid data found to merge.")


def aggregate_and_promote_champions(results_base_path: Path, filter_by: LeadershipBy, strategy_type="ORB"):
    ELITE_DIR = results_base_path / "ELITE_SETUPS"
    leaderboard_dir = results_base_path / "LEADERBOARDS"
    os.makedirs(leaderboard_dir, exist_ok=True)

    if not ELITE_DIR.exists():
        pytest.fail(f"Elite directory not found at {ELITE_DIR}")

    all_summaries = []
    WINNERS_DIR = leaderboard_dir / "WINNERS"

    if WINNERS_DIR.exists():
        shutil.rmtree(WINNERS_DIR)
    WINNERS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. SCAN FILES
    csv_files = list(ELITE_DIR.glob("**/*_results.csv"))
    logger.info(f"\n[AGGREGATOR] Analyzing {len(csv_files)} setups using Weighted Score...")

    for csv_path in csv_files:
        if "WINNERS" in csv_path.parts or "LEADERBOARDS" in csv_path.parts:
            continue

        meta = parse_params_from_results_csv_filename(csv_path.name)
        if not meta:
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        filled = df[df["result"].isin(["WIN", "LOSS"])]
        if filled.empty:
            continue

        wins = len(filled[filled["result"] == "WIN"])
        wr = (wins / len(filled)) * 100
        total_pnl = round(filled["pnl"].sum())

        # Calculate Average Trade (Efficiency Proxy)
        avg_trade = total_pnl / len(filled) if len(filled) > 0 else 0

        # --- DYNAMIC OUTPUT CONSTRUCTION ---
        # Initialize with mandatory common fields
        output = {
            "symbol": meta["symbol"],
            "session": meta["session"],
            "WinRate": round(wr, 2),
            "PnL": total_pnl,
            "AvgTrade": round(avg_trade, 2),
            "Trades": len(filled),
            "rr": meta.get("rr", 0.0),  # Safe get
        }

        # Build Config String & Add Specific Params conditionally
        config_parts = []

        # 1. Structural Params (if present)
        if meta.get("sw") is not None:
            output["sw"] = meta["sw"]
            output["int"] = meta["int"]
            config_parts.extend(
                [
                    f"SW:{meta['sw']}",
                    f"INT:{meta['int']}",
                ]
            )
        # 1. Structural Params (if present)
        if meta.get("off") is not None:
            output["off"] = meta["off"]
            output["atr"] = meta["atr"]
            output["thr"] = meta["thr"]
            config_parts.extend([f"OFF:{meta['off']}", f"ATR:{meta['atr']}", f"THR:{meta['thr']}"])

        # 2. Tap Zone Params (Crucial update)
        if meta.get("mit") is not None:
            output["mit"] = meta["mit"]
            config_parts.append(f"MIT:{meta['mit']}")

        # 3. Common Optional Params
        if meta.get("sl_atr") is not None:
            output["sl_atr"] = meta["sl_atr"]
            config_parts.append(f"SL:{meta['sl_atr']}")

        if meta.get("eb_mult") is not None:
            output["eb_mult"] = meta["eb_mult"]
            config_parts.append(f"EB:{meta['eb_mult']}")

        output.update({"Full_Path": str(csv_path), "Original_Filename": csv_path.name})
        # Finalize Config String
        output["config"] = " | ".join(config_parts)

        all_summaries.append(output)

    if not all_summaries:
        pytest.fail("No valid data found to aggregate.")

    master_df = pd.DataFrame(all_summaries)

    # symbol	session	WinRate	PnL	AvgTrade	Trades	rr	sw	int	mit	sl_atr	Full_Path	Original_Filename	config	Norm_PnL	Norm_RR	Score	Source_File
    columns_order = [
        "symbol",
        "session",
        "WinRate",
        "PnL",
        "Score",
        "Trades",
        "rr",
        "sw",
        "int",
        "mit",
        "sl_atr",
        "Full_Path",
        "Original_Filename",
        "AvgTrade",
        "config",
        "Norm_PnL",
        "Norm_RR",
        "Source_File",
    ]
    master_df = master_df.reindex(columns=columns_order)

    # 2. GROUP AND RANK (The Champion Logic)
    grouped = master_df.groupby(["symbol", "session"])

    for (symbol, session), group in grouped:
        group = group.copy()

        # 1. Normalize PnL (0-100 relative to group best)
        max_pnl = group["PnL"].max()
        max_pnl = max_pnl if max_pnl > 0 else 1.0
        group["Norm_PnL"] = (group["PnL"] / max_pnl) * 100

        # 2. Normalize Risk:Reward (0-100, capped at 2.5R)
        # Note: If RR is missing, it defaults to 0.0, resulting in 0 score contribution
        group["Norm_RR"] = (group["rr"] / 2.5).clip(upper=1.0) * 100

        # 3. Calculate Performance Score
        group["Edge_Margin"] = group.apply(calculate_edge_margin, axis=1)
        group["Norm_Edge"] = (group["Edge_Margin"] / group["Edge_Margin"].max()) * 100

        # ========================================================================
        # Profit Quality (not just total PnL)
        # ========================================================================
        # Penalize inconsistent profit (high PnL from a few lucky trades)

        group["Profit_Per_Trade"] = group["PnL"] / group["Trades"]
        group["Norm_PPT"] = (group["Profit_Per_Trade"] / group["Profit_Per_Trade"].max()) * 100

        # Profit Consistency = how smoothly profits accumulate
        # (High Sharpe-like metric: reward steady grinding over boom-bust)
        group["Profit_Quality"] = (group["Norm_PnL"] * 0.6) + (group["Norm_PPT"] * 0.4)

        # ========================================================================
        # Risk-Adjusted Win Rate
        # ========================================================================
        # A 55% WR at 2.5 RR is MORE impressive than 70% WR at 1.2 RR
        # This accounts for the difficulty of achieving high WR with wide targets

        group["Risk_Adjusted_WR"] = group["WinRate"] * np.sqrt(group["rr"])
        group["Norm_RAWR"] = (group["Risk_Adjusted_WR"] / group["Risk_Adjusted_WR"].max()) * 100

        confidence_func = partial(statistical_confidence, session=session)
        group["Confidence"] = group["Trades"].apply(confidence_func)

        # ========================================================================
        # Expectancy (trader's true north star)
        # ========================================================================
        # Expectancy = (Win% × Avg_Win) - (Loss% × Avg_Loss)
        # Simplified using RR: E = (WR × RR) - (1 - WR)

        group["Expectancy"] = (group["WinRate"] / 100 * group["rr"]) - (1 - group["WinRate"] / 100)
        group["Norm_Expectancy"] = (group["Expectancy"] / group["Expectancy"].max()) * 100

        base_score = (
            group["Norm_Edge"] * 0.30  # Edge over breakeven (critical)
            + group["Profit_Quality"] * 0.30  # Actual profitability
            + group["Norm_Expectancy"] * 0.25  # Long-term expectancy
            + group["Norm_RAWR"] * 0.15  # Difficulty-adjusted win rate
        )

        # Apply confidence multiplier
        group["Score"] = base_score * group["Confidence"]

        group["RR_Penalty"] = group.apply(rr_sample_penalty, axis=1)
        group["Score"] = group["Score"] * group["RR_Penalty"]

        # base_score = (group["WinRate"] * 0.4) + (group["Norm_PnL"] * 0.4) + (group["Norm_RR"] * 0.2)
        #
        # # 4. Apply Confidence Multiplier
        # confidence_mult = (group["Trades"] / 15).clip(upper=1.0)
        # group["Score"] = base_score * confidence_mult

        # 5. Garbage Collection
        garbage_mask = (group["Trades"] < 5) | (group["WinRate"] < 30.0)
        group.loc[garbage_mask, "Score"] = 0.0

        # Sort by Score
        sorted_group = group.sort_values(by="Score", ascending=False)

        # Save Leaderboard
        leaderboard_name = f"{session}_{symbol}_Leaderboard.csv"
        sorted_group.to_csv(leaderboard_dir / leaderboard_name, index=False)

        # E. Identify Champion
        best = sorted_group.iloc[0]

        # If the best score is 0, it means NO config met the minimum criteria
        if best["Score"] <= 0:
            logger.warning(f"⚠️ No qualified champion for {symbol} {session}")
            continue

        # F. Promote Champion
        champion_folder = WINNERS_DIR / f"{session}_{symbol}"
        champion_folder.mkdir(parents=True, exist_ok=True)

        src_csv = Path(best["Full_Path"])
        dest_csv = champion_folder / best["Original_Filename"]
        shutil.copy2(src_csv, dest_csv)

        # Generate Dashboard for specific winner
        html_name = f"Dashboard_TAP_{session}_{symbol}_WINNER.html"
        viewer = BacktestResultsViewer(html_path=champion_folder / html_name)
        viewer.generate_dashboard(dest_csv)

        logger.info(
            f"🏆 CHAMPION: {symbol} ({session}) | Score: {best['Score']:.1f} | PnL: ${best['PnL']} | WR: {best['WinRate']}% | Cfg: {best['config']}"
        )

    # 3. GENERATE MASTER DASHBOARD
    viewer = MasterLeaderboardViewer(
        leaderboard_dir=leaderboard_dir,
        report_name="Dashboard_Zone_Tap_Leaderboard.html",
        strategy_type=strategy_type,
        sort_by=filter_by,
    )
    viewer.generate_master_dashboard()

    merge_leaderboards(leaderboard_dir, "MASTER_LEADERBOARD.csv")

    logger.info(f"✅ Aggregation complete. Weighted Champions located in: {WINNERS_DIR}")
