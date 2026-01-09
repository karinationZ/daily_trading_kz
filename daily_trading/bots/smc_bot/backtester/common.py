import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from daily_trading.bots.smc_bot.backtester.BacktestResultsViewer import BacktestResultsViewer
from daily_trading.bots.smc_bot.backtester.MasterLeaderboardViewer import LeadershipBy, MasterLeaderboardViewer
from daily_trading.settings import logger


def rank_results(df_results):
    # 1. Hard Filters
    df = df_results[df_results['Trades'] >= 25].copy()
    df = df[df['WinRate'] >= 48.0].copy()

    # 2. Normalize Columns
    max_pnl = df['PnL'].max()

    # 3. Calculate Score
    # 40% Weight on Win Rate (Psychological ease)
    # 40% Weight on PnL (Profitability)
    # 20% Weight on RR (Risk profile - Higher RR is generally more robust to slippage)

    df['Score'] = (
            (df['WinRate'] / 100.0 * 0.4) +
            (df['PnL'] / max_pnl * 0.4) +
            (df['rr'] / 2.0 * 0.2)
    )

    return df.sort_values(by='Score', ascending=False)


import re
from typing import Optional, Dict, Any


def parse_params_from_results_csv_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract setup details from filename using a single flexible pattern.
    Supports: 'MBT_NY_sw50_int5_off1_atr2.2_thr0.25_rr1.5_results.csv'
    """
    pattern = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _sw(?P<sw>\d+)
        _int(?P<int>\d+)
        _off(?P<off>\d+)
        _atr(?P<atr>[\d.]+)
        _thr(?P<thr>[\d.]+)
        _rr(?P<rr>[\d.]+)
        (?:_sl(?P<sl_atr>[\d.]+))?            # Optional SL group
        (?:_eb(?:_mult)?(?P<eb_mult>[\d.]+))? # Optional EB or EB_MULT group
        _?results\.csv                        # Flexible results suffix
    """

    match = re.search(pattern, filename, re.IGNORECASE | re.VERBOSE)
    if not match:
        return None

    gd = match.groupdict()

    # 1. Parse and Convert Types
    try:
        symbol = gd["symbol"].upper()
        session = gd["session"].upper()
        sw_val = int(gd['sw'])
        int_val = int(gd['int'])
        off_val = int(gd['off'])
        atr_val = float(gd['atr'])
        thr_val = float(gd['thr'])
        rr_val = float(gd['rr'])

        # Handle Optionals (None if missing)
        sl_atr_val = float(gd['sl_atr']) if gd.get('sl_atr') else None
        eb_mult_val = float(gd['eb_mult']) if gd.get('eb_mult') else None
    except ValueError:
        return None  # Fail gracefully if number conversion fails

    # 2. Build Description (Optional, for UI/Logging)
    parts = [
        f"Symbol: {symbol} | {session}",
        f"SW:{sw_val}",
        f"INT:{int_val}",
        f"OFF:{off_val}",
        f"ATR:{atr_val}",
        f"THR:{thr_val}",
        f"RR:{rr_val}"
    ]
    if sl_atr_val is not None:
        parts.append(f"SL_ATR:{sl_atr_val}")
    if eb_mult_val is not None:
        parts.append(f"EB:{eb_mult_val}")

    # 3. Return Dictionary with ALL Required Access Keys
    return {
        "symbol": symbol,
        "session": session,
        "sw": sw_val,
        "int": int_val,
        "off": off_val,
        "atr": atr_val,
        "thr": thr_val,
        "rr": rr_val,  # Standard Key
        "rr_val": rr_val,  # Legacy Key (requested in your example)
        "sl_atr": sl_atr_val,  # Optional (None if missing)
        "eb_mult": eb_mult_val,  # Optional (None if missing)
        "description": " | ".join(parts)
    }


def aggregate_and_promote_champions(results_base_path: Path, filter_by: LeadershipBy,
                                    strategy_type="ORB"):
    ELITE_DIR = results_base_path / "ELITE_SETUPS"
    leaderboard_dir = results_base_path / "LEADERBOARDS"
    os.makedirs(leaderboard_dir, exist_ok=True)
    # Configuration
    MIN_TRADES = 10  # Hard filter for statistical significance
    MIN_WR = 45.0  # Hard filter for psychological viability

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
        if not meta: continue

        df = pd.read_csv(csv_path)
        if df.empty: continue

        filled = df[df['result'].isin(['WIN', 'LOSS'])]
        if filled.empty: continue

        wins = len(filled[filled['result'] == 'WIN'])
        wr = (wins / len(filled)) * 100
        total_pnl = round(filled['pnl'].sum())

        # Calculate Average Trade (Efficiency Proxy)
        avg_trade = total_pnl / len(filled) if len(filled) > 0 else 0

        output = {
            "symbol": meta['symbol'],
            "session": meta['session'],
            "WinRate": round(wr, 2),
            "PnL": total_pnl,
            "AvgTrade": round(avg_trade, 2),
            "Trades": len(filled),
            "rr": meta['rr'],
            "sw": meta['sw'],
            "int": meta['int'],
            "off": meta['off'],
            "atr": meta['atr'],
            "thr": meta['thr'],
            "sl_atr": meta['sl_atr'],
            "Full_Path": str(csv_path),
            "Original_Filename": csv_path.name,
            "config": f"SW:{meta['sw']} | INT:{meta['int']} | OFF:{meta['off']} | ATR:{meta['atr']} | THR:{meta['thr']}"
        }
        if "sl_atr" in meta:
            output.update({"sl_atr": meta["sl_atr"]})
            output["config"] += f" | SL_ATR:{meta['sl_atr']}"
        if "eb_mult" in meta:
            output.update({"eb_mult": meta['eb_mult']})
            output["config"] += f" | EB:{meta['eb_mult']}"

        all_summaries.append(output)

    if not all_summaries:
        pytest.fail("No valid data found to aggregate.")

    master_df = pd.DataFrame(all_summaries)

    # 2. GROUP AND RANK (The Champion Logic)
    grouped = master_df.groupby(['symbol', 'session'])

    for (symbol, session), group in grouped:
        group = group.copy()
        TARGET_TRADES = 15.0

        # 1. Normalize PnL (0-100 relative to group best)
        max_pnl = group['PnL'].max()
        max_pnl = max_pnl if max_pnl > 0 else 1.0
        group['Norm_PnL'] = (group['PnL'] / max_pnl) * 100

        # 2. Normalize Risk:Reward (0-100, capped at 2.5R)
        group['Norm_RR'] = (group['rr'] / 2.5).clip(upper=1.0) * 100

        # 3. Calculate Performance Score
        base_score = (
                (group['WinRate'] * 0.4) +
                (group['Norm_PnL'] * 0.4) +
                (group['Norm_RR'] * 0.2)
        )

        # 4. Apply Confidence Multiplier
        # Penalizes low sample size without hiding potential gems.
        # 25+ trades -> 1.0x score. 12 trades -> 0.5x score.
        confidence_mult = (group['Trades'] / TARGET_TRADES).clip(upper=1.0)

        group['Score'] = base_score * confidence_mult

        # 5. Garbage Collection (Optional: Zero out statistically insignificant results)
        # Keeps data visible but at bottom of list
        garbage_mask = (group['Trades'] < 5) | (group['WinRate'] < 30.0)
        group.loc[garbage_mask, 'Score'] = 0.0

        # Sort by Score
        sorted_group = group.sort_values(by="Score", ascending=False)

        # Save Leaderboard (Include Score for transparency)
        leaderboard_name = f"TAP_{session}_{symbol}_Leaderboard.csv"
        sorted_group.to_csv(leaderboard_dir / leaderboard_name, index=False)

        # E. Identify Champion
        best = sorted_group.iloc[0]

        # If the best score is -1, it means NO config met the minimum criteria
        if best['Score'] == -1:
            logger.warning(
                f"⚠️ No qualified champion for {symbol} {session} (Min Trades: {MIN_TRADES}, Min WR: {MIN_WR}%)")
            continue

        # F. Promote Champion
        champion_folder = WINNERS_DIR / f"CHAMPION_{session}_{symbol}"
        champion_folder.mkdir(parents=True, exist_ok=True)

        src_csv = Path(best["Full_Path"])
        dest_csv = champion_folder / best["Original_Filename"]
        shutil.copy2(src_csv, dest_csv)

        # Generate Dashboard for specific winner
        html_name = f"Dashboard_TAP_{session}_{symbol}_WINNER.html"
        viewer = BacktestResultsViewer(html_path=champion_folder / html_name)
        viewer.generate_dashboard(dest_csv)

        logger.info(
            f"🏆 CHAMPION: {symbol} ({session}) | Score: {best['Score']:.1f} | PnL: ${best['PnL']} | WR: {best['WinRate']}% | Cfg: {best['config']}")

    # 3. GENERATE MASTER DASHBOARD
    # Note: We pass by_pnl=True purely for visualization in the master table,
    # but the files selected were chosen via Score.
    viewer = MasterLeaderboardViewer(
        leaderboard_dir=leaderboard_dir,
        report_name="Dashboard_Zone_Tap_Leaderboard.html",
        strategy_type=strategy_type,
        sort_by=LeadershipBy.PNL
    )
    viewer.generate_master_dashboard()

    logger.info(f"✅ Aggregation complete. Weighted Champions located in: {WINNERS_DIR}")
