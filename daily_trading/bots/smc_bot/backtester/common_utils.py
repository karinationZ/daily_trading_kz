import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def parse_params_from_results_csv_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract setup details from filename. Supports three formats:
    1. New Buffer Variant: '{symbol}_{sess}_sw{}_int{}_off{}_atr{}_rr{}_eb_mult{}_sl_buffer_{}_results.csv'
    2. Complex (Legacy):   '{symbol}_{sess}_sw{}_int{}_off{}_atr{}_thr{}_rr{}_results.csv'
    3. Tap Zone:           '{symbol}_{sess}_rr{}_mit{}_sl{}_results.csv'
    """

    # --- PATTERN 1: New Buffer Variant (Matches your f-string exactly) ---
    # f"{symbol}_{sess_name}_sw{sw}_int{int}_off{off}_atr{atr}_rr{rr}_eb_mult{eb_mult}_sl_buffer_{sl_buffer}_results.csv"
    pattern_buffer_variant = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _sw(?P<sw>\d+)
        _int(?P<int>\d+)
        _off(?P<off>\d+)
        _atr(?P<atr>[\d.]+)
        _rr(?P<rr>[\d.]+)
        _eb_mult(?P<eb_mult>[\d.]+)
        _sl_buffer_(?P<sl_buffer>[\d.]+)
        _results\.csv
    """

    # --- PATTERN 2: Structural / Complex (Legacy) ---
    pattern_complex = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _sw(?P<sw>\d+)
        _int(?P<int>\d+)
        _off(?P<off>\d+)
        _atr(?P<atr>[\d.]+)
        _thr(?P<thr>[\d.]+)
        _rr(?P<rr>[\d.]+)
        (?:_mit(?P<mit>[\d.]+))?
        (?:_sl(?P<sl_atr>[\d.]+))?
        (?:_eb(?:_mult)?(?P<eb_mult>[\d.]+))? 
        _?results\.csv
    """

    # --- PATTERN 3: Tap Zone (Simplified) ---
    pattern_tap = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _rr(?P<rr>[\d.]+)
        _mit(?P<mit>[\d.]+)
        _sl(?P<sl_atr>[\d.]+)
        _?results\.csv
    """

    # 1. Try New Buffer Variant (Most specific first)
    match = re.search(pattern_buffer_variant, filename, re.IGNORECASE | re.VERBOSE)
    if match:
        gd = match.groupdict()
        try:
            return {
                "symbol": gd["symbol"].upper(),
                "session": gd["session"].upper(),
                "sw": int(gd['sw']),
                "int": int(gd['int']),
                "off": int(gd['off']),
                "atr": float(gd['atr']),
                "rr": float(gd['rr']),
                "eb_mult": float(gd['eb_mult']),
                "sl_atr": float(gd['sl_buffer']),  # Mapping 'sl_buffer' to standard 'sl_atr' key
                "thr": None,  # This pattern doesn't have threshold
                "mit": None,
                "description": (f"Symbol: {gd['symbol'].upper()} | {gd['session'].upper()} | "
                                f"SW:{gd['sw']} EB:{gd['eb_mult']} SL:{gd['sl_buffer']}")
            }
        except ValueError:
            pass

    # 2. Try Complex Pattern (Legacy)
    match = re.search(pattern_complex, filename, re.IGNORECASE | re.VERBOSE)
    if match:
        gd = match.groupdict()
        try:
            mit_val = float(gd['mit']) if gd.get('mit') else None
            sl_val = float(gd['sl_atr']) if gd.get('sl_atr') else None

            return {
                "symbol": gd["symbol"].upper(),
                "session": gd["session"].upper(),
                "sw": int(gd['sw']),
                "int": int(gd['int']),
                "off": int(gd['off']),
                "atr": float(gd['atr']),
                "thr": float(gd['thr']),
                "rr": float(gd['rr']),
                "mit": mit_val,
                "sl_atr": sl_val,
                "eb_mult": float(gd['eb_mult']) if gd.get('eb_mult') else None,
                "description": (f"Symbol: {gd['symbol'].upper()} | {gd['session'].upper()} | "
                                f"SW:{gd['sw']} RR:{gd['rr']}")
            }
        except ValueError:
            pass

    # 3. Try Tap Pattern
    match = re.search(pattern_tap, filename, re.IGNORECASE | re.VERBOSE)
    if match:
        gd = match.groupdict()
        try:
            return {
                "symbol": gd["symbol"].upper(),
                "session": gd["session"].upper(),
                "rr": float(gd['rr']),
                "mit": float(gd['mit']),
                "sl_atr": float(gd['sl_atr']),
                "sw": None, "int": None, "off": None, "atr": None, "thr": None, "eb_mult": None,
                "description": (f"Symbol: {gd['symbol'].upper()} | {gd['session'].upper()} | "
                                f"RR:{gd['rr']} MIT:{gd['mit']} SL:{gd['sl_atr']}")
            }
        except ValueError:
            return None

    return None


def monte_carlo_stress(
        pnl_series: pd.Series,
        runs: int = 2000,
        initial_capital: float = 0.0
) -> pd.DataFrame:
    """
    Monte Carlo stress test on realized PnL.
    Assumes independent trade outcomes.
    """
    pnl = pnl_series.values
    results = []

    for _ in range(runs):
        shuffled = np.random.permutation(pnl)
        equity = initial_capital
        peak = equity
        max_dd = 0.0

        for p in shuffled:
            equity += p
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

        results.append({
            "final_equity": equity,
            "return": equity - initial_capital,
            "max_drawdown": max_dd,
        })

    return pd.DataFrame(results)
