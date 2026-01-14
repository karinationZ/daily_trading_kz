import re
from typing import Any, Dict, Optional


def parse_params_from_results_csv_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract setup details from filename. Supports two formats:
    1. Complex: 'MBT_NY_sw50_int5_off1_atr2.2_thr0.25_rr1.5_results.csv' (Can optionally include _mit)
    2. Tap Zone: 'MNQ_NY_rr2.0_mit0.5_sl0.1_results.csv'
    """

    # --- PATTERN 1: Structural / Complex ---
    # Updated to optionally capture 'mit' (mitigation) and 'sl' (stop loss)
    pattern_complex = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _sw(?P<sw>\d+)
        _int(?P<int>\d+)
        _off(?P<off>\d+)
        _atr(?P<atr>[\d.]+)
        _thr(?P<thr>[\d.]+)
        _rr(?P<rr>[\d.]+)
        (?:_mit(?P<mit>[\d.]+))?          # Optional: Mitigation (Added per request)
        (?:_sl(?P<sl_atr>[\d.]+))?        # Optional: SL Buffer
        (?:_eb(?:_mult)?(?P<eb_mult>[\d.]+))? 
        _?results\.csv
    """

    # --- PATTERN 2: Tap Zone (Simplified) ---
    pattern_tap = r"""
        (?P<symbol>[A-Z0-9]+)_
        (?P<session>[A-Z]+)
        _rr(?P<rr>[\d.]+)
        _mit(?P<mit>[\d.]+)
        _sl(?P<sl_atr>[\d.]+)
        _?results\.csv
    """

    # 1. Try Complex Pattern
    match = re.search(pattern_complex, filename, re.IGNORECASE | re.VERBOSE)
    if match:
        gd = match.groupdict()
        try:
            # Default mit to 0.5 (Standard 50% CE) if missing in filename
            mit_val = float(gd['mit']) if gd.get('mit') else None

            # Default sl to 0.1 (Standard Buffer) if missing in filename
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
                "mit": mit_val,  # Now guaranteed to be a float (default 0.5)
                "sl_atr": sl_val,
                "eb_mult": float(gd['eb_mult']) if gd.get('eb_mult') else None,
                "description": (f"Symbol: {gd['symbol'].upper()} | {gd['session'].upper()} | "
                                f"SW:{gd['sw']} MIT:{mit_val} RR:{gd['rr']}")
            }
        except ValueError:
            pass

            # 2. Try Tap Pattern
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
                # Structural defaults (None implies "Not optimized in this run")
                "sw": None, "int": None, "off": None, "atr": None, "thr": None, "eb_mult": None,
                "description": (f"Symbol: {gd['symbol'].upper()} | {gd['session'].upper()} | "
                                f"RR:{gd['rr']} MIT:{gd['mit']} SL:{gd['sl_atr']}")
            }
        except ValueError:
            return None

    return None
