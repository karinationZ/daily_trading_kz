from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.clients.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.clients.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.clients.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.clients.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.clients.SMCBase import BEARISH, BULLISH, Trend
from daily_trading.bots.smc_bot.clients.strategy.StrategyBase import StrategyBase
from daily_trading.settings import logger


class ContinuationInducementStrategy(StrategyBase):
    """
    Analyzes OHLC data using Smart Money Concept (SMC) for continuation trade signals.
    Designed for scalping in volatile crypto markets with a 6-step process:
    1. Identify Trend 1H (last 24 candles)
    2. Detect Break of Structure (BOS) on 1H; reset on CHOCH.
    3. Detect liquidity pool on 15m, post-BOS, between HH/HL (uptrend) or LH/LL (downtrend) on 1H.
    4. Find unmitigated FVG below (uptrend) or above (downtrend) liquidity pool on 15m.
    5. Confirm price touches the nearest FVG on 15m.
    6. Generate LONG/SHORT signal with entry, SL, and TP (1:2 RR).
    """

    def __init__(self, main_tf: str, debug: bool):
        """
        Initialize SMC strategy with detectors.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for trend and BOS/CHOCH.
            higher_tf (str): Higher timeframe (e.g., '4h') for trend confirmation.
            telegram_token (str): Telegram bot token for sending signals.
            channel_id (str): Telegram channel ID for message delivery.
        """
        super().__init__(main_tf, debug)
        self.detectors = {
            'swing_1h': HighLowDetector(main_tf),  # Detects HH/HL/LL/LH on 1H
            'bos_1h': BosChochDetector(main_tf),  # Detects BOS/CHOCH on 1H
            'liquidity_pool_15m': LiquidityPoolDetector(self.lower_tf_15m),  # Detects liquidity pools on 15m
            'fvg_15m': FvgDetector(self.lower_tf_15m)  # Detects FVGs on 15m
        }

    async def confirm_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime):
        """Step 1: Identify Trend on 1H (last 24 candles)."""
        swing_1h = self.detectors['swing_1h'].calculate_high_low_swings(ohlc_1h, plot=True)
        fig = swing_1h["fig"]
        trend, recent_smc_events = self.get_trend(ohlc_1h, symbol, cache_key, current_time, fig)
        return trend, recent_smc_events, fig