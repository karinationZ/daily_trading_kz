from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import auto, Enum
from typing import Dict, List, Optional
from typing import Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.DetectorsBase import BIAS, DetectorsBase, Trend
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.settings import logger


class TradeStatus(Enum):
    NONE = "NONE"  # Scanning for ORB breakout
    PENDING = "PENDING"  # Limit order sent, waiting for price to hit zone
    FILLED = "FILLED"  # In the market, monitoring SL/TP


class TradeAction(Enum):
    ENTER = auto()
    CANCEL = auto()
    EXIT = auto()
    UPDATE = auto()


@dataclass
class TradeSignal:
    symbol: str
    action: TradeAction
    direction: Optional[BIAS] = None
    price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    size: Optional[int] = None
    entry_timestamp: Optional[datetime] = None
    reason: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    trade_result: Optional[str] = None
    fig: Optional[go.Figure] = None


@dataclass
class SignalSummary:
    direction: str
    trend: Optional[str] = None
    trend_signal: Optional[list] = None
    recent_smc_events: Optional[list] = None
    fig: Optional[go.Figure] = None
    nearest_ob: Optional[dict] = None
    recent_bos_signal: Optional[str] = None
    nearest_fvg: Optional[dict] = None
    title: Optional[str] = None
    trade_signal: Optional[TradeSignal] = None


class StrategyBase(ABC):
    """Base class for trading strategies, providing reusable methods for caching, plotting, and messaging."""
    # Updated Configuration with MCL
    POINT_VALUE = {
        "MNQ": 2.0,
        "MES": 5.0,
        "MGC": 10.0,
        "MBT": 0.10,
        "SIL": 1000.0,
        "MCL": 100.0
    }

    ENTRY_BUFFER = {
        "MNQ": 0.5,
        "MES": 0.5,
        "MGC": 0.2,
        "MBT": 5.0,
        "SIL": 0.010,
        "MCL": 0.02
    }

    SL_BUFFER = {
        "MNQ": 0.5,
        "MES": 0.5,
        "MGC": 0.5,
        "MBT": 5.0,
        "SIL": 0.025,
        "MCL": 0.05
    }

    TICK_SIZES = {
        "MNQ": 0.25,
        "MES": 0.25,
        "MGC": 0.10,
        "MBT": 5.00,
        "MCL": 0.01,
        "SIL": 0.005,
    }

    def __init__(self, main_tf: str, debug=False, telegram_chat_id=None):
        """
        Initialize base strategy with Telegram bot and cache.
        """
        self.main_tf = main_tf
        self.telegram_bot = TelegramBot(chat_id=telegram_chat_id)
        self.smc_base = DetectorsBase(self.main_tf)
        self.event_cache = {}  # Cache: {(symbol, timeframe): {'events': list, 'last_updated': datetime}}
        self.debug = debug
        logger.info(f"StrategyBase initialized with {main_tf}")
        if not self.debug:
            logger.warning("Debug Mode is Off. Internal Telegram messages will not be sent.")

        self.bos_signals = None
        self.recent_bos_signal = None
        self.bos_choch_fig = None

    @staticmethod
    def get_tz_short(timezone: ZoneInfo):
        if timezone == ZoneInfo("America/New_York"):
            tz_short = "NY"
        elif timezone == ZoneInfo("Europe/London"):
            tz_short = "LONDON"
        else:
            tz_short = "ASIA"
        return tz_short

    @staticmethod
    def is_session_started(ts_now, timezone: ZoneInfo) -> bool:
        """
        Check if trading session has started.
        Returns True only if session has started (ignores end time).

        SESSION_MAP times (in local timezone):
        - NY: 9:30 AM - 4:00 PM EST
        - London: 8:00 AM - 2:30 PM GMT
        - Tokyo: 9:00 AM - 3:00 PM JST
        """
        s_info = SESSION_MAP[timezone.key]

        # Get session start
        session_start = ts_now.replace(
            hour=s_info.get("start_hour", 0),
            minute=s_info.get("start_minute", 0),
            second=0,
            microsecond=0
        )

        # Check if session has started
        return ts_now >= session_start

    async def manage_cache_event(self, cache_key: Tuple[str, str], event: str, current_time: datetime) -> bool:
        """Manage cache for a new event, return True if event is new."""
        if cache_key not in self.event_cache:
            self.event_cache[cache_key] = {'events': [], 'last_updated': current_time}
        if event not in self.event_cache[cache_key]['events']:
            self.event_cache[cache_key]['events'].append(event)
            self.event_cache[cache_key]['last_updated'] = current_time
            return True
        return False

    async def send_telegram_message(self, message: str, fig: Optional[go.Figure] = None):
        """
        Send message via Telegram bot.
        """
        if self.debug:
            await self.telegram_bot.send_message(message, fig)
        else:
            logger.info("Skipped sending telegram message. Debug Mode is Off.")

    # TODO: remove it
    async def get_trend(self, ohlc_1h, symbol: str, cache_key: Tuple[str, str], current_time: datetime, fig=None):
        """Step 1: Identify Trend on 1H TF."""
        recent_smc_events = []
        trend = self.smc_base.detect_trend(ohlc_1h)

        trend_event = f"{trend} trend confirmed."
        recent_smc_events.append(trend_event)

        if trend != Trend.NEUTRAL.name:
            if await self.manage_cache_event(cache_key, trend_event, current_time):
                await self.send_telegram_message(f"{symbol} | Step 1: {trend_event}", fig)
        else:
            logger.info(f"Trend is {trend}.")
            if cache_key in self.event_cache:
                self.event_cache[cache_key]['events'] = []
            await self.send_telegram_message(f"{symbol} | Step 1: Neutral trend detected.", fig)

        return trend, recent_smc_events

    # TODO: remove it
    async def confirm_trend(self, detector: HighLowDetector, ohlc: pd.DataFrame, symbol: str,
                            cache_key: Tuple[str, str],
                            current_time: datetime):
        """Step 1: Determine Market Trend on 1H (last 24 candles)."""
        swing_1h = detector.calculate_high_low_swings(ohlc, plot=True)
        fig = swing_1h["fig"]
        trend, recent_smc_events = await self.get_trend(ohlc, symbol, cache_key, current_time, fig)
        return trend, recent_smc_events, fig

    # TODO: remove it
    async def detect_bos_choch(self, detector: BosChochDetector, ohlc: pd.DataFrame, trend: str, symbol: str) -> \
            List[str]:
        """Step 2: Identify the most recent and valid BOS level in the trend direction."""
        recent_smc_events = []
        results = detector.calculate_bos_choch(ohlc, plot=True)
        self.bos_signals = results["signals"]
        self.bos_choch_fig = results["fig"]
        bos_choch_signals = sorted(results["signals"], key=lambda x: x["time"], reverse=True)

        if not bos_choch_signals:
            event = f"No {self.main_tf} BOS or CHOCH signals detected."
            recent_smc_events.append(event)
            return recent_smc_events

        recent_signal = bos_choch_signals[0]
        if recent_signal["type"] == "CHoCH":
            event = "Change of Character detected. Market trend may change."
            self.recent_bos_signal = None
            recent_smc_events.append(event)
            logger.info(f"Step 2: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {event}.", self.bos_choch_fig)
            return recent_smc_events

        expected_bias = Trend.BULLISH.value if trend == Trend.BULLISH.name else Trend.BEARISH.value
        expected_bos_signals = [s for s in bos_choch_signals if s['type'] == 'BOS' and s['bias'] == expected_bias]
        if not expected_bos_signals:
            event = f"No BOS signals detected for {trend} trend."
            recent_smc_events.append(event)
            logger.info(f"Step 2: {event} for {symbol}")
            await self.send_telegram_message(f"{symbol} | Step 2: {event}.", self.bos_choch_fig)
            return recent_smc_events

        self.recent_bos_signal = expected_bos_signals[0]
        event = f"BOS detected at {self.recent_bos_signal['time'].strftime('%Y-%m-%d %H:%M')} for {trend} trend."
        recent_smc_events.append(event)
        logger.info(f"Step 2: {event} for {symbol}")
        await self.send_telegram_message(f"{symbol} | Step 2: {event}.", self.bos_choch_fig)
        return recent_smc_events
