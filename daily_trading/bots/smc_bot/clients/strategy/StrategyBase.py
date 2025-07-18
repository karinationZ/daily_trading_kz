from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import plotly.graph_objects as go

from daily_trading.bots.smc_bot.clients.SMCBase import SMCBase, Trend
from daily_trading.bots.smc_bot.clients.TelegramBot import TelegramBot
from daily_trading.settings import logger, TELEGRAM_BOT_CHAT_ID


@dataclass
class SignalSummary:
    direction: str
    trend: str
    trend_signal: Optional[list] = None
    recent_smc_events: Optional[list] = None
    fig: Optional[go.Figure] = None
    nearest_ob: Optional[dict] = None


class StrategyBase(ABC):
    """Base class for trading strategies, providing reusable methods for caching, plotting, and messaging."""

    def __init__(self, main_tf: str, debug=False, telegram_chat_id=None):
        """
        Initialize base strategy with Telegram bot and cache.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for analysis.
            telegram_token (str): Telegram bot token for sending signals.
            channel_id (str): Telegram channel ID for message delivery.
        """
        self.main_tf = main_tf
        self.lower_tf_15m = '15m'  # Default 15m timeframe for finer analysis
        self.telegram_bot = TelegramBot(chat_id=telegram_chat_id)
        self.smc_base = SMCBase(self.main_tf)
        self.event_cache = {}  # Cache: {(symbol, timeframe): {'events': list, 'last_updated': datetime}}
        self.debug = debug
        logger.info(f"StrategyBase initialized with {main_tf} and {self.lower_tf_15m}")
        if not self.debug:
            logger.warning("Debug Mode is Off. Intermittent Telegram messages will not be sent.")

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
