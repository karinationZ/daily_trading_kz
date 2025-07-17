from abc import ABC
from datetime import datetime
from typing import List, Optional, Tuple

import plotly.graph_objects as go

from daily_trading.bots.smc_bot.clients.TelegramBot import TelegramBot
from daily_trading.settings import logger


class StrategyBase(ABC):
    """Base class for trading strategies, providing reusable methods for caching, plotting, and messaging."""

    def __init__(self, main_tf: str):
        """
        Initialize base strategy with Telegram bot and cache.

        Args:
            main_tf (str): Lower timeframe (e.g., '1h') for analysis.
            telegram_token (str): Telegram bot token for sending signals.
            channel_id (str): Telegram channel ID for message delivery.
        """
        self.main_tf = main_tf
        self.lower_tf_15m = '15m'  # Default 15m timeframe for finer analysis
        self.telegram_bot = TelegramBot()
        self.event_cache = {}  # Cache: {(symbol, timeframe): {'events': list, 'last_updated': datetime}}
        logger.info(f"StrategyBase initialized with {main_tf} and {self.lower_tf_15m}")

    async def manage_cache_event(self, cache_key: Tuple[str, str], event: str, current_time: datetime) -> bool:
        """Manage cache for a new event, return True if event is new."""
        if cache_key not in self.event_cache:
            self.event_cache[cache_key] = {'events': [], 'last_updated': current_time}
        if event not in self.event_cache[cache_key]['events']:
            self.event_cache[cache_key]['events'].append(event)
            self.event_cache[cache_key]['last_updated'] = current_time
            return True
        return False

    async def send_telegram_message(self, message: str, fig: Optional[go.Figure] = None) -> bool:
        """
        Send message via Telegram bot.
        """
        return await self.telegram_bot.send_message(message, fig)

    # def plot_step(self, ohlc: pd.DataFrame, signals: list, step: str, entry: float = None,
    #               sl: float = None, tp: float = None) -> go.Figure:
    #     """
    #     Generate a Plotly figure for a specific step, showing the last 100 candles.
    #
    #     Args:
    #         ohlc (pd.DataFrame): OHLC data (1H or 15m, tail 100 candles used).
    #         signals (list): Detected signals (e.g., HH, BOS, liquidity pool) for the step.
    #         step (str): Step description (e.g., 'Step 1: Trend Detection').
    #         entry (float, optional): Entry price for signal visualization.
    #         sl (float, optional): Stop loss price for visualization.
    #         tp (float, optional): Take profit price for visualization.
    #
    #     Returns:
    #         go.Figure: Plotly figure with candlesticks and signals.
    #     """
    #     fig = make_subplots(rows=1, cols=1)
    #     ohlc_tail = ohlc.tail(100)  # Show last 100 candles for clarity
    #     fig.add_trace(go.Candlestick(
    #         x=ohlc_tail.index,
    #         open=ohlc_tail['open'],
    #         high=ohlc_tail['high'],
    #         low=ohlc_tail['low'],
    #         close=ohlc_tail['close'],
    #         name="OHLC"
    #     ))
    #
    #     for signal in signals:
    #         marker_color = 'blue' if signal.get('bias') == 'BULLISH' else 'red'
    #         marker_symbol = 'triangle-up' if signal.get('bias') == 'BULLISH' else 'triangle-down'
    #         y_level = signal.get('level',
    #                              (signal.get('top') + signal.get('bottom')) / 2 if 'top' in signal else
    #                              signal.get('high', signal.get('low')))
    #         fig.add_trace(go.Scatter(
    #             x=[signal['time']],
    #             y=[y_level * (1.002 if signal.get('bias') == 'BULLISH' else 0.998)],
    #             mode='markers+text',
    #             text=[signal['type']],
    #             textposition='top center' if signal.get('bias') == 'BULLISH' else 'bottom center',
    #             marker=dict(symbol=marker_symbol, size=10, color=marker_color),
    #             name=signal['type'],
    #             showlegend=True
    #         ))
    #
    #     if entry and sl and tp:
    #         last_time = ohlc_tail.index[-1]
    #         fig.add_shape(type="line", x0=last_time, x1=last_time, y0=entry, y1=entry,
    #                       line=dict(color="yellow", width=2, dash="dash"), name="Entry")
    #         fig.add_shape(type="line", x0=last_time, x1=last_time, y0=sl, y1=sl,
    #                       line=dict(color="red", width=2, dash="dash"), name="Stop Loss")
    #         fig.add_shape(type="line", x0=last_time, x1=last_time, y0=tp, y1=tp,
    #                       line=dict(color="green", width=2, dash="dash"), name="Take Profit")
    #
    #     fig.update_layout(
    #         title=f"{step} Analysis",
    #         xaxis_rangeslider_visible=False,
    #         showlegend=True,
    #         height=800,
    #         xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M'),
    #         yaxis=dict(title="Price"),
    #         margin=dict(t=100, b=100, l=50, r=50)
    #     )
    #     return fig

    async def compile_results(self, trend: str, signal: str, recent_smc_events: List[str],
                              symbol: str, fig: go.Figure = None):
        """Compile and return analysis results, send final Telegram message."""
        message = (f"📊 SMC Final Signal for {symbol}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                   f"Signal: {signal}\nTrend: {trend}\n")
        if recent_smc_events:
            message += f"Recent SMC Events: {' '.join(recent_smc_events)}\n"
        cache_key = (symbol, self.lower_tf_15m)
        if cache_key in self.event_cache:
            message += f"Cached Events: {' '.join(self.event_cache[cache_key]['events'])}\n"
        await self.send_telegram_message(message, fig)
