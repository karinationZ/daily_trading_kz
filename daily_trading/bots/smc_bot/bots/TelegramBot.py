import asyncio
import io
from typing import Optional

import aiohttp
import plotly.graph_objects as go
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile
from tenacity import retry, stop_after_attempt, wait_exponential

from daily_trading.settings import *


class TelegramBot:
    """Handles all Telegram bot operations with async support and rate limiting."""

    def __init__(
            self,
            connection_pool_size: int = 20,
            read_timeout: int = 90,
            write_timeout: int = 90,
            parse_mode: Optional[str] = "HTML",
            disable_notification: bool = False,
            chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram bot with connection pool settings.

        Args:
            connection_pool_size (int): Number of concurrent connections
            read_timeout (int): Socket read timeout
            write_timeout (int): Socket connect timeout
            parse_mode (str): Default parse mode for messages
            disable_notification (bool): Default notification setting
            chat_id (str): Target channel ID for messages
        """
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.chat_id = chat_id if chat_id else TELEGRAM_CHANNEL_CHAT_ID
        self.bot = Bot(
            token=TELEGRAM_TOKEN,
            default=DefaultBotProperties(
                parse_mode=parse_mode,
                disable_notification=disable_notification
            ),
            session=AiohttpSession(
                limit=connection_pool_size,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_read=read_timeout,
                    sock_connect=write_timeout
                )
            )
        )
        self.semaphore = asyncio.Semaphore(3)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=lambda e: isinstance(e, TelegramAPIError) and "Too Many Requests" in str(e)
    )
    async def send_message(
            self,
            text: str,
            image: Optional[go.Figure] = None,
    ) -> bool:
        """
        Send message to Telegram channel with optional image.
        """
        if not self.chat_id:
            logger.warning("No channel ID configured, message not sent")
            return False

        async with self.semaphore:
            try:
                text = text[:4096]
                if image:
                    buffer = io.BytesIO()
                    image.write_image(
                        buffer,
                        format="png",
                        width=1600,
                        height=900,
                        scale=2
                    )
                    buffer.seek(0)
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=BufferedInputFile(file=buffer.read(), filename="chart.png"),
                        caption=text
                    )
                else:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=text
                    )

                logger.info(f"Telegram message sent successfully.")
                return True

            except TelegramAPIError as e:
                if "Too Many Requests" in str(e):
                    retry_after = int(str(e).split("retry after ")[-1]) if "retry after" in str(e) else 5
                    logger.warning(f"Rate limit hit, retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    raise
                logger.error(f"Telegram API error: {e}\nFailed to send: {text}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error sending message: {str(e)}\nFailed to send: {text}")
                return False

    async def get_channel_id_from_updates(self):
        try:
            updates = await self.bot.get_updates(timeout=30)
            for update in updates:
                if update.channel_post:
                    channel_id = update.channel_post.chat.id
                    logger.info(f"Detected channel ID: {channel_id}")
                    return channel_id
            logger.warning("No channel posts found in updates.")
            return None
        except TelegramAPIError as e:
            logger.error(f"Failed to get updates: {e}")
            return None

    async def close(self):
        """Close the bot session to release resources."""
        await self.bot.session.close()
