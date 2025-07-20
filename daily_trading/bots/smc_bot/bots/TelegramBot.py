import asyncio
import io
from typing import Optional

import plotly.graph_objects as go
from retrying import retry
from telegram import Bot, InputFile
from telegram.error import TelegramError
from telegram.request import HTTPXRequest

from daily_trading.settings import logger, TELEGRAM_CHANNEL_CHAT_ID, TELEGRAM_TOKEN


class TelegramBot:
    """Handles all Telegram bot operations with connection pooling and rate limiting."""

    def __init__(
            self,
            connection_pool_size: int = 10,
            pool_timeout: int = 60,
            read_timeout: int = 60,
            write_timeout: int = 60,
            parse_mode: Optional[str] = "HTML",
            disable_notification: bool = False,
            chat_id=None
    ):
        """
        Initialize Telegram bot with connection pool settings.

        Args:
            token (str): Telegram bot token
            channel_id (str): Target channel ID for messages
            connection_pool_size (int): Number of concurrent connections
            pool_timeout (int): Connection pool timeout in seconds
            read_timeout (int): Socket read timeout
            write_timeout (int): Socket write timeout
            parse_mode (str): Default parse mode for messages
            disable_notification (bool): Default notification setting
        """
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.chat_id = chat_id if chat_id else TELEGRAM_CHANNEL_CHAT_ID
        # Configure HTTPXRequest with connection pool settings
        request = HTTPXRequest(
            connection_pool_size=connection_pool_size,
            pool_timeout=pool_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout
        )

        # Initialize bot with custom request configuration
        self.bot = Bot(
            token=TELEGRAM_TOKEN,
            request=request,
            get_updates_request=request
        )

        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent sends
        logger.info("TelegramBot initialized")

    @retry(stop_max_attempt_number=10, wait_fixed=5000)
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
                kwargs = {
                    'chat_id': self.chat_id,
                    'disable_notification': self.disable_notification,
                    'parse_mode': self.parse_mode,
                    'write_timeout': self.bot.request.read_timeout
                }

                if image:
                    buffer = io.BytesIO()
                    image.write_image(
                        buffer,
                        format="png",
                        width=1600,  # Wider than default (~700–800)
                        height=900,  # Maintain a good aspect ratio
                        scale=2  # Double resolution (effective DPI)
                    )
                    buffer.seek(0)

                    await self.bot.send_photo(
                        photo=InputFile(buffer, filename="chart.png"),
                        caption=text,
                        **kwargs
                    )
                else:
                    await self.bot.send_message(
                        text=text,
                        **kwargs
                    )

                logger.debug(f"Telegram message sent: {text[:100]}...")
                return True

            except TelegramError as e:
                logger.error(f"Telegram failed to send {text}\n{e}")
                await asyncio.sleep(5)
                return False
            except Exception as e:
                logger.error(f"Unexpected error sending message: {e}")
                return False

    async def get_channel_id_from_updates(self):
        updates = await self.bot.get_updates()
        for update in updates:
            if update.channel_post:
                channel_id = update.channel_post.chat.id
                logger.info(f"Detected channel ID: {channel_id}")
                return channel_id
        logger.warning("No channel posts found in updates.")
        return None
