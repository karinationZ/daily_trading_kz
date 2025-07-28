import asyncio
import io
import re
from typing import Optional

import aiohttp
import plotly.graph_objects as go
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile
from tenacity import retry, stop_after_attempt, wait_exponential

from daily_trading.settings import *


class TelegramBot:
    """Handles all Telegram bot operations with async support and rate limiting."""

    def __init__(
            self,
            connection_pool_size: int = 20,
            read_timeout: int = 180,
            connect_timeout: int = 60,
            total_timeout: int = 300,
            parse_mode: Optional[str] = "HTML",
            disable_notification: bool = False,
            chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram bot with connection pool settings.

        Args:
            connection_pool_size (int): Number of concurrent connections
            read_timeout (int): Socket read timeout
            write_timeout (int): Socket write timeout
            connect_timeout (int): Socket connect timeout
            total_timeout (int): Total request timeout
            parse_mode (str): Default parse mode for messages
            disable_notification (bool): Default notification setting
            chat_id (str): Target channel ID for messages
        """
        self.parse_mode = parse_mode
        self.disable_notification = disable_notification
        self.chat_id = chat_id if chat_id else TELEGRAM_CHANNEL_CHAT_ID

        timeout_config = aiohttp.ClientTimeout(
            total=total_timeout,
            sock_read=read_timeout,
            sock_connect=connect_timeout
        )

        self.bot = Bot(
            token=TELEGRAM_TOKEN,
            default=DefaultBotProperties(
                parse_mode=parse_mode,
                disable_notification=disable_notification
            ),
            session=AiohttpSession(
                limit=connection_pool_size,
                timeout=timeout_config
            )
        )
        self.semaphore = asyncio.Semaphore(3)

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_exponential(multiplier=2, min=3, max=60),
        retry=lambda e: (
                isinstance(e, (TelegramAPIError, asyncio.TimeoutError, aiohttp.ClientError)) and
                any(error_text in str(e).lower() for error_text in [
                    "too many requests", "timeout", "connection", "network"
                ])
        )
    )
    async def send_message(
            self,
            text: str,
            image: Optional[go.Figure] = None
    ) -> bool:
        """
        Send message to Telegram channel with optional image.
        Handles long messages by splitting them into multiple messages.

        Args:
            text (str): Message text
            image (Optional[go.Figure]): Plotly figure to send as an image
        """
        if not self.chat_id:
            logger.warning("No channel ID configured, message not sent")
            return False

        async with self.semaphore:
            try:
                if len(text) > 4096:
                    return await self._send_long_message(text, image)
                return await self._send_single_message(text, image)

            except asyncio.TimeoutError as e:
                logger.error(f"Request timeout while sending message: {text[:100]}... Error: {str(e)}")
                if image:
                    logger.warning("Retrying without image due to timeout...")
                    return await self._send_single_message(f"{text}\n\n[Chart could not be sent due to timeout]", None)
                raise

            except TelegramAPIError as e:
                if "Too Many Requests" in str(e):
                    retry_after = self._extract_retry_after(str(e))
                    logger.warning(f"Rate limit hit, retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    raise
                logger.error(f"Telegram API error: {e}\nFailed to send: {text[:100]}...")
                if image:
                    logger.warning("Retrying without image due to API error...")
                    return await self._send_single_message(f"{text}\n\n[Chart could not be sent due to API error]",
                                                           None)
                return False

            except aiohttp.ClientError as e:
                logger.error(f"HTTP Client error: {e}\nFailed to send: {text[:100]}...")
                if image:
                    logger.warning("Retrying without image due to client error...")
                    return await self._send_single_message(f"{text}\n\n[Chart could not be sent due to network error]",
                                                           None)
                raise

            except Exception as e:
                logger.error(f"Unexpected error sending message: {str(e)}\nFailed to send: {text[:100]}...")
                return False

    async def _send_single_message(self, text: str, image: Optional[go.Figure] = None) -> bool:
        """Send a single message (internal helper)."""
        try:
            if image:
                buffer = io.BytesIO()
                image.write_image(
                    buffer,
                    format="png",
                    width=1200,
                    height=675,
                    scale=1
                )
                buffer.seek(0)

                await asyncio.wait_for(
                    self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=BufferedInputFile(file=buffer.read(), filename="chart.png"),
                        caption=text,
                        parse_mode=ParseMode.HTML
                    ),
                    timeout=90
                )
            else:
                await asyncio.wait_for(
                    self.bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML
                    ),
                    timeout=45
                )

            logger.info(f"Telegram message sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Error sending single message: {str(e)}. Message: {text}")
            return False

    async def _send_long_message(self, text: str, image: Optional[go.Figure] = None) -> bool:
        """Split and send long messages (internal helper)."""
        logger.info(f"Message too long ({len(text)} chars), splitting into chunks...")

        chunks = self._smart_split_text(text, max_length=4000)
        first_chunk = chunks[0]
        if len(chunks) > 1:
            first_chunk = f"Part 1/{len(chunks)}:\n{first_chunk}"

        success = await self._send_single_message(first_chunk, image)
        if not success:
            return False

        for i, chunk in enumerate(chunks[1:], 2):
            await asyncio.sleep(1)
            chunk_with_header = f"Part {i}/{len(chunks)}:\n{chunk}"
            success = await self._send_single_message(chunk_with_header)
            if not success:
                logger.error(f"Failed to send part {i}/{len(chunks)}")
                return False

        logger.info(f"Successfully sent message in {len(chunks)} parts")
        return True

    def _smart_split_text(self, text: str, max_length: int = 4000) -> list[str]:
        """Split text at natural boundaries."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            if len(current_chunk + paragraph + '\n\n') <= max_length:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if len(paragraph) > max_length:
                    lines = paragraph.split('\n')
                    for line in lines:
                        if len(current_chunk + line + '\n') <= max_length:
                            current_chunk += line + '\n'
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                            if len(line) > max_length:
                                while line:
                                    chunks.append(line[:max_length])
                                    line = line[max_length:]
                                current_chunk = ""
                            else:
                                current_chunk = line + '\n'
                else:
                    current_chunk = paragraph + '\n\n'

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_retry_after(self, error_message: str) -> int:
        """Extract retry_after value from error message."""
        try:
            if "retry after" in error_message.lower():
                parts = error_message.lower().split("retry after")
                if len(parts) > 1:
                    match = re.search(r'\d+', parts[1])
                    if match:
                        return int(match.group())
            return 5
        except:
            return 5

    async def get_channel_id_from_updates(self):
        """Retrieve channel ID from bot updates."""
        try:
            updates = await asyncio.wait_for(
                self.bot.get_updates(timeout=30),
                timeout=60
            )
            for update in updates:
                if update.channel_post:
                    channel_id = update.channel_post.chat.id
                    logger.info(f"Detected channel ID: {channel_id}")
                    return channel_id
            logger.warning("No channel posts found in updates.")
            return None
        except asyncio.TimeoutError:
            logger.error("Timeout while getting updates")
            return None
        except TelegramAPIError as e:
            logger.error(f"Failed to get updates: {e}")
            return None

    async def health_check(self) -> bool:
        """
        Check if the bot can communicate with Telegram API.
        """
        try:
            me = await asyncio.wait_for(self.bot.get_me(), timeout=10)
            logger.info(f"Bot health check passed: @{me.username}")
            return True
        except Exception as e:
            logger.error(f"Bot health check failed: {e}")
            return False

    async def close(self):
        """Close the bot session to release resources."""
        try:
            await asyncio.wait_for(self.bot.session.close(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Timeout while closing bot session")
        except Exception as e:
            logger.error(f"Error closing bot session: {e}")


class TelegramBotManager:
    """Context manager for TelegramBot to ensure proper cleanup."""

    def __init__(self, **kwargs):
        self.bot = None
        self.kwargs = kwargs

    async def __aenter__(self):
        self.bot = TelegramBot(**self.kwargs)
        if not await self.bot.health_check():
            logger.warning("Bot health check failed, but continuing...")
        return self.bot

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.bot:
            await self.bot.close()
