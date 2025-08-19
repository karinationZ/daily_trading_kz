import os
import sys

import dotenv

dotenv.load_dotenv()
from loguru import logger

logger.remove(0)
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> <cyan>{function}:{line}</cyan> - {message}\n"
logger.add(sys.stdout, level="INFO", colorize=True, format=fmt)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Telegram configuration from environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_API_KEY")
TELEGRAM_MAIN_CHANNEL_CHAT_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
TELEGRAM_DEBUG_CHANNEL_ID = os.environ.get("TELEGRAM_DEBUG_CHANNEL_ID")
TELEGRAM_MY_DEBUG_BOT_ID = os.environ.get("TELEGRAM_BOT_CHAT_ID")


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CRYPTO_PANIC_TOKEN = os.environ.get("CRYPTO_PANIC_TOKEN")
USE_AI = os.environ.get("USE_AI")
