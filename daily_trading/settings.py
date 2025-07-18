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
TELEGRAM_BOT_CHAT_ID = os.environ.get("TELEGRAM_BOT_CHAT_ID")
TELEGRAM_CHANNEL_CHAT_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
