import os
import sys

import dotenv
from loguru import logger

dotenv.load_dotenv()
logger.remove()
# Define Format with {extra[bot_id]}
fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{function}:{line}</cyan> | "
    "<bold><magenta>{extra[bot_id]:<10}</magenta></bold> - "  # Magenta, no brackets, pipe separator
    "{message}"
)

# 3. Add to stdout
logger.add(sys.stdout, level="INFO", colorize=True, format=fmt)

# 4. Bind a default empty ID for global system logs (so they don't error out)
logger = logger.bind(bot_id="[SYSTEM]")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Telegram configuration from environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_API_KEY")
TELEGRAM_BOT_CHAT_ID = os.environ.get("TELEGRAM_BOT_CHAT_ID")

TELEGRAM_MAIN_CHANNEL_CHAT_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
TELEGRAM_DEBUG_CHANNEL_ID = os.environ.get("TELEGRAM_DEBUG_CHANNEL_ID")

TELEGRAM_NY_MONEY_MAKER_CHANNEL_ID = os.environ.get("TELEGRAM_NY_MONEY_MAKER_CHANNEL_ID")
TELEGRAM_LONDON_MONEY_MAKER_CHANNEL_ID = os.environ.get("TELEGRAM_LONDON_MONEY_MAKER_CHANNEL_ID")
TELEGRAM_ASIA_MONEY_MAKER_CHANNEL_ID = os.environ.get("TELEGRAM_ASIA_MONEY_MAKER_CHANNEL_ID")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CRYPTO_PANIC_TOKEN = os.environ.get("CRYPTO_PANIC_TOKEN")
USE_AI = os.environ.get("USE_AI")

IBKR_PORT = os.environ.get("IBKR_PORT", 4002)
