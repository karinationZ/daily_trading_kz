import os
import sys

from loguru import logger

logger.remove(0)
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{level}</level> <cyan>{function}:{line}</cyan> - {message}\n"
logger.add(sys.stdout, level="INFO", colorize=True, format=fmt)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
