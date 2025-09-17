import os

from loguru import logger

# Make sure logs directory exists
if not os.path.exists("logs"):
    os.mkdir("logs")

# Add file logging with our preferred format
log_format = "[{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}] {message}"

logger.add(
    "logs/running_logs.log",
    format=log_format,
    level="INFO",
)

__all__ = ["logger"]
