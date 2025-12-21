"""Logging configuration."""

from loguru import logger
import sys
from pathlib import Path


def setup_logger(
    log_file: str = "logs/app.log",
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days"
):
    """
    Setup loguru logger with file and console handlers.
    
    Args:
        log_file: Path to log file
        level: Logging level
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    logger.info(f"Logger initialized: {log_file}")


# Initialize default logger
setup_logger()


if __name__ == "__main__":
    # Test logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
