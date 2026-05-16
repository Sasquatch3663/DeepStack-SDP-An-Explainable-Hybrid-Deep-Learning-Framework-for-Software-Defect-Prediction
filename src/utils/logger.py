"""
logger.py

Provides a centralized logging utility for the entire project.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str = "SDP_Logger", log_dir: str = "reports/logs") -> logging.Logger:
    """
    Create and return a logger instance.

    Args:
        name (str): Logger name
        log_dir (str): Directory to store logs

    Returns:
        logging.Logger
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Standalone test
if __name__ == "__main__":
    logger = get_logger()
    logger.info("Logger initialized successfully.")
    logger.debug("Debug message for testing.")