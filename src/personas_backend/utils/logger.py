"""Project-wide logging utilities."""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name (str): Logger name
        log_dir (str): Directory to store log files

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create the logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger
    logger = logging.getLogger(name)

    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create a file handler
    log_filename = f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Configure related loggers if needed
    for module in ["boto3", "botocore"]:
        mod_logger = logging.getLogger(module)
        mod_logger.setLevel(logging.INFO)
        if not mod_logger.handlers:
            mod_logger.addHandler(file_handler)

    logger.info(f"Logger {name} is set up and ready to log.")
    return logger
