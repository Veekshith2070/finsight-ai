import logging
import sys

def setup_logger(name: str = "finsight") -> logging.Logger:
    """ Configures and returns a standardized logger."""
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if setup_logger is called more than once
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Format [Timestamp] [Level] [Module]: Message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Output directly to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Create a default logger instance ready to import
logger = setup_logger()
