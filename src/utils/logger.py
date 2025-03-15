import logging
import sys

def setup_logger(name):
    """Create and configure a logger.

    Args:
        name (str): Name of the logger.

    Returns:
        logger: Configured logger instance.
    """
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create handlers for logging: one for console and one for file
    c_handler = logging.StreamHandler(sys.stdout)  # Console handler
    f_handler = logging.FileHandler('app.log', mode='a')  # File handler

    # Set levels for handlers
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger