import sys
import logging


def get_logger(name, level=logging.INFO):
    # Create logger.
    logger = logging.getLogger(name)

    # Set level.
    logger.setLevel(level)

    # Set up format (level + message).
    utils_logformat = "%(levelname)s: %(message)s"
    formatter = logging.Formatter(utils_logformat)

    # Add handler to log to standard error.
    sth = logging.StreamHandler(stream=sys.stderr)
    sth.setFormatter(formatter)
    logger.addHandler(sth)
    return logger
