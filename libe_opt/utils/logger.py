import sys
import logging


def get_logger(name, level=logging.INFO):
    # Create logger.
    logger = logging.getLogger(name)

    # Set level.
    logger.setLevel(level)

    # Set up format.
    formatter = logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S")

    # Add handler to log to standard error.
    sth = logging.StreamHandler(stream=sys.stderr)
    sth.setFormatter(formatter)
    logger.addHandler(sth)
    return logger
