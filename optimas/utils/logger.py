"""Definition of logging utilities."""

import sys
import logging


def get_logger(name, level=logging.INFO) -> logging.Logger:
    """Get a logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int or str, optional
        Logging level, by default logging.INFO

    Returns
    -------
    logging.Logger

    """
    # Create logger.
    logger = logging.getLogger(name)

    # Set level.
    logger.setLevel(level)

    # Set up format.
    formatter = logging.Formatter(
        fmt="[%(levelname)s %(asctime)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # Add handler to log to standard output.
    sth = logging.StreamHandler(stream=sys.stdout)
    sth.setFormatter(formatter)
    logger.addHandler(sth)
    return logger
