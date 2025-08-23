"""
File used to the define the configurations of logging
"""

import logging

str_to_level = {
    "not_set": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.FATAL
}

def setup_logging(level_str: str) -> None:
    """
    Function used to configure the logging presentation

    Args:
        level_str (str): minimum level that leads to logging.
        Can only be 'not_set', 'debug', 'info', 'warn',
        'warning', 'error', 'critical', 'fatal'

    Returns:
        None
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=str_to_level[level_str]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
